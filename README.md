# Little Gemma

A small, from-scratch C program that loads a model stored in **GGUF** and runs it —
written to *teach* how a modern LLM actually executes, in the spirit of Karpathy's
`llama2.c` but covering a current model (Gemma) and aimed at **CUDA**.

It is a complete pipeline: parse GGUF → BPE tokenize → run the transformer →
generate text. Every stage was validated bit-for-bit against `llama.cpp`.

```
text ──► tokenizer ──► token ids ──► forward ──► logits ──► argmax ──► next token
                          ▲                                                │
                          ╰──────────────── append, repeat ◄───────────────╯
```

## Architecture

```
                      ┌──────────────┐
                      │    run.c     │   CLI: -m <model> [-p "prompt"]
                      └──────┬───────┘
         ╭───────────────────┼───────────────────────────────╮
         ▼                   ▼                               ▼
 ┌───────────────┐   ┌────────────────┐   ┌──────────────────────────────────────┐
 │  tokenizer.c  │   │    model.c     │   │               gguf.c                 │
 │  BPE text↔ids │   │ config+forward │   │ parse → ctx (hdr, kv, tensors, data) │
 └───────────────┘   └───────┬────────┘   └──────────────────┬───────────────────┘
                             │                               │  
                             ▼                               │
                ┌──────────────────────────┐                 │ borrows
                │          quant.c         │◄────────────────╯ quantized
                │ dequantize q3_K/q4_K/... │  weights
                └──────────────────────────┘

    graph.c — a minimal tensor/graph layer (matmul, rmsnorm, softmax, …)
            kept as the "what a compute graph is" teaching reference.
```

**Layering:** GGUF/ggml jargon stays in the lower layers (`gguf.c`, `quant.c`);
the model layer (`model.c`) reads like plain transformer code. The file is read
fully into RAM (no mmap — it errors out rather than silently paging), weights stay
quantized there, and each weight row is unpacked to f32 on the fly during matmul.

### What `model_forward` computes (Gemma 4 / E2B)

```
embed(token) × √d
for each of 35 layers:
    ├─ attention:  rmsnorm → Q,K,V → per-head Q/K-norm → NeoX RoPE
    │              → GQA (8 q-heads, 1 kv-head) with sliding-window OR global mask
    │              → KV cache (layers ≥15 reuse an earlier layer's KV)
    │              → output proj → post-norm → residual
    ├─ feed-forward (GeGLU):  rmsnorm → gelu(gate)·up → down → post-norm → residual
    │              (elastic FFN: width 6144 for layers 0–14, 12288 for 15–34)
    └─ per-layer input (PLE) + per-layer output scale
final rmsnorm → tied logits (× token_embd) → softcap
```

## How to Build

```
cmake -S . -B build
cmake --build build
```

### Release build (recommended — ~3.4× faster than Debug)

```
cmake --build build --config Release
```

Threading is via OpenMP (auto-detected by CMake); the matmul scales across cores.

### CUDA build (optional)

If the CUDA toolkit is found, CMake also builds `run-cuda` (readable f32 matmul)
and `run-cuda-i8r` (int8 matmul, the fast one):

```
cmake --build build --config Release --target run-cuda-i8r
```

Same CLI as `run`. The CPU backend (`model-cpu.c`) and the CUDA backends
(`model-cuda-f32.cu`, `model-cuda-i8r.cu`) implement the same `model.h`; only the compute kernels differ.

## Usage

The CLI is just two flags:

```
run -m <model.gguf> [-p "prompt"]
```

- `-m` only → prints the GGUF dump + config, then exits.
- `-m` + `-p` → also tokenizes the prompt, generates, and reports tok/s.

```
> run -m model.gguf -p "The capital of France is"
The capital of France is Paris.
prompt: 6 tokens in 4.68s (1.28 tok/s)
gen:    2 tokens in 1.48s (1.35 tok/s)
```

## On SIMD (AVX2) — intentionally not implemented

The CPU matmul is plain scalar C parallelized with OpenMP; there are **no hand-written
AVX2/FMA intrinsics**. This is deliberate. Hand-vectorizing the CPU kernels would
be throwaway work, because the real target is **CUDA**: on the GPU the parallelism
comes from thousands of threads (replacing OpenMP) and the per-element math runs in
kernels (replacing what SIMD would do). The weights are already stored quantized
and unpacked inside the matmul — exactly the shape a GPU kernel wants — so the
CPU kernels in `model-cpu.c` (`matmul_q`, `rmsnorm`, `rope_neox`, `softmax`, `gelu`)
double as the reference spec for the CUDA versions in `model-cuda-f32.cu`.

## On mmap — intentionally not used

Most GGUF runners memory-map the model file and let the OS page weights in on
demand. little-gemma instead reads the whole file into RAM up front and refuses to
start if it does not fit. This avoids the complexity — and the silent failure mode —
of paging: with mmap a model that is slightly too large still "loads", then thrashes
as the OS evicts weight pages and re-reads them from disk every token, so the
slowdown is invisible and hard to reason about. The rule here is deliberately
simple: if you have enough memory, you run; if you don't, you get a clear error at
load time instead of a mysterious crawl (`load_gguf` checks the size and bails).

## On CUDA

The whole forward, the kv cache, and every non-matmul kernel live in
`model-cuda.cuh`; the **matmul is the only thing that differs** between the two
GPU backends, so each is a thin file that includes the header and defines just
`matmul_q`. Diff `model-cuda-f32.cu` against `model-cuda-i8r.cu` to see exactly
where the speed comes from.

**`model-cuda-f32.cu` — the readable f32 matmul.** Built up in four steps, each diffed
against the CPU output (byte-identical) before keeping it (E2B tok/s):

1. **matmul on the GPU, the rest on the host** — upload the quantized weights to
   VRAM once; a kernel unpacks each weight row and dots it. (7.9)
2. **activations resident in VRAM** — the whole forward and the KV cache live on
   the device, so only the embedding row (down) and the logits (up) cross the bus. (10)
3. **one warp per output row** instead of one thread — 32× the threads, so the
   small per-layer matmuls actually fill the GPU. (17)
4. **the warp cooperates on each block**, with per-element dequant fused into the
   dot — every lane stays busy and the per-row scratch buffer is gone. (36)

**`model-cuda-i8r.cu` — the int8 matmul** (llama.cpp's `mul_mat_vec_q` idea,
simplified). The activation is the same for every output row, so quantize it to
int8 once per matmul (per 32-element group: a scale, the int8 values, and their
sum). The per-row dot is then done in **integer arithmetic** per weight sub-block —
with the float scale applied once per sub-block instead of once per element — and
each group of four products goes through `__dp4a`, the 4-way int8 dot-product
instruction. Because an integer dot is order-independent, this is **byte-identical**
to the scalar version: pure speedup, zero numerical change. (The int8 *activation*
is lossy, like every GPU inference engine, so the greedy path differs slightly from
f32, but the output stays coherent and accurate.) The int8 idea first shipped as a
byte-load backend, `model-cuda-i8.cu`; the wide-load rework documented below replaced
it — bit-identical output, strictly faster — and the original now lives in git history.

Decode throughput at that first int8 milestone (single token, all layers on GPU):

| model | f32 (`run-cuda`) | int8+dp4a | llama.cpp CUDA |
|-------|-----------------:|----------:|---------------:|
| E2B   | ~35 tok/s        | ~63 tok/s | ~146 tok/s     |
| 12B   | ~15 tok/s        | ~32 tok/s | ~64 tok/s      |

So the int8 path is ~48× over the CPU backend (E2B) and closes the gap to
llama.cpp's heavily-tuned kernels from ~4× to ~2×. (Those `int8+dp4a` numbers are the
starting point; the optimizations documented below — activation-quant dedup, a CUDA
graph, wide weight loads, and the long-tail wins (`run-cuda-i8r`) — then take E2B to
~134 and 12B to ~50 tok/s.) The rest of this section walks the attempts to close the
remaining gap: first two that **did not work**, then those that did — the negative
results are as instructive as the wins.

### What didn't work (and why)

Profiling the matmul (`matmul_i8_kernel`, one warp per output row) with Nsight Compute
on an RTX A5000 shows it runs at only ~25% of peak memory bandwidth, with **two
distinct bottleneck regimes**:

- **Large matmuls** (`lm_head`, late-layer ffn): the load/store unit's issue queue is
  saturated (`lg_throttle`) at ~88% occupancy — too many tiny, *uncoalesced* byte loads.
- **Small matmuls** (most of them, `m ≤ 2048`): bound by raw memory latency
  (`long_scoreboard`) at only ~46% occupancy — one warp per row underfills the GPU.

**Attempt 1 — coalesced loads (`mul_mat_vec_q`-style).** Remap so consecutive lanes read
consecutive bytes (`((uint32_t*)qs)[lane]`) instead of scattered sub-blocks. This *worked*
at the hardware level: global load efficiency went 7→66%, sectors/request 16.9→1.57,
`lg_throttle` essentially vanished. But **tok/s got slightly worse** (E2B 64→62, 12B
Q4_K_M 31→29) across three variants (per-element scaling; segmented reduction; scales
hoisted via shuffle). The catch: the matmuls that are cleanly coalesce-able (q4_K, whose
144-byte blocks are 16-aligned) are the *small, latency-bound* ones — after coalescing,
`long_scoreboard` still dominates at 46% occupancy, so the duration doesn't move. The
large matmuls that *would* benefit are q3_K, whose 110-byte block stride isn't 4-aligned,
so `uint32` loads of its `qs` fault. Lesson: **load efficiency is not the objective
function — wall-clock is.** It improved 9× while tok/s regressed, three times.

**Attempt 2 — split-k for occupancy.** Have `nsplit` warps cooperate on each row (each
summing a strided subset of the k-blocks, combined in shared memory) to raise occupancy
on the small matmuls. Also regressed (uniform `nsplit=2`: E2B 64→60; adaptive, leaving
large matmuls untouched: 60). The split (small) matmuls got *slower*: at a few k-blocks
per row, the fixed per-warp overhead (setup, warp-reduce, shared-mem combine) outweighs
the latency hidden by extra warps. So occupancy wasn't really the limiter at these sizes.

The takeaway: the one-warp-per-row kernel is near a local optimum for incremental changes.
Closing the matmul's remaining ~2× (it runs at ~25% of peak bandwidth) needs a fully tuned
`mul_mat_vec_q` (MMVQ) — coalesced vectorized loads that saturate bandwidth, multiple rows
per warp for latency hiding, arch-specific thread mapping. Decode is batch-1 and
bandwidth-bound, so the int8 *tensor cores* that accelerate llama.cpp's batched matrix×matrix
kernel (`mul_mat_q`, MMQ — used for prompt processing) don't apply here; the win is memory
efficiency, not compute. A full rewrite stayed off the table, but the next two wins were
elsewhere entirely — and then a thin slice of the MMVQ idea (just its wide loads, none of
its restructuring) turned out to fit in ~30 changed lines: `model-cuda-i8r.cu`, below.

### What did work: activation-quant dedup

The win that *wasn't* in the matmul. `quantize_act_kernel` (which quantizes the activation
to int8 before each matmul) was ~13% of GPU time and mostly **redundant**: the forward
hands the *same* activation vector to q/k/v, and another to gate/up, re-quantizing it each
time. `matmul_q_same` skips the quantize and reuses the previous result; the k/v and up
call sites use it. Bit-identical output, and **+2–3% (E2B) / +4% (12B Q4_K_M)** for a few
lines. The lesson worth keeping: the cheap, safe win was *removing redundant work
around* the matmul, not micro-optimizing the matmul itself. (Both threads of this
paragraph pay off again later: `matmul_q_same` was eventually generalized into producer
epilogues that quantize *every* activation exactly once, and the lane-0-only bf16
fallback flagged here as "the remaining lever" turned out to be the single biggest win
of the whole effort — see "The long tail" below.)

### What did work most: a CUDA graph

The biggest single win, and it isn't in any kernel. Profiling showed **~30% of each
token's wall-clock is GPU-*idle*** — the forward issues ~1,000 kernel launches per token
(35 layers × ~29 kernels), and on Windows/WDDM the per-launch latency leaves the GPU
waiting between them (GPU-busy ~10.6 ms vs ~15.4 ms wall → a ~94 tok/s ceiling). The fix
is to **capture the forward into a CUDA graph once and replay it** — ~1,000 launches
collapse into a single graph launch. Result: **E2B 67→80 tok/s (+19%), 12B Q4_K_M
32→35 (+13% over the un-graphed baseline)**, bit-identical output. E2B gains more because
it is more launch-bound; the larger 12B spends relatively more time in actual compute.

Two things make this work:

1. **The graph must be static.** A naive "re-capture every token + `cudaGraphExecUpdate`"
   was *flat* — recording 1,000 nodes costs as much as launching them. The win requires
   capture-*once*, replay-*many*. So every per-token-varying input — the position (which
   drives the KV-cache write offset, the RoPE angle, and the attention range) — is read
   on-device from a one-int `d_pos` buffer that the host updates before each launch. The
   nodes themselves never change. (Attention shared memory is fixed at `max_seq` floats,
   which caps context at ~12k tokens without an online-softmax rewrite.)
2. **Capture needs the per-thread default stream** (`--default-stream=per-thread`; the
   legacy default stream can't be captured), and no `cudaMalloc`/sync mid-capture (two
   tokens run un-captured first so the activation-scratch allocation is already done).
   Both backends use the same single graph path — `run-cuda` (f32) gains less (~+5%)
   because it is more compute-bound, but the forward stays genuinely *common*.

**Net, with dedup + graph: E2B ~80 tok/s, 12B Q4_K_M ~35 tok/s** — the gap to llama.cpp
CUDA at this point was ~1.8× (from the ~2× dp4a starting point above), all of it matmul
memory efficiency.

### What did work in the matmul after all: wide loads (`model-cuda-i8r.cu`)

The matmul win that finally landed — `run-cuda-i8r` ("r" = repacked) — and the failed
attempts above pointed straight at it. The kernel keeps the byte-load backend's exact
structure (one warp per row, same lane→sub-block mapping, same integer dots, same float
order — the output was verified **bit-identical** against `run-cuda-i8` on both models
before that backend was retired); the only change is *how the weight bytes are read*. Each `__dp4a` group was built from 4–16 single-byte loads
plus shift/or assembly; now it is **one aligned `uint32` load** with the packed values
extracted by SIMD-in-word masks (all four q4_K nibbles at once: `q32 & 0x0F0F0F0F`).
That is 4–16× fewer load instructions — aimed at the `lg_throttle` LSU saturation the
profile showed on the large matmuls, and it shortens the stall chain on the small ones too.

Wide loads need 4-byte alignment. q4_K (144 B) and q5_K (176 B) blocks already have it
and are read in place; q3_K (110 B) and q6_K (210 B) do not, so those tensors are
**repacked once on the host at upload** into padded, 4-aligned twins (116/212 B, +5.5%/+1%
bytes). The repack also pre-unpacks q3_K's twisted 6-bit scales into flat int8 — the file
format's layout quirks are paid once per model, not per sub-block per row per token.

Result: **E2B 80→91.5 tok/s (+14%), 12B Q4_K_M 35→47.5 (+35%)**, bit-identical output.
Why this worked when "coalescing" regressed three times: those variants restructured
*which lane reads what* (and paid for it in divergence and shuffles); this keeps the
proven mapping and only widens each load. The lesson is a sharper version of the old one:
the enemy was never the access *pattern* alone — it was the per-byte load instruction
count. (A fourth lever, 2 rows per warp for extra loads-in-flight on the small matmuls,
was tried on top — uniform, adaptive, and template-specialized — and regressed every
time, joining coalescing-remap and split-k as confirmed dead ends.)

**Net, with dedup + graph + wide loads: E2B ~91 tok/s, 12B Q4_K_M ~47 tok/s** — the gap
to llama.cpp CUDA down to ~1.6× on E2B and ~1.35× on 12B (from ~4× at the first f32
kernel). The recurring lesson of this whole section, amended: the cheap, safe wins were
the ones that *removed work* — redundant quantization (dedup), launch overhead (the
graph), per-byte loads and per-block scale untwisting (i8r) — never the ones that
rearranged work.

### The long tail: six more wins

With the matmul's loads fixed, an nsys re-profile showed the new cost structure:
matmul 64.8% (its small instances still latency-bound), `quantize_act` 13.9%
(~290 launches/token), rmsnorm 10.5%, attention 4.4%. Four changes followed, each
gated on tok/s and on unchanged output:

1. **128-bit loads for q4_K/q5_K.** Their block strides (144/176 B) are 16-byte
   multiples, so one `uint4` load replaces four `uint32` loads. 12B (q4_K-heavy)
   +4%; E2B neutral. Bit-identical.
2. **Warp-parallel float fallback.** The bf16 `per_layer_model_proj` (E2B's PLE
   path) was still dotted by lane 0 alone — ~18M *serial* MACs per token hiding in
   a "fallback". All 32 lanes now split the row, two bf16 per `uint32` load.
   **E2B 91→124 tok/s (+36%)** — the biggest single win of the whole CUDA effort,
   and it wasn't in the quantized matmul at all. (12B has no PLE; unaffected.
   Reassociates an f32 sum, so not bit-identical in principle — the generated text
   was unchanged in practice.)
3. **`__launch_bounds__(256, 6)`** on the matmul kernel caps registers so six
   256-thread blocks (48 warps — the sm_86 maximum) fit per SM: the one occupancy
   lever that helps the latency-bound small matmuls without restructuring them.
   E2B +2%. Bit-identical.
4. **Quantize activations where they are born.** ~290 `quantize_act` launches per
   token re-read vectors some kernel had *just written*. The producer kernels
   (rmsnorm, geglu, add, attention) now take a `struct actq` and quantize their own
   output as an epilogue; the f32 backend passes an empty one and the branch
   vanishes. `matmul_q_same` is gone — every matmul input is pre-quantized by its
   producer, and the embedding (the one activation no kernel produces) gets an
   explicit `act_quantize`. E2B +3.6%, 12B +1.7%, bit-identical.
5. **Fused post-norm chain.** Every sub-layer ends "rmsnorm → residual add →
   (output scale)" — up to three kernels and three global-memory round-trips.
   `rmsnorm_add_kernel` does all of it (plus the quantize epilogue) in one. A catch
   worth recording: the naive fusion let nvcc contract the norm-multiply and the
   residual-add into an FMA — different in the last bit, and the greedy path
   *visibly* diverged (163→162 generated tokens). One `__fadd_rn` restores the
   unfused rounding and the bit-identical guarantee. 12B +1.6%; E2B neutral but
   ~100 fewer graph nodes per token.
6. **1024-thread norms.** Profiling the *12B* (not E2B — its 48 dense layers and
   n_embd 3840 weight the costs differently) showed the norms at 16.5% of GPU
   time: every full-width rmsnorm ran as a single 256-thread block — one quarter
   of one SM working, 63 SMs idle. The single-row norms now launch 1024 threads
   (per-head rows stay at 256). Changes the reduction tree, so like the fallback
   fix it is numerically equivalent rather than bit-identical. **12B +4.7%,
   E2B +2.9%.**

**Net: E2B ~139 tok/s, 12B Q4_K_M ~53 tok/s — gaps of 1.05× and 1.20× to llama.cpp
CUDA.** The lesson of the day: profile again after every structural change — and
profile the model you actually care about; each fix exposes the next bottleneck
somewhere new, and twice now the big one was *outside* the kernel everyone stares at.

### The whole arc, in numbers

Decode throughput (tok/s, single-token generation, greedy) after each step, on the same
machine (RTX A5000, Windows/WDDM). Steps 1–4 built up `model-cuda-f32.cu`; steps 5–13 are
the int8 line that became `model-cuda-i8r.cu`. Numbers were measured as each step landed,
in different sessions, so adjacent baselines drift by a tok/s or two (e.g. step 4 f32
re-measured 34.7 later) — each step's delta was verified against its immediate
predecessor at the time. "—" = not measured at that step.

| # | step                                            | E2B Q3_K_M | 12B Q4_K_M |
|---|-------------------------------------------------|-----------:|-----------:|
| 0 | CPU, scalar + OpenMP (`run`)                    |        1.3 |       0.32 |
| 1 | matmul on GPU, rest on host                     |        7.9 |          — |
| 2 | whole forward device-resident                   |       10.3 |          — |
| 3 | one warp per output row                         |       16.7 |          — |
| 4 | warp cooperates per block, fused dequant (f32)  |         36 |         16 |
| 5 | int8 activation, integer sub-block dot          |       54.5 |       20.7 |
| 6 | `__dp4a`                                        |         63 |       31.7 |
| 7 | activation-quant dedup                          |        ~65 |        ~33 |
| 8 | CUDA graph (capture once, replay per token)     |       80.5 |       35.2 |
| 9 | wide weight loads + q3/q6 repack (`i8r`)        |       91.5 |       47.5 |
| 10 | 128-bit q4_K/q5_K loads                        |       91.2 |       49.4 |
| 11 | warp-parallel float fallback (PLE bf16)        |      124.3 |        ~49 |
| 12 | `__launch_bounds__` (6 blocks/SM)              |      126.6 |       49.2 |
| 13 | quantize where born (producer epilogues)       |      134.5 |       50.1 |
| 14 | fused post-norm chain (rmsnorm+add+scale)      |      134.9 |       50.9 |
| 15 | 1024-thread full-width norms                   |      138.8 |       53.3 |
|   | llama.cpp CUDA, same machine (reference)        |        146 |         64 |

## Performance vs llama.cpp (CPU, apples-to-apples)

Both **no CUDA, no SIMD intrinsics, 12 threads**, single-token generation:

| build                                         | generation |
|-----------------------------------------------|-----------:|
| little-gemma (scalar + OpenMP)                | ~1.3 tok/s |
| llama.cpp (SIMD off, CUDA off, `llama-bench`) | 2.33 tok/s |

Only **~1.8×** apart. The gap is *algorithm*, not SIMD: llama.cpp quantizes the
activation to int8 once and does an integer `vec_dot` against the quantized
weights (never materializing f32), plus better cache blocking. little-gemma
dequantizes each weight row to f32 then does an f32 dot. With AVX2 *on*,
llama.cpp reaches ~10–30 tok/s on the same machine — that headroom is what CUDA
is for.

## Lines of Code

| directory | files | code  | comment | blank | total |
|-----------|-------|-------|---------|-------|-------|
| src       | 10    | 2,115 | 200     | 290   | 2,605 |
| include   | 5     | 209   | 82      | 57    | 348   |

## Validation

Built against a CPU `llama.cpp` as an oracle (`llama-eval-callback` dumps every
intermediate tensor): dequantization is bit-exact vs the `gguf` Python package;
the forward pass matches an independent NumPy f32 reference and llama.cpp's logits
(within the f32-vs-quantized-matmul gap); the tokenizer matches `llama-tokenize`
exactly. `test/graph_test.c` (a CTest target) checks the graph kernels.

## License

MIT — see [LICENSE](LICENSE).
