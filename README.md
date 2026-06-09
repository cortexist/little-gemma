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

If the CUDA toolkit is found, CMake also builds a `run-cuda` target:

```
cmake --build build --config Release --target run-cuda
```

Same CLI as `run`. The CPU backend (`model-cpu.c`) and the CUDA backend
(`model-cuda.cu`) implement the same `model.h`; only the compute kernels differ.

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
double as the reference spec for the CUDA versions in `model-cuda.cu`.

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
`model-cuda-common.cuh`; the **matmul is the only thing that differs** between the two
GPU backends, so each is a thin file that includes the header and defines just
`matmul_q`. Diff `model-cuda.cu` against `model-cuda-i8.cu` to see exactly where
the speed comes from.

**`model-cuda.cu` — the readable f32 matmul.** Built up in four steps, each diffed
against the CPU output (byte-identical) before keeping it (E2B tok/s):

1. **matmul on the GPU, the rest on the host** — upload the quantized weights to
   VRAM once; a kernel unpacks each weight row and dots it. (7.9)
2. **activations resident in VRAM** — the whole forward and the KV cache live on
   the device, so only the embedding row (down) and the logits (up) cross the bus. (10)
3. **one warp per output row** instead of one thread — 32× the threads, so the
   small per-layer matmuls actually fill the GPU. (17)
4. **the warp cooperates on each block**, with per-element dequant fused into the
   dot — every lane stays busy and the per-row scratch buffer is gone. (36)

**`model-cuda-i8.cu` — the int8 matmul** (llama.cpp's `mul_mat_vec_q` idea,
simplified). The activation is the same for every output row, so quantize it to
int8 once per matmul (per 32-element group: a scale, the int8 values, and their
sum). The per-row dot is then done in **integer arithmetic** per weight sub-block —
with the float scale applied once per sub-block instead of once per element — and
each group of four products goes through `__dp4a`, the 4-way int8 dot-product
instruction. Because an integer dot is order-independent, this is **byte-identical**
to the scalar version: pure speedup, zero numerical change. (The int8 *activation*
is lossy, like every GPU inference engine, so the greedy path differs slightly from
f32, but the output stays coherent and accurate.)

Decode throughput (single token, all layers on GPU):

| model | f32 (`run-cuda`) | int8+dp4a (`run-cuda-i8`) | llama.cpp CUDA |
|-------|-----------------:|--------------------------:|---------------:|
| E2B   | ~35 tok/s        | ~63 tok/s                 | ~146 tok/s     |
| 12B   | ~15 tok/s        | ~32 tok/s                 | ~64 tok/s      |

So the int8 path is ~48× over the CPU backend (E2B) and closes the gap to
llama.cpp's heavily-tuned kernels from ~4× to ~2×. The remaining ~2× is structural,
and the rest of this section documents two attempts to close it that **did not work** —
kept here because the negative results are more instructive than the wins.

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
Closing the last 2× needs a from-scratch MMQ-style kernel (different tiling, the `q8_1`
activation path, tensor-core-style accumulation) — a large rewrite at odds with this
codebase's goal of staying readable — and is left as future work. Lower-risk wins remain
*outside* the matmul: `quantize_act_kernel` is ~13% of GPU time and redundant (q/k/v
share one activation; gate/up share one), and the bf16 fallback runs lane-0-only.

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
