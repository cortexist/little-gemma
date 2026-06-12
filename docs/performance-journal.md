# The CUDA performance journal

How little-gemma's decode went from 7.9 to 182.5 tok/s (E2B) — and then how the
prompt phase, the context cap, and the kv cache caught up — every step gated on
measured tok/s and on unchanged output, with the failed attempts kept alongside
the wins, because the negative results are as instructive. The architecture this
journal optimizes (two backends sharing one forward, the matmul as the only
seam) is described in the [README](../README.md).

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
~134 and 12B to ~50 tok/s.) The rest of this journal walks the attempts to close the
remaining gap: first two that **did not work**, then those that did — the negative
results are as instructive as the wins.

## What didn't work (and why)

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

## What did work: activation-quant dedup

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

## What did work most: a CUDA graph

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
   nodes themselves never change. (Attention shared memory was fixed at `max_seq`
   floats at this point, which capped context at ~12k tokens; the online-softmax
   rewrite that removed the cap came later — see "The long-context roadmap" below.)
2. **Capture needs the per-thread default stream** (`--default-stream=per-thread`; the
   legacy default stream can't be captured), and no `cudaMalloc`/sync mid-capture (two
   tokens run un-captured first so the activation-scratch allocation is already done).
   Both backends use the same single graph path — `run-cuda` (f32) gains less (~+5%)
   because it is more compute-bound, but the forward stays genuinely *common*.

**Net, with dedup + graph: E2B ~80 tok/s, 12B Q4_K_M ~35 tok/s** — the gap to llama.cpp
CUDA at this point was ~1.8× (from the ~2× dp4a starting point above), all of it matmul
memory efficiency.

## What did work in the matmul after all: wide loads (`model-cuda-i8r.cu`)

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
kernel). The recurring lesson of this whole journal, amended: the cheap, safe wins were
the ones that *removed work* — redundant quantization (dedup), launch overhead (the
graph), per-byte loads and per-block scale untwisting (i8r) — never the ones that
rearranged work.

## The long tail: ten more wins

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
7. **Warp-parallel attention.** A steady-state profile (nsys `--cuda-graph-trace=node`,
   so the graph-replayed kernels are itemized — the default trace only shows warmup)
   put attention at 14.9% and *growing with position*: each score was one thread
   serially dotting a K row (a warp touched 32 different rows — ~1 useful float per
   memory sector), and the whole softmax ran serially on thread 0. Now one **warp**
   per timestep dots with coalesced lane-split loads, and the softmax max/sum are
   block-parallel reductions. **12B +4.9%, E2B +10.4%** at ~200 context — and the
   win grows with context length, which is what vision/audio token streams will need.
8. **Wide scale fetches.** The last loads still issued one byte at a time were the
   q4_K/q5_K sub-block scales (`d_gsm`) and the `d`/`dmin` halves; reading the
   12 scale bytes as 3 words and `d`+`dmin` as one (`d_gsm32`/`d_dm`), plus packing
   the activation's per-group (scale, sum) as a single `float2`, is bit-identical
   and worth ~+1% (12B). A fourth dead-end fell out of the same investigation:
   processing both nibble halves of a q4_K/q5_K sub-block pair in one lane (they
   share the same 32 qs bytes, so it halves the weight loads!) *regressed* ~3-5%
   on both models — the doubled register live-range and halved work-item count
   cost more than the loads saved. Same family as split-k and rows-per-warp:
   in latency-bound kernels, restructuring who-does-what keeps losing to simply
   making each load wider.
9. **Fork/join in the graph.** A graph is a dependency DAG, not a tape — but the
   forward recorded k, v, q (and gate, up) serially, even though k+v read the
   same activation as q and write elsewhere. Recording k+v on a side stream
   between two events makes them genuinely concurrent in the replayed graph: the
   tiny latency-bound k/v matmuls (grid 64 — 12% of one GPU) hide under q for
   free. 12B +2.5%, E2B +2.9%, bit-identical. **The bug that almost shipped:**
   the first version was nondeterministic — the lazy per-tensor weight upload
   (`rweight`) could land *between* the fork events, unordered against the side
   stream, so token 0's k/v occasionally read half-uploaded weights and poisoned
   the KV cache. Five bisection steps (fork without work, work without
   concurrency, hard barriers, pure reorder) cornered it; the fix — repack and
   upload *everything eagerly* before the first forward — also moved that cost
   out of the measured prompt phase. Concurrency bugs don't announce themselves;
   determinism checks (same binary, twice) are part of the gate now.
10. **Greedy pick on the device.** Every token ended with a synchronous download
   of all 262k logits (1 MB) and a CPU scan, serialized with the GPU. A new
   `model_forward_next` runs a 1024-thread argmax kernel and downloads 4 bytes;
   ties break low-index to match the CPU scan exactly, so it is bit-identical.
   The CPU backend implements it as the same scan it always did. **E2B +14.9%,
   12B +4.7%** — the WDDM sync round-trip was worth far more than the bytes.

**Net: E2B ~182 tok/s — 26% *faster* than llama.cpp CUDA (145.0 ± 5.2, re-measured
the same day on the same machine). 12B Q4_K_M ~60 tok/s against llama.cpp's
63.5 ± 0.6: 4.8% behind, on a measurement that pays for longer context than
`llama-bench tg32` does.** The lesson of the
day: profile again after every structural change — and profile the model you
actually care about, in steady state, not just at token 0; each fix exposes the
next bottleneck somewhere new, and repeatedly the big one was *outside* the
kernel everyone stares at.

## The whole arc, in numbers

Decode throughput (tok/s, single-token generation, greedy) after each step, on the same
machine (RTX A5000, Windows/WDDM). Steps 1–4 built up `model-cuda-f32.cu`; steps 5–13 are
the int8 line that became `model-cuda-i8r.cu`. Numbers were measured as each step landed,
in different sessions, so adjacent baselines drift by a tok/s or two (e.g. step 4 f32
re-measured 34.7 later) — each step's delta was verified against its immediate
predecessor at the time. "—" = not measured at that step.

| #  | step                                                 | E2B Q3_K_M  | 12B Q4_K_M |
|----|------------------------------------------------------|------------:|-----------:|
| 0  | CPU, scalar + OpenMP (`run`)                         |         1.3 |       0.32 |
| 1  | matmul on GPU, rest on host                          |         7.9 |          — |
| 2  | whole forward device-resident                        |        10.3 |          — |
| 3  | one warp per output row                              |        16.7 |          — |
| 4  | warp cooperates per block, fused dequant (f32)       |          36 |         16 |
| 5  | int8 activation, integer sub-block dot               |        54.5 |       20.7 |
| 6  | `__dp4a`                                             |          63 |       31.7 |
| 7  | activation-quant dedup                               |         ~65 |        ~33 |
| 8  | CUDA graph (capture once, replay per token)          |        80.5 |       35.2 |
| 9  | wide weight loads + q3/q6 repack (`i8r`)             |        91.5 |       47.5 |
| 10 | 128-bit q4_K/q5_K loads                              |        91.2 |       49.4 |
| 11 | warp-parallel float fallback (PLE bf16)              |       124.3 |        ~49 |
| 12 | `__launch_bounds__` (6 blocks/SM)                    |       126.6 |       49.2 |
| 13 | quantize where born (producer epilogues)             |       134.5 |       50.1 |
| 14 | fused post-norm chain (rmsnorm+add+scale)            |       134.9 |       50.9 |
| 15 | 1024-thread full-width norms                         |       138.8 |       53.3 |
| 16 | warp-parallel attention (coalesced K, par. softmax)  |       153.2 |       55.9 |
| 17 | wide scale fetches + packed (scale,sum) float2       |       154.5 |       56.3 |
| 18 | graph fork/join (k+v beside q, up beside gate)       |       158.9 |       57.7 |
| 19 | greedy argmax on the device (4 B/token, not 1 MB)    |       182.5 |       60.4 |
|    | llama.cpp CUDA, same machine, re-measured at step 19 | 145.0 ± 5.2 | 63.5 ± 0.6 |

One asymmetry to keep in mind when reading the last two rows: `llama-bench tg32`
generates 32 tokens from an empty context (the cheapest attention possible), while
the little-gemma numbers are measured over ~187 generated tokens at positions
23–210, paying attention's per-position growth the whole way. On an identical
workload the comparison shifts slightly further in little-gemma's favor.

## Where things stand, across the whole Gemma 4 family

Same machine, same day, both sides re-measured (little-gemma = `run-cuda-i8r`,
~166–187 generated tokens; llama.cpp = `llama-bench tg32`):

| model | size | params | little-gemma | llama.cpp CUDA | ratio |
|-------|-----:|-------:|-------------:|---------------:|------:|
| E2B Q3_K_M  | 2.35 GiB |  4.65 B | 182.5 | 145.0 ± 5.2 | **1.26×** |
| E4B Q4_K_M  | 4.95 GiB |  7.52 B | 114.4 | 112.9 ± 1.8 | **1.01×** |
| 12B Q4_K_M  | 6.86 GiB | 11.91 B |  60.4 |  63.5 ± 0.6 | 0.95× |

The pattern is the project's thesis in one table. The smaller the model, the more
decode speed is about everything *around* the matmul — launch overhead, sync
round-trips, norms, the PLE path — which 2,000 readable lines can do leanly. The
bigger the model, the more it reduces to one number: sustained DRAM bandwidth
through the quantized matmul, where llama.cpp's arch-tuned kernels still hold a
few percent. The crossover for this codebase currently sits right around E4B.

## The long-context roadmap: four more steps

Decode was fast; everything else still assumed short conversations. The attention
kernel's shared-memory score array capped context at ~12k positions outright, a
prompt token cost a full weight stream just like a generated one (~2 s of prefill
per 256-token image on this machine, far worse on an edge device), and the kv
cache allocated `max_seq` f32 rows for all 35 layers when 20 of them never write
a row and 13 more can never look back past their 512-position window. Four steps,
in dependency order — the first three byte-identical, the last one the project's
single deliberate numerics change. (Perf numbers in this section were measured
with `nvidia-smi --lock-gpu-clocks`; the idle-parked clock swings short runs by
±10% otherwise.)

1. **Online-softmax attention.** Each warp now walks its strided slice of
   timesteps holding a running (max, sum) with per-lane V accumulators in
   registers, rescaling when the max moves; the warps' partials merge once at
   the end. Shared memory drops from `max_seq` floats (the ~12k cap) to
   `nwarp·hd`, independent of context. Verified against a double-precision
   reference: the old and new kernels both sit ~1e-5 from it at every length to
   12,288 (old-vs-new ~1e-6 — pure reassociation), and the new kernel holds
   ~1e-5 at 32,768 and 65,536 where the old one cannot launch at all. **Not
   bit-identical** — reassociation flips greedy near-ties, and the generated
   text visibly changed on all three models — so the gate was relaxed for this
   one step to numeric equivalence plus determinism, with the kernel-level
   harness as the artifact. Faster too: gen E2B +6.3%, E4B +3.6%, 12B +1.6%.
   A companion `model_prefill` skips the head — final norm, the n_vocab×n_embd
   output projection (the model's largest matmul), the argmax sync — for every
   prompt token but the last; the CUDA side keeps two captured graphs
   (layers+head for decode, layers-only for prefill).

2. **Batched prefill.** Prompt tokens don't need their outputs read one at a
   time, so `model_prefill` takes the whole span and the CUDA backends process
   it in 16-token chunks: a `matmul_q_n` seam dots each weight row against all
   16 activation columns in one launch, so the weights cross DRAM **once per
   chunk** instead of once per token (the per-column re-reads hit L1). Each
   column keeps the one-column kernel's accumulation order, so the chunked path
   is **byte-identical** to per-token prefill. On a 2,457-token prompt: E2B
   108 → 371 prompt tok/s (**3.4×**), E4B 76 → 190 (2.5×), 12B 41 → 89 (2.2×);
   a 256-token image's prefill drops from ~2.4 s to ~0.7 s. Chunk size swept
   8/16/32 → 327/369/373; 16 takes ~99% of 32's rate at half the registers.
   **The lesson that shaped the code:** the first draft generalized the decode
   kernels in place (a `rows` parameter, `pos = *d_pos + blockIdx.y`) and decode
   lost 5–10% — *with ptxas reporting identical register counts*. The captured
   graph's latency-bound nodes notice every added instruction. The decode
   kernels are therefore frozen exactly as they were and the chunk path gets its
   own `*_n` entry points, with the big bodies shared through a `__device__`
   function whose row pins to 0 for decode. Decode re-measured equal to within
   0.1% on all three models.

3. **The kv cache stops paying for layers that don't need it.** Two facts about
   Gemma 4's cache were being ignored: layers past `n_kv_start` never write a
   row (they reuse an earlier layer's KV — their pointers are now simply NULL),
   and a sliding-window layer can never look back past its 512-position window,
   so storing `max_seq` rows for it is pure waste. SWA layers now keep a
   **ring** of `sliding_window + 16` rows — position `p` lives in row
   `p % seq`, and the 16 spare rows are load-bearing: a prefill chunk writes
   all of its positions before its queries run, and the pad keeps those writes
   off rows an earlier query in the chunk still needs. Byte-identical, on the
   CPU backend too (its rings are window-exact). The frozen-kernel lesson
   struck a **third** time here: the ring needs one conditional subtract per
   timestep in the attention loop, and adding it to the shared `d_attn` body
   cost the *global* layers 8% at long context — so `RING` became a template
   parameter and the global instantiation keeps its ring-free instruction
   stream, bit-frozen. At an 8K-context server config the cache drops E2B
   852 → 243 MiB and 12B 5,556 → 1,093 MiB; E2B's worst case (131K positions)
   falls from ~10.7 GiB to ~0.5 GiB.

4. **f16 KV on the global layers.** At long context the per-token cost is
   dominated by attention *reads* of the few global full-length KV layers, so
   their rows now store as f16 — capacity and read traffic halved — while the
   SWA rings stay f32 (a few hundred rows save nothing meaningful). This is
   the project's one deliberate numerics change beyond step 1: each KV value
   rounds through f16 once at write, with the CPU's float→half conversion
   bit-matched to the GPU's `cvt.rn.f16.f32` so the backends stay comparable.
   The first attempt read the f16 rows with scalar half loads and *lost* 1.8%
   — attention on this machine is issue-bound, and the added convert
   instructions ate the bandwidth win — flipping to `__half2` paired loads
   (two elements per 32-bit load, the wide-loads lesson yet again) turned it
   into gen +1.3% and prefill +3–4% at 4K context.

## After the roadmap: the encoders learn the same lessons

Multimodal work (see the README) replayed two of this journal's themes in
miniature, so the numbers belong here too. The E2B/E4B vision encoder
(gemma4v, a 16-block ViT — ~1 TFLOP per 266-token image) first ran as host
C: naive loops took **~8 minutes** per image; OpenMP actually linked to the
CUDA targets, cache blocking, file-scoped fast-math (where the *naive* dot
loop vectorizes and a hand-unrolled one defeats MSVC's pattern match), and
hoisting locals out of OpenMP-outlined regions brought it to **14.7 s**; the
same seven-kernel forward on the GPU — with the host path kept in-binary as
numeric oracle (`LG_MEDIA_VERIFY=1`; max |diff| 5.3e-5) — runs it in
**0.8 s**. And the 12B's "thin" encoder-free embedder was quietly re-streaming
its 106 MB f32 patch linear once per patch (the exact disease batched prefill
cures); batching its two matmuls over all patches took it from 3.4 s to
**0.5 s** per image. Ship the readable host implementation first, then offload
with the host as the oracle — that pattern is now load-bearing twice.

## MTP: a draft head that borrows the whole model

Gemma 4's **multi-token prediction** assistant is a tiny transformer (E2B's:
4 blocks, 256 wide, 77M params) that predicts the token *after* next — and
its design is almost parasitic. It owns **no K or V projections at all**:
every block cross-attends straight into the *target's* KV cache (SWA blocks
read the last sliding-window KV layer, the full block the last global one).
Its inputs are the target's own embedding of the freshly chosen token
concatenated with the target's last hidden state; a pre-projection squeezes
that into its width, a post-projection lifts it back out. On the CUDA side
the cross-attention is the frozen `d_attn` template called through two new
wrappers with `pos-1` and `window-1` — a draft for position `pos` sees the
cache exactly as a query at `pos-1` — so no decode entry point changed.

Speculative decoding at **block 2** (one draft per round, deeper blocks
regressed in the fork that pioneered this): draft the successor of the fresh
token, then verify `[token, draft]` as one **B=2 chunk** through the
`matmul_q_n` seam, with the head — for once — evaluated at both rows. Greedy
verification makes the strongest correctness claim available: **the output is
byte-identical to plain greedy decoding**, gated on every change below, plus
run-twice determinism (the multi-graph rule). The draft head itself was
validated the cheap way first: run it as a pure oracle on the CPU backend and
read the acceptance rate — a correct head accepts 70–87% on factual text
(42–58% on prose, where the future is genuinely harder to guess), a wrong one
accepts nothing. It accepted 85.7% on its first run. Both the verify pair and
the draft (~30 launches, position read on-device like decode's `d_pos`)
capture into their own static CUDA graphs.

**The bug this uncovered reaches far beyond MTP.** The B=2 verify should cost
about one decode pass (weights cross DRAM once for the pair — the entire
economics of speculation on a bandwidth-bound device). Measured on the Orin
NX: **2.0×** a decode pass. Step 2 above contains the sentence "the
per-column re-reads hit L1" — and that sentence is *discrete-GPU thinking*.
The chunk matmul's `sub_*` helpers re-load each weight byte once per
activation column; on the Orin the in-place q4_K/q5_K weights are zero-copy
(`cudaHostRegister`, the integrated-GPU path), and **Tegra GPU reads of
host-registered memory are uncached** — every re-read pays full DRAM, so
chunk weight traffic scaled with B. Prefill had been quietly paying up to B×
on the Orin since zero-copy landed. The fix: `sub_*_n<NB>` templates that
load each weight sub-block into registers once and dot all NB columns inside,
float order preserved exactly — byte-identical, decode's subs untouched. The
verify fell to 1.28× a decode pass, **Orin prefill roughly doubled** (~25 →
56 prompt tok/s at 617 tokens), and even the A5000 — whose L1 was supposed to
make re-reads free — gained **+49% prefill** (229 → 341 tok/s). Found with a
20-line `LG_MTP_PROFILE=1` probe that splits the verify into halves with
syncs.

The verdict is the device-scoped story this journal keeps re-learning, now in
one table (story prompt, 256 generated tokens, warm, best of 2; acceptance in
parentheses; `llama-bench tg32` same day, same machine):

**RTX A5000 (Windows/WDDM):**

| model | little-gemma | + MTP | llama.cpp CUDA |
|-------|-------------:|------:|---------------:|
| E2B Q3_K_M | 182.3 | 141.9 (42%) | 148.4 ± 6.6 |
| E4B Q4_K_M | 114.7 | 101.9 (45%) | 116.3 ± 1.3 |
| 12B Q4_K_M |  60.5 |  42.0 (58%) |  64.3 ± 0.3 |

MTP **loses across the board on Windows** — not because the math is wrong
(the output is byte-identical) but because each round needs two D2H syncs the
host cannot avoid (the draft token feeds the verify's inputs), and WDDM
prices every sync in milliseconds against a 5 ms token. The fork that
inspired this work measured the same: on this machine, MTP doesn't pay.

**Jetson Orin NX 16GB (Linux, integrated GPU, zero-copy weights):**

| model | little-gemma | + MTP | llama.cpp CUDA |
|-------|-------------:|------:|---------------:|
| E4B Q4_K_M | 16.80 | **17.76** (49%) | 13.36 ± 0.04 |
| 12B Q4_K_M |  8.27 |   7.79 (51%) |  7.04 ± 0.04 |

On the device that ships, E4B with MTP decodes at **1.33×** llama.cpp — and
the 49%-acceptance prose prompt is the *worst* case; list-style answers
accept 87% and reach 19.8 tok/s (+14.7% over plain). The 12B still loses with
MTP even at ~100% acceptance on easy text: its assistant is 1024 wide with a
537 MB (f16-uploaded) LM head, and a ~33 ms draft against a 121 ms decode
pass eats the margin — keeping the assistant's weights quantized on the
device and routing its big matmuls through the existing int8 machinery is the
known next lever. Per-machine verdicts, per-model verdicts: the same binary,
byte-identical output everywhere, faster only where the hardware says so.

3. **Reuse-layer NULL + sliding-window ring.** The cache was paying ~5× over
   minimum. Layers past `n_kv_start` (20 of E2B's 35) only ever read another
   layer's buffers — they now allocate nothing. Sliding-window layers can never
   attend past `sliding_window` positions — they now keep a ring (`row = p %
   seq`) of `sliding_window + PREFILL_B` rows; the `+PREFILL_B` pad is
   load-bearing, because a prefill chunk writes all 16 positions *before* its
   queries run, and with a window-exact ring the chunk's last write would land
   on the row its first query still needs. Global layers keep full length
   (their `start` is always 0; a full buffer never wraps). Byte-identical,
   including a CPU run to position ~962 through its window-exact ring. An
   8,192-position serve cache: E2B 852 → 292 MiB, 12B 5,556 → 1,236 MiB.
   And the frozen-kernel lesson promptly fired a **third** time: one
   conditional subtract per timestep in the shared attention loop cost ~8% of
   long-context decode, so `d_attn` became a `template<bool RING>` — the
   sliding-window instantiation carries the wrap, the global instantiation
   keeps its previous instruction stream bit for bit.

4. **f16 KV on the global layers.** The one deliberate numerics change, shipped
   with a quality check instead of bit-identity. At a long context the cache —
   capacity and the per-token attention read alike — belongs almost entirely to
   the few global KV-owning layers, so their rows now store as f16: one
   round-to-nearest per value at write (the CPU's converter matches the GPU's
   `cvt.rn` bit for bit), the dot still accumulates f32. The SWA rings stay f32;
   a few hundred rows save nothing meaningful. The instructive part: scalar
   `__half` loads measured *−1.8%* — attention on this GPU is issue-bound at
   these context lengths, and the per-element convert ate the bandwidth saving —
   but loading `__half2` pairs (one 32-bit load per two elements, the same
   make-each-load-wider move as i8r) flipped it to gen +1.3% and prefill +3–4%
   at 4k context, with the win scaling with context and with how
   bandwidth-starved the device is. Text flips greedy ties vs f32-KV as
   expected; quality inspected equivalent; deterministic. The same 8K serve
   cache: E2B 292 → 243 MiB, 12B 1,236 → 1,093 MiB.

Where the cache landed: before the roadmap, E2B at its nominal 131K context
would have needed ~10.7 GiB of f32 KV; it now needs **~0.5 GiB**, dominated by
its two global KV-owning layers — and the attention kernel no longer has an
opinion about how long the context is. One caveat worth carrying forward: every
verdict in this journal is **device-scoped**. This machine's A5000 (~768 GB/s,
launch-latency-prone WDDM) is partly latency- and issue-bound, which is exactly
why batched prefill "only" gained 3.4× and f16 KV needed paired loads to win at
all; on a bandwidth-starved edge device (an Orin NX has ~100 GB/s shared with
the CPU) the same compute-for-bandwidth trades convert at much closer to their
theoretical rates — and the dead ends ruled out above deserve a re-trial there
before being believed.
