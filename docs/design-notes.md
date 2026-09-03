# Design notes

Deliberate non-features and the reasoning behind them.

## On SIMD (AVX2) — intentionally not implemented

The CPU matmul is plain scalar C parallelized with OpenMP; there are **no hand-written
AVX2/FMA intrinsics**. This is deliberate. Hand-vectorizing the CPU kernels would
be throwaway work, because the real target is **CUDA**: on the GPU the parallelism
comes from thousands of threads (replacing OpenMP) and the per-element math runs in
kernels (replacing what SIMD would do). The weights are already stored quantized
and unpacked inside the matmul — exactly the shape a GPU kernel wants — so the
CPU kernels in `model-cpu.c` (`matmul_q`, `rmsnorm`, `rope_neox`, `softmax`, `gelu`)
double as the reference spec for the CUDA versions in `model-cuda-f32.cu`.

The one exception proves the rule: `media.c` (the host vision encoder) compiles
with file-scoped fast-math so the *compiler* may vectorize its reductions —
still no intrinsics, and the naive one-line dot loop turned out to be the
fastest form (a hand-unrolled multi-accumulator version defeats MSVC's
vectorizer pattern match). The LLM side keeps strict fp.

## On mmap — intentionally not used

Most GGUF runners memory-map the model file and let the OS page weights in on
demand. little-gemma instead reads the whole file into RAM up front and refuses to
start if it does not fit. This avoids the complexity — and the silent failure mode —
of paging: with mmap a model that is slightly too large still "loads", then thrashes
as the OS evicts weight pages and re-reads them from disk every token, so the
slowdown is invisible and hard to reason about. The rule here is deliberately
simple: if you have enough memory, you run; if you don't, you get a clear error at
load time instead of a mysterious crawl (`load_gguf` checks the size and bails).

## CPU performance vs llama.cpp (apples-to-apples)

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

## Lines of code

| directory | files | code   |
|-----------|-------|-------:|
| src       | 18    | 10,140 |
| include   | 6     |    567 |

~10,700 lines across `src/` and `include/` (`tools/` not counted). But the
static total is not the number that matters: **the three backends are
mutually exclusive**, so we count by run path — the include closure each
binary actually compiles. The core vendors **nothing** — pure C/CUDA;
media-file decoding lives in `mmcat` in the sibling little-gemma-tools repo.
Every binary is the shared pipeline (GGUF parse, dequant, tokenizer, config,
media orchestration and the PLE/vision/audio embedders, the MTP draft head,
CLI + socket server — 4,382 lines) plus exactly one backend:

| binary        | backend on top of the shared 4,382                                                                    | total |
|---------------|-------------------------------------------------------------------------------------------------------|------:|
| `run`         | `model-cpu.c` + `media-gpu-stub.c`                                                                     | 4,908 |
| `run-cuda`    | `model-cuda.cuh` + `prefill-kernel.cuh` + `mtp-kernel.cuh` + `model-cuda-f32.cu` + `media-kernel.cu`   | 8,282 |
| `run-cuda-i8` | `model-cuda.cuh` + `prefill-kernel.cuh` + `mtp-kernel.cuh` + `model-cuda-i8.cu` + `media-kernel.cu`    | 9,746 |

(`graph.c`/`graph.h`, the teaching tensor/graph layer, are exercised by
`graph_test` only and compile into no run path.) So the program that decodes
ahead of llama.cpp CUDA on the Jetson E4B/12B — multi-turn socket serving,
batched wide-chunk prefill, a ring-buffered f16 KV cache, tensor-core
flash-attention prefill, split-K decode, image and audio understanding, a GPU
vision encoder, an own m16n8k32 tensor-core q4_K/q4_0 prefill kernel, and
byte-identical speculative decoding included — is **about 9,700 lines of
C/CUDA end to end**, tokenizer and all, with no vendored dependency. Of that,
~2,200 lines are the image/audio path (`media.c` + `media-kernel.cu` and their
headers); a text-only int8 build is **~7,500 lines**.

## 2026-09-03 — where the i8 path grew (~6,900 → 9,746)

Since 2026-06-15 (pre-reorganization) the shipped `run-cuda-i8` build went from
~6,900 lines to 9,746 by its include closure — about **+2,850**, in two roughly
equal halves, both functional rather than churn:

- **+1,134 — the E2B/E4B audio path, re-introduced.** `media.c` (562 → 1,070)
  and its GPU kernels (`media-cuda.cu` → `media-kernel.cu`, 381 → 955) plus
  headers. Genuinely new capability — the E-series legacy-encoder audio brought
  back alongside the 12B's encoder-free path, not a refactor.

- **+1,289 — prefill and MTP kernels.** The 2,012-line `model-cuda.cuh` monolith
  split into `model-cuda.cuh` + `prefill-kernel.cuh` + `mtp-kernel.cuh` and grew,
  and the int8 backend (`model-cuda-i8r.cu` → `model-cuda-i8.cu`, 1,108 → 1,602)
  gained the m16n8k32 tensor-core q4_K/q4_0 kernel and the 2-row MTP verify.

**MTP is the win**: byte-identical speculative decode, level-to-ahead of
llama.cpp on the *same* head, and the reduced-vocab head — which llama.cpp has
no equivalent for — faster still; block depth is now a runtime knob whose
optimum lands per model (E2B N=2 → 12B N=4). **Prefill is the hard one** — it
runs through llama.cpp's home turf, arch-tuned tensor-core GEMMs — but the
campaign finally moved the needle on 2026-09-02.

The remaining ~+430 is serving-loop glue (`run.c` 598 → 842: MTP serve
integration, streaming/barge-in, the think budget) and GGUF/host odds and ends.
The July monolith split changed structure, not net lines — the growth is real
encoder and kernel code.
