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

| directory | files | code  |
|-----------|-------|------:|
| src       | 16    | 5,948 |
| include   | 6     |   254 |

~6,200 lines of code in the repository (`tools/` not counted). The core
vendors **nothing** — pure C/CUDA; media-file decoding lives in `mmcat` in
the sibling little-gemma-tools repo. The backends are mutually exclusive, so
no single program is anywhere near that. Each binary is the shared pipeline
(GGUF parse, dequant, tokenizer, config, multimodal embedders, the MTP draft
head, CLI + socket server — 2,711 lines) plus exactly one backend:

| binary        | backend on top of the shared 2,711                         | code lines |
|---------------|------------------------------------------------------------|-----------:|
| `run`         | `model-cpu.c`                                              |      3,079 |
| `run-cuda`    | `model-cuda.cuh` + `model-cuda-f32.cu` + `media-kernel.cu` |      5,128 |
| `run-cuda-i8` | `model-cuda.cuh` + `model-cuda-i8.cu` + `media-kernel.cu`  |      6,058 |

(`graph.c`/`graph.h`, the teaching tensor/graph layer, are exercised by
`graph_test` only.) So the program that decodes ahead of llama.cpp CUDA on the
Jetson E4B/12B — multi-turn socket serving, batched wide-chunk
prefill, a ring-buffered f16 KV cache, tensor-core flash-attention prefill,
split-K decode, image and audio understanding, a GPU vision encoder, an own
m16n8k32 tensor-core q4_K/q4_0 prefill kernel, and byte-identical speculative
decoding included — is **about 6,100 lines of C end to end**, tokenizer and
all, with no vendored dependency.
