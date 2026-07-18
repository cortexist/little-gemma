# Seminar Q&A: the little-gemma vs llama.cpp performance story

Two questions that get to the heart of the decode/prefill split, with honest
answers. Numbers are the 2026-07-17 state (after the split-K decode fix); the
full tables and methodology are in [benchmarks.md](benchmarks.md), the prefill
decomposition in [prefill-performance-journal.md](prefill-performance-journal.md),
and the upstream comparison in [upstream-llama-study.md](upstream-llama-study.md).

## Q1 — What makes little-gemma up to 27% faster at decoding on the Orin? If there are many small tweaks, what's the most remarkable one?

**The framing first, because it's the actual answer:** decode is
*memory-bandwidth-bound*. The weights cross the bus once per token and that read
dominates — llama's E2B decode on our board runs at **95% of the 102 GB/s
ceiling**. When you are that close to the physical wall, no one can win big on
the matvec itself. So the entire contest is **everything that happens per token
that isn't the weight read** — and on a small model spread over just 8 SMs, that
"everything else" is a large slice of wall-clock. That is the remarkable
inversion: people expect the matmul to be the whole story, and for decode on the
edge it is the part nobody can improve.

We win that slice by doing it lean: the whole decode forward captured as one CUDA
graph (few, simple nodes), GPU argmax so we never download the 262k-wide logit
vector to pick the max — 4 bytes come back instead — norms fused with the
residual add, an f16 KV cache, and split-K attention.

**The most remarkable single lever** — and the freshest, measured this week on
that exact board — is the split-K decode attention, "the split that never
split." The kernel splits the KV walk across blocks to keep the schedulers fed,
but a constant capped the *keys per block* at 1024, so below 1024 tokens of
context it collapsed to one block per head and each block's walk grew linearly
with depth. One number, `SPLIT_KEYS 1024 -> 64`, and E4B went 1.17x -> 1.27x,
with the depth droop going from -14% to -0.6%. It is the cleanest illustration
of the whole principle: decode speed is about *keeping the GPU busy between
weight reads*, not the reads themselves.

**The honest caveat:** this is model-dependent, and we *lose* on E2B (0.92x).
E2B is q4_0; on that path llama's matvec genuinely hits 95% of bandwidth where
ours hits 88%. So "the matmul is a wash" is true for the q4_K models (E4B, 12B)
where we then win on overhead — but on q4_0 their matvec is simply more
bandwidth-efficient, and that is the one decode gap we haven't closed.

## Q2 — What makes little-gemma 20% slower at prefill? If there are many gaps, what's the biggest one, and why can't we do exactly what llama.cpp does to reach parity?

Prefill flips everything: it is a large-batch tensor-core GEMM problem —
*compute-bound*, llama's home turf. We closed it from ~0.2x to 0.8x over the
July campaign and then measured the residual to its floor with nsys on both
stacks. The 12B decomposition (per 929-token prefill, ours vs theirs):

| component | ours | llama | gap |
|---|---:|---:|---:|
| q4_K matmul | 3.27s | 3.00s | 0.27s |
| q6_K matmul | 0.91s | 0.57s | 0.34s |
| **attention** | **0.67s** | **0.08s** | **0.59s** |
| elementwise | 0.73s | ~0.55s | ~0.18s |

**The biggest single gap is attention — their flash-attention prefill is ~8x
ours, ~40% of the whole gap.** Not because the math differs, but because their
flash kernel is an integrated pipeline: cp.async/ldmatrix software pipelining,
fragment-instruction economy (`ldmatrix.x4`), and GQA head-packing, with the
softmax rescaling woven through it.

**Why we can't just do the same** has two layers, and both are real:

1. *It isn't a transplantable slice.* We took the one piece that grafts cleanly
   — GQA packing (+1.7%). The rest we tried and **falsified with wiring proof**:
   ldmatrix fragment economy moved nothing (166.3 vs 168.3), K/V staging swapped
   LDG for LDS one-for-one and the profiler counts both, cp.async depth can't
   help a shared-*read* bottleneck. What's left is barrier cadence and softmax
   serialization — a *structural* floor. The residual on the matmul side is the
   same story: it is llama's hand-tuned instruction schedule hiding within-wave
   shared-memory latency, which is **codegen, not a formula** — there is no
   parameter to copy.

2. *"Exactly the same" means their kernels, and that is the one thing we won't
   ship.* We actually did vendor llama's MMQ once — it was faster — and then
   **deliberately deleted it** (-1,829 lines) for our own readable m16n8k32
   kernel, accepting the ~8% end-to-end prefill loss. The project's entire point
   is a dependency-free, teachable runner where every kernel is ours and
   legible. Reaching prefill parity requires adopting ~1,500 lines of dense
   template metaprogramming wholesale — at which point it stops being
   little-gemma. So the 0.8x is a priced, deliberate trade, not a failure to
   figure something out.

One extra Orin-specific twist worth mentioning: at 8 SMs there are no spare warps
to hide that shared-read latency (the "1-CTA-per-SM trap"), so the same codegen
edge that costs llama nothing on a 64-SM desktop costs us more on the edge device
— which is exactly why the residual is a scheduling property, not an algorithmic
one.
