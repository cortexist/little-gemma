# Seminar Q&A: the little-gemma vs llama.cpp performance story

Two questions that get to the heart of the decode/prefill split, with honest
answers. Numbers are the 2026-07-17 state (after the split-K decode fix); the
full tables and methodology are in [benchmarks.md](benchmarks.md), the prefill
decomposition in [prefill-performance-journal.md](prefill-performance-journal.md),
and the upstream comparison in [upstream-llama-study.md](upstream-llama-study.md).

## Q1 — What makes little-gemma up to 27% faster at decoding on the Orin? If there are many small tweaks, what's the most remarkable one?

**The framing:** decode is *memory-bandwidth-bound* — each token reads the
model's weights once and that read dominates. But here is the part that isn't
obvious: **on 8 SMs, hitting the bandwidth wall is not automatic.** Whether a
batch-1 matvec kernel actually *saturates* the bus depends on how many
instructions it spends per weight — and that is exactly where we beat llama.

**The measurement, drilled to the bottom** (`ncu`, same E4B decode, same board,
the single biggest matvec ≈ 44% of decode): our kernel runs at **84.5% of peak
memory bandwidth**; llama's at **45.5%**. Same bytes, same 102 GB/s bus — we
reach it, they leave it half-idle. Across all the matvecs, ours sit at
**67–84%**, llama's at **35–48%**. That gap *is* the 27%.

**Why the bus sits half-idle for llama, and not for us** — ncu names it by which
resource each kernel runs out of first. On llama's kernel, SM (compute)
throughput (**60%**) is *higher* than its memory throughput (45%): it is
**compute-bound** — with only 8 SMs it spends its cycles dequantizing and issuing
narrow loads and can't fire memory requests fast enough to fill the bus. On ours,
memory (**84%**) sits far above SM (**27%**): we're **memory-bound**, the SMs
mostly idle *waiting on the bus* — which is precisely what you want when the bus
is the bottleneck. The cause is our **wide int8 weight loads** — one aligned
16-byte load per dp4a group instead of a fistful of byte loads — so 8 SMs need
far fewer instructions per weight and *can* keep the bus full. llama's
`mul_mat_vec_q` needs more, and at 8 SMs it runs out of SM throughput first.

![Decode matvec memory-bus utilization on the Orin (8 SMs): our matmul_i8r hits
84.5% of peak bandwidth and is memory-bound (SM 27%); llama's mul_mat_vec_q hits
only 45.5% and is compute-bound (SM 60%) — it runs out of instruction throughput
before it fills the bus. Wide int8 loads are why.](fig-decode-membus.svg)

**Why it's Orin-specific** (we're at *parity* on the A5000, 0.99×): the desktop
card has **64 SMs** — 8× the instruction-issue capacity — so llama has plenty to
both dequantize *and* fill its bus; both kernels saturate and we tie. Instruction
efficiency only becomes the binding constraint when SMs are scarce. It also
explains the model-size trend — **E4B 1.27×, 12B 1.12×** — the 12B's bigger
matmuls put more rows on each SM, so even llama's kernel edges closer to
saturation and our lead narrows.

**The most remarkable *recent* lever** is a separate story, and it's what took us
from 1.17× to 1.27× this week. Our decode attention had a *self-inflicted* droop:
the split-K kernel capped *keys per block* at 1024, so below 1024 tokens of
context it collapsed to one block per head and each block's KV walk grew linearly
with depth (−14% shallow→deep). That drag had been *masking* part of the matvec
lead — we were already 1.17× *with* the broken attention. One constant,
`SPLIT_KEYS 1024 → 64`, made attention depth-flat (−0.6%, matching llama), and
the full matvec lead showed through: 1.27×. So the split-K fix did not *create*
the lead — it stopped our own attention from eating it back.

![Decode attention dataflow: the old code capped the block count and grew the
per-block KV walk with depth (−14% droop); the fix (and llama.cpp) cap the walk
and grow the block count, filling the device — flat with
depth.](fig-decode-attention.svg)

**The honest caveat:** the kernel edge is q4_K-specific, and it *inverts* on E2B
(0.92×). E2B QAT is q4_0, and there llama's matvec is the more bus-efficient one
(it reaches ~95% where ours reaches ~88%) — the one decode path where their
kernel saturates the Orin better than ours.

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

**And we have the measurement, not just the argument.** Earlier in the project
(before the July prefill overhaul) we did the maximal version of "do exactly the
same": a **faithful, line-by-line port of llama.cpp's q4_K MMQ kernel** — their
`ldmatrix`/`mma` primitives, their `load_tiles_q4_K` interleave, their
`vec_dot`, validated against an f64 reference to 3.6e-6. On the Orin, E4B
prefill with that port ran **~255 tok/s; our own kernel at the time, ~271; and
llama.cpp itself, 498.** Having llama's *exact matmul* left the gap at ~2×
(0.51×) — and the port was even a touch *slower* than our own kernel, because
scoped to the matmul it drops the parts that actually carry llama's speed
(mmq_x autotune, stream-K balancing, cp.async overlap). The lesson was decisive:
the matmul was never the gap — it's ~half of prefill, and **attention plus the
pipeline are the other half** (the nsys split above puts attention alone at
~40%). That result is *why* we stopped porting their kernel and instead rebuilt
attention (the flash kernel) and the whole prefill pipeline — which is what took
prefill from that ~0.5× to today's **0.8×**. So "why can't we do exactly what
llama.cpp does?" answers itself: we did, line for line, and it wasn't the kernel.

One extra Orin-specific twist worth mentioning: at 8 SMs there are no spare warps
to hide that shared-read latency (the "1-CTA-per-SM trap"), so the same codegen
edge that costs llama nothing on a 64-SM desktop costs us more on the edge device
— which is exactly why the residual is a scheduling property, not an algorithmic
one.

![Prefill flash attention dataflow: identical flash loop on both stacks (load K,
QKᵀ, online softmax, load V, accumulate PV); the ~8× gap is entirely inside the
per-block load+MMA step, where llama uses cp.async pipelining, ldmatrix.x4, and
GQA packing while we do manual shared reads — codegen, not a transplantable
slice.](fig-prefill-flash.svg)
