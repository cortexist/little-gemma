# Stream-K and the q4_K prefill matmul: a research case

This documents why little-gemma ships its **own** `m16n8k32` q4_K prefill kernel
(`matmul_q4k_mma_kernel` in `src/model-cuda-i8r.cu`) rather than vendoring
llama.cpp's `mul_mat_q`, and the experiments that settled it. The vendored kernel
and every experimental variant are preserved on branches (see the end) as the
research record; `main` carries only the clean own kernel.

## The question

For a while we vendored llama.cpp's q4_K MMQ kernel (de-baggaged to NVIDIA
Ampere+ tensor-core-only, ~1,400 lines, in the now-deleted `src/llama-mmq.cu`).
It was faster than every from-scratch kernel we'd written. The question: can our
own clean kernel match it, so we can drop the vendored dependency?

## Baseline: own vs vendored

Our own kernel mirrors the dp4a `sub_q4_K` math but does the 32-element integer
dot with one `mma.sync.m16n8k32.s8` per q4_K sub-block (K=32 aligns exactly with
q4_K's per-32 scale granule and our per-32 activation groups). Minimal signature,
COLS-templated, q4_K stays compressed and is dequantized in-kernel. Fragment
mapping validated in `.scratch/mma4_test.cu` (0 mismatches vs a CPU reference).

| device | own kernel vs vendored llama |
|--------|------------------------------|
| A5000 (sm_86, 64 SMs) | **parity** — own ≤ llama on the q4_K matmul |
| Orin (sm_87, **8 SMs**) | llama ~30% faster on the **matmul component** |

The A5000 parity is what made dropping the dependency look plausible. But **the
journal's "llama wins" was always Orin-scoped, and Orin is the target.** The 8-SM
part is where the gap lives. So the question became: *why* is llama faster on
Orin, and can we close it?

## Diagnosis (ncu on Orin)

The gap is **not** occupancy (both ~16.5%, one CTA/SM — the 8-SM "1-CTA trap"),
**not** memory bandwidth (~46% both), and **not** the math. It is **issue
efficiency under SM starvation**:

- own: 44.6% SM throughput, 4.4 cycles/issued-instruction
- llama: 57.5% SM throughput, 3.46 cycles/issued-instruction

ncu's top stall for our kernel is a **scoreboard dependency on a shared-memory
operation** — warps waiting on the LDS that loads MMA fragments out of the
`sA`/`sB` shared tiles (≈31% of cycles on ffn, 42–45% on the small attn
projections). This is **shared→register read latency**, exposed because 8 SMs at
16.5% occupancy have no spare warps to hide it. llama hides the same latency
better via more instruction-level overlap — its integrated schedule.

## The levers (each measured, each ruled in/out)

**1. `ldmatrix`** (branch `own-q4k-ldmatrix`). Replace the manual shared
fragment reads with the hardware `ldmatrix.sync` collective load. **Result:
neutral-to-worse on Orin; the short-scoreboard stall was unmoved.** The stall
isn't the A-fragment load `ldmatrix` optimizes — exactly the journal's earlier
finding ("not A-load latency, balanced latency-bound"), re-derived.

**2. Split-K, atomic PoC** (branch `own-q4k-splitk`). Split the K-blocks across
`gridDim.z` CTAs to fill the SMs, `atomicAdd` partials. **Net loss
(0.71→0.83–0.95s matmul).** But the per-matmul ncu *confirmed the mechanism*:
genuinely starved matmuls (8%, 25% SM) sped up 15–25% with SM% climbing — while
already-filled ones regressed (over-split + atomic contention). So
wave-starvation is real, but atomics + uniform split are the wrong vehicle.

**3. Stream-K, lock-free** (branch `own-q4k-streamk`). The real thing: distribute
the flattened (tiles × superblocks) work across exactly `nsm` persistent CTAs;
each tile's final superblock is computed by exactly one CTA which writes `dst`
directly (**no atomics** — a tile is never written twice), and the ≤`nsm−1`
boundary tiles write per-CTA scratch slots summed by a backward-walk fixup.
Mirrors llama's protocol, minimalized to q4_K (no MoE/channel/sample). Correct
(matches dp4a at short and long prompts), grid=8 every launch as designed.
**Result: ~neutral end-to-end (slightly slower than plain).** It helps the big
ffn matmuls (−8% component, SM 42→44%) but over-distributes the tiny attn
matmuls (grid=8 is too many CTAs for a few-tile problem → fixup overhead, 5.3%
SM). "Do it only when needed" applies to stream-K itself.

**4. Deep cp.async prefetch** (branch `own-q4k-streamk-pipe`, abandoned). The
hypothesis: stream-K's long continuous per-CTA stream is a better substrate for
deeper pipelining than the old per-tile kernel (whose `LG_Q4K_PIPE3` hit a
register cliff). **Killed by the profile before measuring:** the stall is
shared→register read latency; cp.async only hides global→shared (fill) latency —
it physically cannot speed reads *out of* shared. Wrong lever, same category as
`ldmatrix`.

## Conclusion

Four independent levers all hit the same wall. Even with **perfect SM-fill**
(stream-K's grid=8, zero wave quantization), the ffn tops out at 44% SM vs
llama's 57.5% — the residual is **within-wave shared-read latency**, i.e. llama's
integrated instruction schedule, which is not graftable as a formula (the
journal's long-standing "codegen, not a formula" conclusion, now confirmed for
the m16n8k32 q4_K kernel from every angle).

But the decision turns on the **end-to-end prefill rate**, not the matmul
component. On Orin (328-tok prompt, interleaved reps):

| | prefill rate | vs plain own |
|--|------------:|-------------:|
| vendored llama | 138.1 tok/s | +8.3% |
| **own (plain)** | **127.5 tok/s** | — |
| own + stream-K | 124.7 tok/s | −2.2% |

The matmul-component gap (~30%) dilutes to **~8% end-to-end**, because matmul is
~half of prefill and the rest (attention, norms, rope, plumbing) is identical.
**Decode is unaffected** — llama never touched it. So the trade for shipping our
own kernel is: **~8% slower prefill (decode unchanged) for −1,829 lines + the
vendored dependency + the multi-backend readability debt entirely gone.** For a
codebase whose point is that everything is ours and teachable, that is a reward.
None of the three optimization levers improved the end-to-end rate, so we ship
the plain own kernel.

## Methodology notes (the expensive lessons)

- **Measure end-to-end, not a sub-component.** The "30%" was the matmul-only gap;
  the product number is 8%. The component is where the *mechanism* lives, the rate
  is where the *decision* lives — pull the rate first.
- **Verdicts are device-scoped.** A5000 parity did not transfer to Orin; the 8-SM
  part flips results that look settled on the 64-SM desktop. Never generalize
  dev-box → target.
- **Tegra clocks aren't lockable** — absolute times swing run-to-run (llama
  measured 0.51–0.69s across runs). Trust ncu *ratios* (SM%, cyc/issue), which are
  clock-independent; treat absolute tok/s only within a single interleaved run.
- **A small regression is only a problem if it can't raise the ceiling.** Stream-K
  is −2% in isolation, which would condemn it as a standalone lever — but the right
  question was whether its persistent-CTA structure opened headroom for the latency
  fix. It didn't (the bottleneck is shared-read, which neither stream-K nor its
  enabled cp.async-depth addresses), so the −2% was not a foundation. The principle
  is sound; here it pointed away.

## Branches (preserved research record)

All off the clean base (commit before this consolidation), with the vendored
`llama-mmq.cu` present for A/B:

- `own-q4k-m16n8k32` — the plain own kernel (what shipped to `main`).
- `own-q4k-ldmatrix` — lever 1.
- `own-q4k-splitk` — lever 2 (atomic split-K PoC).
- `own-q4k-streamk` — lever 3 (the lock-free stream-K + fixup; the most complete
  experimental kernel, retained as the reference implementation).
- `own-q4k-streamk-pipe` — lever 4 (deep cp.async, abandoned mid-build).
