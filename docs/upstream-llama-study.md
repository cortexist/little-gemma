# What llama.cpp changed since 2026-04, and what transfers

A source-level study of upstream `llama.cpp` between **2026-04-15**
(`b3d758750` — the Cortexist fork's merge-base) and **2026-07-17** (`b10054`),
done when the fork was retired. The question was two-part: *are our
"vs llama.cpp" numbers measured against a stale reference*, and *what did
their maintainers learn that we can use*.

Companion to [benchmarks.md](benchmarks.md) (the numbers) and
[prefill-performance-journal.md](prefill-performance-journal.md) (our own
campaign).

## 1. The reference is not stale — and the "upstream is faster" delta was ours

**Finding: upstream's three months of code changes are performance-neutral
on both our devices for Gemma 4.** Details and the isolating experiment are
in [benchmarks.md](benchmarks.md#upstream-cross-check-llamacpp-b10054-2026-07-17);
in short: Orin (both source-built) is parity, and on the A5000 an apparent
+2.3–4.4% for upstream evaporated once the CI-vs-local build and the CUDA
version were held fixed — same-toolchain April-vs-July code is neutral-to-
slightly-negative.

This matches upstream's own account. **PR #24127** ("CUDA: refactor MMQ
kernel configuration", 2026-07-13, JohannesGaessler) is the only large MMQ
change in the window, and its author states plainly: *"On my NVIDIA hardware
I am seeing no changes to performance beyond statistical fluctuations."*
The window's one real decode win — **PDL** (#22522, programmatic dependent
launch, kernel-launch overlap, claimed ~10% tg on an RTX PRO 6000) — is
gated to **CC ≥ 90 (Hopper+)** and cannot touch Ampere.

**Methodology lesson (the expensive one):** a locally-built binary and a
vendor CI binary are *different measurements of the same code*. Our first
pass attributed a build-configuration artifact to three months of someone
else's engineering. When comparing across projects, hold the toolchain
fixed or you are benchmarking your compiler flags.

## 2. The MMQ restructure is tunability, not tuning

Upstream replaced the monolithic 4,282-line `mmq.cuh` with a 1,495-line core
plus `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh` and per-architecture config
tables (`mmq-config-{ampere,blackwell,pascal,cdna,rdna2,rdna4}.cuh`), driven
by a `struct ggml_cuda_mmq_config {type, nthreads, occupancy, I, J,
sram_layout, K_vram, stream_k, fallback}`.

It is tempting to read that as "they tuned per architecture." They did not:

- **The Ampere table encodes April's hardcoded values.** `nthreads=256`,
  `I=128`, `J_max=128` are exactly what April computed via
  `get_mmq_y_host`/`get_mmq_x_max_host`. No Ampere-visible tuning change.
- **There is no intra-Ampere differentiation.** `ggml_cuda_mmq_get_config_ampere(type, J, fallback)`
  takes **no `cc` argument** (`mmq.cuh:225-242`) — sm_80/86/87/89 get
  byte-identical tiles. Our Orin (sm_87) and A5000 (sm_86) run the *same*
  config. Any device-vs-device difference in llama is therefore a *regime*
  effect, not a tuning-bucket effect. (An earlier hypothesis of ours — "Tegra
  fell into a generic untuned bucket" — is falsified by this signature.)
- **The SRAM-layout abstraction is a repackaging.** `ggml_cuda_mmq_sram_layout`
  + `ggml_cuda_mmq_get_sram_stride()` compute numerically identical strides
  to April's `MMQ_MMA_TILE_X_K_*` macros; the bank-conflict rule ("K % 2 == 1
  for dp4a or K % 8 == 4 for mma") is verbatim from April.
- **load-tiles / vec-dot are line-for-line April** modulo renames
  (`mmq_y`→`I`, `mmq_x`→`J`).

**What it buys them** (the author's stated roadmap, worth knowing because it
is where their next wins will come from): per-(arch, dtype, batch-size)
configurability so that small-batch tuning **for speculative decoding**,
replacing the legacy `__dp4a` 4-byte SRAM layout with 16-byte loads
(*"~10% end-to-end for e.g. P40"*), and FP16-in-SRAM for Volta stop being
whack-a-mole.

**Two convergences worth noting**, because they independently confirm our own
conclusions: their Ampere config targets `occupancy=1, nthreads=256` — one
256-thread CTA per SM, 2 warps/scheduler — which is exactly the operating
point our journal repeatedly landed on and called a structural floor. And
their `fallback` flag is our `%64`-rounding bounds-check problem under
another name.

## 3. Stream-K: we implemented and falsified the version upstream had already replaced

This is the sharpest lesson in the study.

Our [stream-k-experiment](stream-k-experiment.md) built stream-K for the q4_K
prefill matmul, mirroring llama's protocol: *"distribute the flattened
(tiles × superblocks) work across exactly `nsm` persistent CTAs … grid=8
every launch as designed."* Result: neutral-to-negative, closed on both
devices. The doc's own closing lesson: **"'Do it only when needed' applies to
stream-K itself"** — it helped the big ffn matmuls (−8% component) and
over-distributed the tiny attention matmuls (fixup overhead, 5.3% SM).

We mirrored **April's** protocol — which pinned the grid unconditionally:

```cpp
const dim3 block_nums_stream_k(nsm, 1, 1);          // fork base, mmq.cuh:4115
```

**Ten days after our merge-base**, PR #22298 (2026-04-25) replaced exactly
that with a wave-quantization gate — the thing our own doc said was missing:

```cpp
// mmq.cuh:1334-1343 (current upstream)
const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90
                               ? ntiles_dst : nsm, 1, 1);
const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;
```

`stream_k=true` in the config table is a **capability flag, not a mode**. At
every launch llama measures how well the tile grid fills the SMs and picks:
**one CTA per tile** when the grid already fills them (≥90%) — and then
`fixup_needed` evaluates false, so **the fixup kernel never launches at all**
— or **persistent stream-K** (grid = `nsm`, with fixup) only when the last
wave is ragged. It is also *why upstream needs no per-device config here*:
the adaptation is dynamic, off the live SM count, not baked per arch.

**The lesson is not "stream-K works after all."** It is that we falsified a
lever in a form its author had already abandoned, because we ported the
protocol from a snapshot instead of asking what it looked like now. Our own
data (ffn helps, attn hurts) *was* the gate's motivation, discovered
independently and then left on the floor.

**Closed 2026-07-17: the gate is not worth porting, because we already solve
the problem it solves — more cheaply.** llama's row-tile is frozen at `I=128`
for *every* Ampere entry in the config table (§2), so when the tile grid does
not cover the SMs they cannot make tiles smaller — the only lever left is to
split K and pay a fixup pass. That is what stream-K *is*: a workaround for a
fixed tile size. Our tile is not fixed — `launch_q4k_mma` shrinks
warps-per-CTA (`model-cuda-i8.cu:1061`), and the loop is already self-gating on
exactly llama's condition:

```c
while (wpc > 1 && (long)((m + 32*wpc - 1) / (32*wpc)) * ncol < 2 * sms) wpc >>= 1;
```

Shrinking the tile fills the SMs with no fixup, no cross-CTA partials, and no
scratch buffer. So stream-K measuring −2.2% (Orin) / −5–9% (A5000) was not bad
luck: it is a *second* mechanism for an already-solved problem, carrying
overhead the first one doesn't.

The shrink is deliberately **off on discrete**, and that decision re-measured
correct today (A5000 E4B prefill, `LG_Q4K_WPC=-1` forces it): **3,846 → 3,661
tok/s, −4.8%.** The reason is in the code comment and it is sound — shared
memory >50 KB pins residency at one CTA/SM either way, so a smaller CTA is the
same number of waves with half the warps and half the latency hiding.

That leaves exactly one stream-K-shaped gap: **small matmuls on discrete**,
where we decline to shrink and the tile count doesn't cover 64 SMs (a 12B
global-layer k-projection is ~30 tiles = 47% fill, and llama's gate *would*
route it to stream-K). It is not worth chasing: with V reusing K's projection,
those k/v matmuls are ~1–2% of matmul work against ffn's ~12288-wide tensors,
so even halving them is a fraction of a percent of prefill. **Stream-K stays
closed on both devices, now for a structural reason rather than an empirical
one.**

## 4. The transferable win: grow the KV split with depth, not the walk

**This is the actionable finding, and it explains a scoreboard we thought
was a mystery.**

The difference is which quantity they let grow with context depth.

Upstream's batch-1 decode attention (Ampere, f16 KV → the `mma_f16` kernel)
**pins per-block work to a fixed KV tile (`nbatch_fa`, 128 keys) and grows the
block count with depth**: the grid is
`min(max_blocks_per_sm * nsm, ntiles_KV * ntiles_dst)` where
`ntiles_KV = ceil(K->ne[1] / nbatch_fa)` and `max_blocks_per_sm` comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (`fattn-common.cuh:1113-1133`),
with a cheap LSE-combine/fixup merging the partials
(`fattn-common.cuh:723-913`). Each CTA's serial walk is therefore ~constant in
depth; deeper context buys *more CTAs*, not longer ones, until the KV axis
fills a whole wave of the device (thousands of tokens). That is why their
`tg128` loses only ~1–4% from depth 0 → 512, measured. (Their SWA cache also
holds `K->ne[1]` constant for 5 of every 6 Gemma layers — as our f16 SWA rings
do.)

We do the inverse: **we pin the block count and grow per-block work.**

Ours (`src/cuda/model-cuda.cuh:501-517`):

```c
#define MAXSPLIT 8
#define SPLIT_KEYS 1024
int n_split = min(MAXSPLIT, max(1, (T + SPLIT_KEYS - 1) / SPLIT_KEYS));
```

The comment above it calls this *"device-adaptive"*. **It contains no device
property.** It is `ceil(T/1024)`, capped at 8 — context-adaptive, not
device-adaptive. The grid is launched at `n_head × MAXSPLIT` (fixed, for the
captured graph) and blocks with `split >= n_split` early-exit, so the *working*
parallelism is `n_head × n_split`:

| context T | n_split | working blocks (8 q-heads, B=1) | **keys walked per block** |
|---|---|---|---|
| 512 | **1** | **8** | **512** (llama: ~128) |
| 1024 | **1** | **8** | **1024** (llama: ~128) |
| 4096 | 4 | 32 | 1024 |
| ≥ 8192 | 8 | 64 | ≥1024 |

Three facts fall out of that table:

1. **Below 1024 tokens `n_split == 1` — the split-K kernel degenerates to one
   block per head**, i.e. the pre-split-K kernel, and per-block work grows
   *linearly* with depth (512 keys at d512, 1024 at d1024) where llama's stays
   at ~128. **This is our decode depth droop** (E2B −18% from depth 0→512
   where llama loses ~1%), worst on E2B precisely because the smallest weights
   make attention the largest share of decode.
2. **`MAXSPLIT 8` is the Orin's SM count.** The split was developed and
   measured on the Orin at 4k (`.scratch/flash_decode_test.cu`, "~3.3x
   decode-attn on the Orin @4k") and never generalized — at 4k on 8 SMs it
   gives 32 blocks over 8 SMs, a good fill, and at ≤1k the 8 heads still cover
   the 8 SMs. The Orin is the one device where these constants are *accidentally
   right*, which is part of why **we beat llama on Orin decode**.
3. **On the A5000 the same constants leave decode attention at 8 working
   blocks on 64 SMs with a 4× longer serial walk than llama's** — part of why
   **we lose A5000 decode (0.70–0.81×)**, worst on E2B (0.70×).

**The fix is small and graph-safe.** `n_split` is computed *inside* the kernel
from `T` (line 515), so it can vary per token without breaking the captured
graph; only `MAXSPLIT` (the launch geometry) is fixed at capture. So: raise
`MAXSPLIT`, and replace the `SPLIT_KEYS` granule with llama's rule — target
`n_head × n_split ≈ occupancy × nsm`, subject to a minimum keys-per-split
floor so short contexts don't over-split. `multiProcessorCount` is already
queried elsewhere in the codebase (`model-cuda-i8.cu:1314`).

**Caveats before anyone ships it:** changing `n_split` changes the reduction
order (the split-K decode is already the "relaxed/reassociated" numerics
class, so this is a numerics-gated change under the f16-KV battery), and the
MTP verify rides the same attention — its byte-identity-by-construction
requires verify and decode to use the *same* reduction order, which holds as
long as both derive `n_split` from the same formula. **Acceptance % is the
tripwire** (it has caught this class twice already).

## 5. Upstream's blind spot is the edge — including our exact workload

Searching the MMQ/FA PRs and their discussions for jetson/orin/tegra/sm_87/
iGPU/unified-memory returns **nothing**. The test hardware in this window was
RTX 3090/4090, P40, RTX PRO 6000, DGX Spark, MI50/RX 6800. Jetson enters the
record only as breakage:

> **Issue #24457** (2026-06-11): *"FA+MTP crash: ggml_cuda_flash_attn_ext_mma_f16
> fatal error … on SM87 (Jetson Orin)"* — root cause: FA specializations for
> GQA ratios 1/2 had been pruned for *"unnecessarily high compilation time/
> binary size"*, unaware GQA=2 was live, which *"inadvertently broke Gemma 4
> E4B MTP."* Fixed 2026-06-30 (#25148).

That is *our* model, *our* speculative feature, on *our* board, broken for
three weeks by a compile-time optimization. It is the clearest available
statement of why an independent, edge-first runner has room to exist — and a
reminder that upstream's defaults are tuned by people measuring on hardware
we do not ship.

The same asymmetry explains the regime split cleanly: upstream's Ampere
config is device-blind, so their (real, but small) discrete-GPU wins come
from mechanisms that pay in an L2/occupancy-bound regime and are neutral in
the Orin's DRAM-bandwidth-bound one — the same physics our own journal found
when swapxy paid +6.6% on the A5000 and zero on Tegra's zero-copy path.

## 6. MTP: upstream diverges from us on chained-draft position

Upstream branches **specifically for Gemma-4-style shared-memory assistants**
(`common/speculative.cpp:1589-1596`):

```cpp
} else if (is_mem_shared) {
    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
    // ref: .../huggingface/transformers/.../gemma4_assistant.md ...L36-L37
    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
} else {
    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
}
```

All chained drafts share `n_past`, cited to the HuggingFace `gemma4_assistant`
doc; the advancing `n_past + i + 1` form is upstream's path for *non*-shared
drafters. **We advance** (`src/run.c:61`: `mtp_draft_chain(..., pos + j - 1)`),
which by upstream's own note diverges from block-3 up — our default. The
theory looked strong: our chained draft's rope angle would be off by one
position, and worse, a draft at `pos+1` attends `[0, pos]` while the verify
has not yet written row `pos` — a stale read. And it is exactly the class of
bug that hides forever, because greedy verify absorbs bad drafts: **output
stays byte-correct while acceptance silently rots** (this has bitten us twice
— the f16-ring draft, the warp-norm verify).

**Measured verdict: FALSIFIED, no effect.** Orin, E4B block-3, prose prompt,
serve mode, 4 turns with the first discarded (`bench/mtp-pos-ab.sh`):

| | acceptance | decode | reply sha1 |
|---|---:|---:|---|
| baseline (`pos + j - 1`) | 471/1056 = **44.6%** | 20.8 tok/s | `dcda20e1c1b2b99f` |
| patched (`pos`, upstream's rule) | 473/1052 = **45.0%** | 20.9 tok/s | `dcda20e1c1b2b99f` |

+0.4 points is run-to-run noise. Identical SHAs re-confirm the byte-identity
invariant across both rules. **Our advancing rule stands; no change made.**
For a single chained draft the one-position rope drift is evidently too small
to move the argmax, and whatever draft 2 reads at row `pos` does not degrade
it measurably. The divergence may still matter at block-4+ (more chained
drafts, larger drift) — untested, and the config is not our default.

Two things this run *did* establish: the E4B prose MTP numbers in
[mtp.md](mtp.md) (~30% acceptance, 1.12× Orin prose) are **prompt-specific
and pessimistic** — this prose prompt gives 44.6% and **1.27×** (20.8 vs 16.4
plain). And a negative result cost ~20 minutes because the acceptance
tripwire makes the experiment self-verifying.

Other upstream MTP design notes worth having on record:

- **Variable draft length.** Upstream drafts up to `--draft-max` with per-step
  top-k(10) sampling and a `p_min` confidence early-exit
  (`speculative.cpp:1544-1577`) — it stops drafting when the head is not
  confident, instead of always emitting `B-1` tokens. Our fixed block-N always
  pays for `N-1` drafts. On the low-acceptance prose regime (E4B's 256-wide
  head, ~30%) an early-exit would cut wasted draft compute directly.
- **Verification is exact in both.** Upstream accepts only the target
  sampler's own picks (`common_sampler_sample_and_accept_n`), so greedy output
  equals non-speculative decoding — the same guarantee we make, reached the
  same way.
- **Context design.** Upstream runs the assistant as a *second* `llama_context`
  (`LLAMA_CONTEXT_TYPE_MTP`) whose iswa cache aliases the target's K/V
  pointers via a share callback — "only context+compute are new". We hang the
  assistant off the target model and draft inside the target context. Theirs
  composes with a generic speculative framework (ngram/eagle3/dflash) and
  supports multi-slot batched drafting; ours is single-in-flight but pays no
  second-context bookkeeping.
- **E2B/E4B centroid head.** Upstream's converter **drops** the MaskedEmbedder
  centroid tensors ("Skipping … so that convert can end normally") and runs
  the dense tied head — as we do. The retired fork implemented the reference
  centroid top-k routing. Since upstream (dense) and the fork (centroid) both
  measure ~64% acceptance at B=2, the head type appears not to drive
  acceptance.

## Open leads this study produced

1. ~~Depth-scaled KV split for decode attention~~ (§4) — **SHIPPED, and it was
   the biggest decode win in the project's history.** `SPLIT_KEYS 1024 → 64`,
   one constant: A5000 E2B **+44.7%** / E4B **+25.5%** / 12B **+16.7%**, Orin
   **+15.0% / +9.1% / +3.8%**; depth droop **−14.3% → −0.6%**; desktop decode
   went from 0.70–0.81× to **parity (1.01–1.02× on E2B/E4B)** and the Orin
   lead widened to **1.27× (E4B)**. All gates green (determinism, coherence,
   prefill control unchanged, MTP acceptance 43.9–45.9% vs 44.6% baseline).
   Full writeup in [performance-journal.md](performance-journal.md#the-split-that-never-split-2026-07-17).
   The prediction that the Orin would see little was **wrong** — it gained too;
   more resident blocks hide latency even on 8 SMs.
2. ~~Efficiency-gated stream-K~~ (§3) — **closed, not worth porting.** We
   already fill the SMs by shrinking the tile (`wpc`), which llama cannot do
   (their `I` is frozen at 128 per-arch), so stream-K is their workaround for
   a constraint we don't have. The one residual case (small matmuls on
   discrete, where we decline to shrink) is ~1–2% of matmul work.
3. ~~Chained-draft position~~ — **tested and falsified** (§6). Closed.
4. **Variable draft length / confidence early-exit** (§6) — upstream stops
   drafting when the head's top-k confidence is low instead of always paying
   for `N-1` drafts. Cheap, and aimed exactly at our weakest MTP regime (the
   E4B 256-wide head on prose). The only MTP idea from this study still open.
5. **Re-measure the MTP table on a prompt panel** — §6's run shows the
   published E4B prose figures are one prompt's numbers and understate the
   feature (44.6% / 1.27× vs the documented ~30% / 1.12×). Content-dependence
   is already the headline claim; the table should show a range, not a point.
