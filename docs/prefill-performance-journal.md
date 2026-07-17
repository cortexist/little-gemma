# Prefill performance journal

Focused log for the prefill workstream: closing the prefill gap vs llama.cpp.
Decode and MTP are out of scope; every change gates on **Paris cohesion**,
byte-identity vs the legacy path, warm prefill tok/s, and decode unchanged.

The 2026-06 Orin ILP sessions (column-group mma pairing, matmul CUDA graph,
fork/join A/B — all exhausted or failed) live on branch `prefill-q4k-ilp` in
this file's earlier incarnation; `docs/stream-k-experiment.md` covers the
own-kernel-vs-vendored decision.

---

## Session: A5000 first (2026-07-01) — serve wide chunks + balanced split, 2.7–3.0×

Scope change: optimize the **A5000 dev box first**, validate on Orin after. On
the A5000, little-gemma sat at **0.26×/0.21×** llama.cpp prefill (12B/E4B; see
`llama.cpp.diagrams/tensorsharp-llama.cpp-benchmark.md`) — its worst relative
showing, behind even TensorSharp (a managed C# runtime) on both models.

**Benchmark:** `.scratch/bench-serve.ps1` — the real serve path (socket, one
connection per turn so every turn is same-shape, like llama-bench pp), first
turn discarded, warm best/avg of the rest. 12B & E4B Q4_K_M.

### The three discoveries

1. **The serve path never ran wide chunks.** `LG_WIDE_CHUNK` was only wired
   into `model_prefill` (one-shot `-p`, system prefix). `model_prefill_mixed` —
   every actual serve turn — packed text to a hard `PREFILL_B`=128. All prior
   "wide chunk" serve measurements, on any device, varied nothing.

2. **On the A5000 the gap is SM fill, not weight streaming.** Warm profile
   (12B, 928-token turn): matmul 1.47 s of a 1.74 s turn; attention 0.15 s,
   elementwise 0.10 s. Weights are VRAM-resident on a discrete card (the
   Orin's once-per-chunk DRAM-pass rationale doesn't transfer); at 128-wide
   chunks the fat q4_K kernel launches 16–120-CTA grids on a 64-SM card at
   1 CTA/SM (ncu: 16.7% occupancy, reg+shmem limited) — most of the machine
   idles. llama.cpp's `mul_mat_q` on the same card: same 16.7% occupancy,
   52% vs our 45.7% SM throughput — per-kernel near-parity. The end-to-end 4×
   was launch shape, exactly what ub=512 buys llama.

3. **CORRECTNESS: the SWA ring spare was broken for every wide chunk.**
   `kvcache_init` sized the ring spare from `g_wide_chunk` — but it runs
   before `wide_chunk_init()` ever gets called (first prefill), so the spare
   was ALWAYS 128 regardless of `LG_WIDE_CHUNK`, and media spans up to 1120
   never had a matching spare at all. A chunk of B > spare starting past
   `window+spare` wraps the ring over rows its own first queries still need —
   silent SWA corruption. **Proven on device:** 4096-token high-entropy
   prompt, narrow vs wide diverged pre-fix, byte-identical post-fix.
   - Latent on main for: media sessions whose spans land past the first ring
     wrap, and any `LG_WIDE_CHUNK` user; on branch `prefill-q4k-ilp` (Orin
     default 768) it additionally bites every prompt ≳1150 tokens.
   - **Gate lesson:** a byte-identity gate on a REDUNDANT prompt (a repeated
     paragraph) hid this — greedy argmax margins absorbed the corruption
     twice before a random-words prompt exposed it. Identity gates need
     high-entropy prompts.

### The changes

- **Ring fix:** spare = `g_prefill_max_b`, resolved BEFORE the rings are
  sized (`wide_chunk_init()` at the top of `kvcache_init` and
  `ensure_scratch`; `serve()` opens the media projector and calls
  `model_prefill_reserve()` before `kvcache_init`). Cost: the ring grows by
  (spare−128) rows × kv_dim × 4 B per SWA layer — 12B ≈ +300 MB at spare 1024.
- **Serve text packs to the wide budget** (same knob and buffers as one-shot).
- **Balanced chunks:** split n into ceil(n/TB) near-equal chunks instead of
  TB + skinny tail (1428 → 2×~720, not 1024+404); chunks >128 round up to a
  multiple of 64 — `matmul_q_n`'s single fat launch needs cols%64==0 —
  padding a few tail tokens beats per-64-column launches for the whole chunk
  (the pad machinery pre-existed). Same number of weight passes as greedy,
  or fewer (the tail chunk merges instead of straggling).
- **Defaults:** discrete 1024 (best of the sweep on both models at
  928/1428/4096 tokens), integrated stays 768. `LG_WIDE_CHUNK=128` = legacy
  narrow; the env value now rounds to a multiple of 64 (was 128).

### Results (A5000, warm serve, Q4_K_M)

| model / turn | before | after | llama.cpp | ratio before → after |
|---|---:|---:|---:|---:|
| 12B, 928 tok  | 533  | **1430** | 2247       | 0.24× → **0.64×** |
| 12B, 1428 tok | 565  | **1426** | ~2160      | 0.26× → **0.66×** |
| E4B, 928 tok  | 1020 | **3050** | ~5000 (FA) | 0.21× → **0.61×** |
| E4B, 1428 tok | ~1000| **2890** | ~5200      | ~0.20× → **0.56×** |

One-shot cold 4096 tokens: 12B 429 → 853, E4B 575 → 976, E2B 1367 → 2405.
vs TensorSharp: E4B now AHEAD (3050 vs 2567), 12B 1430 vs 1735.

**Gates passed:** byte-identity vs legacy-128 at ~4096 tokens (12B, E4B, E2B,
and the f32 backend at 2163); Paris on 12B/E4B both backends; decode unchanged
(44.5–45.2 vs 43.1–45.6 tok/s at 928-ctx); MTP serve smoke (armed, 94.7%
acceptance, correct counting output).

Known pre-existing issue (NOT from this change; identical on pristine main):
E2B Q3_K_M in `.data/` generates "]" garbage on the Paris prompt on the i8
CUDA backend — needs its own investigation.

### Round 2 (same day): launch policy + warmup — 12B → 1550, first turn → warm

The "next levers" list, executed:

1. **Occupancy: falsified on the A5000 too** (was already a wash on Orin).
   `matmul_q4k_mma<32>` is 124 regs / 46.6 KB shared — a genuine 2 CTAs/SM,
   4 warps/scheduler — and it LOSES: 1429 → 1211 (−15%). The loss is doubled
   weight re-streaming (ncol doubles), and even with that absorbed (swapxy
   below) it still trails 64-col at 1494 vs 1523. The kernel is
   latency-balanced, not occupancy-starved, on BOTH device classes. Closed.
2. **The C32 loss exposed the real lever: grid-axis order.** With grid
   (row-tiles, col-tiles), CTAs issue x-fastest — one column-tile's whole
   m-sweep per wave, so every column-tile re-streams the entire weight
   matrix from DRAM (an ffn matmul paid ~15 × 33 MB). **swapxy** puts
   column-tiles on x: all column-tiles of one row-tile run in the same wave
   and share its ~550 KB of weights through L2. +6.6%. Discrete-only
   default (Tegra zero-copy weight reads bypass L2). `LG_Q4K_SWAPXY=0/1`.
3. **Warps-per-CTA shrink OFF on discrete** (+1.4%): the shrink covers the
   8-SM Orin, but a 128-thread CTA still occupies a whole A5000 SM (shared
   >50 KB caps residency either way) with half the warps. `LG_Q4K_WPC`.
4. **Serve engine warmup**: without `-sys`, a 2-token throwaway prefill at
   startup absorbs weight upload + repacks + buffer allocation. First turn
   333 → **1525** tok/s (12B), 335 → **3155** (E4B) — warm from turn one.
   Two tokens because run.c is backend-shared and the CPU backend pays per
   token; the one-time costs are all-or-nothing at the first chunk.

**Round-2 results (A5000, warm serve, 928-token turns):** 12B 1430 → **1550**
(0.69× llama), E4B 3050 → **3280** (0.65×). 12B @1428 tok: 1426 → 1527.
Byte-identical vs the old launch order at 4096 tokens; Paris; decode
untouched. Kernel time (nsys, ~2870-token prefill): q4k 1.139 → 1.016 s,
q6k 0.267 → 0.250 s.

### Round 3 (same day): the wiring audit — re-testing the old "neutrals"

Motivated by the serve-wide-chunk discovery (a knob that measured nothing
for weeks), the questionable prior verdicts were re-tested on the A5000
under a new rule: **no verdict without proof the knob engaged**
(instruction/launch-level evidence or a poison test).

| prior verdict (Orin) | A5000 re-test, wiring-proven | outcome |
|---|---|---|
| stream-K "~neutral" | ported to current kernel (branch `prefill-a5000-streamk`); nsys shows 284 streamk + 284 fixup launches, zero plain launches | **loses**: 1486 (colmajor walk) / 1426 (rowmajor) vs 1565 plain. colmajor > rowmajor re-confirms the L2-order effect inside stream-K; but at 180–720-CTA grids there is no wave quantization left to fix, and the fixup pass is pure cost. Closed on BOTH devices. |
| ILP-branch kernel body (qs pipelining, dual-tile interleave, 8-wide pairing; Orin matmul component 6.0→4.38 s) | built `prefill-q4k-ilp` as-is in a worktree; A/B at IDENTICAL 128-chunk launch shapes | **exactly neutral**: 402.9/401.3 vs 402.6/399.8 tok/s. The Orin component win hid Tegra's uncached zero-copy weight latency, which doesn't exist here. At 768-wide chunks that branch is 12% SLOWER than ours — entirely its chunk splitting (768+128s+pad vs balanced), not the kernel. |
| C32 / 2-CTA occupancy "wash" | round 2 | **loses 15%** properly wired — original direction confirmed. |
| ldmatrix "neutral" | not re-run | skipped: three-for-three confirmations above; premise check needs warp-stall ncu (still permission-blocked). |

**Audit conclusion:** the old kernel-level verdicts survive verification —
the mis-wired/mis-aimed experiments were all in the LAUNCH PLUMBING
(serve chunking never wired, ring silently corrupting, grid-axis order,
warp shrink), which is where this branch's ~3× came from. The kernel body
itself is at a genuine local optimum for this architecture family.

### Round 4 (same day): warp-stall counters land — the A-staging pipeline

Perf-counter permission enabled → first REAL stall profile on the A5000,
and it rewrote the diagnosis. Per-instruction warp sampling on the ffn
launch: **long_scoreboard 1.47 cycles/instr** (top stall by 10×), landing
on the first consumer of the A-staging's global qs `LDG.128` — and
**short_scoreboard 0.09**, i.e. the Orin's shared-read bottleneck is a
non-issue here. The June "cp.async/prefetch physically can't help"
reasoning was Orin-scoped; on this card the global weight fetch IS the
stall. Device-scoped verdicts, once again, in the other direction.

**Fix:** software-pipeline the A-staging in both mma kernels — the next
sub-block pair's qs words (and the next superblock's first pair + header)
go in flight before the current pair's mma phase, whose tensor-core cycles
cover the latency. Same loads, same order: byte-identical, verified at
4096 tokens on both devices. No spills (q4k 238 regs, q6k 254); occupancy
was already 1 CTA/SM so the register growth costs nothing.

| | before | after |
|---|---:|---:|
| ncu ffn launch: long_scoreboard | 1.47 cyc/instr | **0.66** |
| ncu ffn launch: duration | 2.11 ms | **1.78 ms** |
| A5000 12B warm serve | 1544 | **1670** (0.74× llama) |
| A5000 E4B warm serve | 3280 | **3423** |
| Orin 12B one-shot 910 tok | 105 | **120 (+15%)** |
| Orin E4B one-shot 910 tok | 216.5 | **236 (+9%)** |

The Orin gains exceed the A5000's — uncached zero-copy host reads have
even more latency to hide. This supersedes the `prefill-q4k-ilp` branch's
qs-pipeline attempt (measured neutral-on-A5000 / modest-on-Orin from a
different structure). Cumulative today: A5000 12B 533 → 1670 (**3.13×**),
E4B 1020 → 3423 (**3.36×**); Orin 12B 102 → 120, E4B 192 → 236.

### Round 5 (post-merge, same day): the elementwise + attention levers

Worked the "next levers" list on merged main:

1. **2-superblock-deep qs prefetch: WASH** (1682 vs 1694 — the pf/pg
   rotation copies eat the win). 1-deep stays.
2. **Warp-cooperative activation quantization (+3.7%/+5.2%).** The serial
   `d_quant_group` walks 32 consecutive floats per THREAD — a warp touches
   32 different 128-B lines per step (12.5% sector efficiency), and
   `d_quant_block` idled 31/32 threads in the geglu epilogue. The flash-
   output quantize (960×4096) ran at 59 GB/s / 267 µs. `d_quant_group_warp`
   (lane-per-element, shuffle trees; fmax + int-sum exactly associative →
   byte-identical) behind prefill-only entry points; decode's captured
   kernels stay frozen. 12B 1694 → 1757, E4B 3423 → 3602.
3. **Flash K-block staging (+3.2% at 4096 ctx, neutral at 928).** ncu on
   `flash_attn_n<256,ring,f32>`: long_scoreboard 11.5 cyc/instr, all on
   `F2FP.PACK_AB` — the f32→f16 pack consuming per-fragment global K loads
   (each K element also read twice, once per query sub-tile). Staging each
   32-key block into shared as f16 (one coalesced sweep, 16.9 KB, still 2
   CTAs/SM, byte-identical) drops long_sb to 2.5 and the kernel 954→856 µs.
   The kernel is now **mio_throttle-bound** (5.2 — the strided
   half-at-a-time V loads): that's the next attention lever if needed.
4. **Orin validation (branch `explore-levers` on cortex): byte-identical,
   12B 120 → 125 (+4.2%), E4B 236 → 246 (+4.4%), decode 6.86 unchanged.**

**Standing after round 5:** A5000 12B 928-turn **~1760** (0.78× llama),
1428-turn 1724, 4096 cold 941; E4B **3602** (~0.69×). Orin 12B **125**,
E4B **246**. Day cumulative: A5000 3.3×/3.5×, Orin 1.23×/1.28×.

### Round 6 (same day): the exhaustion sweep — every remaining lever measured

All candidates tried, every verdict wiring-verified; falsifications are
results too:

| lever | result |
|---|---|
| **V re-staged through the freed sK buffer** (transposed f16, PV fragment = one aligned LDS.32, zero extra shared) | **falsified**: kernel 856 → 991 µs. The stage sweep sits in a window with no work to hide behind — the direct V loads at least overlapped the mma FLOPs. long_sb went back up (2.5 → 3.6). |
| **V fragments pipelined one n-iteration ahead** (the qs-pipeline pattern, third application) | **neutral**: 871 vs 856 µs, mio unchanged at 5.24 — the compiler already schedules those unrolled loads. mio_throttle is instruction COUNT; no pipelining fixes that. |
| **2048-wide chunks** (PREFILL_MAX_B 2048, 4096-token prompt: 2×2048 vs 4×1024) | **neutral-to-negative**: 927 vs 937 tok/s, byte-identical. swapxy's L2 sharing already captured the weight-reuse win; 1024 stays. |
| **2-superblock-deep qs prefetch** (round 5) | wash — rotation copies eat it. |
| rmsnorm chunk-twin with warp-quant epilogue | not built: the real-aq prefill norms are ~4 ms per 2870 tokens; ceiling ≈ +0.25% for a duplicated kernel. |
| q6_K wpc rebalance for L2 | not isolated (the wpc knob is shared with q4_K); ceiling ≈ +1% on the 15%-of-matmul q6_K share. |

**f16 SWA rings — assessed, recommended AGAINST (for now).** On inspection
the payoff shrank: the K-stage already removed the f32→f16 conversion from
the hot path; f16 rings would halve K/V ring bytes (latency, not the mio
instruction-count wall) and save ring memory, worth ≈ +1–3% prefill and
+2–3% decode at long context — against breaking the byte-identity gate on
DECODE (SWA layers) and touching every ring read/write path. Revisit only
if long-context decode becomes the priority.

## Session: the Orin push — Phase 0 (2026-07-02): measure before planning

Goal: parity-or-better vs llama.cpp prefill ON THE ORIN. Phase 0 re-measured
everything on pinned clocks (MAXN + `jetson_clocks`; GPU locked 918 MHz)
before any plan — and the corrections were large.

**Corrected baselines (warm serve, 929-token turns, same day/clocks):**

| | little-gemma | llama.cpp (fa1, `-mmp 0`) | ratio |
|---|---:|---:|---|
| 12B | **162.3** (was believed 125) | 216.8 | **0.75×** |
| E4B | **381.4** (was believed 246) | 510.3 | **0.75×** |

The one-shot `-p` numbers understated by 30–55%: each process paid the q6/q3
repack inside the prompt timer and ran on unpinned clocks. Decode stays
ahead (12B 8.0 vs 7.55). Also: 12B llama-bench needs dropped page caches +
`-mmp 0` on the 16 GB board or NvMap alloc fails.

**Gap anatomy (12B: ours 5.72 s vs llama 4.28 s — find 1.44 s):**
matmul 4.19 s (q4k 3.27 + q6k 0.91), attention 0.67, elementwise 0.73.

**ncu ON THE ORIN IS SAFE** with single-launch, minimal-section captures
(SpeedOfLight + WarpStateStats, ~13 passes; ncu 2024.3/JetPack current) —
the old board resets don't reproduce. That unlocks device-truth stall data:

- q4k ffn launches: **SM 53.4%** — up from June's 44.6% (the A-staging
  pipeline transferred); llama's kernel measured 57.5% here, so the big
  matmul is within ~8% of their efficiency now. SMALL launches still stall
  hard on long_scoreboard (0.9–1.2): Orin's uncached zero-copy qs loads
  outlast the 1-deep prefetch window (A5000's L2 latency did not). The
  2-deep prefetch that WASHED on the A5000 is exactly what this wants —
  device verdicts invert, fourth occurrence.
- q6k: SM 38%, **lg_throttle 0.58** + long_sb — the staging is ld32-only
  because `block_q6_Kr` is 4-byte-aligned. The repack layout is OURS: pad
  it to 16 bytes and stage with uint4 (4× fewer LSU instructions) — the
  June wide-loads lesson, unapplied to this kernel.
- Re-falsified with today's kernels: `LG_NO_ZEROCOPY` device-copied weights
  (±swapxy) are all ~2–4% WORSE on E4B — same DRAM either way, and 4 MB of
  L2 doesn't hold the working set. The reject-list verdict holds.

**Phase 1 plan (ordered by expected yield):**
1. q6_K 16-aligned repack + uint4 staging (0.91 s at ~2.4× its BW floor).
2. Integrated-gated deeper A-staging prefetch (2-deep, maybe 3) for the
   small-launch long_sb.
3. Flash + elementwise Orin captures → targeted fixes (0.67 + 0.73 s pool).
4. Chunk-width sweep on Orin under the corrected serve harness
   (`/tmp/bench-serve-orin.sh` methodology; port into `.scratch/`).
Deep-read of llama.cpp's Orin-side non-matmul path (their ~0.5–0.9 s vs our
1.55 s) is the candidate for a fan-out workflow if 1–4 fall short of parity.

## Orin push — Phase 1a (2026-07-02): two falsifications, a deep-read, and the real plan

**Falsified on-device (both would have shipped on paper):**
- **2-deep A-staging prefetch:** dead neutral on BOTH devices (Orin 162.4 vs
  162.3; E4B 380.9 vs 381.4; A5000 1759 — the earlier −0.7% was noise). The
  "deeper window covers uncached latency" hypothesis fails: at 2
  warps/scheduler there is nothing to fill the wait with, however early the
  load issues.
- **q6_K 16-aligned repack + uint4 staging:** e2e flat and the q6k kernel
  measured SLOWER (0.912 → 0.938 s) — the 4× LSU-instruction cut was repaid
  with interest by +6.7% bytes. lg_throttle was real but the kernel sits at
  bandwidth-balance; conservation of misery. Reverted.

**The board-reset mystery is solved:** ncu + the 12B on the 16 GB board =
NvMap OOM → watchdog reboot (reproduced; the board reset during a 12B flash
capture, and a post-reboot retry failed gracefully with NvMap error 12).
Every E4B capture runs clean. **Rule: Orin ncu captures use E4B only.**

**llama.cpp deep-read (33-agent fan-out, adversarially verified, 14
confirmed mechanisms).** Negative knowledge first — all verified in source:
their mmq has NO cp.async (the journal's old "cp.async depth" hypothesis is
retired); prefill never runs under CUDA graphs; there is no elementwise
fusion advantage (≈12 launches/layer vs our ≈14, and our
quantize-in-producer fusion is ahead of their separate q8_1 kernels); the
fork's turboquant CUDA is decode-only — we are losing to stock upstream
mmq/fattn maturity, nothing exotic. Surviving transplant candidates:

1. **Wide-N mma restructure** (~0.3–0.6 s): llama auto-sizes column tiles to
   128 and streams the weight matrix 8× per 929-token turn; we stream it 15×
   (COLS=64) over uncached zero-copy weights. COLS=128 needs llama's tile
   shape — ONE 16-row tile per warp (acc stays 64 f/thread), half-buffered B,
   >48 KB shared opt-in — not a template bump (which would spill). Orin-gated;
   byte-identity preserved (same per-element k-order).
2. **Flash GQA packing** (~0.15–0.25 s): serve both Q-heads of a KV group per
   CTA (llama's ncols2=2), halving K-stage sweeps and strided V loads. Orin
   flash capture (E4B): SM 25.8%, mio_throttle 4.7 — the same
   instruction-count wall GQA packing halves. Media/bidir mask carries over;
   media smoke required.
3. **Elementwise per-kernel bandwidth pass** (~0.2–0.35 s pool): ~25 GB of
   traffic at ~35 GB/s effective; float4/one-pass fixes guided by E4B
   captures. Fusion-hunting explicitly excluded (see negative knowledge).

Mid-points sum to ~1.0–1.2 s against the 1.44 s parity gap — reachable if
most land, with llama's remaining per-pass codegen edge (57.5 vs 53.4% SM)
explicitly NOT chased.

## Orin push — Phase 1b (2026-07-02): the transplant verdicts + the real decomposition

**The decisive measurement of the phase: nsys on llama-bench itself** (both
models, same board/clocks/token-count as our benches). Per 929-token 12B
prefill, kernel-time decomposition, ours vs theirs:

| component | ours | llama | gap |
|---|---:|---:|---:|
| q4_K matmul | 3.27 s | 3.00 s | 0.27 s |
| q6_K matmul | 0.91 s | 0.57 s | 0.34 s |
| **attention** | **0.67 s** | **0.08 s** | **0.59 s** |
| elementwise (incl. their separate q8_1 quantizes) | 0.73 s | ~0.55 s | ~0.18 s |

Three surprises: the week's pipeline work already closed the q4_K "codegen
wall" to **9%**; elementwise is near-parity as the deep-read predicted; and
**their flash attention is 8× ours** — f16 KV cache + cp.async/ldmatrix
pipeline + GQA packing. Attention is 41% of the whole gap.

**Lever verdicts (all wiring-proven):**
- **Wide-N restructure (deep-read #3): FALSIFIED on the Orin it targeted** —
  12B 163.0 → 147.1 (−9.8%), E4B −8.3% (and −13.5% on A5000). Losing the
  2-tile B-fragment reuse costs more than the 15→8 weight-pass cut saves;
  consistent with the device-copy falsification — weight traffic is NOT the
  binding constraint. (Fork-implemented, all correctness gates passed;
  preserved in the session record, not in the tree.)
- **rope frequency table: KEPT, +0.4% both models** (e8d9418) — the chunk
  rope was 84%-SM-saturated recomputing ~2M powf per launch; table built by
  the same powf → byte-identical.
- **Flash GQA packing (deep-read #4): KEPT, 12B +1.7%** (4299013) —
  163.0 → 165.8. Structurally the first slice of llama's 8× flash edge;
  E4B is gqa-4 (unpacked for now).

**Scoreboard after Phase 1b (Orin, warm serve, pinned clocks):**
12B **165.8** vs llama 216.8 (**0.76×**); E4B **382.9** vs 510.3 (0.75×).
A5000 unchanged at ~1770 / ~3600.

**What parity now requires — the honest ledger.** Byte-identity-preserving
levers still open: their q6_K edge (0.34 s, mechanism not yet isolated) and
scraps. Numerics-gated levers: f16 SWA rings + an async flash rewrite
(attention 0.57 → ~0.2–0.3 s; changes decode numerics), rmsnorm_add_n
reduction restructure (~0.1 s). Sum of EVERYTHING lands ~4.7 s ≈ 197 tok/s ≈
0.91× — true parity additionally needs the last 0.27 s of q4_K (their mmq
instruction schedule, priced and declined) or new ideas. The strategy
decision above this line belongs to the user.

## Orin push — Phase 2 (2026-07-02): option (b) — the numerics gate opens for attention

User decision: pursue the attention gap (their flash is 8×) with the
numerics-gated levers, under the f16-KV-precedent gate battery
(determinism + quality A/B + MTP==plain preserved) instead of byte-identity
vs f32.

**f16 SWA rings (33127a9): KEPT — the enabler.** The last f32 K/V goes
half; ring memory halves; the prefill flash sheds the per-fragment f32→f16
pack; decode's split-K reads half the bytes. Gate battery caught a real
bug byte-identity never would have: the device MTP draft kept reading the
rings as f32 — output stayed byte-correct (greedy verify absorbs bad
drafts) while acceptance silently collapsed 94% → 1.1%. Fixed (draft
dispatch follows kv->f16); MTP==plain preserved at 93.8%.

| | Orin prefill | Orin decode | A5000 prefill |
|---|---|---|---|
| 12B | 165.8 → **168.3** | 8.0 → **8.7 (+8.8%)** | 1770 → **1799** |
| E4B | 382.9 → **387.6** | 27.8 → **30.0 (+7.9%)** | 3602 → **3683** |

The decode win is the sleeper: rings were the decode-bandwidth path.

**Stage 2 — K/V shared staging in the packed f16 flash: FALSIFIED, and it
finally names the flash wall.** Fork-implemented (byte-identical, both legs
toggleable), Orin A/B: K+V staging 166.0 (−1.4%), K-only 168.0 (neutral)
vs 168.3 baseline. The V-restage physics failed a second time, and the
K-stage neutrality closes the re-read/latency theory. The real reason
staging cannot work here: **it swaps LDG for LDS one-for-one, and
mio_throttle counts both** — the kernel is memory-INSTRUCTION-count bound,
which is why GQA packing (fewer instructions) paid and staging (same
instructions + a sweep) cannot. llama's flash spends ~4× fewer fragment
instructions via `ldmatrix.x4` — fragment ECONOMY, not data placement, is
the remaining flash lever (the June ldmatrix falsification was on the
matmul A-path, a different site with a different diagnosis).

## Orin push — Phase 2 close-out (2026-07-02)

Final two rounds under the open numerics gate:

- **ldmatrix fragment economy: FALSIFIED** (166.3 vs 168.3) — even 4× fewer
  fragment instructions (64 LDSM replacing the fragment LDS traffic,
  byte-identical) moved nothing on the staging base. With staging
  (placement) and ldmatrix (instruction count) both dead, the flash
  kernel's floor is structural — barrier cadence + softmax serialization —
  and llama's 8× is their whole architecture, not a transplantable slice
  beyond the GQA packing already taken.
- **Warp-per-row chunk norms: KEPT (2615300)** — the block-per-row norms
  ran one row per SM (24% SM); a warp per row feeds every scheduler.
  Orin 12B +2.9%, **E4B +7.7%** (PLE norms compound); numerics-gated,
  all checks green, MTP 93.8% preserved.

**Final scoreboard (warm serve, 929-token turns, pinned clocks):**

| | Orin prefill | vs llama | Orin decode | A5000 prefill |
|---|---:|---|---:|---:|
| 12B | **173.2** | **0.80×** (216.8) | **8.7** (llama 7.55) | **1811** |
| E4B | **417.5** | **0.82×** (510.3) | **30.0** | **3718** |

> **ERRATUM (2026-07-16):** the decode column is wrong. The E4B "30.0" is an
> E2B QAT value (E4B serve decode is ~16.4 pinned), and the 12B 8.7 does not
> reproduce (8.0 sustained). The prefill columns are correct and reproduce
> within 1%. See the 2026-07-16 settle entry below and docs/benchmarks.md.

Option (b) delivered +4.5% / +9.0% prefill on top of Phase 1b plus ~+8%
decode (the f16-ring sleeper). Campaign total from the 2026-07-01 start:
A5000 3.4× / 3.6×; Orin 12B 102 → 173 and E4B 192 → 417 against the same
llama build. Every identified lever is now measured-and-kept or
measured-and-falsified; the remaining 0.2× is llama's mmq instruction
schedule (~0.27 s) and their flash architecture (~0.35 s) — both priced,
both requiring their kernels wholesale, both declined. Loose ends parked:
E4B G=4 GQA pack (needs a row restructure), mmcat frame-decode failure on
ffmpeg 4.4 (tools repo), the pre-existing E2B Q3_K garbage bug.

### Where this leaves prefill

The whole 2026-07-01/02 arc, all gates green on both devices:

| | day start | now | vs llama.cpp |
|---|---:|---:|---|
| A5000 12B warm serve | 533 | **~1760** | 0.24× → **0.78×** |
| A5000 E4B warm serve | 1020 | **3602** | 0.21× → **~0.69×** |
| Orin 12B / E4B | 102 / 192 | **125 / 246** | — (records) |

Post-sweep, every stall class in the two dominant kernels is either fixed
or measured at its structural floor: the mma kernel is balanced-stalled at
2 warps/scheduler (the "codegen, not a formula" wall, now confirmed with
counters from five angles), and the flash kernel is mio-instruction-count
bound on strided V. The residual vs llama.cpp is their ground-up MMQ
schedule; closing it means writing that kernel, which the stream-K
experiment already priced (docs/stream-k-experiment.md) and the project
declined. This design family is exhausted — further prefill work should
wait for new evidence or new hardware.
3. ~~Orin re-validation~~ **DONE (same day, worktree `~/lg-wide` on cortex,
   branch applied via patch series):**
   - Byte-identity narrow-vs-default at 4096 tokens (ring wrap exercised
     on-device): **BYTE-IDENTICAL** — the ring fix holds on the target.
   - 12B one-shot 910 tok, interleaved ×2: narrow 102.4/102.8 vs default
     **104.8/105.0** (+2.3%); gen 6.58 → 6.86 tok/s. No regression.
   - E4B: narrow 192.2/192.5 vs default **216.4/216.6 (+12.6%)** — past the
     README's ~183 record; balanced-768 cuts a 910-token turn from 8 weight
     passes to 2, the dominant cost under zero-copy weights.
   - Serve smoke: `engine warmup: 1.17s` at startup, first 19-token turn at
     0.28 s, Paris. 12B loads fine with the larger rings on the 16 GB board.
   The branch is validated on both devices.


---

## 2026-07-02 — TTFT: media prefills as it arrives (serve layer, no kernels)

The prefill-rate campaign built the engine; this spends it on latency.
serve() used to buffer the whole turn — every frame embedded on arrival but
prefilled only after the text line — so a video turn paid its entire prefill
inside time-to-first-token. Now each media span prefills as soon as its
frame is decoded (causality needs only completed prefixes; the bidirectional
window never leaves a span), with a one-span holdback ruled by a single
non-blocking MSG_PEEK (`sock_pending`):

- socket idle after embedding → the client is between messages; the held
  span prefills now, hiding under that pause;
- next frame already queued → the held span flushes ahead of it;
- the text line queued → the held span packs into the tail's single mixed
  call — a camera's frame+question burst is byte-for-byte the old packed
  turn, so the flagship zero-gap case cannot regress.

Packing only moves chunk boundaries, which the campaign's identity gates
proved unobservable in output. Measured (ttft = line sent → first reply
byte, client-side on the Orin, server-side `t1−t0` on the A5000; 3 turns
each, all stable to ±0.01 s):

| case | deferred | streamed | |
|---|---:|---:|---|
| Orin 12B, 6-frame video, 0.5 s arrival gaps | 8.79 s | **2.93 s** | 3.0×; turn wall 11.3 → 9.1 s |
| Orin E4B, camera burst (frame+line, zero gap) | 1.820 s | 1.819 s | =, glued (packed call 0.72 s both) |
| A5000 12B, 6-frame video (local mmcat) | 0.28 s | 0.14 s | 2.0× |
| A5000 12B single image / audio, E4B image | 0.13 s | 0.13 s | =, glued |

Gates: replies byte-identical on both devices across text / image / video /
audio / E4B-image; MTP acceptance counts exactly equal; text serve at the
campaign numbers (A5000 1802.8, Orin E4B 417 tok/s) — the text path is
structurally the same single tail call. The `turn:` stat line appends a
trailing `ttft %.2fs` field (existing bench regexes have no end anchor).
MAX_SEG span buffering is gone; the only per-turn bound left is the context.

For the ledger, two TTFT levers surfaced and left on the table:
- the first pick after any prefill costs ~66 ms on the A5000 (~4 decode
  steps) in BOTH builds — the tail's padded 32-wide chunk and the first
  decode drain serially in it; graph instantiation is once-per-process
  (nsys: 2 instantiates, 300 µs graph launches), so this is real GPU work,
  not API overhead. Worth an ncu look before believing any single theory.
- the legacy E4B vision encoder (~1.1 s/frame on Orin) now dominates the
  camera-burst ttft — the model waits on the eyes, not the other way round.

## 2026-07-02 — TTFT lever 2: the vision encoder on tensor cores

With streaming prefill landed, the E4B camera burst spent 1.1 of its 1.82 s
ttft inside the legacy gemma4v encoder — the model was waiting on the eyes.
A 5-reader code audit + nsys agreed on the cause: the encoder ran entirely
as f32 CUDA-core FMA (~21% of even the f32 peak, ~5% of the f16-TC peak),
with the f16 weights widened to f32 per use, and an attention kernel
(one block per query×head, 2 FMAs per 5-shuffle dot) that cost more wall
time than all seven GEMMs together — 592 ms vs 473 ms of the 1.1 s frame.
Weight residency was ruled out with evidence (encoder weights were already
a cudaMalloc device copy on both devices; the zero-copy-uncached story
belongs to the main model blob only).

Rewrite (media-cuda.cu, same file, same layer structure):
- `k_gemm` → m16n8k16 tensor cores, f16 inputs / **f32 accumulation**
  (tighter than llama.cpp's f16-accumulate cuBLAS encoder). W stays the f16
  it is in memory; A rounds to f16 at shared-staging time. 8 warps, 64×64
  tile, m16×n32 warp stripes, 64-deep k slabs.
- `k_attn` → tensor-core flash: one CTA = 64 queries × head, each warp owns
  16 query rows end-to-end (own S tile, own online softmax, own O tile — no
  cross-warp reductions). K/V stream through shared as f16, V staged
  transposed; P refragmented in registers (c-frag == a-frag positions for
  k16). Static shared — the old 48 KB dynamic-shmem patch cap is gone.
- im2col + position add moved on-device (`k_im2col`, `k_posadd`): the raw
  0.9 MB frame uploads instead of a host-built 7.2 MB f32 pair, and the
  position table uploads once at init.

Measured (624×480 = 1170 patches → 130 tokens, warm):

| | before | after | |
|---|---:|---:|---|
| Orin embed/frame | 1.1 s | **0.2 s** | 5.5× |
| Orin camera-burst ttft | 1.82 s | **0.92 s** | prefill (0.72 s) now dominates |
| Orin k_attn / k_gemm per frame | 592 / 473 ms | 22.5 / 146 ms | 26× / 3.2× |
| A5000 embed/frame | ~0.2 s | <0.1 s | image ttft 0.12 → 0.07 s |

Gates: LG_MEDIA_VERIFY max |diff| 0.0055 (A5000) / 0.0064 (Orin) — the f16
input-rounding class, as predicted, and consistent across devices; E4B
note.png reply word-for-word identical to the f32 build; Orin camera reply
identical; E4B video reply coherent with the frame counter read correctly;
the 12B unified path is host-only and untouched.

Left on the table, in order of current ttft share:
- the per-frame LLM span prefill pads 130 media tokens to a 192-column wide
  chunk plus a ~32-column marker chunk (~224 cols for ~136 real positions)
  — now the largest term in the 0.92 s camera ttft (0.72 s packed call);
- k_gemm keeps ~2× headroom (no cp.async pipeline; FFN slabs re-streamed
  ceil(np/64)× — subsumed if it ever gets the A-staging treatment the LLM
  q4k kernel got);
- encoder work on its own stream could overlap span prefill on the A5000
  (stream-serialized today; less relevant on the 8-SM Orin).

## 2026-07-02 — TTFT lever 3: media spans pack with their text

The packer in model_prefill_mixed gave any media span wider than PREFILL_B
(128) its own %64-rounded wide chunk, orphaning the text around it: a camera
turn (opener + 130-row frame + question, ~157 real positions) prefilled as
three padded launches — 32 + 192 + 32 = 256 columns. But the balanced budget
for that turn is 192, and a whole span packs inside it next to its text; the
special case predated wide chunks and init-sized buffers, which already
guarantee any chunk up to g_prefill_max_b fits every buffer and the SWA ring
spare. The fix DELETES the special case: spans pack greedily with text to
the balanced budget like small spans always did, and only a span wider than
the whole budget still stretches its own chunk (up to g_prefill_max_b;
beyond that the old causal sub-chunk fallback stands). The packer got
shorter.

| Orin E4B camera burst | before | after |
|---|---:|---:|
| packed prefill call | 0.72 s (256 cols) | **0.49 s (192 cols)** |
| client ttft | 0.92 s | **0.685 s** |

Session ledger for that turn: 1.82 s at dawn (f32 encoder + 3-launch
prefill) → 0.92 s (tensor-core encoder) → **0.685 s** (span packing) — 2.7×.

Gates: all five A5000 battery replies byte-identical (chunk boundaries are
output-invisible, re-proven end to end); MTP acceptance counts exactly
equal (76/76, 16/24, 28/60, 118/210 — the B=3 tripwire); text serve at
campaign numbers on both devices (A5000 12B 1812 avg, Orin E4B 417.4,
Paris OK — the text path takes the same branches as before); Orin camera
reply identical, 6-frame video coherent.

Still open, smaller: encoder k_gemm ~2x residual (cp.async), the %64 wide-
chunk rounding (192-for-157 keeps ~35 pad columns; a %32 wide launch path
would save ~0.07 s but complicates the fat-launch contract), the ~66 ms
A5000 first-pick, and encoder-stream overlap for bursts. The camera turn now
spends 0.2 s seeing and 0.49 s reading — both terms at structural-ish floors
without new kernel families.

## 2026-07-02 — TTFT lever 4: the 12B unified media path on GPU

The unified (gemma4uv/gemma4ua) path — the likely production path — still
encoded on the HOST: an image was ~11 GFLOP of OpenMP matmats (~0.4 s/frame
on the Orin CPU), audio a frame-at-a-time matvec loop, both serial in front
of every span prefill. There is no transformer to port: image = LayerNorm of
the raw 48x48 patch → [6912→3840] linear → bias/LN/pos/LN/rms → [3840→3840]
projection; audio = rms of a raw 40 ms frame → one [640→3840] linear. On the
tensor-core k_gemm from lever 2 that is two launches and change per image,
and the audio loop becomes ONE batched GEMM (all shapes %64-clean). Both
embedders now try the GPU first with the host kept as oracle/fallback, same
seam as legacy; im2col got range parameters ([0,1] unified vs [-1,1] legacy,
host op order preserved).

| Orin 12B | before | after |
|---|---:|---:|
| image embed (130 tok) | ~0.4-0.5 s (host) | **~0.02 s** |
| camera-burst ttft | ~1.7 s | **1.14 s** |
| 6-frame paced video ttft | 2.93 s | **2.34 s** |
| 6-frame paced video wall | 9.07 s | **6.65 s** |

The video pipeline is now GPU-saturated: per-frame "embed" logs read ~1.0 s
because the blocking D2H drains the PREVIOUS span's in-flight prefill — the
encoder itself is ~20 ms and encode is entirely off the critical path.

**Oracle forensics worth remembering.** The Orin verify first read 0.15 where
the A5000 read 0.012 on the same input — a 12x device inconsistency that
looked like a kernel bug. Dumping all four legs settled it: **the GPU rows
are bit-identical across A5000 and Orin (max |diff| 0.0)**; it is the two
HOST references that disagree with each other by 0.15 (fast-math OpenMP,
x86 AVX vs ARM NEON, on unnormalized rows with |x| up to ~50 — ~3e-3
relative). The host leg the Orin ran in production until today WAS the
0.15 leg; the GPU path is more consistent than what it replaces. Rule:
when LG_MEDIA_VERIFY reads high on one device only, suspect the reference.

Gates: image reply word-for-word identical to the host path (A5000, 12B
note.png); audio reply re-phrased but heard the same sentence (the expected
greedy sensitivity to ~3e-3 embedding drift — quality-equivalent class);
both deterministic across repeated runs; Orin camera reply identical, video
reply reads the counter; E4B legacy battery reply IDENTICAL after the
dispatch refactor; CPU `run` target still builds (stubs extended).

## 2026-07-02 — 12B audio input: two experiments close the question

Post-lever-4 the audio input path is at its floor server-side (embed ~0, one
batched GEMM, span packs with the text, GPU rows bit-identical across
devices). Two experiments probed what is left:

**Causal audio: FALSIFIED with proof.** If audio tokens tolerated a causal
mask, a live microphone could prefill DURING speech (each 40 ms token is a
pure function of its own frame — no receptive field to recompute). They do
not: LG_NO_IMG_BIDIR on three clips degrades every one (12B, A5000 —
"the road never comes to an end" -> "When will the fall never come to an
end?", a speech clip heard as "a beep or chirp", plus channel-token
stutter). Bidirectional attention within an audio span is load-bearing,
same as vision — an utterance span can only prefill WHOLE, after it ends.

**Chunked utterance spans: a latency/quality DIAL, measured.** The protocol
already takes several 'A' frames per turn; each becomes its own bidir span
and prefills on arrival (the lever-1 holdback). Orin 12B, the 3.1 s alice
clip, live-capture shape:

| chunking | post-utterance ttft | heard |
|---|---:|---|
| whole (1 span) | 0.86 s | "the farm never come to an end" (baseline garble) |
| 2 x ~1.5 s | **0.48 s** | "the poem never came to an end" — gist intact, noun drifts |
| 4 x ~0.8 s | 1.09 s (burst) | "what's coming, what's coming" — collapsed, ran to the GEN cap |

So ~1.5-2 s chunks buy roughly half the post-utterance latency at a mild,
real comprehension cost that will accumulate with utterance length; finer
chunks destroy the percept. Client-side policy, zero server work — the safe
default stays whole-utterance. Remaining open on audio: the combined A+V
template order (mp4 soundtrack + frames, parked in little-gemma-tools), and
for a voice product the loop is now bounded by DECODE (7.7 tok/s Orin 12B),
not input.

## 2026-07-02 — combined audio+video: verified, and the answer is a boundary

The parked question was which ORDER a video and its soundtrack should take
in the turn. The answer, after seven arrangements on the 12B unified path
(Orin, greedy, testcard frames + a known spoken clip; each modality alone
answers correctly): NONE of them work —

| arrangement | audio comprehension |
|---|---|
| audio alone — question before OR after the span | hears the sentence |
| frames then audio (mmcat today) | confabulates from the visuals |
| audio then frames | confabulates |
| interleaved (frames / half the audio / frames / rest) | confabulates |
| frames, question, audio (the model card recommendation) | confabulates |
| a "soundtrack:" framing tag before the span | confabulates |
| BLACK frames + audio | confabulates ("I am going to be a doctor") |
| video turn, then audio turn (separate turns) | confabulates |
| camera turn, then voice turn | denies having ears outright |

Vision survives audio presence; audio does not survive vision — an
asymmetric capture, and it is CONTEXT-scoped, not turn-scoped (an earlier
camera turn silences a later voice turn: the model answers "I am a
text-based AI and cannot hear"). No reference implementation exists to
disagree — the fork cannot even load a unified A+V mmproj (clip.cpp "TODO:
support both audio and video"), and the model card's placement advice
covers separate tasks. Conclusion: a capability boundary of this
checkpoint, not a template bug.

Consequences, shipped in little-gemma-tools (b82d853): mmcat's audio policy
is now TURN-scoped — audio-only turns keep the model's native ears
(unchanged); when any file in the turn has video, the soundtrack's speech
fuses onto the question as a whisper transcript (whisper_caption now
returns the words), and without whisper the span is DROPPED with a loud
warning, because a poisoned span helps nobody. Drop leg verified end to end
(A+V mp4: video described, zero confabulation; wav-only unchanged); the
whisper leg is coded on the proven note_caption pattern, untested here (no
whisper-cli on this box).

Production guidance for the voice+camera Orin product: within one session
the model's own ears work only while the context is vision-free. A session
that mixes camera and voice needs whisper for ALL speech input (native
hearing reserved for audio-only sessions), or separate sessions per
modality. Harnesses: ~/ttft_av.py (arrangements), ~/ttft_2t.py (split
turns) on cortex; scratchpad copies kept with the campaign files.

## 2026-07-02 — live voice: streaming transcripts and barge-in

The whisper-everywhere consequence of the A+V exclusion raised the obvious
objection: a 10 s utterance must not mean 10 s of listening followed by one
monolithic whisper pass while everything waits. Two pieces close it:

**The prompt was already appendable.** A turn stays open until its newline
line; T frames append text to it at any time. So a voice client streams the
transcript INTO THE OPEN TURN while the user is still talking: whisper runs
over the growing utterance every ~1 s, and words two consecutive passes
agree on (LocalAgreement-2 — a confirmed prefix never changes) are
committed as T frames. By end-of-speech only the unstable tail is left:
post-utterance cost is one whisper pass over the last chunk, not the clip.
This is voicecat in little-gemma-tools (424ed72) — mic via ffmpeg, energy
VAD, single-threaded and mic-clocked, native-audio-span fallback for
vision-free sessions, --stdin-pcm/--rt for deterministic testing (a stub
whisper exercised the whole pipeline before a real one existed).

**Barge-in is one byte.** MEDIA_BARGE (0x02, cd479d3): sent while a reply
streams, the per-token peek the serve loop already did (client_signal, same
syscall pattern — decode untouched: A5000 1783 text / MTP 156/156 @ 131,
Orin 417.5, Paris OK) consumes it, closes the turn on the wire with
<turn|>, and falls through to the next turn. The cut-off reply STAYS in
the context — the model knows what it didn't get to say — and remembering
"I was interrupted" belongs to the client: voicecat opens the next turn
with a note. Measured on the Orin: "history of Rome in detail" barged at
32 tokens, then "(interrupting) short version please" answered with a
one-sentence Rome summary — context retained, interruption honored.

Also fixed along the way: a latent cmd.exe quoting bug in the whisper
invocation shared by mmcat/voicecat (a popen /c string that starts with a
quoted path gets its outer quotes stripped; wrapped in an extra pair).
Remaining for production voice: install whisper.cpp on the Orin (model
size sets the commit cadence), AEC/headset assumption for open-air mics,
and the TTS side of barge (stop speaking when barged — downstream of
voicecat's stdout).

## 2026-07-02 — TTFT vs llama-server, apples to apples

The campaign's prefill ratios came from llama-bench, a kernel-level rate.
TTFT is a SERVING number, so the fair reference is llama-server, measured
with the same client-side definition both stacks use: last request byte →
first streamed token (ttft_llama.py: /completion & /v1/chat/completions,
stream on, cache_prompt OFF — the fork streams Gemma thought-channel tokens
as delta.reasoning_content, and prompt caching would silently zero repeated
prompts). Same GGUFs, warm servers, first turn discarded, pinned clocks,
drop_caches + --no-mmap for the Orin 12B. Token parity on text confirmed by
prompt_n (928 vs our 929 — BOS accounting). The full table lives in the
README (TTFT section); the shape of it:

- TEXT tracks the known ratio, slightly kinder to us: ours ~0.86× in
  serving (A5000 12B actually WINS, 0.51 vs 0.53 s) vs 0.80× at the kernel
  level — llama-server's stack costs more per turn than our socket loop.
- MEDIA inverts it: image+question bursts run 1.5–2.4× FASTER here (Orin
  E4B 0.69 vs 1.65 s, Orin 12B 1.15 vs 2.11 s). The 12B rows are
  near-token-equal (157 vs 150 — the fork's gemma4uv also encodes
  natively), so that one is kernel-for-kernel honest: the TC encoder + span
  packing + one-launch turns is the margin. The E4B rows differ by POLICY
  (the fork upscales to a 252-token minimum → 293 tokens vs our native
  130) — report both numbers, never hide the asymmetry.
- llama's server-reported prompt_ms EXCLUDES its media encoder: ttft −
  prompt_ms ≈ 0.25 s (E4B) / 0.43 s (12B) on the Orin image rows is their
  encode+pick. Our prefill stat includes everything after the burst.
- Paced arrival (video streamed over a socket) has NO llama-server
  counterpart — one POST carries the whole request — so the streaming-
  prefill wins (video ttft 8.79 → 2.34 s) are vs our own deferred baseline
  and are reported that way.

Harness kept: .scratch/ttft_llama.py (also on cortex ~/). Pitfalls it
guards: cache_prompt default drift, reasoning_content deltas, timings only
in the final stream chunk, and the warm-up turn (first llama turn runs
~2-7x slow on Orin while buffers settle).

## 2026-07-02 — Orin power study: full utilization at half the power budget

Question: does little-gemma push the MAXN (40 W) envelope? Method: tegrastats
at 500 ms, timestamped, phase-marked (harness kept as .scratch/power_run*.sh +
parse_power.py, also on cortex ~/); clocks pinned except the DVFS idle cell;
VDD_IN is total module input — what a battery sees.

| phase | sec | VDD_IN avg | peak | CPU_GPU_CV | SOC | GPU% |
|---|---:|---:|---:|---:|---:|---:|
| idle, pinned clocks, no server | 20 | 7.26 W | 7.33 | 1.95 | 2.40 | 0 |
| idle, 12B resident | 20 | 7.23 W | 7.37 | 1.95 | 2.36 | 0 |
| prefill 12B (5x928-tok turns) | 30 | 19.48 W | 20.79 | 10.10 | 4.20 | 99 |
| decode 12B (1024 tok) | 132 | 22.22 W | 23.00 | 10.75 | 5.46 | 99 |
| camera bursts 12B (x6) | 16 | 21.34 W | 23.00 | 10.66 | 4.97 | 98 |
| decode 12B + MTP (1024 tok) | 114 | 20.67 W | 21.14 | 10.52 | 4.70 | 99 |
| prefill E4B (5x928) | 12 | 18.69 W | 19.38 | 9.55 | 4.08 | 97 |
| decode E4B (1024 tok) | 65 | 20.04 W | 21.07 | 9.47 | 4.95 | 99 |
| idle, DVFS (nvpmodel re-applied) | 20 | 6.15 W | 6.30 | 0.79 | 2.42 | 0 |
| idle, re-pinned | 8 | 7.45 W | 7.64 | 2.08 | 2.40 | 0 |

Findings:

- **Peak 23.0 W against a 40 W profile, at GR3D 99%.** Full utilization at
  ~55% of the power budget is exactly what a bandwidth-bound workload looks
  like: the SMs are busy WAITING on LPDDR, not burning ALU power. Decode
  (every weight streamed per token) is the hungriest phase and shows the
  highest SOC rail (5.46 W — the memory controller); compute-denser prefill
  is 2.7 W CHEAPER despite the same 99% utilization. Perf on this board is
  memory-limited, never power- or thermal-limited.
- **Energy per output token (VDD_IN x time / tokens):** 12B plain
  **2.85 J/tok** (22.22 W, 7.8 tok/s); 12B + MTP **2.29 J/tok** (20.67 W,
  9.0 tok/s at 45.7% prose acceptance) — speculative decoding is 16% faster
  AND 20% cheaper per token, since accepted drafts share weight passes;
  E4B **1.27 J/tok**. Prefill: ~113 J per 1k prompt tokens (12B),
  ~45 J/1k (E4B). A camera burst turn (12B, 130-tok frame + short reply)
  is ~56 J.
- **jetson_clocks costs ~1.1 W at idle** (7.26 vs 6.15 W DVFS; CPU_GPU_CV
  1.95 vs 0.79 W). For battery products, leave DVFS on — under sustained
  load it reaches max clocks anyway; pinning is a BENCH discipline, not a
  deployment setting. Idle floor is ~6.2 W even in DVFS (carrier board +
  SOC rails; deeper sleep states untouched here).
- CPU stays ~0% in every phase — the runner idle-waits on the GPU by
  design. Whisper (voicecat) will claim CPU headroom without touching the
  GPU budget.
- Battery arithmetic for a 100 Wh pack: ~16 h idle (DVFS, model resident),
  ~4.5 h of continuous 12B conversation, ~5 h with MTP, ~7.9 h E4B — and a
  10%-duty camera assistant (one burst every ~10 s over DVFS idle) lands
  around ~9 h.

## 2026-07-02 — TTFS: time to the first speakable sentence

Voice-to-voice products (TTS on the output) do not feel TTFT; they feel the
moment the first complete sentence exists, because that is when a
sentence-chunking TTS can start speaking — the industry calls the
end-to-end version voice-to-voice latency / time-to-first-audio, and the
LLM-side component first-sentence latency. TTFS = TTFT + (first-sentence
tokens at decode rate) + (the WHOLE thought channel first, if the model
thinks — thought is not speakable). Both harnesses now measure it
(.scratch/ttft_send.py, .scratch/ttft_llama.py): a scanner fires at the
first [.!?] followed by whitespace/closer in the SPEAKABLE stream, with
<|channel>…<channel|> spans and special tokens stripped; unterminated
thought spans correctly suppress it. llama-server needs --reasoning-format
none so its stream is byte-symmetric with our wire (its default parser
routes these models' entire replies into delta.reasoning_content).
Questions were made prose-shaped ("…explain in two or three sentences…" /
"Describe this image in two sentences.") so the first sentence is real.
Full table in the README; the shape:

- TEXT is the clean kernel comparison: identical prompt bytes → the
  identical greedy first sentence on both stacks (29 tok on the 12B, 19 on
  the E4B), so TTFS differences are pure TTFT + decode-rate. On the A5000
  our decode edge erases llama's TTFT lead within one sentence (1.27 vs
  1.28 s); on the Orin their prefill lead survives but shrinks (E4B: 0.32 s
  of TTFT gap becomes 0.22 s of TTFS gap).
- IMAGE inverts by an order of magnitude: ours 0.78/0.25 s (A5000
  12B/E4B) and 5.22/1.98 s (Orin) vs llama 8.9/4.7 s and 98.6/≈18 s. Two
  compounding causes: the media-TTFT levers, and the fork's chat template
  eliciting a 466–717-token thinking process before the first speakable
  word (~92 of llama's 98.6 s on the Orin 12B is thought at 7.4 tok/s).
  Our serve template gets a terse thought from the same weights. Even
  with their thought subtracted entirely, our media TTFS leads — but the
  headline lesson is that TEMPLATE-INDUCED THOUGHT LENGTH dwarfs every
  kernel in the stack for voice latency. Keeping thinking terse via -sys
  is a first-class voice optimization; MTP (which chews thought ~1.2–2×
  faster) is another.
- Ours-A5000 TTFS cells are composed (client AF_UNIX python is unavailable
  on Windows): ttft-stat + decode-time × char-fraction to the fire point in
  the captured reply — decode is constant-rate, so the composition is
  sound. Orin cells are direct client-side clocks.
- The Orin llama-E4B-image cell was lost to the board's SECOND NVMe
  read-only event of the day (mid-run, last cell); estimated from the
  A5000 thought length at Orin decode rate. The recurrence strengthens
  the suspect-the-NVMe-medium note in the ops ledger.
**Addendum (same day, post power-cycle):** the lost Orin llama-E4B-image
cell measured 35.8 s (481-token thought — the ≈18 s estimate was 2×
optimistic), vs ours 1.98 s. And the read-only mystery moved: smartmontools
(now installed on the board) reports ZERO media/data-integrity errors and an
empty error log — but 73 °C on the NVMe near-idle, with heavy accumulated
thermal-transition time. The two same-day read-only events both followed
hours of sustained bench load: the suspect is now NVMe THERMAL throttling /
link stall, not a failing medium. Fix is airflow or a heatsink pad on the
SSD, and the ops rule stands: check `mount` shows rw before trusting any
Orin run.

## 2026-07-02 — the appending prompt, showcased: a simulated audio stream

voicecat + a stub ASR (little-gemma-tools test/stub-whisper.sh — prompt-aware,
timestamped segments, sleeps 0.12x realtime; no ears, no external code) ran an
11.4s utterance into the Orin 12B two ways, timelines stamped client-side:

STREAMING (commit-ms 1000):        NAIVE (one ASR pass at the end):
  t= 2.2s  +2 words committed        (silent while "listening")
  t= 3.4s  +2 words
  t= 4.5s  +3 words
  t= 5.6s  +2, window -2.5s
  t= 7.5s  +5 words
  t= 8.7s  +2, window -2.5s
  t=10.6s  +5, window -2.5s,
           turn closed               t=11.5s  turn closed (everything at once)
  t=11.36s reply streams             t=12.01s reply streams

The turn text fills WHILE the user is still speaking (the runner's appendable
turn — 'T' frames into the open turn), the audio window trims behind the
confirmed words, and the final ASR pass at end-of-speech covers only the
~1.9s unconfirmed tail instead of the whole clip. The 0.65s advantage over
naive is entirely that final-pass shrink at the stub's 0.12x-RT cost; with
real whisper on the Orin (0.3-0.5x RT) the naive tail pass on 10s of speech
is 3-5s and the streaming tail stays under a second — and the gap grows
linearly with utterance length. Two voicecat fixes made the showcase honest
(tools de365f8): --rt paces to frame deadlines so ASR passes overlap capture
like a live mic, and whisper runs with timestamps so confirmed segments trim
out of the window (trimmed text rides --prompt as context). Real whisper.cpp
on the board remains the one missing production piece (external-source
install pending user approval).
**Addendum — real ears (whisper.cpp, CUDA base.en, user-installed at
~/repos/whisper.cpp):** the full pipeline runs end-to-end on the 12B: a 30 s
simulated stream commits 46 words live (t=5.6/10.9/16.0/19.4/26.1 s), trims
the window four times, and closes 1.0 s after end-of-speech. Three tunings
came out of the real measurements (tools cceae5c): mid-passes gate off once
trailing silence accumulates; the final pass is REUSED from the last
mid-pass when no voiced frame followed it (last_voice tracking); and the
commit cadence rule is ≥3× one ASR invocation (default now 2500 ms —
at 1000 ms the loop fell 12 s behind a 30 s utterance).

The honest surprise: this CUDA base.en costs ~0.55 s per invocation FLAT —
30 s of audio transcribes in barely more than 2 s of audio does, because
process+model load dominates. So on THIS stack the naive single end-pass is
~0.7 s faster than streaming, at any practical utterance length. Streaming
transcripts earn their keep as UX (live partial understanding), as bounded
memory, and as LATENCY the moment ASR slows down: CPU whisper (the
deployment that keeps the GPU exclusively for the LLM — likely the real
product choice), larger whisper models, or long dictation. And the
production upgrade that flips the verdict unconditionally is a PERSISTENT
whisper (server mode / linked libwhisper): it deletes the per-pass load
cost that is the entire naive advantage.

## 2026-07-02 — dictation: the appendable turn prefills while the user speaks

The last unhidden latency was the LONG spoken instruction: a 929-token turn
costs 5.37 s of prefill on the Orin 12B, all of it after the user stops
talking. The turn was already appendable ('T' frames) and voicecat already
streams confirmed transcript into it — the server just let that text SIT in
the chat buffer until the closing line. Now it flushes: accumulated 'T'
text past SERVE_TEXT_FLUSH (256 chars) prefills incrementally, split at the
last SPACE with the space kept in the remainder. That split is EXACT, not
approximate: SentencePiece pieces never cross a space, so the fragmented
token stream is byte-identical to tokenizing the whole turn at once —
proven directly (930 ids in every mode, replies byte-identical).

Orin 12B, the 929-token instruction (ttft_dictate.py, client-side clocks):

| delivery | ttft after last word | note |
|---|---:|---|
| single line (baseline) | 5.37 s | all prefill post-utterance |
| burst of 'T' frames | 6.72 s | nothing to hide behind + small-chunk pads (burst clients should send a line) |
| paced at 30 words/s | **0.55 s** | prefill fully hidden under the 30.7 s of speaking |

At a real dictation pace (~2.5 words/s) the hiding is even more
comfortable. TTFS follows: 9.23 -> 4.41 s, the remainder being thought +
first sentence at decode rate. Gates: replies byte-identical across all
three deliveries with EQUAL id counts (the tokenization invariant, observed
not assumed); text bench at campaign numbers (A5000 1824, Paris OK); MTP
156/156. voicecat needed NO change — its commits now trigger the flushes
automatically. With true streaming ASR (persistent whisper) this completes
the chain: speech -> transcript -> kv cache, all concurrent, and the answer
starts a beat after the user stops.
**Addendum — production decode config + plan B (measured, deterministic):**
with MTP block-3 the dictation scenario's decode runs 9.3 tok/s on the 12B
(59.3% acceptance on this reply) and 18.7 tok/s on the E4B (41.4%); first
token is untouched by MTP. Full matrix, seconds after the last spoken word:
12B deferred +5.37 token / +8.13 sentence, 12B streaming +0.55 / +3.31;
E4B deferred +2.23 / +3.20, E4B streaming **+0.26 / +1.24**. The E4B answers
this prompt with NO thought channel — its post-token gap is the sentence
alone, while the 12B's is mostly thought; template-induced thought length
remains the biggest TTFS variable. Timing diagrams (UML lifelines, both
models, all measured): **docs/dictation-timing.html** — self-contained,
open it in any browser.
## 2026-07-02 — QAT E2B: q4_0 support, and the prefill that launched nothing

Google released a QAT build of the E2B and the loader refused it:
`model: missing attn_q/attn_k for layer 15`. Two file-level surprises, one
real bug of ours.

**The file.** Despite the `Q4_K_XL` name, every quantized tensor is plain
**q4_0** (ggml type 2) — QAT trains *around* a simulated quantizer, and the
simplest scheme (symmetric, one f16 scale per 32) is the one you simulate
exactly; the release checkpoint IS the quantized weights, so q4_K's
sub-block scale/min machinery would buy nothing. Same 4.5 bits/weight.
And the 20 kv-shared layers (n_kv_start=15) are **pruned** of
attn_k/attn_v/attn_k_norm — they only read their source layer's cache, so
the dead projections are simply absent. Support was mechanical: q4_0 in
quant.c (+f32/i8 kernel subs), and the geometry loop inherits the source
layer's n_head_kv when a shared layer has no attn_k (the forwards already
never compute k/v past n_kv_start).

**The bug.** With q4_0 wired, the i8 backend still generated garbage —
while the f32 backend answered *Paris* on the same file. The bisect that
closed it: `LG_NO_MMA=1` made the i8 backend correct too. `matmul_q_n`'s
comment promised "everything else falls to the dp4a chunk kernel", but the
dp4a loop only ran under LG_NO_MMA — a type with no mma route (q4_0, q8_0,
q3/q5_K, untwinned floats) fell off the end of the function and launched
**nothing**. Stale d_out, garbage KV; decode is correct, so the model
confidently continues from trash. Prefill-only corruption is the
signature. This retroactively explains the parked "E2B Q3_K garbage" —
q3_K has no mma route either; that was never a quant-quality problem. The
fallthrough is now unconditional (and LG_NO_MMA just skips the mma gates).

**q4_0 done properly.** The first scalar subs decoded at 7 tok/s: 18-byte
blocks can't take 4-byte loads, and the Orin reads the in-place blob
uncached. q4_0 now joins the q3_K/q6_K repack club — 20-byte 4-aligned
`block_q4_0r` on device memory, dp4a subs (`__vsub4` with 0x08 bias, the
q6 pattern). Integer dots are order-exact, so decode/verify byte-identity
holds by construction.

**Orin, pinned clocks, warm.** E2B QAT: decode 26.3 tok/s plain (1.6x the
E4B's 16.2, as its size predicts), 31.5 tok/s on MTP prose at 78%
acceptance (the release's own mtp head loads and drafts). The user-facing
serve line works end to end: mmproj answers a red frame with "Red" (100
tokens, 0.3s encode), text turns coherent. E4B regression 271 tok/s
prefill, A5000 3722 — both at campaign numbers, Paris OK. **Open lever:**
q4_0 has no mma prefill route, so E2B prefills at 48 tok/s on the dp4a
fallback vs the E4B's 271 — an own m16n8k32 q4_0 kernel (simpler epilogue
than q4_K: no mins, no super-block) is the priced next step if QAT E2B
graduates toward production.

## 2026-07-02 — q4_0 joins the tensor cores: the mma flavor and the shape-shifting repack

"Since we added the support, we should support it well." The dp4a fallback
left QAT E2B prefilling at 48 tok/s on the Orin while the E4B did 271 —
q4_0 had no mma route. Now it does, and the trick was to not write a new
kernel at all.

**One layout to feed them all.** q4_0 now repacks into a **q4_K-shaped
144-byte superblock** (`block_q4_0m`): 8 consecutive 32-element blocks,
nibbles rearranged into q4_K's lo/hi pair layout, and the 8 f16 scales
sitting exactly where q4_K keeps its d/dmin/scales header. Two
static_asserts pin the contract (same sizeof, same qs offset). The payoff:
every q4_K walker reads it with its existing index math. The dp4a subs
became eight-line siblings of `sub_q4_K` (signed nibbles via `__vsub4`, so
the −8 rides the integer dot — no min term); the mma kernel gained a
compile-time `Q40` flavor whose only differences are the per-sub-block
scale decode (`d[sj]` from the same four header words) and a min-free
epilogue. The q4_K instantiation is untouched — proven, not assumed:
stash-A/B on a high-entropy prompt, **byte-identical**. MTP acceptance on
the E2B (75.7%, same reply text) confirms the reshaped decode/verify subs
kept their by-construction identity.

**Numbers (warm serve, 912-token turns).** Orin: 48 → **798 tok/s**
(16.6x the dp4a route, 1.9x the E4B's 417 — the half-size model finally
prefills like one). A5000: **6,682 tok/s** (1.8x E4B's 3,722). Decode rode
along: the uint4-friendly layout took plain decode 26.3 → 28.5 and the MTP
verify 37.7 → 20.1 ms/round → structured MTP decode 31.5 → **55.3 tok/s**.
Camera turn ttft 0.20s. mma-vs-dp4a replies diverge mid-text on the
high-entropy prompt — the same relaxed prefill class as q4_K's mma,
both coherent.

**A measurement trap, twice avoided.** One-shot `-p` charged the in-process
repack to the prompt timer: the A5000 "measured" 215 tok/s mma vs 200 dp4a
— both numbers are mostly repack. Serve mode (repack in engine warmup)
tells the truth; this is the same lesson the Orin one-shot benches taught
in Phase 0, now confirmed on the discrete card. The byte-wise nibble weave
also cost 3.2s of warmup; a word-wise rewrite (two mask-and-or lines per
16 bytes) brought it to 1.51s, gated byte-identical output.

## 2026-07-02 — dictation timing, plan C: the lightest Gemma closes the loop in 0.7s

Same protocol as the 12B/E4B matrix (929-token spoken instruction via
ttft_dictate.py, client-side clocks, MTP block-3, warmup turn discarded,
pinned clocks): QAT E2B on the Orin.

| model | deferred ttft / ttfs | streamed 30 w/s ttft / ttfs | reply decode |
|---|---:|---:|---|
| 12B | 5.37 / 8.13 s | 0.55 / 3.31 s | 9.3 tok/s, thought + sentence |
| E4B | 2.23 / 3.20 s | 0.26 / 1.24 s | 18.7 tok/s, no thought |
| **E2B QAT** | **1.15 / 1.76 s** | **0.10 / 0.72 s** | **31.6 tok/s @ 39.5% acc, no thought** |

Streamed dictation on the E2B: first token 0.10 s after the last spoken
word, first speakable sentence 0.72 s — the whole 930-token prefill hides
under the 30.6 s of speaking and what remains is one flush plus a
19-word sentence at 31.6 tok/s. Even DEFERRED delivery is now sub-2s to a
speakable sentence (811 tok/s turn prefill), so on this model streaming
buys a second, not five. Burst re-measured: 1.67/2.29 s — still pays the
small-chunk pads, plain line stays the right call for burst clients.
Like the E4B, the E2B answers with no thought channel; its ttfs−ttft gap
IS the sentence. Acceptance on this prose reply is 39.5% (the 256-wide
draft head), decode still nets 31.6 tok/s. Deferred runs are
reproducible to the millisecond (1.148/1.144 back-to-back).
docs/dictation-timing.html now carries the E2B rows (strip chart C).

## 2026-07-02 — the 8GB question: peak memory of the E2B dictation stack

"Would an Orin Nano Super 8GB run this?" Measured on the NX 16GB
(mempeak.sh: per-process smaps_rollup + nvmap iovmm clients + a
MemAvailable sampler), and the measuring taught us more than the number.

**Metric lesson first.** On Jetson, no single counter tells the truth:
VmHWM counts evictable file pages, and MemAvailable UNDER-counts nvmap
(GPU allocations landed while the watermark barely moved). The honest
figure is **anonymous RSS + nvmap** — neither is reclaimable.

**Two fixes fell out.** (1) ensure_weights pinned the whole 2.4GB blob on
Tegra while every q4_0 tensor also carried a repacked device copy — for
models whose bulk types are all repacked (the QAT E2B), the blob now
skips device residency entirely; the ~340MB f32 residual uploads
per-tensor (`weights_repacked()` seam, per backend). (2) That alone saved
nothing real, because gguf.c malloc+fread'd the blob — anonymous memory
the kernel can never reclaim. The data section is now **mmap'd** on POSIX
(PROT_READ|PROT_WRITE MAP_PRIVATE — Tegra's cudaHostRegister refuses
read-only pages, and that register failure silently fell back to the
blob copy that OOMs the 12B; a loud warning now marks that path). Blob
pages are file-backed and evictable; after warmup only the embedding
rows stay hot. All gates: E2B byte-identical through both changes,
12B/E4B registered zero-copy path intact (Paris), Windows keeps fread.

**The numbers (E2B QAT + MTP head, dictation workload):**

| component | unreclaimable |
|---|---:|
| little-gemma serve: nvmap (repack 2.3GB + residual + rings/scratch) | 3.00 GB |
| little-gemma serve: anonymous host RSS | 0.11 GB |
| whisper.cpp CUDA base.en, per pass (nvmap 0.42 + anon ~0.25) | ~0.7 GB |
| **stack peak, concurrent dictation + ASR** | **~3.9 GB** |

Plus up to 2.4GB of *evictable* blob page cache the OS keeps only while
it has room. Before the two fixes the same stack held ~5.5GB
unreclaimable (pinned blob + repack, both resident by construction).

**Verdict for the Nano Super 8GB:** ~3.9GB stack + ~1GB headless L4T
leaves ~2.5GB of slack on a 7.4GB-usable board — comfortable, with room
for the mmproj vision encoder (~0.7GB) on top. Perf will scale with the
Nano's lower GPU clocks, but memory is no longer the question. (Not
verified on the actual SKU — this is the NX 16GB measuring an 8GB
budget.)

## 2026-07-02 — the last number: TTS, and the voice loop closes at ~1.2s

TTFS told us when the first sentence is READY; the user experience ends at
when it is HEARD. Piper (en_US-ozgirl_v6-step18500, onnxruntime CPU — the
GPU stays reserved for the LLM) measured on the Orin, tts_time.py:

| mode | latency |
|---|---:|
| cold one-shot (python + espeak + onnx load + synth) | 2.47 s |
| service, per sentence (persistent process, lines in / raw PCM out) | **0.49 s** |

The load dominates the cold path, so piper runs as a service — same
verdict as whisper, and piper's stdin-line mode already IS the service:
one process, each line synthesized as it arrives. Steady-state it turns
the dictation reply's 21-word first sentence into 9.0–9.7 s of audio in
0.49 s (~18x realtime, three runs within 10 ms), all on CPU, 360 MB RSS
(stack + piper ≈ 4.3 GB — still comfortable in the 8 GB budget).

**The full equation, E2B QAT on the Orin, streamed dictation:** last
spoken word → first token 0.10 s → first sentence complete 0.72 s → first
audio byte **≈ 1.2 s**. Ears (whisper), brain (little-gemma + MTP), and
voice (piper) all on-device; while the 9 s first sentence plays, the rest
of the reply decodes 30x faster than it needs to. Refinements if 1.2 s
ever matters: piper synthesizes per sentence in one ONNX call, so
clause-level splitting would shave the 0.49, and TTS overlaps decode
already — the loop is speech-paced from here, not compute-paced.

## 2026-07-02 — the LLM is the splitter: a voice system prompt cuts first audio to 0.82s

Generating 9s of audio in one piper call was the visible waste; the real
finding is WHERE splitting can happen. The glue layer can flush to piper
at any punctuation mark — piper synthesizes whatever line it is handed —
but measurement shows the baseline E2B writes 21-word sentences with NO
commas: a clause-flusher has nothing to flush. Only the LLM can create
the split points. So the split policy lives in a system prompt, loaded
once at startup via the existing `-sys` (docs/voice-sys.txt): spoken
prose, no markdown, five-to-ten-word clauses, point first.

E2B follows it beautifully. The dictation reply became "Paris became the
capital of France in 1806. Napoleon Bonaparte moved the capital there.
He wanted a central location for his empire." — short sentences, zero
markdown to scrub, and the first speakable unit fell from 21 words to 8:

| | first token | first speakable | + piper | first audio |
|---|---:|---:|---:|---:|
| baseline | 0.102 s | 0.714 s | 0.49 s | 1.21 s |
| voice-sys | 0.101 s | **0.549 s** | **0.27 s** | **0.82 s** |

Piper scales ~linearly with audio length (0.27s for 5s of clause audio
vs 0.49s for 9.4s), so shorter first units pay off twice — earlier text
AND faster synthesis. The 5s first clause plays while sentence two is
long since decoded; the loop stays speech-paced.

Two honest notes. (1) Style pressure showed the small model's edges: the
voice-sys answer is crisper AND confidently wrong ("in 1806" — Paris
predates Napoleon as capital by centuries), where the baseline was vague
but defensible. (2) Prompt-only is the bootstrap; the production path is
the finetune — teach the model human pause placement (often SHORTER than
grammatical clauses) instead of asking a 2B to also be its own editor.

## 2026-07-03 — same silicon, different species: the cascade vs Moshi on the A5000

Moshi (Kyutai) is the natural reference point for conversational
latency: a full-duplex speech-to-speech foundation model, no text stage
at all. Measured on this A5000: TTFB to first audio ~0.13 s warm
(trials 252/129/135 ms; ~23 ms connection handshake, whether it's
included in the TTFB clock still to be confirmed). Moshi on an Orin is
impractical to set up, so the field was levelled the other way: the
little-gemma voice pipeline on the A5000, all three Gemma 4 models.

Protocol is the Orin dictation matrix verbatim — the 929-token spoken
instruction, typed 'T' frames at 30 words/s, voice-sys prompt, MTP
block-3, warmup turn discarded — with one port: Windows CPython has no
AF_UNIX, so the client is native C (.scratch/ttfc_dictate.c), a
byte-exact mirror of ttft_dictate.py that also clocks the FIRST CLAUSE
(any of ,;:.!? — the unit the TTS leg consumes). TTS is piper
en_US-kristin-medium on the host CPU (i7-8086K; the GPU belongs to the
LLM), persistent service, first-byte clocked per clause
(.scratch/tts_time_win.py). Medians of 3-4 runs, streamed delivery:

| model | first token | first clause | + piper | last word → first audio |
|---|---:|---:|---:|---:|
| 12B Q4_K_M | 0.090 s | 0.241 s | 0.109 s | **0.35 s** |
| E4B Q4_K_M | 0.065 s | 0.132 s | 0.108 s | **0.24 s** |
| E2B QAT q4_0 | 0.121 s | 0.29 s | 0.193 s | **0.48 s** |

Deferred delivery (whole line at turn end — no streaming to hide
behind) still closes at 0.79 / 0.41 / 0.43 s respectively. Replies were
byte-identical across deliveries per model, reconfirming the
space-boundary exactness on a second OS and a second client
implementation. Server crosschecks all on spec: deferred prefill ~6,970
/ 3,650 / 1,743 tok/s (E2B/E4B/12B), decode with MTP 145 / 118 / 69
tok/s at 22.5 / 39.3 / 52.4% acceptance.

Three readings. (1) Moshi is 2-4x ahead, and the gap is structural, not
an engineering deficit: a speech-native model starts vocoding without
waiting for a speakable text unit, while the cascade's floor is one
decoded clause plus one VITS call (~0.2 s at these rates). (2) In
exchange, the cascade stays inside the human turn-gap band (~0.2-0.5 s)
with a general-purpose text LLM underneath — tools, reading, 12B-class
answers vs Moshi's ~7B speech backbone — and swappable voices. (3) On
the A5000, model size has almost stopped mattering: 2B to 12B spans
first-clause 0.13-0.29 s, and the E2B is the SLOWEST end-to-end because
its clause says "in 1806." — which verbalizes to 3.9 s of audio and
0.19 s of synthesis. Once streaming hides prefill, content dominates
silicon. Corollary: the whole datacenter card + desktop CPU buys only
~1.7x over the 20W Jetson (0.48 vs 0.82 s on the same E2B) — in the
conversational regime the pipeline is floor-bound, not compute-bound.

One small inversion tells on desktop DVFS: E2B deferred first-clause
(0.232 s) beats paced (0.29 s). Its prefill is so fast (929 tokens in
0.13 s) that streaming hides nothing, while the 30 s paced send window
lets the unlocked GPU park at 210 MHz idle clocks and the reply pays
the ramp. The Orin never shows this — jetson_clocks pins it.

Pitfalls, both with teeth. (1) The first 12B pass read 3.9 s ttft —
uniformly 7.5x slow (prefill, decode, MTP verify alike) with SM 100%,
clocks a healthy 1920 MHz, and the memory controller near 0%. Cause:
`pkill` doesn't exist in Git Bash (masked by 2>/dev/null), three
servers stacked to 24.2/24.5 GB, and Windows WDDM PAGES oversubscribed
cudaMalloc memory per-launch instead of failing loudly. That counter
signature (SM pegged, DRAM idle) is the tell; `taskkill //F //IM` and a
memory.used check between servers is the rule. (2) Python's
BufferedReader.read(n) on a pipe blocks for the FULL n bytes — a
clause of PCM (~44 KB) sat invisible behind a 64 KB read while the
clock ran; os.read has available-bytes semantics. Both fixed before any
number above was recorded.

## 2026-07-03 — the scale, completed: piper weighs 0.31 GB

The 8GB study weighed little-gemma and whisper the honest way
(anonymous + nvmap) but piper only got a VmRSS glance (360 MB) from the
timing session — the very metric the study warned against. Measured
properly on the Orin (smaps_rollup Anonymous sampled at 50 ms through a
service session, including the 21-word / 9.4s-audio sentence as the
worst case): **309 MB peak anonymous**, VmHWM 340 MB — only ~30 MB of
the glance was evictable file pages, so RSS happened to be nearly
honest this time. CPU-only onnxruntime, so no nvmap side at all.

The peak lands during the LONGEST sentence — VITS activations scale
with output length — so the voice-sys clause-splitting loop actually
runs piper lighter than this ceiling.

Updated arithmetic for the Nano Super 8GB: 3.9 GB (LLM + ASR) + 0.31 GB
(TTS) ≈ **4.2 GB unreclaimable** for the full voice loop — ears, brain,
and mouth — leaving ~2.2 GB of slack on a 7.4GB-usable headless board,
still with room for the vision encoder. The conclusion survives its
last missing number.

## 2026-07-03 — piper on the GPU: measured, and the CPU-first bet was right

With the LLM leg down to 0.1–0.3 s, TTS is 30–50% of the voice loop — so
the obvious question got measured: would GPU piper help? First, the
archaeology: upstream DID attempt GPU. `--cuda` wires
CUDAExecutionProvider, and `cudnn_conv_algo_search: HEURISTIC`
(voice.py) is a scar — ORT's default exhaustive conv autotune makes the
first GPU call take seconds, someone hit it and dialed it down. It never
became the default because the target hardware (Rhasspy, Raspberry Pi)
has no CUDA, the pip default is CPU-only onnxruntime, and a
process-per-utterance CLI pays the CUDA context cost every call.

The A/B (A5000, ORT-gpu 1.27 prepended via PYTHONPATH, CUDA-13 +
cuDNN-9 DLLs borrowed from the moshi venv's torch/lib,
.scratch/piper_gpu_ab.py; kristin-medium, warm medians of 5):

| clause | CPU (i7-8086K) | CUDA (A5000) |
|---|---:|---:|
| 3.9 s of audio | 0.185 s | 0.116 s (1.6×) |
| 2.2 s of audio | **0.089 s** | 0.112 s (0.8× — loses) |
| first call | 0.081 s | 0.640 s |

The tell is ORT's own warning: **28 Memcpy nodes** inserted — VITS keeps
shape/duration/alignment ops on CPU, so every synthesis round-trips the
bus repeatedly, and launch+copy overhead swamps ~20 MB of conv math on
short utterances. GPU only wins as audio length grows. **The crossover
sits at our clause length**: voice-sys clause splitting deliberately
moved TTS into the short regime, so GPU TTS is worthless for this
pipeline even on a free datacenter GPU. On the Orin it is doubly moot:
no pip onnxruntime-gpu for Jetson aarch64, and the iGPU would contend
with LLM decode for the same LPDDR5.

Two traps for the record: the first A/B run silently fell back to CPU
(the provider DLL's *transitive* cudnn dependency resolves via PATH, not
add_dll_directory — and the "CUDA" numbers matching CPU was the only
tell; the script now hard-asserts the active provider), and piper's
session must be checked with get_providers(), never trusted.

The real lever for the TTS share is streaming synthesis: first-byte
currently equals whole-clause synthesis because VITS decodes the whole
utterance in one ONNX call. A chunked vocoder decode would put first PCM
at tens of milliseconds on the same CPU — the Moshi lesson (Mimi streams
by construction) applied to the cascade's mouth.

## 2026-07-03 — the streaming vocoder ships: first PCM is O(1), loop 0.82 → 0.65 s

Same day, lever pulled. VITS's two halves have opposite constraints: the
text encoder + duration + flow need the WHOLE clause (prosody is
global), but they're cheap; the HiFi-GAN upsampler is most of the
compute and is a pure convnet — finite receptive field, no attention, no
recurrence — so it can decode in overlapped chunks whose interiors are
exact. piper's published ONNX is monolithic, but the graph names its
decoder (`/dec/…`), and exactly ONE tensor crosses into it (z·mask
feeding dec.conv_pre). `.scratch/vits_split.py` finds that boundary
automatically and extracts enc/dec halves from any stock piper voice —
no retraining, no re-export; verified on kristin-medium and the ozgirl
finetune (same boundary, both).

Chunked decode (`.scratch/piper_stream.py`, drop-in for
`piper --output-raw`): overlap R=16 frames saturates exactness — max
sample diff 4e-07 vs the monolithic decode, pure fp32 noise, both
machines both voices. First chunk K=10 frames = 0.12 s of audio.

End-to-end through the service on the Orin (ozgirl, includes espeak +
enc + chunk decode + pipe; `.scratch/tts_stream_time.py`):

| clause | stock piper first byte | streaming | audio |
|---|---:|---:|---:|
| 8-word voice-sys clause | ~0.27 s | **0.10 s** | 4.8 s |
| 21-word baseline sentence | 0.49 s | **0.15 s** | 9.1 s |

First byte no longer scales with clause length (enc's linear term is
tiny); synthesis stays 6–13× realtime despite the overlap redundancy
(the CPU cores were idle anyway). A5000 bench: 0.036–0.065 s first byte,
2.5–5× over baseline, same exactness.

**Voice loop arithmetic (E2B QAT, Orin, streamed dictation):** last word
→ first clause 0.549 s + streaming TTS 0.10 s ≈ **0.65 s first audio**
(was 0.82; composed from same-box measured legs, the same method as the
0.82). The ablation ladder now reads: baseline 1.21 → voice-sys 0.82 →
+streaming vocoder 0.65. Even WITHOUT the voice-sys prompt the loop
would hit 0.714 + 0.15 ≈ 0.87 s — the vocoder alone nearly matches what
prompt engineering bought, and they compose.

One design note: streamed chunks can't be peak-normalized (that needs
the whole clause); the service ships VITS's native level with a fixed
gain. And one measurement note: the in-process bench understated first
byte by the espeak phonemization cost — the service-level number (0.10)
is the honest one; always clock through the pipe.

Productionized same day in our piper fork (cortexist/piper1-gpl, branch
vits-streaming, 7e03c15): `python -m piper.split <voice>` writes the
halves, `PiperVoice.load(streaming=True)` + `synthesize_stream()` or the
CLI's `--output-raw --stream` consume them; boundary discovery is
automatic, tests build a miniature VITS graph by hand (no bundled
voice needed), full suite 22/22. i7 CLI first byte 0.19 → 0.05 s /
0.37 → 0.09 s. Upstream PR candidate once proven general. Repo
convention note: piper models now live under local/ (gitignored), not
data/ — both machines.

## 2026-07-03 — the missing reference: llama.cpp on the QAT E2B

The prefill finals table had one hole: no llama.cpp number for the QAT
E2B (the Windows attempt died in llama-completion's chat-template code;
llama-bench never applies templates, so it was always the right tool).
On the Orin, fork build 83efbcc79 loads the QAT file fine.

Same-day pair under pinned clocks (stored + restored via jetson_clocks):
llama-bench pp929 871 (fa0) / **1,021 tok/s** (fa1, best); little-gemma
serve re-measured in the same session: warm turns 813.7-815.4 →
**~815 tok/s**. Ratio **0.80×** — dead consistent with 12B 0.80× / E4B
0.82×. Methodology check: the E4B llama reference reproduced at 524 vs
~509 recorded (3% day drift) — which is exactly why the pair had to be
same-day; the recorded 798 against today's 1,021 would have understated
us at 0.78×.

Also captured while pinned: llama tg32 **37.8 tok/s** (fa1) vs our
recorded 28.5 plain decode — llama.cpp LEADS on q4_0 decode (~0.75×),
plausibly because q4_0 is their oldest and most-tuned quant while ours
rides the q4_K-shaped repack path. Flagged, not printed: our side of
that pair needs a pinned-clocks re-measure first. Honest weak spot
either way — the E2B decode win memory (“25% faster”) was the OLD Q3_K
model, and does not transfer to the QAT release.

Harness preserved: .scratch/bench-serve-orin.sh (the PS1 bench's bash
twin — one connection per turn, first discarded).

## 2026-07-04 — a little temperature: sampling lands without breaking a single gate

Greedy-only was a design pillar (byte-identity, MTP verify equality, the
acceptance tripwires), so sampling enters through the narrowest possible
door: a `model_pick` hook in model.h. NULL — the default — leaves every
backend's device-argmax path untouched, bit for bit; `-temp` installs a
host sampler (temperature over top-k, truncated at top-p) and each
backend calls it on the softcapped logits row instead of the argmax —
plain decode and the MTP verify alike, no call-site changes anywhere.

The MTP composition is the part worth writing down: the verify samples
the TARGET's own distribution at every block position, and a draft is
accepted only when the sample agrees. The emitted tokens are exactly
target samples — speculation still changes only tokens/s, never the
distribution, no rejection-sampling correction needed. Acceptance drops
as the price (E2B planets: 76.6% greedy → 28–46% at temp 1.0) but the
batching stays net-positive. Drafts remain greedy — they only gate
batching; the pick is what lands.

Defaults come from the model itself: Gemma 4 ggufs ship
`general.sampling.*` (top-k 64, top-p 0.95), read as the `-topk`/`-topp`
fallbacks. `-seed` fixes the run (same backend ⇒ byte-reproducible,
verified); without it a time-derived seed prints to stderr. Cross-backend
identity remains a greedy-only property — a boundary case in the
cumulative distribution can flip on ~1e-6 logit differences.

Gates, all green: default greedy byte-identical to the pre-change binary
with the 76.6% acceptance tripwire unchanged; seed-42 runs identical
twice; different seeds diverge into distinct coherent replies; CPU and
f32 backends compile with the hook; serve mode samples end-to-end.

One build trap for the record: installing CUDA 13.1/13.3 deleted the
CUDA_PATH_V13_0 env var and repointed CUDA_PATH, so the VS2019-generated
project (pinned to 13.0 targets) failed with an empty toolkit dir — and
half-fixing it (V13_0 only) mixes 13.3 headers with 13.0 cudart
(LNK2019: cudaGetDeviceProperties_v2). Export BOTH to v13.0 to build.
Also caught: yesterday's "rebuild" was an MSBuild no-op (binary mtime
unmoved) — check the timestamp, not the success banner.

## 2026-07-04 — the third species measured: HF speech-to-speech on the same board, same brain

huggingface/speech-to-speech (pip 0.2.10) is the closest published
relative of our pipeline: a cascaded, open-weight, python-glue stack —
VAD → STT → LLM over HTTP → TTS. On the Orin we ran it against the SAME
brain we use (llama-server, GPU, the E2B QAT gguf — our llama.cpp fork
build serves /v1/responses fine), with faster-whisper (CPU int8) ears
and MMS-TTS (CPU VITS) mouth. Probe = end of speech (last live-paced
16 kHz chunk) → first reply audio byte, spoken question synthesized by
ozgirl, six trials per session.

| configuration | warm TTFB (median) | spread | cold first turn |
|---|---:|---|---:|
| defaults | ~1.6 s | 0.9–2.2 s | 3.4–23 s |
| tuned (--stream_batch_sentences 1) | **~0.9 s** | 0.67–1.69 s | ~2 s |

Same board, same LLM, our pipeline composes 0.65 s — and the delta is
the paper's thesis in one experiment, because everything else is held
equal: their chain runs serially AFTER end of speech (full ASR pass,
full prompt prefill, first-sentence decode, whole-first-sentence VITS
synthesis), while ours hides ASR and prefill under the speech itself
and streams the vocoder. Their default even batches THREE sentences
before the first TTS call (stream_batch_sentences=3 — the difference
between 1.6 and 0.9 medians). Their variance couples to reply length
(first byte waits on the whole first sentence); ours is ~O(1).

Also learned: torch 2.12 cu130 from PyPI cannot see the Tegra iGPU
(cuda.is_available()=False), so their STT/TTS run CPU — the same
division of labor we chose deliberately, reached by necessity. VAD
min_silence is only 64 ms; the latency is compute, not hang time. The
YouTube-demo feel is plausible on a desktop GPU with short replies; on
a Jetson with an unmodified llama-server, ~0.9 s median is the honest
tuned number. Setup traps for reproduction: apt-pandas/pip-numpy ABI
clash (fix: pip install --user --ignore-installed pandas), the package
needs --responses_api_api_key even for local servers, one client
connection per process lifetime, and pkill -f self-match over ssh bit
again. Probe preserved as bench/s2s_probe.py.

## 2026-07-04 — HF s2s, part two: the TTS is tighter than the LLM, and long input breaks it

Reading their installed sources answers where the engineering went. The
TTS layer is TWO-TIER: the CPU-viable backends (facebookMMS — what we
benchmarked — and the melo/chatTTS/kokoro family) synthesize the WHOLE
sentence in one call and chop it into 512-sample blocks afterwards —
transport chunking, stock-piper behavior, first byte waits for the
sentence. But their flagship backends genuinely stream: qwen3
(faster-qwen3-tts) prefills the full target text then decodes codec
tokens incrementally, 8 codec steps per chunk, with a CUDA-graph
captured decode loop; pocket (Kyutai) yields 10–20 ms chunks by
construction. The streaming lives in codec-token TTS architectures that
stream natively — nobody retrofits streaming onto VITS the way our
latent-boundary split does, and their streaming tier wants a GPU. The
LLM side, by contrast, is a plain HTTP client with post-hoc nltk
sent_tokenize and a batch-3-sentences default: sentence-level flushing
only, no clause policy, no model-side split pressure.

Long input, measured honestly: it BREAKS at defaults. A 31.8 s natural
spoken question gets segmented at comma-length breath pauses (silero +
min_silence 64 ms), and the pipeline REPLIES INTO THE USER'S ONGOING
SPEECH — our probe clocked first audio 25–28 s BEFORE end of speech.
Raising min_silence did not save it: silero assigns weak probabilities
to our synthetic voice and only ever caught ~0.9 s FRAGMENTS (even the
short-question benchmark ran on an 0.884 s fragment — which means the
0.9 s TTFB we measured had an artificially LIGHT ASR leg; a real mic
would score somewhat worse). The scaling legs, measured in isolation:
faster-whisper base int8 on the Orin CPU takes 3.9 s for 31.8 s of
audio (their tiny.en default ≈ 2 s), plus a full prompt prefill after
end of speech — so even with perfect VAD their 30-second-utterance
TTFB projects to ~3.5–5 s, growing linearly with utterance length,
where ours is 0.65 s flat (the 929-token instruction is the measured
case). Prefill-under-speech and streaming ASR commits are exactly the
difference, as predicted.

## 2026-07-05 — Figure 5: TTFB vs prompt length, and it's all ASR placement

*(Figure numbers here follow the paper's current numbering: the board photo
became Figure 1 on 2026-07-07, shifting this sweep from 4 to 5.)*

Swept the §5.9 comparison across length (docs/fig-ttfb-vs-length.svg,
paper §5.9). Six coherent questions of increasing length on one Orin NX,
same E2B QAT weights, each leg clocked separately.

**Ours is flat.** Streamed `ttft` (prefill-under-speech) held 0.09–0.12 s
across all six lengths; the deferred (one-shot line) `ttft` rose only
0.09 → 0.20 s at 90 words. So at conversational lengths (11–90 words) the
LLM prefill is NOT a differentiator — deferred prefill of a short prompt is
already sub-0.2 s. The streaming win (5.37 → 0.55 s) was a 929-TOKEN
artifact; for a normal spoken question, prefill is cheap either way.

**The divergence is the ASR leg.** faster-whisper base-int8 on the six
synthesized clips: 1.20 s at 3.8 s of audio → 3.52 s at 32.1 s (the 32 s
point reproduces the old 3.9 s / 31.8 s anchor). HF runs this serially
AFTER end of speech; ours streams commits DURING speech, leaving one
constant final pass. HF perfect-VAD floor = measured base ASR + 0.45 s
downstream (prefill + first-sentence decode + first MMS-TTS, anchored to
their measured 0.9 s tuned point) → ~1.6 s at 5 s, ~3.9 s at 30 s. Ours:
0.65 s (streamed text) / ~1.0 s (live-mic, +streaming-ASR final commit),
flat. The chart's whole gap is ASR placement; everything else is roughly
equal and roughly constant.

**Measurement gotchas (cost most of the session):**
- Whisper temperature-fallback 5x-inflated synthetic speech: a 5.5 s clip
  took 27 s at defaults, 5.3 s with `temperature=[0.0],
  condition_on_previous_text=False`. Run-on / truncated / repeated
  synthetic text triggers the retry machinery. Use temp0-no-fallback and
  COHERENT stimulus.
- Orin shared-memory SoC: a GPU serve running alongside 3–4x's a CPU-bound
  int8 whisper pass (LPDDR5 bandwidth contention). Measure the ASR leg with
  serve+app STOPPED and jetson_clocks pinned — otherwise non-monotonic
  garbage (16→20→19 s for 5→10 s clips). This is the same "measure anon,
  isolate the SoC" lesson as the memory bench notes.
- Every ~/ test clip (alice, t2s, t11s, demo10/30) was the SAME repeated
  piper phrase "Would the form never come to an end?" — whisper collapses
  repetition at temp0, so prefix-slicing them is degenerate. Needed fresh
  coherent questions synthesized on the spot.
- Harnesses: bench/ttfb_vs_length.py (ours + deferred),
  bench/asr_leg_vs_length.py (base/tiny ASR), bench/gen-fig-ttfb-vs-length.py
  (moved from docs/ on 2026-07-07).
- Version pin: all HF speech-to-speech measurements in this journal
  (2026-07-03/04 entries and this one's composed floor) were taken against
  **speech-to-speech v0.2.10** (PyPI, released 2026-06-11). Their codebase
  moves quickly — reinstalls must pin `speech-to-speech==0.2.10` to
  reproduce these numbers.

## 2026-07-07 — the decode pair closed: near-parity, and the 0.75× scare retires

The one measurement Appendix B still owed: our E2B QAT plain decode,
paired same-session with llama-bench under pinned clocks (the 2026-07-03
entry had flagged the unpaired 28.5 as unprintable). Clock-state sanity
first: llama pp929 reproduced at 1,020.95 vs the recorded 1,021 —
conditions identical to the reference session.

The pair: llama tg32 **37.91 ± 0.12** (fa1); ours **36.4 tok/s**, dead
flat across 7 serve turns (21-tok prompt, 78-tok reply, one connection
per turn, first discarded — decode showed no warmup at all). Ratio
**0.96×, near-parity**. The old 28.5 — and the ~0.75× "llama.cpp LEADS"
reading it implied — was a stale-conditions artifact; a same-session
26% swing is exactly why the 07-03 entry refused to print the ratio
from a one-sided number. §5.2 now carries the row with its protocol
stated (78-tok replies vs tg32, not the 256-token protocol of the
E4B/12B rows).

Bench-harness hygiene from the same session: `bench-serve.ps1` had
absolute `C:\Users\Zero` paths headed for the public repo — now
`$PSScriptRoot`-relative with an `LG_BENCH_MODEL` override. And two ssh
traps re-confirmed the hard way: `cd x && server & client` puts the cd
inside the backgrounded subshell (clients ran from $HOME), and
`pkill -f "run-cuda-i8 -m"` matches the remote shell's own command line
and kills the session — use `pkill -x`.

## 2026-07-16 — float4 chunk rmsnorm: a small but real prefill win on E4B

Branch `prefill-e4b-orin`. The 2026-07-02 sweep left the elementwise pool
at ~0.18 s of the 1.44 s Orin gap with no isolated verdict. ncu on the
chunk norms (E4B, A5000): `rmsnorm_w_n_kernel` 43% memory throughput /
28% SM at 30% occupancy; `rmsnorm_add_w_n_kernel` 74% / 16%. Both are
scalar one-float-per-lane loops over `n` — each warp reads `x` twice
(rowsum + multiply), `w` once, writes once, with 80–120 scalar iterations
per row (n=2560/3840).

Change: float4 vectorize both loops in `rmsnorm_w_n_kernel` and
`rmsnorm_add_w_n_kernel` (and `d_warp_rowsum2` with them). The multiply
loop keeps the exact per-element arithmetic order → byte-identical; the
rowsum regroups each lane's addends (same shuffle tree after) — the
project's numerics-gated class, verified byte-identical on a high-entropy
random-words prompt for E4B/12B/E2B (the redundant-paragraph trap from
the ring-spare bug is why the gate uses random words).

Measured (warm serve, 928-token turns, first discarded):

| device | model | before | after | delta |
|---|---|---:|---:|---:|
| A5000 | E4B | 3715.5 | 3755.0 | **+1.1%** |
| Orin  | E4B | 415.5  | 419.3  | **+1.0%** |
| A5000 | 12B | 1806.8 | 1805.5 | ~0 (noise) |
| Orin  | 12B | 172.8  | 173.5  | +0.4% |
| A5000 | E2B QAT | 7127.4 | 7151.5 | +0.3% |
| Orin  | E2B QAT | 825.5  | 830.8  | +0.6% |

Kernel-level (A5000): `rmsnorm_w_n_kernel` 129→113 µs (−12%),
`rmsnorm_add_w_n_kernel` 67→57 µs (−15%); memory throughput 43→49% and
74→89% respectively. The E4B win is the largest because n_embd=2560 puts
the norms at ~10% of prefill time; 12B's bigger matmuls dilute it to
noise, and E2B's tiny rows leave little to recover. llama.cpp's
`rms_norm_f32` already loads f32x4 — this closes that slice of the
elementwise gap. The rest of the pool (geglu, rope, the small
`rmsnorm_kernel`) is unchanged.

## 2026-07-16 — the settle campaign: one methodology, every number re-measured

The documentation had accumulated mutually inconsistent numbers (README
decode tables, the Phase-2 scoreboard, the fork's README), so both stacks ×
both devices × all three models were re-benched in one sitting under one
written methodology — serve harness for us, `llama-bench` best-of-fa at
matched depth (`tg128 -d 512`) for llama, pinned `jetson_clocks` on the
Orin. Results and methodology now live in **docs/benchmarks.md**, which
supersedes all earlier tables; raw logs in `.scratch/settle-2026-07-16/`;
harnesses committed as `bench/settle-bench.sh` / `bench/settle-bench.ps1`.

What the settling found, beyond fresh numbers:

- **The Phase-2 "Final scoreboard" decode column was wrong** (erratum added
  in place): its E4B decode 27.8→30.0 values are E2B QAT values — today's
  E2B measures 27.8 at the 929-deep tail and 30.0 sustained, exactly — and
  its 12B 8.7 doesn't reproduce (8.0 sustained, 7.5 tail). A model-label
  mixup in the campaign's decode cells. Prefill columns reproduce within 1%.
- **The retired README decode table ("A5000 ≈1.0× llama") was a harness
  artifact**: one-shot `-p` decode reads ~20% higher than serve mode on
  Windows (per-token detokenize + socket write; negligible on Linux). The
  serve numbers were stable all along — 12B A5000 49.9 in the 2026-07-02 MTP
  entry, 49.7 today.
- **The old "E2B 25% faster than llama.cpp" claim was Q3_K-era; the QAT
  E2B picture is a DEPTH story**: near parity at shallow context (0.96×,
  the 2026-07-07 pinned pair, 36.4 vs tg32 37.9) but 0.80× sustained to 1k
  — our decode slows ~18% with depth where llama's fattn loses ~1%. The
  split-K decode attention's depth scaling on E2B (attention is a large
  share when weights are small) is the one remaining decode gap, both
  devices (0.70–0.80× sustained).
- The claims that survive, cleanly: **Orin decode ahead on E4B (1.17×) and
  12B (1.08×); prefill 0.76–0.82× everywhere** (six pairs, two devices).
- Depth costs llama little on these models (SWA caps most layers' KV):
  `tg128` d0→d512 is −1% (E4B Orin) to −4% (12B A5000). Depth was not the
  explanation for any of the discrepancies; harness and labels were.
