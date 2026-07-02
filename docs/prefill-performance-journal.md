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