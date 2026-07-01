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

### Next levers

1. **Flash V path** — mio-throttle from strided `fa_rd1` single-half V
   loads; staging V costs 16 KB more shared (→1 CTA/SM, occupancy halves),
   so it needs a smarter layout (e.g. V staged f16 through the freed sS
   space, or a kv_dim-major V ring). Matters more as context grows.
2. **f16 SWA rings (PROPOSAL, numerics-changing)** — would remove the
   f32→f16 conversion entirely, halve K/V ring bytes AND ring memory, and
   make decode attention reads cheaper too. Precedent: the f16-KV global
   layers step shipped with a quality gate instead of byte-identity. Needs
   a user decision (changes decode numerics on SWA layers).
3. **q6_K L2 tuning** — its 3 MB row-tiles overflow a 4-row-tile wave's L2
   share even swapped; a wpc/rows-per-CTA rebalance might buy a few %.
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
