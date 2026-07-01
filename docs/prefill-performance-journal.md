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

### Next levers (A5000)

1. **The kernel itself** — now the whole residual gap (~1.6×). At full fill
   it runs 45.7% SM, 2 warps/scheduler, 1 CTA/SM (68.6 KB shared, >128
   regs/thread). The Orin 2-CTA occupancy experiment (`LG_Q4K_OCC`, net wash
   THERE) is unexplored on the A5000's different shared/L2 budget. llama's
   same-shape kernel reaches 52% SM plus stream-K scheduling.
2. **q6_K twin** (14.4% of MACs) — same treatment as whatever wins on q4_K.
3. **First-turn cold cost** (12B first turn 344 vs 1430 warm — weight repack
   + clock ramp; llama-bench discards it, serve users pay it).
4. **Orin re-validation:** balanced chunking should be neutral-to-better
   (equal-or-fewer weight passes) and the ring fix is a correctness must,
   but ring memory grows (~+215 MB, 12B at spare 768) — re-run the Orin
   suite before merging there.
