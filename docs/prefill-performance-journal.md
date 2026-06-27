# Prefill performance journal (q4_K ILP branch)

Focused log for the **`prefill-q4k-ilp`** workstream: closing the prefill gap vs
llama.cpp on Jetson Orin NX with **gemma-4-12b-it-Q4_K_M.gguf**. Decode and MTP
are out of scope; every change gates on **Paris cohesion**, warm prefill tok/s,
and decode unchanged.

**Benchmark gate:** `.scratch/bench-prefill.sh` — cold `"The capital of France is "`
(must emit **Paris**), warm ~910-token `"word "` × 900.

**Model / binary:**
- Model: `/home/one/repos/cortexist/llama.cpp/.scratch/gemma-4-12b/gemma-4-12b-it-Q4_K_M.gguf`
- Binary: `build/run-cuda-i8`

**Target:** llama.cpp ~203 tok/s warm prefill (pp512/1024). Our best ~114 tok/s
(~1.75× gap). Matmul ~65% of wall time on this model.

---

## Permanent reject list (do not retry without new evidence)

| Approach | Result | Why |
|----------|--------|-----|
| q4_K device copy on 12B/Orin | ~25% regression | ~5.8 GB extra footprint; zero-copy host path is the design |
| cp.async qs from zero-copy host weights (Tegra) | Garbage / cohesion fail | Uncached host-mapped qs |
| stream-K as first lever | — | User scope: not first |
| Full `chunk_layers` CUDA graph | ~308 tok/s, garbage gen | Multi-stream + flash; reverted (see below) |
| `#pragma unroll` on variable-step paired-h loop | Cohesion fail | Must not unroll outer h loop when step is variable |
| COLS=128 fat tiles | ~108 tok/s | Register / occupancy |
| Fused half-loops (4 mmas/column group) | ~110 tok/s | `dscH`/`afragH`/`c[2][2]` register pressure |
| PREFILL_B=48 / B=64 one-tile (historical) | Occupancy trap | See main `performance-journal.md` |

---

## Baseline at branch start (session 1)

| Metric | Value |
|--------|-------|
| Warm prefill | ~103 tok/s |
| Matmul (profile) | ~6.0 s |
| Decode | ~8.5 tok/s |
| vs llama | ~2.0× |

Profile split: matmul ~69%, attention ~7%, elementwise ~9%.

---

## Shipped on `prefill-q4k-ilp` (kernel / infra)

| Commit | Change | Effect |
|--------|--------|--------|
| e28f2a7 | Wide chunks + fork/join + initial pipelining | Foundation |
| 8ac6c1f | B-fragment software pipeline + cp.async qs revert | Correctness |
| 7c392e1 | qs register pipeline across sjp | ~114 tok/s, matmul ~5.29 s |
| e1c1dc4 | Dual-tile mma before epilogue | Matmul win |
| f9bea16 | Hoist bp_t, pipeline header scales, **LG_WIDE_CHUNK=768** default | ~114 tok/s |
| 481ab47 | GPU q4_K token embed (`lg_i8_embed_q4k_chunk`) | Neutral; removes CPU+H2D |
| 1e5b9b4 | (embed related) | — |
| 929e3b7 | **2-wide** paired column-group mmas + `LG_FORCE_WPC` | Matmul **~4.38 s**, ~114 tok/s |
| *(this session)* | **8-wide** column-group mmas (`LG_MMA_PAIR`, default **8**) | +1–2 tok/s vs 2-wide in sweep |
| *(this session)* | `q4k_colgroup_mmas<8,64>` full unroll + prewarm moves host API out of launch | Infra for graphs |

### Key kernel constants (`matmul_q4k_mma_kernel`)

- `SB_COL=272`, `SA_ROW=48`, `SBX=9` (bank-conflict pads)
- Wide chunk: `COLS=64`, `gridDim.y = g_pf_cols/64` (768 → 12 column tiles)
- q4_K weights: zero-copy **host** memory, uncached on Orin — wide chunks cut
  **passes**, not L2 reuse across tiles

### Useful env vars

| Variable | Purpose |
|----------|---------|
| `LG_WIDE_CHUNK` | Chunk width (default 768 on integrated GPU) |
| `LG_MMA_PAIR` | Column-group mma batching 1–8 (default 8) |
| `LG_FORCE_WPC` | Skip warps-per-CTA shrink |
| `LG_NO_GPU_EMBED` | CPU embed fallback |
| `LG_NO_MMA` | dp4a A/B |
| `LG_NO_FLASH` | Scalar attention |
| `LG_PREFILL_PROFILE` | Stage timing (slows run) |
| `LG_MM_GRAPH` | Opt-in matmul-only CUDA graph (**broken**, see below) |
| `LG_NO_MM_GRAPH` | Force direct matmul launches |
| `LG_NO_PREFILL_FORK` | Sequential k/v and ffn_up on main stream (A/B: ~neutral, matmul +4% without fork) |

---

## Option 1: Column-group mma pairing (1–8-wide) — EXHAUSTED

**Goal:** Issue more independent `mma.sync` chains before float epilogue (llama
wins via ILP at lower occupancy, not higher CTA count).

**Implementation:** Templated `q4k_colgroup_mmas<NW, COLS>` with prefetch of
next column-group B-frags; `switch` dispatch on `c_mma_pair` (1–8); specialization
`q4k_colgroup_mmas<8, 64>` fully unrolled for wide chunk.

**Sweep (two-run average, Paris ✓, decode ~8.5 tok/s):**

| `LG_MMA_PAIR` | Warm tok/s (approx) |
|---------------|---------------------|
| 1 | 109.0 |
| 2 | 108.6 |
| 3 | 109.0 |
| 4 | 109.3 |
| 5–7 | ~109–110 (first sweep; dispatch bug fixed afterward) |
| **8** | **110.7** |

Matmul profile: pair=8 **4.66 s** vs pair=2 **4.77 s** (~2% matmul win).

**Also tried:** tile-major vs group-major mma ordering — neutral; COLS=64
full-unroll vs generic template — neutral.

**Shipped default:** `LG_MMA_PAIR=8` via `__constant__ c_mma_pair`.

**Lesson:** Wider pairing helps monotonically up to 8 (all groups in one half
before epilogue) but only ~1–2 tok/s e2e — not the ~90 tok/s needed vs llama.
Register budget at NW=8 is acceptable (no cohesion failures).

**Note:** Absolute tok/s varies ~107–115 with thermal/state; same-session A/B
ratio is the reliable metric on Orin (clocks not lockable).

---

## Option 2: Matmul-only CUDA graph — FAILED (opt-in stub kept)

**Motivation:** Full `chunk_layers` graph was reverted earlier (~308 tok/s,
garbage) likely due to side-stream fork/join + flash. Matmul-only graph captures
each `matmul_q_n` launch sequence per `(weight tensor, d_out buffer, stream)` and
replays on later chunks — elementwise and attention stay outside.

**Implementation:** `mm_graph_replay()` in `model-cuda-i8.cu`; split
`matmul_q_n_launch()` (capture-safe) from wrapper; moved `cudaMemcpyToSymbol(c_mma_pair)`,
`cudaFuncSetAttribute` (dynamic shared carveout), and `sms` init into
`lg_i8_prewarm_mma()` (called from `lg_i8_prewarm_weights()`).

**Result:** **Correctness failure** even on first warm run (capture-only, no replay).

| Mode | Paris / cohesion | Warm tok/s | Matmul profile |
|------|------------------|------------|----------------|
| Default (graph off) | ✓ coherent | ~108 tok/s | ~4.8 s |
| `LG_MM_GRAPH=1` | ✗ multilingual garbage | ~210–216 tok/s (fake) | ~0.74 s |

Decode unchanged (~8.5 tok/s) when graph off.

**Attempted fixes:**
1. Move host API (`cudaMemcpyToSymbol`, `cudaFuncSetAttribute`) to prewarm — still broken
2. Main-stream-only graph (skip side-stream k/v) — still broken

**Hypotheses (unconfirmed):**
- Graph capture + mma kernel + zero-copy q4_K host weights on Tegra
- Per-matmul graphs missing producer→consumer visibility for `g_xq`/`g_xds`
  (unlikely on same stream; decode graph works with similar pattern)
- Illegal ops still on capture path (none found after prewarm move)

**Status:** **Disabled by default.** Enable with `LG_MM_GRAPH=1` for experiments;
use `LG_NO_MM_GRAPH=1` to force direct launch. Code kept for future Orin/x86 A/B.

---

## Option 2 (historical): Full `chunk_layers` CUDA graph — FAILED

Captured entire wide-chunk layer loop (~1000 launches). Symptoms: ~308 tok/s,
garbage generation (`---R--R---…`). Sync-after-capture did not fix. Reverted;
only `lg_i8_prewarm_weights()` kept.

**Likely cause:** Multi-stream capture (fork/join side stream + flash prefill path)
not replaying faithfully on Orin.

---

## Other experiments (this workstream)

| Experiment | Result |
|------------|--------|
| Wide-chunk CUDA graph for `chunk_layers` | Broken — reverted |
| q4 device copy | ~25% regression — rejected |
| cp.async qs (Tegra) | Garbage — rejected |
| LG_WIDE_CHUNK 512/640/768/896 | 768 default ~best |
| GPU Q4_K embed | Neutral ~114 tok/s |
| Paired 2-wide column mmas | Matmul 4.38 s — shipped |
| 8-wide column mmas | +1–2 tok/s — shipped as default |

---

## Profile snapshot (12B Q4_K_M, LG_WIDE_CHUNK=768, graph off)

Typical warm 896 chunk tokens (`LG_PREFILL_PROFILE=1`):

```
matmul ~4.7–4.9 s (~65%)
attention ~0.65 s (~7%)
elementwise ~1.0 s (~11%)
```

q4_K MAC share ~85.6%; q6_K ~14.4%.

---

## Option 3: Prefill fork/join A/B (`LG_NO_PREFILL_FORK`) — NEUTRAL

**Goal:** Decode hides k/v and ffn_up beside larger q/gate matmuls via side stream.
Test whether prefill benefits similarly or whether event/sync overhead dominates.

**Implementation:** `LG_NO_PREFILL_FORK=1` runs k/v and ffn_up on the main stream
in `chunk_layers` only; decode `forward_layers` unchanged.

**Result (12B Q4_K_M, warm ~910 tok, Paris ✓, decode ~8.5 tok/s):**

| Mode | Warm tok/s | Matmul | Elementwise |
|------|------------|--------|-------------|
| Fork on (default) | ~108.0 | 4.84 s | 1.07 s |
| `LG_NO_PREFILL_FORK=1` | ~107.6 | 5.04 s (+4%) | 0.92 s |

**Conclusion:** Fork/join is **neutral to slightly positive** for prefill on 12B —
disabling it regresses matmul (~4%) with no e2e win. **Keep fork enabled.**
Env knob kept for future graph-capture experiments (side stream was suspect in
Option 2).

---

## Next levers (priority order)

1. **Remote microbench + ncu on x86** — safe profiling (ncu on Orin caused board reset)
2. **More mma ILP** — diminishing returns; occupancy trap documented
3. **Matmul graph** — blocked on correctness; revisit with global capture or fork-off A/B
4. **Attention / elementwise** — ~11% elementwise launch overhead remains

---

## ncu / Orin warnings

- `ncu` on Orin NX caused hard resets (`nvgpu_regops_exec`, CBB `PWRDOWN_ERR`)
- Prefer `LG_PREFILL_PROFILE=1` on device; use x86 + `ncu` for kernel deep dives

---

*Last updated: 2026-06-27 — Options 1–3 done (8-wide pairing shipped, matmul graph
failed opt-in stub, fork/join A/B neutral).*
