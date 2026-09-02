# little-gemma code-quality review (2026-09-02)

Readability + smaller-codebase pass over the whole C/CUDA engine, reviewed against `main`
file-by-file. Scope is **quality only** — duplication, dead code, conciseness/idiom, readability,
style consistency, resource-management boilerplate. **No correctness or performance changes**;
every item below is behavior-preserving. Line numbers are `main` at review time.

## Health summary

The codebase is in good shape. Across all 15 files reviewed there is **essentially no dead code**
— no `#if 0`, no commented-out blocks, and every apparent orphan symbol is a live backend-seam
forward declaration. Comments are dense but carry real rationale. The redundancy that exists is
**mechanical**: repeated idioms, kernel/function-family boilerplate, and a handful of shared
math/layout definitions maintained in two or three places.

Realistic behavior-preserving reduction is roughly **500–650 LOC (~5–6%)**, but the larger prize
is consistency and removing copy-paste-drift hazards (the kind that already produced `s` vs `st`
reduction vars, `-1e30f` vs `FLT_MAX` sentinels, and three half-to-float decoders).

Coverage: heavy CUDA — `model-cuda-i8.cu`, `model-cuda.cuh`, `media-kernel.cu`, `prefill-kernel.cuh`,
`mtp-kernel.cuh`, `model-cuda-f32.cu`; core C — `media.c`, `run.c`, `gguf.c`, `model.c`,
`model-cpu.c`, `quant.c`, `mtp.c`, `tokenizer.c`, `graph.c`.

---

## 1. Cross-file wins — do these first (one change, many files)

These are the highest-value items precisely because a per-file pass understates them.

### CUDA
- **`warp_sum` / `warp_max` helper** → `model-cuda.cuh`. The hand-rolled shuffle-reduce loop
  `for (o=16;o>0;o>>=1) x += __shfl_*_sync(...)` appears **~16 times** across every CUDA file, in
  two inconsistent spellings (`__shfl_down` + broadcast vs `__shfl_xor`; `0xffffffffu` vs `~0u`).
  One shared pair retires all of them. **#1 cleanup.**
- **`swa_start(pos, window)`** — the sliding-window start ternary, ~11 copies (prefill + decode).
- **`env_flag(name)`** — the `static int x=-1; if (x<0) x=getenv(...)!=NULL;` cache, ~11 copies.
- **Named constants**: `NEG_INF` (`-1e30f`, scattered incl. media/prefill/cuh), `FULL_MASK`,
  `KPAD=8` (flash bank pad), `GGML_TYPE_COUNT` (bare `40` in i8).
- **`GRID_STRIDE` macro** (media-kernel, 12×), **`block_reduce_sum`** (media-kernel 3–4× + cuh),
  **`sigmoidf`** (media-kernel 3×).

### C / shared-with-CUDA
- **Shared CPU-math header (biggest C-side lever).** `rmsnorm`, `rope_neox`, `gelu`, `softmax`,
  `matmul_q`, `f16_to_f32`, `kv_at` are duplicated verbatim between `model-cpu.c` and `mtp.c`
  (~70 lines), and the half-to-float decode exists a **third** time in `quant.c` (`fp16_to_fp32`).
  → one internal `math-cpu.h` (or export from `quant.h`), included by both TUs. The files' own
  comments already assert they are bit-identical.
- **Shared quant block-layout header.** `QK_K`, `K_SCALE_SIZE`, the six `block_q*` structs, and the
  `_Static_assert` layout checks are declared twice — `quant.c` and `model-cuda.cuh:42–50` — in two
  styles that can silently diverge. → hoist to `include/quant.h`, keep the asserts in one TU.
- **`now_sec`** — `run.c:42` is byte-identical to `now_sec_dev` in `mtp-kernel.cuh:38`. → one home
  (e.g. `mtp-internal.h`), drop `_dev`.
- **q3_K `kmask` unpack** — the `kmask1/kmask2` + `aux[]` shuffle recurs in `quant.c`,
  `model-cuda-f32.cu`, and `model-cuda-i8.cu` (host vs device). Previously deemed intentional;
  at minimum cross-reference the copies in a comment.

---

## 2. Per-file findings (top items, grouped)

### src/cuda/model-cuda-i8.cu (1602 LOC · est. −180–220)
- Delete the **6 single-column subs** (`sub_q4_K`…`sub_q4_0m`, :125–214) as `_n<1>` instantiations
  of their templated twins — `matmul_i8r_kernel` calls `_n<1>` into `float s[1]`. **~90 LOC, biggest single win.**
- Merge the two near-identical **mma launchers** `launch_q4k_mma`/`launch_q6k_mmq` (:1241 / :1468) — ~30.
- Collapse the **three chunk-dispatch loops** in `matmul_q_n` (:1514/:1531/:1555) into one over a launch callable — ~25.
- `dp4a8()` + reuse existing `q6w()` helper for the 8-way dp4a / q6 unpack pasted across the subs — ~35.
- `nsub_for(type)` (:496 ≡ :826), `d_gsm32` → wrap `d_gsm32r` (:101), templated `mma_writeback<COLS>` (:1219 ≡ :1440).
- Readability: name the A-tile byte magics (`2*2048`, `2*2*16*SA_ROW`), `SBX*8` vs `SBX*sizeof(float2)`, `GGML_TYPE_COUNT`.

### src/cuda/model-cuda.cuh (1372 LOC · est. −80–90)
- Template the **8 `kv_write` kernels** into one `template<DT,RING,N>` (:622–661) — ~30; launch-site 4-way branch collapses too.
- Macro-generate the **8 `attn_*` wrappers** from `d_attn` (:447–487) — ~25; byte-identical PTX, decode graph unaffected.
- `warp_sum`/`warp_max` (cross-file), `d_block_sumsq` helper (:124 ≡ :150), call `model_has_ple` instead of inlining 3× (:1096/:1127/:1158).
- Small: `float4` scale helpers (:210/:248), `d_quant_block_warp` for the geglu epilogue.

### src/cuda/prefill-kernel.cuh (813 LOC · est. −40–60)
- Collapse the **7 near-identical `{f16,f32}×{ring,full}` launch ladders** to a dispatch macro (:432/:497/:634/:648/:660/:567) — ~25.
- `ring4()` helper (:242 ≡ :382), fold the two `cudaFuncSetAttribute` carve blocks, `swa_start`/`env_flag` (cross-file).
- Readability: `KPAD=8` for the scattered `+8` bank pad; `NEG_INF`; fix stale "above" comments (:13/:95, the `attn_*_n` twins now live in `model-cuda.cuh`); run the dense flash region through the file's normal formatter.
- **NOTE:** the `LG_FLASH_REG` register-resident kernel is the flash-pipeline work (branch
  `prefill-e4b-flash-pipeline`), NOT a dead experiment — do not delete it. When that branch merges,
  dedup its shared skeleton (staging/mask/ring/epilogue) against `flash_attn_n_kernel` into
  `__device__` helpers.

### src/cuda/media-kernel.cu (955 LOC · est. −40–55)
- `GRID_STRIDE` macro (12×), `block_reduce_sum` (3–4×, `s`/`st` drift), free-lists iterate the `bufs[]` alloc tables instead of hand-listing.
- `sigmoidf` (3×), promote the local `AGG` grid-dim macro to file scope, one `NEG_INF`, `#define DH 64`, drop the `& 63` no-op mask (:154), move the mid-file `#include <cfloat>` (:642) up.

### src/cuda/model-cuda-f32.cu (138 LOC · est. −3–5)
- Collapse the `matmul_q_n`/`matmul_q_spec` twin loops (F1). **Otherwise leave it** — it's the readable oracle; its interface stubs are seam-mandated, not dead.

### src/cuda/mtp-kernel.cuh (425 LOC · est. −12–18)
- Two helpers for the verbatim blocks shared by `mtp_draft_device`/`mtp_draft_chain_device`: embed-fill (:341 ≡ :386) and readback+d2t (:367 ≡ :400) — the latter gives the d2t comment one home.
- Hoist the repeated `window`/`shm` locals (:295–308), fix stale "two wrappers" comment (three exist, :128).

### src/media.c (1070 LOC · est. −60–90)
- Free-ladders → array+loop / `goto cleanup`: `v_embed_image` 8-ptr list ×3 (:496/:512/:518), `a_blocks_host` 9-ptr ×2 (:991/:996), the small alloc-check-free idiom ×6.
- Extract `prenorm_rows` (×6), `postnorm_residual` (×5), the two half-step FFNs (:880 ≈ :962), `im2col_patch` (:288 ≈ :362); unify the verify-fallback triad behind the fn-pointer the image path already uses (:548/:1040/:1062).
- `clampf`/`imin`/`imax` (~10 hand-rolled, keep ternary for NaN), `memset` for the zero loops (:447/:500), `sizeof(float)` for the literal `4` (6×), `LN_EPS`/`RMS_EPS`, `A_REL_CTX=12`, one π constant.

### src/run.c (842 LOC · est. −40–70)
- Dedup: `now_sec` (cross-file), MTP stat line (:582 ≡ :776), Unix-socket setup (:337 ≈ :604), ▁-marker translation loop (:123 ≡ :164), the `MSG_PEEK` block (`client_signal`/`sock_pending`).
- `slurp_file` helper (:288–320), table-drive the 13-branch arg-parse ladder (:793), collapse the inconsistent cleanup staircase to one `goto cleanup` (:243–255 vs :296–313).
- Name the buffer magics (`TOK_CAP 4096` ×12, `LINE_CAP`, `MAX_FRAME_BYTES`, `SYS_MAX_BYTES`); rename the two `full` uses (:353/:412).

### src/gguf.c (599 LOC · est. −25–35)
- `find_alignment` → `gguf_get_u32(..., 32)` (10 lines → 2, :238). `RD(f,x)` read-exact macro (~8×). `align_up`/`align_down` helpers (5×). Clarify `print_value`'s string-array `break` (:508). Already table-driven where it counts (good).

### src/model.c (210 LOC · est. −20–30)
- `make_key(buf,n,arch,suffix)` (6×) + one `arch_i32_array(...)` helper collapse the repeated
  "build `<arch>.<suffix>` → find_meta → scalar-or-INT32-array" preamble across `arch_ff`/`arch_int0`/`load_ffn_lens` (and locate `feed_forward_length` once, not twice). Merge `arch_u32`/`arch_f32` twins. Document the caller-frees contract on `model_init`'s error returns.

### src/cpu/model-cpu.c (497 LOC · est. −25–40)
- Move `f16_to_f32`/`kv_at` to the shared header (cross-file); prefer `quant.c`'s readable half-float.
- `add_vec`/`scale_vec` helpers for the triplicated sublayer epilogue (:402/:412/:421) and the scattered scale loops. `blk_name()` for the `blk.%d.%s` boilerplate (:128 ≡ :139). Drop the `kv_max` alias (:316) and the stale double comment (:45–50); move the stranded rmsnorm doc-comment (:15) to its function; `kv_elt_size(f16)` for the `?2:4` magic.

### src/quant.c (343 LOC · est. −30–45)
- **Traits table**: collapse the three parallel switches `ggml_type_name`/`ggml_blck_size`/`ggml_type_size` (:97–141) to one `{type,name,blck,size}` table. Move block structs/`QK_K` to `quant.h` (cross-file). Factor the q4_K/q5_K shared prologue (:165 ≈ :184). Drop the `dequantize_into` pass-through alias + the repeated guard+malloc tail. Name `Q4_ZP=8`/`Q6_ZP=32`; fix `q[j - 0]` (:161).

### src/tokenizer.c (288 LOC · est. −10–18)
- `snprintf("<0x%02X>")` for the hand-packed byte-fallback (:227); name token-type codes 3/4; hoist the count/fill predicate; one `emit` helper for the four bounds-checked appends; break the one-line heap sift-up + `bigram_swap` helper.

### src/graph.c (225 LOC · est. −25–40)
- `unary(g,name,op,x)` + `binary(g,name,op,a,b)` builders collapse the four near-identical op builders (:72–126). One `{name, fn}` **op-metadata table** replaces the two parallel `OP_*` switches in `graph_compute`/`op_name` (:185/:197) — adding an op becomes one line. Biggest win for its size.

---

## 3. Leave as-is — deliberate house style (all agents agreed)

- **Frozen-decode twins**: `_n` vs non-`_n`, warp-per-row vs block-per-row, rope table vs inline
  `powf`, pinned-row-0 variants. Merging them changes decode's **captured CUDA graph** — keep.
- **Interface stubs** in `model-cuda-f32.cu` / the CPU stub — seam-mandated, not dead.
- **`split_attn_kernel`** re-deriving the `d_attn` softmax loop (~40 LOC) — perf-critical, different
  register/split layout.
- **The `LG_FLASH_REG` register flash** — live flash-pipeline work (dedup vs base flash on merge, don't remove).
- **Profiling-narrative comments** (i8 `:905–929` etc.) — earn their place; move to a design doc only
  if size becomes a hard target.
- `mtp_attn_*` decode-style wrappers — deliberate parallelism with decode, don't template.

---

## 4. Suggested order

1. **Shared definitions** — `math-cpu.h` (CPU math), `quant.h` (block layout + traits table), and the
   CUDA `warp_sum`/`swa_start`/`env_flag` helpers + named constants. Biggest cross-file payoff, touches
   nearly every file, low risk.
2. **`model-cuda-i8.cu`** single-column-sub deletion (~90) + `quant.c` traits table.
3. **`model-cuda.cuh`** `kv_write` template + `attn_*` macros (~55).
4. **Media** (host + kernel) free-ladder → loop and the norm/FFN helpers; **`graph.c`** op table; **`run.c`** dedups.
5. Per-file small items + naming/constants.

Gate every change on the existing byte-identity / coherence tests (`bench/coherence-screen`) — most
of the above are pure refactors, but the CUDA kernel-family collapses (`kv_write`, `attn_*`) must keep
the decode graph byte-identical, which they do by construction (template/macro expansion).

## 5. Estimated reduction

| area | files | est. LOC |
|---|---|---:|
| CUDA | 6 | ~340–430 |
| C | 9 | ~235–366 |
| **total** | **15** | **~500–650 (~5–6%)** |

Second-round targets (not reviewed here): the C headers, `bench/`, `tools/`, `test/`.

---

## Addendum — flash-family re-review (post-rebase onto canonical main, 2026-09-02)

After rebasing `prefill-e4b-flash-pipeline` onto canonical `main` (which already carries the
register-resident `flash_attn_r_kernel`), both flash kernels sit in one file. Re-review:

**Neither flash kernel is removable.** `flash_attn_n_kernel` covers the HD-512 global layers,
G=1/4, and the default path; `flash_attn_r_kernel` is HD-256 SWA G=2 only (the register-resident
winning path). Their divergent core — the warp→data mapping (shared `sS`/`sP` round-trips vs
register-resident softmax) — is the reason both exist; keep it inline.

**Shared skeleton → `__device__` helpers (behavior-preserving):**
- `stage_K(...)` — the KSTAGE f32→f16 K loader, same logic in both (n `:178`, r `:316`; differ only
  by thread count and HD vs HD+8 stride).
- `ring4(k0, seq)` — the RING index quad `{r0,r1,r8,r9}` (n `:243` ≡ r `:392`).
- `swa_start(pos, window)` — the window-start ternary (the cross-file helper, also in §1).
- `stage_Q(...)` — the padded sQ fill (n `:162`, r `:306`).

**Cleanups introduced by the V-stage change:**
- `flash_r_shm` (`:414`): the ternary is now vacuous — both branches return `32*(hd+8)*2` — so
  collapse it to `64*(hd+8)*2 + 32*(hd+8)*2`.
- `sK`/`sV` alias the same shared offset (`:295`/`:296`), correct because the f32 K-stage and the
  f16 V-stage are mutually exclusive, but it reads as two live buffers; a one-line "union, mutually
  exclusive" comment (or an actual union) makes the intent obvious.

**Merge-prep:** the r-kernel is the better HD-256 path but still gated behind `LG_FLASH_REG`. To
ship it as default, flip that for the SWA layers and validate E2B/12B; the n-kernel's HD-256 G=2
branch then becomes dead (n stays for HD-512, G=1, G=4). Coherence screen on the rebased build:
**PASS** (0 degenerate, 14/24 exact, 2 known-correct hard divergences — identical to pre-rebase).

---

## Sequencing note (relative to PR #16, the flash pipeline)

This review was taken against `main` (`d1c4c8f`). PR #16 (gelu + flash V-stage/pad + screen)
touches only **2 of the 15 reviewed files** — `src/cuda/model-cuda.cuh` and
`src/cuda/prefill-kernel.cuh` — plus new files under `bench/coherence-screen/` (not reviewed). The
other 13 files' findings and line numbers are independent of PR #16 and apply as-is.

**Do the cleanup AFTER PR #16 merges, based on the merged `main`** — not before. Cleaning first
would refactor those two files out from under the PR (hard rebase). Landing the feature first means
the cleanup is the only change to those files and cannot conflict; at that point re-anchor just:
- `prefill-kernel.cuh` — the flash-family helpers (the addendum above is already branch-relative), and
- `model-cuda.cuh` — the geglu region (the gelu float4 moves the launch sites).

The 13 non-overlapping files can be cleaned in any order.
