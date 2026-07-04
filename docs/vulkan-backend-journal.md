# Vulkan backend journal

Focused log for the `run-vulkan` workstream: a backend for machines with no
CUDA at all — written and measured on its actual target, a Windows laptop
with an AMD Radeon integrated GPU (Vega-class, no dot4), 16 GB of shared
DDR, and the CPU reference (`run`, scalar+OpenMP) decoding E4B Q4_K_M at
1.1 tok/s. Every change gates on **byte-identical output vs the CPU
reference** (full stdout diff, not just the answer line) and, where a batch
width is involved, byte-identity across widths. `LG_VK_VERIFY=1` is the
running oracle: every GPU matmul recomputed on the host, max |gpu−host|
reported — healthy is ~1e-4 against 1e2-scale outputs, which is summation
order only (the dequantized values are bit-identical by construction).

A laptop iGPU measurement caveat up front, learned mid-session: identical
binaries swing ~25% run to run (clocks, thermals — the A5000 clock-floor
story from the README, worse because it's a laptop). Single short runs
cannot support micro-comparisons; the numbers below are the honest medians
of repeated runs, and one regression scare (batch verify "slowing down" 238
→ 300 ms on an unchanged pipeline) was exactly this noise.

---

## Session: bring-up (2026-07-03) — 0.8 → 2.5 tok/s, zero-copy weights

Architecture choice: `model-cpu.c`'s readable forward stays on the host
(norms, RoPE, attention over the host KV cache, `model_kv_host = 1` so the
host MTP draft works untouched); the quantized matmul — nearly all of a
token's weight traffic and compute — becomes a compute shader, one `glslc`
compilation per weight type, dequant ported line-for-line from `quant.c`.
On an APU, "device memory" is the same DRAM, so the activation handoff each
way is a memcpy, not a bus.

### Discoveries

1. **Zero-copy weights work on the AMD proprietary driver.** The GGUF blob
   imports in place via `VK_EXT_external_memory_host` (gguf.c now
   page-aligns and page-pads the blob to make that legal), bound as ≤2 GiB
   storage-buffer windows every 1 GiB with each tensor's byte offset pushed
   as a constant — sidestepping both `maxStorageBufferRange` (the 5.3 GB
   blob exceeds any binding) and `minStorageBufferOffsetAlignment` (tensor
   offsets are only 32-aligned). A 16 GB laptop cannot afford the model
   twice; this is the Orin's zero-copy instinct arrived at from the other
   side.

2. **The naive shader coalesced nothing: ~4 GB/s.** v1 gave each lane its
   own 32-element group, so adjacent lanes read ~144 bytes apart — almost
   every byte fetch was its own transaction. Relaying the lanes so adjacent
   lanes read adjacent words of the weight stream (32 lanes cover a q4_K
   superblock's 128 qs bytes contiguously; q6_K/q8_0 at 16-bit granularity
   because 210- and 34-byte blocks are only 2-aligned) took decode 0.8 →
   ~1.3 tok/s.

3. **Logits readback from write-combined memory cost a 2×.** The out
   buffer's memory type was chosen like the weights' (avoid HOST_CACHED so
   GPU reads don't take the APU snoop path) — but `out` is the one buffer
   the CPU *reads*, and CPU reads from write-combined memory are the slow
   direction. Splitting the policy (GPU-read buffers avoid HOST_CACHED,
   CPU-read buffers prefer it) took decode ~1.3 → 2.5 tok/s. One enum, 2×.

4. **Snooped host reads were NOT the bottleneck.** The A/B everyone expects
   to matter — zero-copy imported host pages vs weights uploaded into
   driver GTT (`LG_VK_NO_IMPORT=1`) — measured 2.49 vs 2.57 tok/s. Upload
   costs 5.3 GB of extra RAM and a long warmup for ~3%; zero-copy stays the
   default.

State at session end: decode 2.5 tok/s (2.3× CPU), prefill = decode rate
(each prompt token streams all 5.3 GB), output token-identical to `run`.

---

## Session: the -i8-spirit push (2026-07-04) — prefill 1.4 → 7.5, decode 3.0, MTP

Goal: "do whatever you can," in the CUDA `-i8` path's spirit. The profile
said: decode sits at the device's measured DRAM stream rate (~16 GB/s
effective), so the levers are prefill (bandwidth amortization), sync
round-trips, and the CPU attention tail — plus checking whether an int8
kernel is even worth building on this silicon.

### Discoveries

1. **Chunked prefill needs the SWA rings padded by the chunk width.** A
   chunk writes all B of its KV rows before any of its attention runs; with
   an exact-window ring, the chunk's last row lands on a position its own
   FIRST token still attends. `window + CHUNK_MAX` rows make every
   clobbered row unreachable by any token of the chunk — the same padding
   the CUDA cache applies, rediscovered from the same failure shape (the
   model-cpu.c comment even warns about it).

2. **The first wide kernel was no faster than the narrow one.** NB=16
   columns per weight pass should make prefill ~free per extra token;
   instead a 13-token chunk cost ~8 narrow passes. Cause: `acc[b] += wv *
   x[col_b + e]` is NB *scalar* loads per weight element (columns live k
   floats apart). Fix: transpose x on upload ([k][NB], dead columns padded
   with column 0) so one element's NB values are contiguous and read as
   NB/4 vec4s. Strided stores into write-combined memory defeat the
   combining, so the transpose goes through a malloc'd scratch and lands as
   one memcpy.

3. **One row per workgroup drowned the chunk in L2 traffic.** Each output
   row's workgroup reads all k×NB×4 bytes of x — ~250 GB of L2 traffic per
   chunk across a big matmul's rows, seconds on an iGPU's L2. TM=4-row
   tiles load each x quad once and MAC it into 4 rows' accumulators:
   prefill 4.4 → 8.0 tok/s on the short prompt. Row tiling then turned out
   to pay on the *matvec* too (fewer workgroups, fewer x re-reads): the
   narrow pipeline also compiles TM=4 now, decode ~3.0–3.2.

4. **Scalar f16→f32 owned long contexts.** At pos ~800 the 898-token
   prompt prefilled *slower* than a short one (5.7 tok/s): the attention
   loops converted the f16 global-layer KV rows element by element, every
   q-head of a GQA group re-converting its kv-head's rows, every token of
   the chunk again — tens of millions of bit-twiddling calls per token.
   Staging each source layer's rows to f32 once per chunk (position-
   indexed, shared by all 16 tokens × 8 heads, reusing the staging across
   the layers that share a KV source) made attention pure f32 math: 898-tok
   prefill 5.7 → 7.5 tok/s, long-context decode 2.1 → 2.4.

5. **MTP batch verify needs its own pipeline width.** `model_forward_spec`
   was LG_MTP_N sequential forwards (no speedup by definition); it now runs
   the block as ONE chunk — but through the NB=16 pipeline a 3-column
   verify pays 16 columns of MAC and *loses*. A middle NB=4 family fixes
   that. Fused q/k/v and gate/up submits (one fence round-trip each, ~250
   submits/token instead of ~380) round out the session.

### Dead end, deliberately not built: the int8 path

The CUDA `-i8` win is dp4a — 4 int8 MACs per instruction. The query that
decides whether that transfers:
`integerDotProduct4x8BitPackedSignedAccelerated = **false**` on this device
(all 8-bit dot flags false — Vega-class APU, no v_dot4). Packed int math
would lower to the same scalar multiply-adds the float path already
executes, plus activation quantization on top. So: investigated, measured
as unbuildable-with-benefit on this hardware, documented instead of
vendored — `stream-k-experiment.md`'s conclusion shape. On RDNA2+ the same
query answers true, and the pipeline-family structure (NB=1/4/16 per type)
is where a q8-activation family would slot in.

### Where it stands (E4B Q4_K_M, identical greedy tokens on every row)

| step | decode | prefill (898-tok) |
|------|-------:|------------------:|
| `run` (CPU reference)              | 1.1 | ~1.4 |
| v1 shader                          | 0.8 | 0.8 |
| + coalesced lanes, cached readback | 2.5 | 2.5 |
| + chunks, fused submits, tiles, staged KV | **3.0** | **7.5** |
| + `-mtp`, counting                 | **3.5** | — |

MTP honesty: the batch verify costs ~1.6× a plain forward here (the NB=4
kernel's padded column + B× the CPU-side work), so counting-shaped output
wins ~1.2–1.35× and prose (~25% acceptance on E4B's small head) breaks
even. The README's rule applies twice over on a laptop: benchmark warm, in
serve mode, on your own workload.

### Gate lessons

- Chunked prefill and batch verify are bit-identical to token-at-a-time *by
  construction* (each wide column accumulates in the narrow kernel's
  element order), and the gates confirmed it: `LG_VK_PREFILL_B=1` vs `=16`
  full-stdout diff clean, and both diff clean against `run`.
- The verify oracle caught nothing after bring-up — every bug this session
  was a *performance* bug, invisible to numeric gates. The profile-shaped
  question ("what would make a 13-token chunk cost 8 passes?") did the
  work the oracle couldn't.

### Known next step

The CUDA journey's big one, unchanged: move attention and the small ops
onto the GPU so a token is one submit instead of ~250, and the KV cache
lives on-device. That flips `model_kv_host`, which drags the MTP draft head
along — the same coupled migration the CUDA backends did in
`performance-journal.md`.
