# Speculative decoding (MTP)

Gemma 4 ships a **multi-token-prediction assistant**: a tiny transformer
(E2B's is 4 blocks, 256 wide, 77M parameters) that predicts the token *after*
next. Its design is almost parasitic — it owns no K or V projections at all;
every block cross-attends directly into the **target model's KV cache**, and
its inputs are the target's own embedding of the freshly chosen token plus
the target's last hidden state. `mtp.c` implements it; `-mtp` turns it on:

```
run-cuda-i8 -m gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf -mtp mtp-gemma-4-E4B-it.gguf -p "..."
```

Each round drafts and then verifies the block as one small batch — the
weights cross memory once for it, so an accepted draft is nearly a free
token. Verification is greedy, which buys the strongest property speculative
decoding can have: **the output is byte-identical to plain greedy decoding,
always** (the split-K verify makes it byte-identical *by construction*) — the
only things that move are tokens/s and the acceptance rate the stats line
reports. What MTP is worth is **content-dependent** — the draft head lands
~37% of free-prose drafts (E4B, matched QAT head), ~76% of code, ~99% of
counting — and, since the 2026-07-19 **2-row verify kernel** dropped the
Orin's verify cost to ~1.0–1.2× a decode step, worth having everywhere.
Measured in the **socket server**, steady-state, E4B QAT (2026-07-19; the
12B rows predate the new kernel):

| model | counting | code | image-only turn | prose |
|-------|---------:|-----:|----------------:|------:|
| E4B QAT (Orin / A5000) | **2.33×** / 2.10× | **1.91×** / — | **1.52×** / — | **1.40×** / 1.26× |
| 12B (A5000 / Orin, pre-2-row) | 2.16× / 1.85× | — | — | 1.35× / 1.19× |

(Orin E4B QAT: 20.7 plain → 29.9 prose, 40.7 code, 48.6 counting, 31.5
describing an image — every one ahead of `llama-server`'s `draft-mtp` on
the same model, same day; block-depth sweep and llama pairs in
[benchmarks.md](benchmarks.md).) Block depth is `-DLG_MTP_N` (default 3):
prose peaks at 2–3, code at 4, counting at 4–5.

**A measurement trap worth recording:** the draft head pays a one-time ~3.6 s
CUDA-graph warmup on its *first* call, and a one-shot `run -p` charges all of
it to that single short run — an earlier version of these tables did exactly
that and wrongly concluded *"MTP loses on Windows."* Benchmark MTP in serve
mode with the first turn discarded. The full story, including the
uncached-zero-copy bug the verify exposed in the chunk matmul, is in the
[performance journal](performance-journal.md).
