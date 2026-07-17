# Speculative decoding (MTP)

Gemma 4 ships a **multi-token-prediction assistant**: a tiny transformer
(E2B's is 4 blocks, 256 wide, 77M parameters) that predicts the token *after*
next. Its design is almost parasitic — it owns no K or V projections at all;
every block cross-attends directly into the **target model's KV cache**, and
its inputs are the target's own embedding of the freshly chosen token plus
the target's last hidden state. `mtp.c` implements it; `-mtp` turns it on:

```
run-cuda-i8 -m gemma-4-E4B-it-Q4_K_M.gguf -mtp gemma-e4b-assistant-mtp.gguf -p "..."
```

Each round drafts and then verifies the block as one small batch — the
weights cross memory once for it, so an accepted draft is nearly a free
token. Verification is greedy, which buys the strongest property speculative
decoding can have: **the output is byte-identical to plain greedy decoding,
always** (the split-K verify makes it byte-identical *by construction*) — the
only things that move are tokens/s and the acceptance rate the stats line
reports. What MTP is worth is **sharply content-dependent**: at block-3, a
fully-accepted round emits three tokens for ~one decode pass, so structured
output (counting, lists, tables) roughly **doubles**; free prose lives or
dies on the draft head's acceptance rate — the 12B's 1024-wide head lands
~49% of prose drafts, E4B's 256-wide head only ~30%. Measured in the
**socket server**, steady-state (2026-07):

| model | counting (100% acc) | prose |
|-------|--------------------:|------:|
| E4B (A5000 / Orin) | **2.08×** / **1.83×** | 1.10× / 1.12× |
| 12B (A5000 / Orin) | **2.16×** / **1.85×** | 1.35× / 1.19× |

(A5000 12B: 61.6 → 133 tok/s counting, 49.9 → 66.9 prose; Orin 12B:
8.4 → 15.5 counting, 7.8 → 9.3 prose.)

**A measurement trap worth recording:** the draft head pays a one-time ~3.6 s
CUDA-graph warmup on its *first* call, and a one-shot `run -p` charges all of
it to that single short run — an earlier version of these tables did exactly
that and wrongly concluded *"MTP loses on Windows."* Benchmark MTP in serve
mode with the first turn discarded. The full story, including the
uncached-zero-copy bug the verify exposed in the chunk matmul, is in the
[performance journal](performance-journal.md).
