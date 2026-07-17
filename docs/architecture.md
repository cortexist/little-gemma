# Architecture

```
                     ┌─────────────────────────────────┐
                     │              run.c              │  CLI · socket server · client
                     │ greedy and speculative decoding │  -m -mm -mtp [-p | -s | -c]
                     └────────────────┬────────────────┘
        ╭───────────────┬─────────────┼────────────────┬───────────────╮
        ▼               ▼             ▼                ▼               ▼
 ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
 │ tokenizer.c │ │   model.c   │ │   media.c   │ │    mtp.c    │ │   gguf.c    │
 │ BPE text↔ids│ │ config and  │ │ image/audio │ │ the MTP     │ │ parse → ctx │
 └─────────────┘ │ per-layer   │ │ → embedding │ │ draft head  │ │ (hdr, meta, │
                 │ geometry    │ │ rows        │ └─────────────┘ │ tensors,    │
                 └─────────────┘ └─────────────┘                 │ data blob)  │
                                                                 └─────────────┘
     weights stay quantized in the gguf blob; everyone reads them through
                    quant.c (dequantize q3_K/q4_K/q8_0/…)

         model.h — one API, one forward, three interchangeable backends:

 ┌───────────────┐  ┌──────────────────────────────────────────────────────┐
 │  model-cpu.c  │  │              model-cuda.cuh (shared)                 │
 │ scalar+OpenMP;│  │ forward, kv cache (rings, f16), CUDA graphs, decode  │
 │ doubles as the│  │ — plus prefill-kernel.cuh (flash, chunked prefill),  │
 │ reference spec│  │ mtp-kernel.cuh (verify + device draft), and the GPU  │
 │               │  │ vision/audio encoder (media-kernel.cu)               │
 └───────────────┘  ├───────────────────────────┬──────────────────────────┤
                    │     model-cuda-f32.cu     │    model-cuda-i8.cu      │
                    │   readable f32 matmul     │  int8 dp4a + wide loads  │
                    └───────────────────────────┴──────────────────────────┘

 tools/  — socket_cat (the raw wire); media files → frames via mmcat (sibling repo little-gemma-tools)
 graph.c — a minimal tensor/graph layer (matmul, rmsnorm, softmax, …) kept as
           the "what a compute graph is" teaching reference (graph_test only).
```

**Layering:** GGUF/ggml jargon stays in the lower layers (`gguf.c`, `quant.c`);
the model layer (`model.c`) reads like plain transformer code. The file is read
fully into RAM (no mmap — it errors out rather than silently paging), weights stay
quantized there, and each weight row is unpacked to f32 on the fly during matmul.

## What `model_forward` computes (Gemma 4 / E2B)

```
embed(token) × √d
for each of 35 layers:
    ├─ attention:  rmsnorm → Q,K,V → per-head Q/K-norm → NeoX RoPE
    │              → GQA (8 q-heads, 1 kv-head) with sliding-window OR global mask
    │              → KV cache (layers ≥15 reuse an earlier layer's KV)
    │              → output proj → post-norm → residual
    ├─ feed-forward (GeGLU):  rmsnorm → gelu(gate)·up → down → post-norm → residual
    │              (elastic FFN: width 6144 for layers 0–14, 12288 for 15–34)
    └─ per-layer input (PLE) + per-layer output scale
final rmsnorm → tied logits (× token_embd) → softcap
```

## On CUDA

The whole forward, the kv cache, and every non-matmul kernel live in
`model-cuda.cuh`; the **matmul is the only thing that differs** between the two
GPU backends, so each is a thin file that includes the header and defines just
`matmul_q`. Diff `model-cuda-f32.cu` against `model-cuda-i8.cu` to see exactly
where the speed comes from.

Getting the speed was a journey of gated steps — profiling-led rewrites that
*failed* and earned their write-ups, a CUDA graph against WDDM launch
latency, wide weight loads, a long-context roadmap (online-softmax attention,
batched prefill, a kv cache at ~5% of its old footprint), GPU media encoders,
speculative decoding, a tensor-core prefill push (an own int8 `mma` chunk
matmul, flash attention for the prompt phase, split-K decode for long
context) — and then a prefill overhaul that took both devices to their
structural floor: balanced wide serve chunks, an L2-aware launch order,
software-pipelined weight staging, and warp-cooperative activation
quantization, every step gated byte-identical. The full logs, dead ends and
bisections included, are [performance-journal.md](performance-journal.md)
and [prefill-performance-journal.md](prefill-performance-journal.md).
The complete voice pipeline — mic → whisper → serve → streaming piper, with
runnable commands and every measurement harness — is
[voice-pipeline.md](voice-pipeline.md) + [`bench/`](../bench/).

## Validation

Built against a CPU `llama.cpp` as an oracle (`llama-eval-callback` dumps every
intermediate tensor): dequantization is bit-exact vs the `gguf` Python package;
the forward pass matches an independent NumPy f32 reference and llama.cpp's logits
(within the f32-vs-quantized-matmul gap); the tokenizer matches `llama-tokenize`
exactly. `test/graph_test.c` (a CTest target) checks the graph kernels.
