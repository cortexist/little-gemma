# Little Gemma

A small, from-scratch **C program on CUDA** that loads a Gemma 4 model from
**GGUF** and runs it — written to *teach* how a modern LLM actually executes,
in the spirit of Karpathy's `llama2.c`, but covering a current model end to
end: parse GGUF → BPE tokenize → run the transformer → generate text. Every
stage validated bit-for-bit against `llama.cpp`.

```
text ──► tokenizer ──► token ids ──► forward ──► logits ──► argmax ──► next token
                          ▲                                                │
                          ╰──────────────── append, repeat ◄───────────────╯
```

**~6,100 lines of C/CUDA, no vendored dependencies** — and that includes
multi-turn socket serving, image and audio understanding, a GPU vision
encoder, tensor-core flash-attention prefill, a ring-buffered f16 KV cache,
and byte-identical speculative decoding ([MTP](docs/mtp.md)).

## Performance vs llama.cpp

Same day, same GGUFs, same machines; little-gemma measured on its serving
path, llama.cpp with `llama-bench` (best of `-fa 0/1`, decode at matched
context depth). Full tables, methodology, and history:
**[docs/benchmarks.md](docs/benchmarks.md)**.

**Decode** (tokens/s, batch 1) — ahead on the Jetson, the device this
project targets, and at parity on desktop:

| device | model | little-gemma | llama.cpp | ratio |
|--------|-------|-------------:|----------:|------:|
| Jetson Orin NX | E4B Q4_K_M | **17.9** | 14.1 | **1.27×** |
| Jetson Orin NX | 12B Q4_K_M | **8.3** | 7.4 | **1.12×** |
| Jetson Orin NX | E2B QAT q4_0 | 34.5 | 37.4 | 0.92× |
| RTX A5000 | E4B / 12B / E2B | **117.6** / 58.0 / **213.9** | 116.1 / 62.5 / 209.3 | **1.01×** / 0.93× / **1.02×** |

**Prefill** (929-token prompts) — **0.8×** llama.cpp, consistently:

| device | model | little-gemma | llama.cpp | ratio |
|--------|-------|-------------:|----------:|------:|
| Jetson Orin NX | E4B / 12B / E2B | 426 / 174 / 834 | 524 / 217 / 1,020 | 0.80–0.82× |
| RTX A5000 | E4B / 12B / E2B | 3,703 / 1,782 / 7,222 | 4,846 / 2,207 / 8,785 | 0.76–0.82× |

The pattern is the project's thesis: decode speed is mostly everything
*around* the matmul — launch overhead, syncs, norms, the PLE path, how the
KV walk is split across the GPU — which a few thousand readable lines can do
leanly. Prefill runs through llama.cpp's home turf (arch-tuned tensor-core
GEMMs); the 2026-07 campaign closed it from ~0.2× to 0.8× and measured the
rest to its structural floor. [MTP](docs/mtp.md) multiplies decode on top
(~2× structured / ~1.1–1.35× prose), output byte-identical. On media turns,
time-to-first-token **inverts in our favor** (1.5–2.4×) — GPU encoder plus
arrival-overlapped prefill; see [docs/benchmarks.md](docs/benchmarks.md).

## Build

```
cmake -S . -B build
cmake --build build --config Release
```

CPU build (`run`) needs only a C compiler; OpenMP is auto-detected. If the
CUDA toolkit is found, CMake also builds `run-cuda` (readable f32 matmul)
and `run-cuda-i8` (int8 + tensor cores — the fast one):

```
cmake --build build --config Release --target run-cuda-i8
```

All three implement the same `model.h`; only the compute kernels differ.

## Run

```
run-cuda-i8 -m gemma-4-E4B-it-Q4_K_M.gguf -p "The capital of France is"
```

Serve conversations over a Unix-domain socket (multi-turn KV cache, raw
token stream out — details in [docs/serving.md](docs/serving.md)):

```
run-cuda-i8 -m model.gguf -s /tmp/lg.sock          # server (Ctrl-C to stop)
echo "What is the capital of France?" | nc -N -U /tmp/lg.sock
run -c /tmp/lg.sock                                # or the bundled client
```

Options: `-mm mmproj.gguf` adds image/audio input over the socket
([docs/multimodal.md](docs/multimodal.md), via
[`mmcat`](../little-gemma-tools)); `-mtp assistant.gguf` adds speculative
decoding with byte-identical output ([docs/mtp.md](docs/mtp.md));
`-sys file` prefills a system turn once at server start; `-temp`/`-topk`/
`-topp`/`-seed` sample instead of greedy. On Windows the same code serves
`%TEMP%\lg.sock` and the build ships its own socket clients.

## Documentation

- [docs/architecture.md](docs/architecture.md) — module map, the forward
  pass walkthrough, backend layering, validation.
- [docs/benchmarks.md](docs/benchmarks.md) — canonical numbers, methodology,
  TTFT/TTFS, reconciliation of older figures.
- [docs/serving.md](docs/serving.md) — the socket protocol, CLI reference,
  Windows clients.
- [docs/multimodal.md](docs/multimodal.md) — encoder-free 12B vision/audio,
  the E2B/E4B legacy encoders, streaming/dictation prefill.
- [docs/mtp.md](docs/mtp.md) — Gemma 4's multi-token-prediction head,
  byte-identical speculative decoding.
- [docs/design-notes.md](docs/design-notes.md) — why no SIMD, why no mmap,
  CPU-vs-llama.cpp apples-to-apples, lines-of-code ledger.
- [docs/performance-journal.md](docs/performance-journal.md) and
  [docs/prefill-performance-journal.md](docs/prefill-performance-journal.md)
  — the full optimization logs, failed experiments included.
- [docs/voice-pipeline.md](docs/voice-pipeline.md) — mic → whisper → serve →
  streaming TTS, with runnable harnesses in [`bench/`](bench/).

## License

MIT — see [LICENSE](LICENSE).
