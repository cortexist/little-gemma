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

**Generation with [MTP](docs/mtp.md) on** (tokens/s) — the shipped
configuration, and the number that matters in use; output is
**byte-identical to plain greedy decoding, always**:

| device | model | plain | +MTP chat | +MTP structured |
|--------|-------|------:|----------:|----------------:|
| Jetson Orin NX | E4B QAT | 20.7 | **29.9** | **48.6** |
| Jetson Orin NX | 12B QAT | 9.8 | **14.5** | **20.4** |
| RTX A5000 | E4B QAT | 134 | **168.8** | **282** |
| RTX A5000 | 12B QAT | 70.7 | **101.5** | **143.7** |

The gain is content-dependent — Orin E4B by turn type: prose 29.9, image
description 31.5, code 40.7, fully predictable output 48.6 (57.8 at
block-4) — and ahead of `llama-server`'s own `draft-mtp` at every point
measured (prose 24.5, code 34.1, image 28.8).

**Decode** (tokens/s, batch 1, speculation off) — ahead on the Jetson, the
device this project targets, and at parity on desktop:

| device | model | little-gemma | llama.cpp | ratio |
|--------|-------|-------------:|----------:|------:|
| Jetson Orin NX | E4B QAT q4_0 | **20.7** | 18.7 | **1.11×** |
| Jetson Orin NX | 12B QAT q4_0 | **9.8** | 9.05 | **1.08×** |
| Jetson Orin NX | E2B QAT q4_0 | 34.5 | 37.4 | 0.92× |
| RTX A5000 | E4B / 12B / E2B | 134 / 70.7 / **213.9** | 136.6 / 70.7 / 209.3 | 0.98× / **1.00×** / **1.02×** |

**Prefill** (929-token prompts) — **0.8×** llama.cpp, consistently:

| device | model | little-gemma | llama.cpp | ratio |
|--------|-------|-------------:|----------:|------:|
| Jetson Orin NX | E4B / 12B / E2B | 474 / 193 / 834 | 553 / 232 / 1,020 | 0.82–0.86× |
| RTX A5000 | E4B / 12B / E2B | 4,335 / 2,067 / 7,222 | 5,254 / 2,365 / 8,785 | 0.82–0.87× |

The pattern is the project's thesis: decode speed is mostly everything
*around* the matmul — launch overhead, syncs, norms, the PLE path, how the
KV walk is split across the GPU — which a few thousand readable lines can do
leanly. Prefill runs through llama.cpp's home turf (arch-tuned tensor-core
GEMMs); the 2026-07 campaign closed it from ~0.2× to 0.8× and measured the
rest to its structural floor. On media turns,
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
run-cuda-i8 -m gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf -p "The capital of France is"
```

The default E4B and 12B are unsloth's **QAT q4_0** builds (with their
matched MTP heads, `mtp-gemma-4-{E4B,12B}-it.gguf`) — QAT-trained for
q4_0, and faster than Q4_K_M on both stacks.

Serve conversations over a Unix-domain socket (multi-turn KV cache, raw
token stream out — details in [docs/serving.md](docs/serving.md)):

```
run-cuda-i8 -m model.gguf -s /tmp/lg.sock          # server (Ctrl-C to stop)
echo "What is the capital of France?" | nc -N -U /tmp/lg.sock
run -c /tmp/lg.sock                                # or the bundled client
```

Options:

- `-mm mmproj.gguf` — image/audio input over the socket, via
  [`mmcat`](../little-gemma-tools) ([docs/multimodal.md](docs/multimodal.md)).
- `-mtp assistant.gguf` — speculative decoding, byte-identical output
  ([docs/mtp.md](docs/mtp.md)).
- `-sys file` — prefill a system turn once at server start.
- `-think N` — cap the reasoning channel: `0` off (structural — prompt control
  of thinking is inert on Gemma 4), `N` up to N tokens, omitted unlimited
  ([docs/serving.md](docs/serving.md#controlling-the-reasoning-channel-think)).
- `-temp`/`-topk`/`-topp`/`-seed` — sample instead of greedy.

On Windows the same code serves `%TEMP%\lg.sock` and the build ships its own
socket clients.

## Documentation

- [docs/architecture.md](docs/architecture.md) — module map, the forward
  pass walkthrough, backend layering, validation.
- [docs/benchmarks.md](docs/benchmarks.md) — canonical numbers, methodology,
  TTFT/TTFS, reconciliation of older figures.
- [docs/seminar.md](docs/seminar.md) — Q&A: why decode is faster and prefill
  is slower than llama.cpp, and the single most remarkable lever in each.
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
