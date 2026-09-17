# Little Gemma

A small, from-scratch **C program on CUDA** that loads a Gemma 4 model from
**GGUF** and runs it — written to *teach* how a modern LLM actually executes,
in the spirit of Karpathy's `llama2.c`, but covering a current model end to
end: parse GGUF → BPE tokenize → run the transformer → generate text.
Tokenization and dequantization are checked for exact agreement; forward
computations are compared with independent references using numerical
tolerances. See [validation](docs/architecture.md#validation).

```
text ──► tokenizer ──► token ids ──► forward ──► logits ──► argmax ──► next token
                          ▲                                                │
                          ╰──────────────── append, repeat ◄───────────────╯
```

**~9,900 lines of C/CUDA for the shipped int8-CUDA build, no vendored
dependencies** — 5,003 CPU · 8,410 f32-CUDA · 9,880 int8-CUDA, counted per
binary as the include closure the compiler actually sees (the three backends
are mutually exclusive, so no single program is their sum). That 9,880
includes multi-turn socket serving, image and audio understanding, a GPU
vision encoder, tensor-core flash-attention prefill, a ring-buffered f16 KV
cache, and byte-identical speculative decoding ([MTP](docs/mtp.md)).

## Performance vs llama.cpp

**Orin NX 16GB, 2026-09-06:** cache-only prefill and Q4_0 decode
specialization together, MAXN with pinned clocks. Same QAT GGUFs on both
engines. Full methodology, depth sweeps, and historical measurements:
**[docs/benchmarks.md](docs/benchmarks.md)**.

**Warm generation** (tokens/s, socket serving, greedy). Means give equal
weight to prose, C code, and French; each prompt uses the median of two warm
turns after a discarded warmup cycle. MTP depths 2–5 were swept independently
for full and selected heads; each column shows its best three-prompt mean.

| QAT model | Plain mean | Full-head MTP mean (N) | Selected 16K MTP mean (N) | Selected prose | Selected C code |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B | 44.0 | 54.5 (3) | **60.4 (3)** | 61.5 | 63.3 |
| E4B | 25.9 | 35.4 (4) | **38.1 (4)** | 33.0 | 48.5 |
| 12B | 11.7 | 17.5 (4) | **19.4 (4)** | 17.7 | 22.9 |

Selected heads use **16,384 FP16 rows** from the matched assistant, chosen by
`LG_MTP_IDS`; see [vocabulary selection](docs/mtp-vocab-trim.md). Set runtime
`LG_MTP_N=3` for E2B, `LG_MTP_N=4` for E4B/12B with these selected heads.
The default remains N=3. On the separate 930-token fixture, the best tested
depths are **E2B N=3, E4B/12B N=2**. These choices depend on the prompt;
the full sweeps are in the benchmarks. Packed selected Q4_0 was dropped
because its small gain did not justify the extra code.

**Warm prefill** (tokens/s, 930 input tokens per fresh serving connection;
first turn discarded, median of four). MTP uses selected heads at N=3/4/4
for E2B/E4B/12B. llama.cpp is the best of
`llama-bench -p 930 -n 0 -fa 0,1 -r 3`.

| QAT model | little-gemma plain | With selected MTP | llama.cpp | Plain / llama |
| --- | ---: | ---: | ---: | ---: |
| E2B | 2,578.8 | 2,578.7 | 1,021.8 | 2.52× |
| E4B | 858.0 | 858.0 | 554.6 | 1.55× |
| 12B | 202.5 | 202.4 | 231.6 | 0.87× |

Warmup is per process; these rates do not require reusing a conversation's
prefix cache. The first turn also performs lazy allocation and graph setup.

**Plain decode reference** (tokens/s, 64 generated tokens starting at context
32, median of two warm runs for little-gemma; llama-bench mean of three,
best of flash attention off/on):

| QAT model | little-gemma | llama-bench |
| --- | ---: | ---: |
| E2B | 44.32 | 37.87 |
| E4B | 25.83 | 18.95 |
| 12B | 11.83 | 9.24 |

The little-gemma API probe includes greedy argmax and token readback;
llama-bench feeds random tokens without sampling. Context and work count
match, but this is not an identical serving workload. Use the serving table
above for application rates. Context-930 results are in the benchmarks.

**Historical RTX A5000** results remain 134 / 70.7 / 213.9 tok/s plain decode
and 4,335 / 2,067 / 7,222 tok/s prefill for E4B / 12B / E2B. The combined
changes have not been timed on that workstation. Earlier llama-server MTP
and media time-to-first-token comparisons remain in the benchmark history.

Both improvements keep the existing kernels readable: prefill stops after
the last required KV write, and Q4_0 decode gives the compiler constant
format and block strides. MTP verification still runs the full transformer.

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
  streaming TTS. Measurement harnesses live in the protected research repository.

## License

MIT — see [LICENSE](LICENSE).
