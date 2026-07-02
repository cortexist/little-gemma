# Little Gemma

A small, from-scratch C program that loads a model stored in **GGUF** and runs it —
written to *teach* how a modern LLM actually executes, in the spirit of Karpathy's
`llama2.c` but covering a current model (Gemma) and aimed at **CUDA**.

It is a complete pipeline: parse GGUF → BPE tokenize → run the transformer →
generate text. Every stage was validated bit-for-bit against `llama.cpp`.

```
text ──► tokenizer ──► token ids ──► forward ──► logits ──► argmax ──► next token
                          ▲                                                │
                          ╰──────────────── append, repeat ◄───────────────╯
```

## Architecture

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
 │ scalar+OpenMP;│  │ forward, kv cache (rings, f16), CUDA graphs, chunked │
 │ doubles as the│  │ prefill, B=2 verify, device MTP draft — plus the GPU │
 │ reference spec│  │ vision encoder (media-cuda.cu)                       │
 └───────────────┘  ├───────────────────────────┬──────────────────────────┤
                    │     model-cuda-f32.cu     │    model-cuda-i8.cu     │
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

### What `model_forward` computes (Gemma 4 / E2B)

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

## How to Build

```
cmake -S . -B build
cmake --build build
```

### Release build (recommended — ~3.4× faster than Debug)

```
cmake --build build --config Release
```

Threading is via OpenMP (auto-detected by CMake); the matmul scales across cores.

### CUDA build (optional)

If the CUDA toolkit is found, CMake also builds `run-cuda` (readable f32 matmul)
and `run-cuda-i8` (int8 matmul, the fast one):

```
cmake --build build --config Release --target run-cuda-i8
```

Same CLI as `run`. The CPU backend (`model-cpu.c`) and the CUDA backends
(`model-cuda-f32.cu`, `model-cuda-i8.cu`) implement the same `model.h`; only the compute kernels differ.

## Usage

```
run -m <model.gguf> [-mm <mmproj.gguf>] [-mtp <assistant.gguf>] [-p "prompt" | -s <socket>]
run -c <socket>
```

- `-m` only → prints the GGUF dump + config, then exits.
- `-m` + `-p` → one-shot demo: tokenizes the prompt, generates, reports tok/s.
  Text only — media arrives over the socket; see "Multimodal" below.
- `-mm` → also load a multimodal projector, so `-s` sessions accept image and
  audio frames.
- `-mtp` → also load a gemma4-assistant draft head and decode speculatively;
  see "Speculative decoding" below. Output is identical, only speed changes.
- `-sys <file>` → prefill the file as a system turn ONCE at server start;
  every `-s` session then begins with it already in context — the
  skills-in-context pattern without the TTFT tax. The saved cache rows are
  restored at session start, so each conversation begins byte-identically
  fresh no matter what a previous session did to the rings.

```
> run -m model.gguf -p "The capital of France is"
The capital of France is Paris.
prompt: 6 tokens in 4.68s (1.28 tok/s)
gen:    2 tokens in 1.48s (1.35 tok/s)
```

### Serving over a Unix-domain socket (`-s`)

`-s` turns the runner into a tiny conversation server — no HTTP, no JSON, no web
security surface; transport concerns (TCP exposure, TLS, auth) belong to `nc`,
`socat`, and whatever chat server or agent sits downstream:

```
run-cuda-i8 -m model.gguf -s /tmp/lg.sock        # Ctrl-C to stop
echo "What is the capital of France?" | nc -N -U /tmp/lg.sock
```

(The `-N` matters: it half-closes the socket at stdin EOF, which is how the
server learns the conversation is over — without it both sides wait forever;
`-q 0` instead quits on a timer and can cut the answer off mid-stream.)

The protocol is the simplest thing that works, designed so raw media frames can
join it later without breaking anything:

- **A connection is a conversation.** The kv cache lives for the connection, so
  multi-turn context is free; close the connection (or just stop typing into
  the client) to end the session. One conversation is served at a time — decode
  is batch-1, and the listen backlog is the queue.
- **A line is a user turn.** Each newline-terminated line is wrapped in the
  Gemma chat template and prefilled.
- **The reply is the raw token stream**, special tokens included — the thinking
  channel, the markers, and the closing `<turn|>` that downstream tools can
  split turns on. The server filters nothing; presentation is the client's job.
- **Input is symmetric: control tokens pass through.** Special-token text in a
  line (`<|tool>`, `<|tool_response>`, `<|think|>`, even `<turn|>`) encodes as
  the real control tokens — so a client can speak the full Gemma 4 protocol,
  tool calling included, with no tool-calling code in the runner. The flip
  side is deliberate too: the server does not neutralize tags, so sanitizing
  untrusted text belongs upstream, exactly where TLS and auth already live.
- stdout/stderr are logging only; per-turn stats go to stderr.
- A turn is capped at 1,024 output tokens (`SERVE_GEN`): greedy decoding has
  no sampler and no repetition penalty, so a degenerate loop would otherwise
  spin until the context filled (observed: 8,098 tokens of the same sentence).
  A capped turn ends with a visible `[SERVE_GEN cap]<turn|>`.
- A clean half-close (e.g. `nc -N` after stdin EOF) still receives the full turn
  in flight; only a hard reset aborts generation early — checked between
  tokens, so one ~5–17 ms forward is the most work a dead client can waste.

**Don't read benchmark numbers out of one-question sessions.** The per-turn
tok/s swings wildly for the same question — 13, then 61, then 134 — and that
is the GPU, not the server: an idle GPU parks at its clock floor (210 of
2100 MHz on the A5000 here), and a one-question session is only ~25–35
forwards, so the whole session fits inside the clock ramp. (The one-time
weight repack/upload is no longer part of this: the server absorbs it in a
2-token warmup before it starts listening, so even the first turn runs warm.)
The numbers in the tables below are steady state: pipeline several turns into
one connection, or pin the clocks with `nvidia-smi --lock-gpu-clocks`.

AF_UNIX works on Windows 10+ with the same code (`afunix.h`); the socket path's
directory must exist (`/tmp/...` is a Linux path — on Windows use e.g.
`%TEMP%\lg.sock`). On native Windows, the usual clients are simply not there:
no socat, `ncat` can't speak AF_UNIX, and even Python's `socket` module doesn't
expose it. The build therefore ships its own:

```
socket_cat %TEMP%\lg.sock                          # the wire, verbatim
run -c %TEMP%\lg.sock                              # the conversation, cleaned up
```

`socket_cat` (`tools/socket_cat.c`) is the missing netcat: stdin to the socket,
socket to stdout, no protocol — you see the raw token stream exactly as sent,
`<turn|>` markers and all. `run -c` is the protocol-aware client: it pumps
stdin lines as turns and knows where each streamed reply ends.

## Multimodal

The gemma-4-12B takes images and audio with **no encoder at all** — its mmproj
file is 11 tensors (167 MB): vision is one linear layer over 48×48-pixel
patches plus three norms and a learned 2-axis position table; audio is one
linear layer over raw 16 kHz waveform sliced into 640-sample (40 ms) frames.
No vision transformer, no conformer, no mel spectrogram. `media.c` implements
exactly that (the math mirrors llama.cpp's `gemma4uv`/`gemma4ua` graphs), the
CUDA builds run it on tensor cores (`media-cuda.cu` — an image is two GEMM
launches and change, an audio clip's frame-at-a-time matvec loop becomes one
batched GEMM; the GPU rows are bit-identical across devices, and the host
path stays in as oracle and fallback), and the resulting embedding rows
prefill through the same batched-chunk path as text — they are just rows the
tokenizer didn't make. Media embeddings are *not* `√n_embd`-scaled; only
real token lookups are.

The runner never reads media **files**. The split mirrors the socket design:
file formats — JPEG entropy coding, WAV chunk soup, resampling, resize
filters — belong to an upstream tool, and the model runner handles only what
its own tensors define. That tool is **`mmcat`**, in the sibling repo
[`little-gemma-tools`](../little-gemma-tools): it auto-detects each file with
`ffprobe` and decodes with `ffmpeg` (so it takes mp4/mkv/webm/jpg/png/wav/… and
an mp4's video+soundtrack from one argument), then sends the raw shape the model
wants (u8 RGB at 48-multiple dimensions, mono 16 kHz s16 PCM in whole frames) as
typed length-prefixed frames followed by the question line, and streams the
answer. Keeping it a separate repo is the point — the core stays dependency-free
C/CUDA; the file-decoding dependency (ffmpeg) lives with the tool, not the engine.

```
run-cuda-i8 -m gemma-4-12B-it-Q4_K_M.gguf -mm mmproj-F16.gguf -s %TEMP%\lg.sock
mmcat %TEMP%\lg.sock photo.jpg "What is in this image?"
mmcat %TEMP%\lg.sock clip.mp4  "What happens, and what is said?"
```

A turn over the socket is zero or more typed frames — media spans, and text
frames that land between them — then the usual newline-terminated text line;
a text-only client speaks the same protocol it always did. The interleaved
text is what makes a multi-image turn legible to the model: given two bare
images it answers "there is only one image", but with timestamp text between
them it counts both frames and compares them correctly — a video tool emits
exactly that shape. The server wraps each media span in the model's marker
tokens, so the model sees what it was trained on:
`<|turn>user\n<|image>` *(192-ish embedding rows)* `<image|>{text}<turn|>\n<|turn>model\n`.

Media also **prefills as it arrives**: a span's kv rows are seated the moment
its frame is decoded — causality needs only completed prefixes, and the
bidirectional window never leaves a span — so the cache fills while the
client is still sending, and once the question lands only the line itself is
left to prefill. One embedded span is held while its verdict is unknown: an
idle socket means the client is between messages (a video tool decoding its
next frame), so the held span prefills under that pause; a queued next frame
flushes it; the queued text line packs it into the tail's single call, so a
camera's frame+question burst costs exactly what the packed turn always did.
On the Orin, a six-frame video turn (12B) answers in 2.9 s instead of
8.8 s — same reply, byte for byte.

Text is optional when media frames are sent: a spoken question, or a written
note shown to the camera, is a complete turn by itself —

```
mmcat %TEMP%\lg.sock question.wav
```

— and the model answers the *voice*, because past the projection it cannot
tell a sentence that arrived as sound from one that arrived as keystrokes.

The E2B/E4B mmproj files instead carry conventional encoders — the **legacy
path**. Their 16-block vision transformer (`gemma4v`) is implemented: on the
CUDA backends it runs on the GPU on tensor cores (`media-cuda.cu`, m16n8k16
with f16 inputs and f32 accumulation — ~0.2 s per 130-token frame on the
Orin, under 0.1 s on a desktop card), with the host implementation kept in
the binary as numeric oracle (`LG_MEDIA_VERIFY=1` runs both per image and
prints the max difference; ~6e-3, which is the f16 input rounding) and as
the only path on the CPU build. Their 12-block audio
conformer was implemented, verified against the reference, and **dropped**:
it underperforms in practice, and production pipelines pair these models
with a dedicated STT (Whisper) anyway — audio frames at a legacy mmproj say
so and point at the 12B, which hears. The conformer lives in git history.

## Speculative decoding (MTP)

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
[performance journal](docs/performance-journal.md).

## On SIMD (AVX2) — intentionally not implemented

The CPU matmul is plain scalar C parallelized with OpenMP; there are **no hand-written
AVX2/FMA intrinsics**. This is deliberate. Hand-vectorizing the CPU kernels would
be throwaway work, because the real target is **CUDA**: on the GPU the parallelism
comes from thousands of threads (replacing OpenMP) and the per-element math runs in
kernels (replacing what SIMD would do). The weights are already stored quantized
and unpacked inside the matmul — exactly the shape a GPU kernel wants — so the
CPU kernels in `model-cpu.c` (`matmul_q`, `rmsnorm`, `rope_neox`, `softmax`, `gelu`)
double as the reference spec for the CUDA versions in `model-cuda-f32.cu`.

The one exception proves the rule: `media.c` (the host vision encoder) compiles
with file-scoped fast-math so the *compiler* may vectorize its reductions —
still no intrinsics, and the naive one-line dot loop turned out to be the
fastest form (a hand-unrolled multi-accumulator version defeats MSVC's
vectorizer pattern match). The LLM side keeps strict fp.

## On mmap — intentionally not used

Most GGUF runners memory-map the model file and let the OS page weights in on
demand. little-gemma instead reads the whole file into RAM up front and refuses to
start if it does not fit. This avoids the complexity — and the silent failure mode —
of paging: with mmap a model that is slightly too large still "loads", then thrashes
as the OS evicts weight pages and re-reads them from disk every token, so the
slowdown is invisible and hard to reason about. The rule here is deliberately
simple: if you have enough memory, you run; if you don't, you get a clear error at
load time instead of a mysterious crawl (`load_gguf` checks the size and bails).

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
bisections included, are **[docs/performance-journal.md](docs/performance-journal.md)**
and **[docs/prefill-performance-journal.md](docs/prefill-performance-journal.md)**.

Where things stand — same day, same machine, same prompt (little-gemma =
`run-cuda-i8`, 256 generated tokens, warm; llama.cpp = `llama-bench` tg32,
best of fa0/fa1). **Plain decode**; MTP multiplies on top (the table above):

**RTX A5000 (Windows), re-measured 2026-07:**

| model | size | params | little-gemma | llama.cpp CUDA | ratio |
|-------|-----:|-------:|-------------:|---------------:|------:|
| E4B Q4_K_M  | 4.95 GiB |  7.52 B | 118.8 | 117.4 ± 1.5 | **1.01×** |
| 12B Q4_K_M  | 6.86 GiB | 11.91 B |  62.3 |  66.2 ± 0.2 | 0.94× |

**Jetson Orin NX 16GB (Linux, integrated GPU, zero-copy weights):**

| model | little-gemma | llama.cpp CUDA | ratio |
|-------|-------------:|---------------:|------:|
| E4B Q4_K_M | 16.80 | 13.36 ± 0.04 | **1.26×** |
| 12B Q4_K_M |  8.27 |  7.04 ± 0.04 | **1.17×** |

The pattern is the project's thesis in two tables. The smaller the model, the
more decode speed is about everything *around* the matmul — launch overhead,
sync round-trips, norms, the PLE path — which a few thousand readable lines
can do leanly. The bigger the model, the more it reduces to sustained DRAM
bandwidth through the quantized matmul, where llama.cpp's arch-tuned kernels
still hold a few percent on desktop (the 12B's 0.94×). The E4B crossed into
a win when the f16 SWA rings landed; on the edge device this project
actually targets, every row is ahead, and MTP widens the lead.

**Prefill** (prompt processing) was long the honest weak axis — llama.cpp
prefills through arch-tuned tensor-core GEMMs at large batch — and the
2026-07 pushes closed most of it on BOTH devices. Warm serve prefill,
~930-token turns, same-day llama.cpp:

| device | model | prompt tok/s | vs llama.cpp |
|--------|-------|-------------:|--------------|
| A5000 | 12B | 533 → **~1,820** | 0.81× (was 0.24×) |
| A5000 | E4B | 1,020 → **~3,720** | ~0.70× (was 0.21×) |
| Orin  | 12B | ~102 → **~173** | **0.80×** |
| Orin  | E4B | ~192 → **~417** | **0.82×** |

Most steps were gated byte-identical; the f16 SWA rings and warp-row norms
ship under the same determinism + quality gate as the original f16-KV step
(and bought decode ~+8% on the Jetson as a side effect). The residual is
llama.cpp's ground-up MMQ instruction schedule and flash-attention
architecture — measured to their floors with warp-stall counters and
deliberately not vendored back (`docs/stream-k-experiment.md`,
`docs/prefill-performance-journal.md`). In interactive serving it rarely
shows — turns are short, `-sys` removes the skills re-prefill, the GPU
encoder removed the image one — but on very long documents llama.cpp still
wins the wait.

**TTFT** (time to first token, the metric a user feels) — measured against
`llama-server` with the same client-side definition on both stacks: last
request byte sent → first streamed token, warm server, first turn
discarded, prompt caching off, same GGUFs, token parity checked
(2026-07-02, medians of warm turns):

| device | model | turn | little-gemma ttft (s) / prefill (tok/s) | llama-server ttft (s) / prefill (tok/s) | input tokens (lg / llama) |
|--------|-----|------|----------------------------:|----------------------------:|:-------------------------:|
| A5000 | 12B | 929-tok text | **0.51** / 1,803 | 0.53 / 1,753 | 928 / 928 |
| A5000 | E4B | 929-tok text | 0.25 / 3,717 | **0.23** / 4,070 | 928 / 928 |
| A5000 | 12B | image+question | **0.20** / 827 | 0.29 / 715 | 150 / 157 |
| A5000 | E4B | image+question | **0.13** / 1,317 | 0.30 / 1,229 | 150 / 293 |
| Orin | 12B | 929-tok text | 5.37 / 173 | **4.64** / 200 | 929 / 928 |
| Orin | E4B | 929-tok text | 2.23 / 417 | **1.91** / 488 | 929 / 928 |
| Orin | 12B | image+question | **1.15** / 132 | 2.11 / 92 | 150 / 157 |
| Orin | E4B | image+question | **0.69** / 308 | 1.65 / 210 | 150 / 293 |

The prefill column is tok/s over the whole turn: ours from the serve stat
(everything after the burst, media included), llama's from its own
server-reported `prompt_n/prompt_ms` — which *excludes* its media encoder
(visible as ttft − prompt_ms ≈ 0.25–0.43 s on the Orin image rows). The
last column is the **number of input (prompt) tokens** each stack processed
for the same turn — a count, not a rate: text rows differ only by BOS
accounting, image rows differ where preprocessing policy differs. Prefill
rate and TTFT are related but deliberately not 1:1 — the first-decode step,
chunk padding, encoders, and arrival overlap all live in one and not the
other; **[docs/prefill-vs-ttft.md](docs/prefill-vs-ttft.md)** walks through
it. Text turns track the known kernel ratio (ours ~0.86× in serving, a
touch better than llama-bench's 0.80× — their server stack costs more than
ours).
**Media turns invert it**: 1.5–2.4× ahead, and the 12B image rows are
near-token-equal (both encode natively), so that win is not a policy
artifact. The E4B token counts differ by upstream policy — the fork
upscales every image to a 252-token minimum; mmcat sends 624×480 natively
as 130. Paced arrival (video frames over a socket, spans prefilled while
the clip streams) has no llama-server counterpart at all — those numbers
(Orin 12B six-frame video ttft 8.79 → 2.34 s) are measured against our own
deferred baseline.

**TTFS** (time to first *speakable* sentence — when a sentence-chunking TTS
can start talking, the metric a voice product feels; thought channels are
not speakable and delay it at decode rate). Same clocks and discipline,
prose-shaped questions so the first sentence is a real one:

| device | turn | little-gemma ttfs | llama-server ttfs | first sentence |
|--------|------|------------------:|------------------:|:--------------:|
| A5000 | 12B, 929-tok text | **1.27 s** | 1.28 s | 29 tok, same words |
| A5000 | E4B, 929-tok text | **0.51 s** | 0.54 s | 19 tok, same words |
| A5000 | 12B, image+question | **0.78 s** | 8.9 s | thought: ~0 vs 528 tok |
| A5000 | E4B, image+question | **0.25 s** | 4.7 s | thought: ~0 vs 466 tok |
| Orin | 12B, 929-tok text | 9.23 s | **8.47 s** | 29 tok, same words |
| Orin | E4B, 929-tok text | 3.43 s | **3.21 s** | 19 tok, same words |
| Orin | 12B, image+question | **5.22 s** | 98.6 s | thought: short vs 717 tok |
| Orin | E4B, image+question | **1.98 s** | 35.8 s | thought: ~0 vs 481 tok |

Text rows are the clean kernel story: identical prompt bytes produce the
identical first sentence on both stacks, so TTFS = TTFT + the same tokens
at each stack's decode rate — on the A5000 our decode edge erases llama's
TTFT lead by the end of the sentence (1.27 vs 1.28 s), on the Orin their
prefill lead survives but shrinks. The image rows are the product story,
and two effects compound: the media-TTFT levers above, and **the thought
channel** — llama-server's chat template elicits a 466–717-token "thinking
process" from these models before the first speakable word, paid at decode
rate; our serve template gets a terse thought from the same weights. Even
subtracting their thought entirely, our media TTFS still leads — but the
lesson generalizes: for voice, the prompt template's effect on thought
length dwarfs every kernel in the stack (see `-sys`).

## Performance vs llama.cpp (CPU, apples-to-apples)

Both **no CUDA, no SIMD intrinsics, 12 threads**, single-token generation:

| build                                         | generation |
|-----------------------------------------------|-----------:|
| little-gemma (scalar + OpenMP)                | ~1.3 tok/s |
| llama.cpp (SIMD off, CUDA off, `llama-bench`) | 2.33 tok/s |

Only **~1.8×** apart. The gap is *algorithm*, not SIMD: llama.cpp quantizes the
activation to int8 once and does an integer `vec_dot` against the quantized
weights (never materializing f32), plus better cache blocking. little-gemma
dequantizes each weight row to f32 then does an f32 dot. With AVX2 *on*,
llama.cpp reaches ~10–30 tok/s on the same machine — that headroom is what CUDA
is for.

## Lines of Code

| directory | files | code  |
|-----------|-------|------:|
| src       | 16    | 5,948 |
| include   | 6     |   254 |

~6,200 lines of code in the repository (`tools/` not counted). The core
vendors **nothing** — pure C/CUDA; media-file decoding lives in `mmcat` in
the sibling little-gemma-tools repo. The backends are mutually exclusive, so
no single program is anywhere near that. Each binary is the shared pipeline
(GGUF parse, dequant, tokenizer, config, multimodal embedders, the MTP draft
head, CLI + socket server — 2,658 lines) plus exactly one backend:

| binary        | backend on top of the shared 2,658                       | code lines |
|---------------|----------------------------------------------------------|-----------:|
| `run`         | `model-cpu.c`                                            |      3,032 |
| `run-cuda`    | `model-cuda.cuh` + `model-cuda-f32.cu` + `media-cuda.cu` |      4,697 |
| `run-cuda-i8` | `model-cuda.cuh` + `model-cuda-i8.cu` + `media-cuda.cu` |      5,539 |

(`graph.c`/`graph.h`, the teaching tensor/graph layer, are exercised by
`graph_test` only.) So the program that out-decodes llama.cpp CUDA on the
Jetson on every model it runs — multi-turn socket serving, batched wide-chunk
prefill, a ring-buffered f16 KV cache, tensor-core flash-attention prefill,
split-K decode, image and audio understanding, a GPU vision encoder, an own
m16n8k32 tensor-core q4_K prefill kernel, and byte-identical speculative
decoding included — is **about 5,500 lines of C end to end**, tokenizer and
all, with no vendored dependency.

## Validation

Built against a CPU `llama.cpp` as an oracle (`llama-eval-callback` dumps every
intermediate tensor): dequantization is bit-exact vs the `gguf` Python package;
the forward pass matches an independent NumPy f32 reference and llama.cpp's logits
(within the f32-vs-quantized-matmul gap); the tokenizer matches `llama-tokenize`
exactly. `test/graph_test.c` (a CTest target) checks the graph kernels.

## License

MIT — see [LICENSE](LICENSE).
