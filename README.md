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
 └─────────────┘ │ per-layer   │ │ → embedding │ │ draft head  │ │ (hdr, kv,   │
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
                    │     model-cuda-f32.cu     │    model-cuda-i8r.cu     │
                    │   readable f32 matmul     │  int8 dp4a + wide loads  │
                    └───────────────────────────┴──────────────────────────┘

 tools/  — socket_cat (the raw wire) · media_cat (files → frames + question)
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
and `run-cuda-i8r` (int8 matmul, the fast one):

```
cmake --build build --config Release --target run-cuda-i8r
```

Same CLI as `run`. The CPU backend (`model-cpu.c`) and the CUDA backends
(`model-cuda-f32.cu`, `model-cuda-i8r.cu`) implement the same `model.h`; only the compute kernels differ.

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
  every `-s` session then begins with it already in context. This is the
  skills-in-context pattern without the TTFT tax: a client that reconnects
  per exchange (a robot, an agent pipeline) stops re-paying the skills
  prefill on every connection — the saved cache rows are restored at session
  start (a long previous session can wrap the sliding-window rings over
  them), so each conversation starts byte-identically fresh.

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
run-cuda-i8r -m model.gguf -s /tmp/lg.sock        # Ctrl-C to stop
echo "What is the capital of France?" | nc -N -U /tmp/lg.sock
```

(The `-N` matters: it half-closes the socket when stdin ends, which is how the
server learns the conversation is over. Without it, plain `nc` holds its write
side open, the server politely waits for your next turn, and both sides sit
there forever. Don't use `-q 0` instead — that one quits on a timer and can
cut the answer off mid-stream.)

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
tok/s will swing wildly for the same question — 13, then 61, then 134, then 60
tok/s — and that is the GPU, not the server. The first session after load pays
one-time warmup (weight repack and upload, CUDA graph capture). After that the
variance is clock ramping: an idle GPU parks at its floor (210 of 2100 MHz on
the A5000 here), and a one-question session is only ~25–35 forwards — ~0.2 s
of work at full clock — so the whole session can fit inside the ramp, and the
reported rate is just where in the ramp it landed. Run sessions back-to-back
fast enough and some catch the previous session's still-raised clocks, hence
the up-and-down. The numbers in the tables below are steady state: pipeline
several turns into one connection and the rate reaches the table's value by
the second turn (or pin the clocks while measuring with
`nvidia-smi --lock-gpu-clocks`, which needs an elevated shell).

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
exactly that (the math mirrors llama.cpp's `gemma4uv`/`gemma4ua` graphs), and
the resulting embedding rows prefill through the same batched-chunk path as
text — they are just rows the tokenizer didn't make. Media embeddings are
*not* `√n_embd`-scaled; only real token lookups are.

The runner never reads media **files**. The split mirrors the socket design:
file formats — JPEG entropy coding, WAV chunk soup, resampling, resize
filters — belong to an upstream tool, and the model runner handles only what
its own tensors define. `media_cat` (`tools/media_cat.c`) decodes files into
the raw shape the model wants (u8 RGB at 48-multiple dimensions, mono 16 kHz
s16 PCM in whole frames), sends them as typed length-prefixed frames followed
by the question line, and streams the answer:

```
run-cuda-i8r -m gemma-4-12B-it-Q4_K_M.gguf -mm mmproj-F16.gguf -s %TEMP%\lg.sock
media_cat %TEMP%\lg.sock -i photo.jpg "What is in this image?"
media_cat %TEMP%\lg.sock -a clip.wav  "Transcribe this."
media_cat %TEMP%\lg.sock -t "Frame at 0:01: " -i f1.jpg -t " Frame at 0:02: " -i f2.jpg \
          "What changes between the frames?"
```

A turn over the socket is zero or more typed frames — media spans, and `-t`
text that lands between them — then the usual newline-terminated text line;
a text-only client speaks the same protocol it always did. The interleaved
text is what makes a multi-image turn legible to the model: given two bare
images it answers "there is only one image", but with timestamp text between
them it counts both frames and compares them correctly. A video tool emits
exactly that shape. The server wraps each media span in the model's marker tokens, so
the model sees what it was trained on:
`<|turn>user\n<|image>` *(192-ish embedding rows)* `<image|>{text}<turn|>\n<|turn>model\n`.
The runner validates geometry, not content — junk pixels are junk whether or
not a valid JPEG wrapped them, and decoding junk is the tool's job anyway.
At ~2 MB per maximal image, a local socket moves a frame in about a
millisecond; the prefill it triggers costs hundreds of times more.

Text is optional when media frames are sent: a spoken question, or a written
note shown to the camera, is a complete turn by itself —

```
media_cat %TEMP%\lg.sock -a question.wav
```

— and the model answers the *voice*, because past the projection it cannot
tell a sentence that arrived as sound from one that arrived as keystrokes.

The E2B/E4B mmproj files instead carry conventional encoders — the **legacy
path**. Their 16-block vision transformer (`gemma4v`) is implemented: on the
CUDA backends it runs on the GPU (`media-cuda.cu`, ~0.8 s per image on a
desktop card), with the host implementation kept in the binary as numeric
oracle (`LG_MEDIA_VERIFY=1` runs both per image and prints the max
difference) and as the only path on the CPU build. Their 12-block audio
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
run-cuda-i8r -m gemma-4-E4B-it-Q4_K_M.gguf -mtp gemma-e4b-assistant-mtp.gguf -p "..."
```

Each round drafts one token and verifies `[token, draft]` as a single
two-position batch — the weights cross memory once for the pair, so an
accepted draft is nearly a free token (spec blocks deeper than 2 measured as
regressions, so one draft per round it is). Verification is greedy, which
buys the strongest property speculative decoding can have: **the output is
byte-identical to plain greedy decoding, always** — the only things that move
are tokens/s and the acceptance rate the stats line reports. Acceptance is
content-driven: ~100% on counting, ~68–83% on code, ~55–74% on prose — the
harder the next token is to guess, the lower it falls.

Whether MTP *pays* comes down to acceptance × verify cost, and on every target
measured it pays. A B=2 verify costs nearly one decode pass, so a held draft is
almost a free token. Measured in the **socket server** — steady-state, the way
it actually runs (a one-shot `-p` mismeasures it; see the trap below) — both
models speed up on **both** the A5000 and the Jetson:

| model | counting (~100%) | code (~68–83%) | prose (~55–74%) |
|-------|-----------------:|---------------:|----------------:|
| E4B (A5000 / Orin) | 1.58× / 1.50× | 1.39× / 1.26× | 1.23× / 1.16× |
| 12B (A5000 / Orin) | 1.63× / 1.43× | 1.48× / 1.30× | 1.42× / 1.22× |

The 12B's wider draft head (1024 vs E4B's 256) accepts more on hard content, so
it gains the most. **A measurement trap worth recording:** the draft head pays
a one-time **~3.6 s CUDA-graph warmup on its *first* call**, mid-generation — and
a one-shot `run -p` charges all of it to that single short run, which makes MTP
look like a regression. An earlier version of these tables did exactly that and
wrongly concluded *"MTP loses on Windows."* In the server the warmup is paid
once, at the first turn, and amortizes to nothing — so MTP must be benchmarked
in serve mode with the first turn discarded. Greedy verification keeps the
output **byte-identical to plain decoding**, and the split-K verify makes
`-mtp == plain` byte-identical *by construction* — the full story, including the
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
`matmul_q`. Diff `model-cuda-f32.cu` against `model-cuda-i8r.cu` to see exactly
where the speed comes from.

Getting the speed was a journey of two dozen gated steps: two profiling-led
rewrites that *failed* (and earned their write-ups), a CUDA graph against WDDM
launch latency, wide weight loads, a long tail of wins mostly *outside* the
matmul everyone stares at — E2B from 7.9 to 182.5 tok/s — then a long-context
roadmap (online-softmax attention, batched prefill, a kv cache at ~5% of its
old footprint), GPU media encoders, and speculative decoding, which flushed
out an uncached-zero-copy bug whose fix alone bought +49% prefill on desktop
and ~2× on Jetson. Then a tensor-core prefill push — an int8 `mma` chunk
matmul (q4_K then q6_K), wider 32-token chunks, and an L2-aware K/V-sharing
attention — roughly doubled Jetson prefill again, and a **tensor-core flash
attention** for the prompt phase added +44% on top (E4B prefill ~183 tok/s,
12B ~98). The decode side gained a **split-K (FlashDecoding)** attention for
long context, and speculative decoding finally reached the **socket server**
(it had been quietly running plain) — where, on the short structured outputs
the edge target produces, it speeds decode up to ~1.6×. The full log, dead ends
and bisections included, is **[docs/performance-journal.md](docs/performance-journal.md)**.

Where things stand — both sides re-measured the same day on the same machine,
same prompt (little-gemma = `run-cuda-i8r`, 256 generated tokens, warm;
llama.cpp = `llama-bench tg32`). These rows are **plain decode**; MTP is a
further 1.2–1.6× on top of them (the table above):

**RTX A5000 (Windows):**

| model | size | params | little-gemma | llama.cpp CUDA | ratio |
|-------|-----:|-------:|-------------:|---------------:|------:|
| E2B Q3_K_M  | 2.35 GiB |  4.65 B | 182.3 | 148.4 ± 6.6 | **1.23×** |
| E4B Q4_K_M  | 4.95 GiB |  7.52 B | 114.7 | 116.3 ± 1.3 | 0.99× |
| 12B Q4_K_M  | 6.86 GiB | 11.91 B |  60.5 |  64.3 ± 0.3 | 0.94× |

**Jetson Orin NX 16GB (Linux, integrated GPU, zero-copy weights):**

| model | little-gemma | llama.cpp CUDA | ratio |
|-------|-------------:|---------------:|------:|
| E4B Q4_K_M | 16.80 | 13.36 ± 0.04 | **1.26×** |
| 12B Q4_K_M |  8.27 |  7.04 ± 0.04 | **1.17×** |

The pattern is the project's thesis in two tables. The smaller the model, the
more decode speed is about everything *around* the matmul — launch overhead,
sync round-trips, norms, the PLE path — which a few thousand readable lines can
do leanly. The bigger the model, the more it reduces to sustained DRAM
bandwidth through the quantized matmul, where llama.cpp's arch-tuned kernels
still hold a few percent on desktop. On top of plain decode, MTP adds a further
1.2–1.6× — serve-mode, byte-identical output — on **both** the desktop and the
Jetson (the earlier "loses on Windows" reading was a one-shot-`-p` warmup
artifact, corrected above). On the edge device this project actually targets,
every row is ahead, and MTP widens the lead.

Those tables are **decode** — the project's strong axis. **Prefill** (prompt
processing) is the honest weak axis: it is weight-bandwidth-bound, and
llama.cpp prefills at batch ~512 through arch-tuned tensor-core GEMMs that no
few-thousand-line kernel matches. The push above — the int8 `mma` chunk matmul,
wider chunks, K/V sharing, the bank-conflict pads, and then a tensor-core flash
attention for the prompt phase — closed most of the gap on the Jetson: E4B
~55 → **~183** prompt tok/s, 12B ~21 → **~98** (a ~2,000-token prompt), moving
the ratio to llama.cpp from ~10× behind to **~2.6×**. It is still their axis,
and the wins are gated to the same output up to a late greedy tie. It rarely
shows in interactive serving — turns are short, `-sys` removes the skills
re-prefill, the GPU encoder removed the image one — but on long documents
llama.cpp still wins the wait.

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
| src       | 15    | 5,807 |
| include   | 6     |   254 |

~6,060 lines of code in the repository (the original 3,000 exploring ceiling was
retired when the sandbox phase began; `tools/` and the vendored
`vendor/stb_image.h` — compiled only into `media_cat` — are not counted). The
backends are mutually exclusive, so no single program is anywhere near that.
Each binary is the shared pipeline (GGUF parse, dequant, tokenizer, config,
multimodal embedders, the MTP draft head, CLI + socket server — 2,657 lines)
plus exactly one backend:

| binary        | backend on top of the shared 2,657                       | code lines |
|---------------|----------------------------------------------------------|-----------:|
| `run`         | `model-cpu.c`                                            |      3,025 |
| `run-cuda`    | `model-cuda.cuh` + `model-cuda-f32.cu` + `media-cuda.cu` |      4,645 |
| `run-cuda-i8r`| `model-cuda.cuh` + `model-cuda-i8r.cu` + `media-cuda.cu` |      5,405 |

(`graph.c`/`graph.h`, the teaching tensor/graph layer, are exercised by `graph_test`
only.) So the program that out-decodes llama.cpp CUDA on the Jetson on every
model it runs — multi-turn socket serving, batched prefill, a ring-buffered
f16 KV cache, tensor-core flash-attention prefill, split-K decode, image and
audio understanding, a GPU vision encoder, an own m16n8k32 tensor-core q4_K
prefill kernel, and byte-identical speculative decoding included — is **about
5,400 lines of C end to end**, tokenizer and all, with no vendored dependency.

## Validation

Built against a CPU `llama.cpp` as an oracle (`llama-eval-callback` dumps every
intermediate tensor): dequantization is bit-exact vs the `gguf` Python package;
the forward pass matches an independent NumPy f32 reference and llama.cpp's logits
(within the f32-vs-quantized-matmul gap); the tokenizer matches `llama-tokenize`
exactly. `test/graph_test.c` (a CTest target) checks the graph kernels.

## License

MIT — see [LICENSE](LICENSE).
