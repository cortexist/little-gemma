# Serving over a Unix-domain socket (`-s`)

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

## CLI reference

```
run -m <model.gguf> [-mm <mmproj.gguf>] [-mtp <assistant.gguf>] [-p "prompt" | -s <socket>]
run -c <socket>
```

- `-m` only → prints the GGUF dump + config, then exits.
- `-m` + `-p` → one-shot demo: tokenizes the prompt, generates, reports tok/s.
  Text only — media arrives over the socket; see [multimodal.md](multimodal.md).
- `-mm` → also load a multimodal projector, so `-s` sessions accept image and
  audio frames.
- `-mtp` → also load a gemma4-assistant draft head and decode speculatively;
  see [mtp.md](mtp.md). Output is identical, only speed changes.
- `-sys <file>` → prefill the file as a system turn ONCE at server start;
  every `-s` session then begins with it already in context — the
  skills-in-context pattern without the TTFT tax. The saved cache rows are
  restored at session start, so each conversation begins byte-identically
  fresh no matter what a previous session did to the rings.
- `-think N` → bound the reasoning channel. `0` = none, `N` = up to N tokens,
  omitted = unlimited (default). See [Controlling the reasoning
  channel](#controlling-the-reasoning-channel).

## Controlling the reasoning channel (`-think`)

Gemma 4's thinking models emit a reasoning channel:
`<|channel>thought[reasoning]<channel|>[answer]`. The **12B always emits it**
(wrapping even "The capital of France is Paris"); the **E4B never does** (plain
answer). The client hides the channel from display, but it still occupies the
KV cache and — when the model fills the reasoning span — costs TTFS before the
first speakable word.

**Prompt-level control of thinking does not work.** Measured on the 12B
`Q4_K_M`: enabling `<|think|>` in the system turn, disabling it, and instructing
"reply directly, no thinking channel" **all produce byte-identical output** —
the model ignores the request. (This matches the general llama.cpp experience
that turning thinking off by prompt is unreliable.) So `-think` controls it
**structurally** instead, independent of whether the model obeys:

- **`-think 0`** — seed an empty `<|channel>thought\n<channel|>` onto the model
  turn. The model cannot open a channel it has already been handed closed, so
  its first generated token is the answer: no reasoning, best TTFS, no channel
  in the stream at all. This is the reliable off-switch.
- **`-think N`** (N > 0) — let the model reason, but force-inject `<channel|>`
  once the reasoning span reaches N tokens: *a little* thinking, then the
  answer. The dial for the TTFS-vs-reasoning-depth tradeoff.
- **omitted** — unbounded; the model's native behavior, byte-unchanged.

The seed is **gated on having seen the model use the channel this session**, so
on a model that answers plainly (E4B) `-think` is a clean no-op — it is never
handed a closed channel it wasn't going to open (which would make it emit a
stray marker). On the 12B the first turn is the model's native (client-hidden
empty) thought; every turn after is seeded clean.

> **Note for a future reasoning-capable checkpoint** (e.g. the T2000 era). The
> shipped 12B `Q4_K_M` emits an *empty* reasoning span — it puts its
> step-by-step in the *answer*, after `<channel|>` — so `-think N>0`'s force-cap
> has nothing to cap here and is a no-op; only `-think 0` changes anything. The
> cap mechanism is implemented and sound, but **re-verify it against a
> checkpoint that actually fills the reasoning span** before relying on the
> budget. Don't rediscover this: prompt control is inert, the span is empty on
> this build, and `-think` is the structural handle.

## The protocol

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
- Decoding is greedy by default — the byte-identity property every gate in
  this repo leans on. `-temp T` samples instead: temperature over the top-k,
  truncated at top-p, with `-topk`/`-topp` defaulting to the model's own
  recommendation shipped in the gguf (`general.sampling.*` — Gemma 4 says
  top-k 64, top-p 0.95) and `-seed N` for reproducible runs (the chosen seed
  prints to stderr either way). Under MTP the verify samples the target's
  distribution at each position and accepts a draft only when the sample
  agrees, so speculation still changes only tokens/s, never the distribution
  (acceptance drops accordingly; greedy remains the benchmarked config).
- A turn is capped at 1,024 output tokens (`SERVE_GEN`): greedy decoding has
  no repetition penalty, so a degenerate loop would otherwise spin until the
  context filled (observed: 8,098 tokens of the same sentence). A capped turn
  ends with a visible `[SERVE_GEN cap]<turn|>`.
- A clean half-close (e.g. `nc -N` after stdin EOF) still receives the full turn
  in flight; only a hard reset aborts generation early — checked between
  tokens, so one ~5–17 ms forward is the most work a dead client can waste.

## Benchmarking etiquette

**Don't read benchmark numbers out of one-question sessions.** The per-turn
tok/s swings wildly for the same question — 13, then 61, then 134 — and that
is the GPU, not the server: an idle GPU parks at its clock floor (210 of
1695 MHz on the A5000 here), and a one-question session is only ~25–35
forwards, so the whole session fits inside the clock ramp. (The one-time
weight repack/upload is no longer part of this: the server absorbs it in a
2-token warmup before it starts listening, so even the first turn runs warm.)
Steady state means pipelining several turns into one connection, or pinning
the clocks; the project's canonical methodology is in
[benchmarks.md](benchmarks.md).

## Windows clients

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
