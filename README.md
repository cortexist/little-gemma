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
                      ┌──────────────┐
                      │    run.c     │   CLI: -m <model> [-p "prompt"]
                      └──────┬───────┘
         ╭───────────────────┼───────────────────────────────╮
         ▼                   ▼                               ▼
 ┌───────────────┐   ┌────────────────┐   ┌──────────────────────────────────────┐
 │  tokenizer.c  │   │    model.c     │   │               gguf.c                 │
 │  BPE text↔ids │   │ config+forward │   │ parse → ctx (hdr, kv, tensors, data) │
 └───────────────┘   └───────┬────────┘   └──────────────────┬───────────────────┘
                             │                               │  
                             ▼                               │
                ┌──────────────────────────┐                 │ borrows
                │          quant.c         │◄────────────────╯ quantized
                │ dequantize q3_K/q4_K/... │  weights
                └──────────────────────────┘

    graph.c — a minimal tensor/graph layer (matmul, rmsnorm, softmax, …)
            kept as the "what a compute graph is" teaching reference.
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
run -m <model.gguf> [-p "prompt" | -s <socket>]
run -c <socket>
```

- `-m` only → prints the GGUF dump + config, then exits.
- `-m` + `-p` → one-shot demo: tokenizes the prompt, generates, reports tok/s.

```
> run -m model.gguf -p "The capital of France is"
The capital of France is Paris.
prompt: 6 tokens in 4.68s (1.28 tok/s)
gen:    2 tokens in 1.48s (1.35 tok/s)
```

### Serving over a Unix-domain socket (`-s`)

`-s` turns the runner into a tiny conversation server — no HTTP, no JSON, no web
security surface; transport concerns (TCP exposure, TLS, auth) belong to `socat`
and to whatever chat server or agent sits downstream:

```
run-cuda-i8r -m model.gguf -s /tmp/lg.sock        # Ctrl-C to stop
echo "What is the capital of France?" | socat - UNIX-CONNECT:/tmp/lg.sock
```

The protocol is the simplest thing that works, designed so raw media frames can
join it later without breaking anything:

- **A connection is a conversation.** The kv cache lives for the connection, so
  multi-turn context is free; close the connection (or just stop typing into
  socat) to end the session. One conversation is served at a time — decode is
  batch-1, and the listen backlog is the queue.
- **A line is a user turn.** Each newline-terminated line is wrapped in the
  Gemma chat template and prefilled.
- **The reply is the raw token stream**, special tokens included — the thinking
  channel, the markers, and the closing `<turn|>` that downstream tools can
  split turns on. The server filters nothing; presentation is the client's job.
- stdout/stderr are logging only; per-turn stats go to stderr.
- A clean half-close (e.g. socat after stdin EOF) still receives the full turn
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

## On SIMD (AVX2) — intentionally not implemented

The CPU matmul is plain scalar C parallelized with OpenMP; there are **no hand-written
AVX2/FMA intrinsics**. This is deliberate. Hand-vectorizing the CPU kernels would
be throwaway work, because the real target is **CUDA**: on the GPU the parallelism
comes from thousands of threads (replacing OpenMP) and the per-element math runs in
kernels (replacing what SIMD would do). The weights are already stored quantized
and unpacked inside the matmul — exactly the shape a GPU kernel wants — so the
CPU kernels in `model-cpu.c` (`matmul_q`, `rmsnorm`, `rope_neox`, `softmax`, `gelu`)
double as the reference spec for the CUDA versions in `model-cuda-f32.cu`.

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
roadmap: online-softmax attention (the ~12k context cap gone), batched prefill
(3.4× prompt speed), and a kv cache at ~5% of its old footprint. The full log,
dead ends and bisections included, is
**[docs/performance-journal.md](docs/performance-journal.md)**.
Where things stand, same machine, same day, both sides re-measured
(little-gemma = `run-cuda-i8r`, ~166–187 generated tokens; llama.cpp =
`llama-bench tg32`):

| model | size | params | little-gemma | llama.cpp CUDA | ratio |
|-------|-----:|-------:|-------------:|---------------:|------:|
| E2B Q3_K_M  | 2.35 GiB |  4.65 B | 182.5 | 145.0 ± 5.2 | **1.26×** |
| E4B Q4_K_M  | 4.95 GiB |  7.52 B | 114.4 | 112.9 ± 1.8 | **1.01×** |
| 12B Q4_K_M  | 6.86 GiB | 11.91 B |  60.4 |  63.5 ± 0.6 | 0.95× |

The pattern is the project's thesis in one table. The smaller the model, the more
decode speed is about everything *around* the matmul — launch overhead, sync
round-trips, norms, the PLE path — which 2,000 readable lines can do leanly. The
bigger the model, the more it reduces to one number: sustained DRAM bandwidth
through the quantized matmul, where llama.cpp's arch-tuned kernels still hold a
few percent. The crossover for this codebase currently sits right around E4B.

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

| directory | files | code  | comment | blank | total |
|-----------|-------|-------|---------|-------|-------|
| src       | 10    | 2,910 | 415     | 346   | 3,671 |
| include   | 5     | 213   | 105     | 59    | 377   |

3,123 lines of code in the repository — the self-imposed exploring ceiling of
3,000 was crossed by the long-context roadmap (online softmax, batched prefill,
the KV ring, f16 KV). The backends are mutually exclusive, so no single program
is anywhere near that. Each binary is the shared pipeline (GGUF parse, dequant,
tokenizer, config, CLI + socket server — 1,517 lines) plus exactly one backend:

| binary        | backend on top of the shared 1,517        | code lines |
|---------------|-------------------------------------------|-----------:|
| `run`         | `model-cpu.c`                             |      1,815 |
| `run-cuda`    | `model-cuda.cuh` + `model-cuda-f32.cu`    |      2,294 |
| `run-cuda-i8r`| `model-cuda.cuh` + `model-cuda-i8r.cu`    |      2,498 |

(`graph.c`/`graph.h`, the teaching tensor/graph layer, are exercised by `graph_test`
only.) So the program that decodes E2B 26% faster than llama.cpp CUDA — multi-turn
socket serving, 3.4× batched prefill, and a ring-buffered f16 KV cache included —
is **2,498 lines of C end to end**, tokenizer and all.

## Validation

Built against a CPU `llama.cpp` as an oracle (`llama-eval-callback` dumps every
intermediate tensor): dequantization is bit-exact vs the `gguf` Python package;
the forward pass matches an independent NumPy f32 reference and llama.cpp's logits
(within the f32-vs-quantized-matmul gap); the tokenizer matches `llama-tokenize`
exactly. `test/graph_test.c` (a CTest target) checks the graph kernels.

## License

MIT — see [LICENSE](LICENSE).
