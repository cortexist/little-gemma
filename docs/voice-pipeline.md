# The voice pipeline — full commands and reproduction guide

Everything behind the "last word → first audio in 0.65 s" result, as
runnable commands. Three processes and two pipes: **voicecat** (VAD +
streaming whisper, committed words become `'T'` frames in the open turn)
→ **little-gemma serve** (the LLM; prefills while you speak, replies as
a raw token stream) → **clause_pipe** (strips scaffolding, emits one
line per clause) → **piper `--stream`** (first PCM after one decoder
chunk) → **aplay**.

## Prerequisites

| piece | source | note |
|---|---|---|
| Jetson Orin NX 16GB | JetPack 6.2.1, NVMe | the reference board; any CUDA GPU works |
| little-gemma | this repo | `cmake -B build && cmake --build build -j` → `build/run-cuda-i8` |
| voicecat | sibling repo `little-gemma-tools` | needs `ffmpeg` for mic capture |
| whisper.cpp | upstream, CUDA build | model `ggml-base.en.bin` |
| piper | our fork `cortexist/piper1-gpl`, branch `vits-streaming` | `pip install -e .` |
| LLM weights | `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` — Unsloth's GGUF of Google's QAT release [TODO: pin HF link] | 12B/E4B: `Q4_K_M` ggufs |
| MTP draft head | `mtp-gemma-4-E2B-it.gguf` [TODO: document conversion provenance] | optional but measured with it |
| TTS voice | any piper voice, e.g. `en_US-kristin-medium` (HF `rhasspy/piper-voices`) | our Orin numbers used a private finetune of the same size |

One-time voice preparation (writes `<voice>.enc.onnx` / `<voice>.dec.onnx`):

```bash
python3 -m piper.split ~/voices/en_US-kristin-medium.onnx
```

## The pipeline

```bash
# 1. the brain — serve mode, voice system prompt prefilled once at start
./build/run-cuda-i8 \
    -m   gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
    -mtp mtp-gemma-4-E2B-it.gguf \
    -sys docs/voice-sys.txt \
    -s   /tmp/lg.sock &

# 2. ears | glue | mouth — one shell pipeline, mic to speaker
voicecat /tmp/lg.sock --rt \
    --whisper-bin   ~/repos/whisper.cpp/build/bin/whisper-cli \
    --whisper-model ~/repos/whisper.cpp/models/ggml-base.en.bin \
  | python3 bench/clause_pipe.py \
  | piper -m ~/voices/en_US-kristin-medium.onnx --output-raw --stream \
  | aplay -r 22050 -f S16_LE -t raw -c 1 -
```

Variants:

- **Model tier**: swap `-m`/`-mtp` for the E4B or 12B pair — nothing else
  changes.
- **Camera**: add `-mm <mmproj>` to the server and feed frames as `'I'`
  frames (see README serve protocol). Note the measured A/V exclusion:
  with vision in context the 12B stops hearing native audio — speech
  must ride as whisper transcripts (voicecat's default when a whisper
  model is given).
- **Barge-in**: `--barge-note "<text>"` makes voicecat interrupt the
  reply when the user speaks over it.
- **No `aplay`** (e.g. piping to a network sink): the stream is mono
  s16le at the voice's sample rate (22050 Hz for medium voices).

## Measuring it (bench/)

Every number in the journals traces to one of these. Common discipline:
client-side clocks only, first turn after model load discarded, fresh
conversation per measurement, kernels benched under pinned clocks
(`sudo jetson_clocks --store f; sudo jetson_clocks; ...; sudo
jetson_clocks --restore f`), llama.cpp ratios only from same-day pairs.

| harness | measures | protocol |
|---|---|---|
| `ttft_dictate.py` (POSIX) | TTFT/TTFS under streamed dictation | `ttft_dictate.py /tmp/lg.sock line929s.txt <words/frame> <frames/s>` — `0 0` = deferred, `6 5` = paced 30 words/s |
| `ttfc_dictate.c` (Windows, no AF_UNIX in CPython) | + first-clause clock | `cl ttfc_dictate.c ws2_32.lib winmm.lib`, same args |
| `line929s.txt` | the standard 929-token spoken instruction | 918 filler words + a question |
| `tts_time.py` / `tts_stream_time.py` | TTS first byte, stock vs streaming | persistent service, clocked through the pipe (in-process clocks omit espeak) |
| `piper_mem.py` / `mempeak.sh` | peak memory, the Jetson way | smaps_rollup Anonymous + nvmap — never VmHWM/MemAvailable |
| `bench-serve-orin.sh` / `bench-serve.ps1` | serve-mode prefill tok/s | N identical turns, one connection each, first discarded |
| `clause_pipe.py` | (glue, not a probe) | also the harness for the end-to-end single-run measurement |

Reply byte-identity across delivery modes is the correctness gate: if a
streamed run's reply differs from the deferred run's, the measurement
does not count.

## Honest gaps

- The 0.65 s headline is composed from same-box measured legs (first
  clause 0.549 s + streaming TTS first PCM 0.10 s); the single-command
  pipeline above is the harness for verifying it as one system — that
  run is still to be done (whisper's final-commit adds ~0.3-0.5 s in
  live-mic operation until the persistent-whisper upgrade lands).
- MTP draft-head gguf provenance/conversion steps are not yet written.
- Orin voice-loop numbers used a private voice finetune; kristin-medium
  is the public stand-in (same architecture and size class).
