# The voice pipeline — full commands and reproduction guide

Everything behind the "last word → first audio in 0.65 s" result, as
runnable commands. Three processes and two pipes: **voicecat** (VAD +
streaming whisper, committed words become `'T'` frames in the open turn)
→ **little-gemma serve** (the LLM; prefills while you speak, replies as
a raw token stream) → **clausecat** (strips scaffolding, emits one
line per clause) → **piper `--stream`** (first PCM after one decoder
chunk) → **aplay**.

## Prerequisites

| piece | source | note |
|---|---|---|
| Jetson Orin NX 16GB | JetPack 6.2.1, NVMe | the reference board; any CUDA GPU works |
| little-gemma | this repo | `cmake -B build && cmake --build build -j` → `build/run-cuda-i8` |
| voicecat, clausecat | sibling repo `little-gemma-tools` | voicecat needs `ffmpeg` for mic capture |
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
voicecat /tmp/lg.sock \
    --whisper-bin   ~/repos/whisper.cpp/build/bin/whisper-cli \
    --whisper-model ~/repos/whisper.cpp/models/ggml-base.en.bin \
  | clausecat \
  | piper -m ~/voices/en_US-kristin-medium.onnx --output-raw --stream \
  | aplay -r 22050 -f S16_LE -t raw -c 1 -
```

To replay a recording as if it were live (for reproducible runs), swap
the mic for `--stdin-pcm --realtime` and pipe mono 16 kHz s16 PCM in;
`--realtime` deadline-paces the file at wall-clock rate so ASR passes overlap
"capture" exactly as they would against a real mic. (A live mic needs no
pacing flag — the hardware paces itself.)

Variants:

- **Model tier**: swap `-m`/`-mtp` for the E4B or 12B pair — nothing else
  changes.
- **Variety**: add `-temp 1` to the server for sampled replies (top-k/top-p
  default to the model's own recommendation in the gguf; `-seed N` makes a
  run reproducible). All published numbers are greedy — the default.
- **Camera**: add `-mm <mmproj>` to the server and feed frames as `'I'`
  frames (see README serve protocol). Note the measured A/V exclusion:
  with vision in context the 12B stops hearing native audio — speech
  must ride as whisper transcripts (voicecat's default when a whisper
  model is given).
- **Barge-in**: `--barge-note "<text>"` makes voicecat interrupt the
  reply when the user speaks over it.
- **No `aplay`** (e.g. piping to a network sink): the stream is mono
  s16le at the voice's sample rate (22050 Hz for medium voices).

## Speakerphone mode — open-air mic + speaker on one card

The pipeline above assumes the mic doesn't hear the speakers (headset, or
AEC upstream). For an open-air rig — one box, its own mic and speaker,
barge-in included — the AEC has to be real. Findings and wiring, measured
on a ReSpeaker Lite (Seeed USB fw v2.0.5) on the Orin:

- **The Lite's onboard XMOS AEC does not cancel its own USB playback.**
  Speech played out its speaker comes back through its mic at −15 dB RMS
  and whisper transcribes it near-verbatim. The Seeed forum reports the
  same through fw v2.0.7. Its two capture channels are bit-identical (no
  raw/processed split).
- **Host-side AEC cannot work on the stock capture.** The firmware's
  NS/AGC applies time-varying gain that decorrelates the echo from the
  playback reference before the host ever sees it: speex gets ~2–7 dB
  ERLE, webrtc under 1 dB — nothing downstream can cancel what that DSP
  has scrambled (a steady tone cancels at 29.7 dB; speech doesn't —
  that's the signature).
- **The fix that ships: Seeed's v2.0.6 debug firmware +
  `far-field-service`.** The reflash
  (`sudo dfu-util -R -e -a 1 -D ffva_ua_v2.0.6_output_proc0_ref0.bin`,
  UPGRADE partition — the untouched FACTORY image is the fallback, so the
  risk is small) exports a **true raw mic on channel 1** (verified
  tone-flat). The cancelling then runs in **far-field-service**
  (little-gemma-tools), the audio authority: ONE process owns both
  directions of the card over an AF_UNIX socket, so webrtc AEC3 gets a
  sample-aligned in-process reference — the browser's trick, native. On
  this deaf channel (~30 dB low) the linear filter learns little, but
  AEC3's output gate holds the TTS residual near the noise floor
  (clean tap −78 dB during full-volume playback, whisper-blank):
  ghost-free listening while the mouth speaks; barging wants a firm
  voice. Full design, wire, build notes (webrtc-audio-processing 1.x
  from source — jammy's 0.3.1 predates AEC3) and the paid-for real-time
  rules: **`docs/far-field-service.md` in little-gemma-tools**. (An
  earlier PipeWire-module route, `respeaker-aec.sh`, is kept only as a
  reference — it managed ~13 dB where the in-process gate reaches the
  floor.)
- **Measurement traps** (each cost us a wrong conclusion once):
  PipeWire's echo-cancel source is born at 33% volume — a near-silent
  mic that looks *exactly* like perfect AEC (control-test near-end
  speech before believing any AEC result); the XMOS noise suppressor
  eats steady tones, which looks like AEC if you test with a sweep
  (test with speech); nobody touches the board mid-test (handling it
  changes the echo path and couples vibration into the mics); a
  whisper-transcribable residual is not a failed gate — the gate is
  the VAD threshold, since voicecat transcribes only after VAD opens;
  and a replug renumbers ALSA cards (address the board as `hw:Lite`).

One command, cold to conversation (`script/speakerphone.sh` in
little-gemma-tools). The block below is the reference-box invocation,
verbatim; adjust paths to your checkout (any piper voice works — run
`python3 -m piper.split` on it once so the `.enc`/`.dec` pair sits
beside the `.onnx`):

```bash
export LG_BIN=$HOME/repos/cortexist/lg-duplex/build/run-cuda-i8
export LG_MODEL=$HOME/repos/cortexist/llama.cpp/.scratch/gemma-4-e2b/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
export LG_MTP=$HOME/repos/cortexist/llama.cpp/.scratch/gemma-4-e2b/mtp-gemma-4-E2B-it.gguf
export LG_VOICE=$HOME/repos/piper/local/model/en_US-ozgirl_v6-step18500.onnx
export PIPER=$HOME/repos/piper/.venv/bin/piper
export LG_WHISPER_BIN=$HOME/repos/whisper.cpp/build/bin/whisper-cli
export LG_WHISPER_MODEL=$HOME/repos/whisper.cpp/models/ggml-base.en.bin
sh ~/repos/cortexist/lgt-duplex/script/speakerphone.sh /tmp/lg.sock
```

`speakerphone.sh` brings up **far-field-service** via `far-field.sh
start` (the audio service is persistent — ctrl-c ends the conversation,
the card keeps its owner; `far-field.sh stop` is the off switch), starts
the LLM server only when `/tmp/lg.sock` is absent and a `whisper-server`
when none answers on :8642 (those two ARE torn down on ctrl-c if it
started them), and picks the TTS sample rate up from the voice's
sidecar json.

Two voicecat features carry this mode (both new):

- `--whisper-url` — ASR passes POST to the resident whisper-server
  instead of spawning whisper-cli: ~0.36 s/pass vs 0.55 (whisper's
  30 s-padded encoder is the floor, not process spawn — a streaming
  ASR model behind the same URL is the next step down), so
  `--commit-ms 1100` streams commits mid-utterance.
- `--mouth-synth` / `--mouth-play` — voicecat OWNS the TTS: clause
  splitting runs in-process (clausecat's policy verbatim), piper is
  spawned once and stays warm (cold start is ~4 s on the Orin),
  speaking mux frames; the player is a `far-field-service --speak`
  client, and a barge simply KILLS it — the service treats a mouth's
  hangup as the cut and flushes everything it queued, so the sound
  stops within the sink's ~100 ms tail. A `set_voice` sentinel down
  piper's stdin (its `M`-frame echo) marks the exact stale/live
  boundary, and an audible-horizon clock (bytes handed to the player
  over the stream rate) keeps "the mouth is speaking" true for the
  whole utterance even though the synth finishes seconds early — the
  in-reply barge bar (`--barge-mult`) rides that state. Verified: of
  four replies in flight around a barge, the played audio contained
  only the post-barge one.

The assembled loop is:

```
far-field-service --tap /tmp/ff.sock          # clean 16 kHz mic
  | voicecat /tmp/lg.sock --stdin-pcm ...     # VAD, whisper, turns, barge
      --mouth-synth "piper ... --output-mux --stream"
      --mouth-play  "far-field-service --speak /tmp/ff.sock --rate 22050"
```

verified end-to-end on the Orin (E2B QAT + MTP): multi-turn open-air
conversation, clean TTS, barge cutting mid-word with the queued backlog
dying at the cut — on the debug firmware above, ahead of the XVF3800
4-mic array this wiring was built to receive.

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
| `clause_pipe.py` | (policy sandbox, not a probe) | python twin of clausecat (little-gemma-tools) — try split-policy changes here first; the two are kept byte-identical by differential feed |

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
