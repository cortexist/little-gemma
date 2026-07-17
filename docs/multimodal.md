# Multimodal

The gemma-4-12B takes images and audio with **no encoder at all** — its mmproj
file is 11 tensors (167 MB): vision is one linear layer over 48×48-pixel
patches plus three norms and a learned 2-axis position table; audio is one
linear layer over raw 16 kHz waveform sliced into 640-sample (40 ms) frames.
No vision transformer, no conformer, no mel spectrogram. `media.c` implements
exactly that (the math mirrors llama.cpp's `gemma4uv`/`gemma4ua` graphs), the
CUDA builds run it on tensor cores (`media-kernel.cu` — an image is two GEMM
launches and change, an audio clip's frame-at-a-time matvec loop becomes one
batched GEMM; the GPU rows are bit-identical across devices, and the host
path stays in as oracle and fallback), and the resulting embedding rows
prefill through the same batched-chunk path as text — they are just rows the
tokenizer didn't make. Media embeddings are *not* `√n_embd`-scaled; only
real token lookups are.

## Media files stay upstream: `mmcat`

The runner never reads media **files**. The split mirrors the socket design:
file formats — JPEG entropy coding, WAV chunk soup, resampling, resize
filters — belong to an upstream tool, and the model runner handles only what
its own tensors define. That tool is **`mmcat`**, in the sibling repo
[`little-gemma-tools`](../../little-gemma-tools): it auto-detects each file with
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

## Turn shape

A turn over the socket is zero or more typed frames — media spans, and text
frames that land between them — then the usual newline-terminated text line;
a text-only client speaks the same protocol it always did. The interleaved
text is what makes a multi-image turn legible to the model: given two bare
images it answers "there is only one image", but with timestamp text between
them it counts both frames and compares them correctly — a video tool emits
exactly that shape. The server wraps each media span in the model's marker
tokens, so the model sees what it was trained on:
`<|turn>user\n<|image>` *(192-ish embedding rows)* `<image|>{text}<turn|>\n<|turn>model\n`.

Text does too — the turn is appendable, and it **prefills while it is still
being dictated**: streamed `'T'` text (voicecat's confirmed transcript, or
any client's) flushes into the kv cache in word-boundary chunks as it
accumulates, which is exact because SentencePiece pieces never cross a
space — the split token stream is byte-identical to tokenizing the whole
turn at once (verified: identical replies, identical id counts). A spoken
929-token instruction on the Orin 12B answers **0.55 s** after the last
word instead of 5.4 s; the prefill didn't get faster, it got *hidden under
the speaking*.

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

## The E2B/E4B legacy encoders

The E2B/E4B mmproj files instead carry conventional encoders — the **legacy
path**. Their 16-block vision transformer (`gemma4v`) is implemented: on the
CUDA backends it runs on the GPU on tensor cores (`media-kernel.cu`, m16n8k16
with f16 inputs and f32 accumulation — ~0.2 s per 130-token frame on the
Orin, under 0.1 s on a desktop card), with the host implementation kept in
the binary as numeric oracle (`LG_MEDIA_VERIFY=1` runs both per image and
prints the max difference; ~6e-3, which is the f16 input rounding) and as
the only path on the CPU build. Their 12-block audio
conformer was implemented, verified against the reference, and **dropped**:
it underperforms in practice, and production pipelines pair these models
with a dedicated STT (Whisper) anyway — audio frames at a legacy mmproj say
so and point at the 12B, which hears. The conformer lives in git history.
