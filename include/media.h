#ifndef MEDIA_H
#define MEDIA_H

#include <stdint.h>

// Multimodal embedders ("mmproj"): turn RAW media data into rows of LLM
// embeddings, ready to prefill as if they were token embeddings (except they
// are NOT sqrt(n_embd)-scaled — the model scales only real token lookups).
//
// little-gemma does not read media FILES. Decoding, resizing and resampling
// belong to an upstream tool (mmcat, in little-gemma-tools) that delivers model-ready
// raw data over the socket: u8 RGB with both sides multiples of the patch
// size, and mono 16 kHz s16 PCM padded to whole frames. What stays here is
// exactly what the mmproj tensors define — the patch layout, the
// normalization, and the projector math.
//
// This implements the gemma-4-12B "unified" design, where there is no encoder
// at all: vision is one linear layer over 48x48 pixel patches (plus norms and
// a learned 2-axis position table), audio is one linear layer over raw 640-
// sample (40 ms @ 16 kHz) waveform frames. The whole projector is 11 tensors.
// The E2B/E4B mmproj files carry a legacy 16-block vision transformer (also
// implemented — media.c + media-kernel.cu) and a 12-block audio conformer,
// which is not: legacy audio is rejected (use Whisper upstream, or the 12B).

struct media;

// Load an mmproj GGUF. n_embd is the LLM's embedding width; the projector's
// output width must match. Returns NULL (with a message on stderr) on a
// legacy encoder-stack mmproj or any mismatch.
struct media *media_open(const char *path, int n_embd);
void media_free(struct media *md);

// The geometry the raw data must arrive in (derived from the tensors).
int media_patch(const struct media *md);    // pixels per patch side (48)
int media_frame(const struct media *md);    // samples per audio frame (640)

// The token text that wraps a media span in the prompt (gemma4 vocabulary).
#define MEDIA_IMG_BEG "<|image>"
#define MEDIA_IMG_END "<image|>"
#define MEDIA_AUD_BEG "<|audio>"
#define MEDIA_AUD_END "<audio|>"

// Embed raw media: returns a malloc'd [n_tokens][n_embd] row matrix (caller
// frees) and the token count, or NULL if the geometry is wrong.
//  image: u8 RGB, w and h multiples of media_patch() (one token per patch)
//  audio: mono 16 kHz s16 PCM, n_samples a multiple of media_frame()
//         (one token per frame; the tool zero-pads the tail)
float *media_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens);
float *media_embed_audio(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens);

// ---- the socket media frame -------------------------------------------------
// A user turn over the socket may start with typed frames, each:
//   u8 MEDIA_FRAME_MAGIC, u8 type ('I' | 'A' | 'T'), u16 w, u16 h, u32 len,
//   payload
// (little-endian; w,h are used by 'I' and zero otherwise; len is the payload
// byte count: w*h*3 for u8 RGB, 2*n_samples for s16 PCM, UTF-8 bytes for 'T').
// 'T' carries text that lands BETWEEN media spans — what makes a video turn
// possible: "0:01" frame "0:02" frame ... The newline-terminated text line
// that follows the frames closes the turn, exactly as before — a turn that
// starts with a printable byte is plain text, so the old protocol is a strict
// subset of this one.
#define MEDIA_FRAME_MAGIC  0x01
#define MEDIA_FRAME_HDR    10
#define MEDIA_FRAME_IMAGE  'I'
#define MEDIA_FRAME_AUDIO  'A'
#define MEDIA_FRAME_TEXT   'T'

// Barge-in: one bare byte (no header) a client may send WHILE a reply streams.
// The server stops decoding at the next token, closes the turn on the wire
// with "<turn|>", and the session continues — the context keeps the cut-off
// reply, so the model knows what it did and didn't get to say. A voice client
// sends this the moment its user starts talking over the answer; whether the
// next turn mentions the interruption is the client's choice (a text note).
#define MEDIA_BARGE        0x02

// Listener probe (duplex prototype): a 'P' frame sent while the user turn is
// still OPEN asks "if the turn ended right here, what would you say?" — the
// engine dry-runs the turn close (payload = a steering suffix appended inside
// the user turn first; w = max tokens to decode, 0 for the default), replies
// mid-turn with "<|probe>text<probe|>\n", and rolls the context back, so a
// probed conversation is byte-identical to an unprobed one. This is what lets
// a voice client arbitrate duplex behavior — nod, backchannel, take the turn —
// with the model's own judgment while its user is still speaking. Send it only
// while listening (never with a reply pending); payload must be >= 1 byte
// (send "\n" for a bare end-point peek).
#define MEDIA_FRAME_PROBE  'P'

#endif // MEDIA_H
