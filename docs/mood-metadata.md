# Mood metadata through the voice pipeline — design note

*Status: designed, deferred. 2026-07-04. The plumbing is a ~day of work; the
gating item is a finetune. Written down now because humanoids and visual
assistants are drawing attention fast, and this requirement may arrive
sooner than planned.*

## The requirement

The LLM annotates its own speech with expression/mood. The voice changes
with it — a multi-speaker piper voice carrying happy/sad/neutral speakers
switches per clause — and the mood tag travels the whole pipe alongside the
phoneme timings, so a face rig or a mechanical actuator can pose the body
to match the voice, in time with it.

```
LLM (mood tags in the reply stream)
  → clausecat            parses the tag, attaches it to the clause
  → piper                selects the speaker, echoes the tag into the mux
  → demux / browser      mood events beside viseme events, same clock
  → animation, actuators
```

## What already works, by earlier design decisions

- **The runner needs no changes, ever.** serve emits the raw token stream
  and filters nothing — presentation is the client's job. Whatever markers
  the model emits arrive intact at clausecat.
- **Multi-speaker voices already flow through the streaming split.**
  `piper.split` passes all graph inputs to the encoder half, so `sid`
  survives, and `synthesize_stream()` feeds it per utterance. A
  happy/sad/neutral multi-speaker onnx works with chunked decode today.
  Durations are speaker-conditioned, but the schedule is computed after the
  speaker is chosen, so phoneme timings stay correct per mood.
- **The mux framing is extensible on purpose.** `[kind][u32 len][payload]`
  — every consumer (piper.demux, the voicedemo page) skips unknown kinds,
  so a new frame type is purely additive; old consumers keep working.
- **The demo mouth is `currentColor` line art** — tinting the face by mood
  is a CSS one-liner.

## The two doors to build

**1. The clause policy becomes parse-don't-strip.** Today `speakable()`
removes *anything* in angle brackets (`<[^<>]{1,24}>`) — a mood marker like
`<mood:happy>` would be eaten silently. The filter becomes a parser:
recognized metadata is extracted and attached to the clause; everything
else is stripped as before. Note the policy currently lives in THREE synced
copies — `clausecat` (little-gemma-tools), `bench/clause_pipe.py` (the
python sandbox), and voicedemo's inline port — and this change is the
third pressure (after barge-in mouth-ownership and the pause finetune)
toward consolidating it in one place.

**2. piper gains a structured line mode.** The stdin protocol has no
per-line control channel (`-s SPEAKER` is process-wide). The fork adds
JSON-lines input:

```json
{"text": "I'd love that,", "speaker_id": 2, "meta": "happy"}
```

`synthesize_stream()` already takes `speaker_id` per call, so the CLI
change is small. The opaque `meta` is echoed into the mux as a new frame
kind — `M`, emitted before that clause's `A` frame — so mood rides the
same single pipe as audio and timings, ordered ahead of the speech it
colors. demux prints it as an event line; the browser page reacts to it
like any other frame.

## Schema decisions (made now so they don't get remade)

- **Mood is clause-scoped and sticky**: a tag applies from its clause until
  the next tag. The actuator never interpolates guesswork; it holds the
  last commanded expression.
- The tag vocabulary maps 1:1 to speaker ids in the voice (plus a neutral
  default when the model emits nothing).
- The `M` payload is the raw meta string — the pipe does not interpret it;
  meaning belongs to the two ends (the finetuned model and the rig).

## Reliability: the finetune, not the prompt

Small models cannot be trusted to emit strict formats from prompting alone
(known from tool calling; measured here when style pressure made the 2B
confidently wrong). Mood tags join the existing finetune plan — human pause
placement and early clause boundaries are the same species of task:
**speech-production annotations woven into generated text**. One training
story covers all three, and strict-format reliability is exactly what a
finetune buys.

## Pointers

- Clause policy copies: `little-gemma-tools/src/clausecat.c`,
  `bench/clause_pipe.py`, `little-gemma-tools/script/voicedemo.py`.
- Streaming + mux: piper fork `cortexist/piper1-gpl`, branch
  `phoneme-stream` (`piper.mux`, `piper.demux`, `piper.split`).
- The running demo of the full path: `docs/voice-pipeline.md` and
  `little-gemma-tools/script/voicedemo.py` (viseme mouth included).
