# Mood metadata through the voice pipeline — design note

*Status: PLUMBING SHIPPED 2026-07-05; the finetune remains the gating item
for the model emitting tags reliably. The mechanism chosen is in-band
tool-call control lines (below), which replaced the JSON-lines idea from the
first draft of this note.*

*UPDATE 2026-07-09 (duplex branch): the MODEL-facing grammar moved to the
inline tag form `[[emotion:happy]]` — 5 tokens vs the tool-call line's 17,
and semantically an annotation, not a call. `docs/voice-sys.txt` teaches the
tag; `clausecat --route-emotion` (and voicedemo) translate it into the very
control line documented below, so piper and everything downstream of the
orchestrator are unchanged — this doc's piper/mux/demux design stands.
Measured bonus: under the tag grammar the E4B emits emotion spans PROMPT-ONLY
(sad→neutral→happy tracking a story arc, exact spans) — the tool-call form
needed the 12B. The old form still parses (`--allow-control-token`), so mixed
sessions migrate gracefully.*

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

## The mechanism, as shipped

The LLM emits a tool-call span; clausecat passes allowed spans through as
their own lines; piper consumes them and reports downstream:

```
<|tool_call>call:set_voice{speaker_id:<|"|>happy<|"|>}<tool_call|>
```

- `clausecat --allow-control-token '<|tool_call>call:set_voice{*}<tool_call|>'`
  passes matching spans through verbatim, ordered against the clauses;
  non-matching spans are dropped WHOLE (fixing a latent leak: a tool call's
  payload used to reach TTS as speech). Twin: `bench/clause_pipe.py`,
  differential feed `bench/clause_cases.txt`.
- piper (fork branch `phoneme-stream`): `piper.control` parses the line
  (marker-token dialect agnostic); `set_voice` retargets the speaker by name
  (the voice's `speaker_id_map` — e.g. happy/sad/neutral) or number for every
  following clause; `--output-mux` writes it as an `M` frame BEFORE the audio
  it colors; `piper.demux` prints `meta<TAB>payload` events.
- Multi-speaker voices condition their decoder on the speaker embedding — a
  SECOND boundary tensor — so the split carries all crossing tensors and the
  streaming decode feeds conditioning whole to every chunk (verified on
  en_US-arctic-medium, 18 speakers: distinct audio per speaker, chunked ==
  monolithic to fp32 noise).

voicedemo is wired too (2026-07-05): its inline splitter runs the same
span machine, forwards `set_voice` lines to piper, and the page shows the
active voice next to the lipsync — one-stroke faces for
happy/sad/neutral/angry, a bust plus the name otherwise, in the mouth's
line-art language. In mux mode piper's validated `M` frame drives it; in
raw mode the app reports the value itself.

## The original design sketch (superseded where it disagrees)

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

## Reliability: prompt-only is a tier question (measured 2026-07-05)

With the tool taught in `voice-sys.txt` (exact span + the four emotion
names), raw probes on the Orin:

- **E2B QAT**: "please comfort me, and sound sad" produces pure prose with
  no tool-call attempt — *implicit* emotional content does not elicit the
  span. But a **direct instruction works**: "tell me a story in happy voice"
  emitted the span and turned the face happy in the live demo (user-observed,
  2026-07-05). The 2B follows explicit commands, not tone inference.
- **12B**: emits the span **exactly**, first try, both directions — sad
  prompt → `set_voice{…sad…}` then prose; "sound cheerful" →
  `set_voice{…happy…}` then the joke — and correctly emits *no* span on a
  neutral factual question ("what is two plus two"). On the 12B tier the
  emotion face works today, prompt-only.

So the finetune is the gating item **for implicit emotion on the small
models only** (community tool-calling finetunes of the same class — e.g.
supergemma on HF — are the known remedy), and mood tags join the existing
finetune plan there. Human pause placement and early
clause boundaries are the same species of task: **speech-production
annotations woven into generated text**. One training story covers all
three, and strict-format reliability is exactly what a finetune buys.

(Operational note: the 12B does not co-reside with the E2B serve on a 16GB
Orin — 5.5GB + 7.1GB blobs OOM NvMap; probing the 12B means stopping the
E2B stack first, and the 12B load wants `drop_caches`, same as llama-bench.)

## Pointers

- Clause policy copies: `little-gemma-tools/src/clausecat.c`,
  `bench/clause_pipe.py`, `little-gemma-tools/script/voicedemo.py`.
- Streaming + mux: piper fork `cortexist/piper1-gpl`, branch
  `phoneme-stream` (`piper.mux`, `piper.demux`, `piper.split`).
- The running demo of the full path: `docs/voice-pipeline.md` and
  `little-gemma-tools/script/voicedemo.py` (viseme mouth included).
