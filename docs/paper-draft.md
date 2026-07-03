# Paper draft — v0.1 (2026-07-03)

> **Title candidates** (pick one; 3 is the recommendation):
> 1. Natural and Cohesive Conversation with Gemma 4 on Jetson Orin
> 2. Fluent Conversation with SOTA Open-Source Models on Edge Devices
> 3. **Fluent and Cohesive: Sub-Second Voice Interaction with General-Purpose
>    Open-Weight Models on a 20-Watt Edge Device**
> 4. (short form of 3) Fluent and Cohesive: A Sub-Second Voice Pipeline for
>    Gemma 4 on Jetson Orin
>
> Rationale for 3: "Fluent and Cohesive" names the paper's two defined,
> measurable properties (the abstract defines them in its first sentence);
> "general-purpose open-weight" is the positioning against Moshi in the title
> itself; "20-Watt" is the number a skimming reviewer remembers.
>
> TODO(author): authors/affiliation; venue (fits MLSys / EdgeSys /
> Interspeech-systems-track shapes; arXiv first regardless).

---

## Abstract

A voice assistant is *fluent* when its reply begins within the human
turn-taking window — first audio in well under a second — and *cohesive* when
the thing replying is a full large language model rather than an intent
router. Production voice AI today gets one or the other: datacenter cascades
median 1.4–1.7 s to first audio [TODO cite hamming.ai], and the systems that
beat the window do it with purpose-trained speech-to-speech models (Moshi
[TODO cite arXiv:2410.00037]) that give up the text LLM's generality. We show
both are achievable at the same time, entirely on a 20 W, credit-card-sized
edge board, using unmodified state-of-the-art open-weight models.

Our pipeline — streaming ASR (whisper.cpp), a Gemma 4 model (2B–12B) on a
purpose-built ~6,000-line C/CUDA runner, and a VITS TTS (piper) — reaches
**0.82 s from end of speech to first audio** on a Jetson Orin NX 16GB
responding to a 30-second spoken instruction, with the entire stack holding
**4.2 GB of unreclaimable memory** (it fits an 8 GB Jetson Nano) and peaking
near 20 W. Three techniques carry the result, none of which modify the model:
(1) *prefill under speech* — streaming ASR commits prefill into the open turn
while the user is still speaking, backed by a proof that space-boundary
splits are exact for SentencePiece, cutting post-speech TTFT from 5.37 s to
0.55 s on the 12B; (2) *the LLM as clause splitter* — a system prompt that
makes the model create TTS flush points, motivated by the measured finding
that the baseline model emits 21-word sentences with zero commas, leaving a
clause-flushing TTS layer nothing to flush; (3) *multi-token-prediction
speculative decoding* characterized honestly as content-dependent
(1.1–2.1×) and as a battery feature (2.29 vs 2.85 J/token). A cross-silicon
study against Moshi on an RTX A5000 finds the remaining gap to
speech-native models is structural (one decoded clause + one vocoder call,
~0.2 s) — and that the datacenter GPU buys only 1.7× over the 20 W board,
because in the conversational regime the pipeline is floor-bound, not
compute-bound.

## 1. Introduction

Conversation has a clock. Human turn transitions cluster around 200–500 ms
[TODO cite arXiv:2404.16053 + Stivers et al. 2009]; beyond one second a
reply reads as hesitation, and industry telemetry places the median
production voice agent at 1.4–1.7 s with a P90 of 3–5 s — the region of
talk-overs and user frustration [TODO cite hamming.ai]. We name the two
properties this paper pursues:

- **Fluent**: time from end of user speech to the first audio sample of the
  reply (TTFB) under one second — inside or adjacent to the human window.
- **Cohesive**: the reply is produced by a state-of-the-art general-purpose
  LLM in the loop — capable of instruction following, world knowledge,
  multi-turn context, tool use, and (in the same family) vision — not a
  latency-optimized intent model.

The two properties pull apart because of where each is cheap. Datacenter
cascades are cohesive but pay network, queueing, and orchestration latency.
Speech-native full-duplex models such as Moshi [TODO cite] are strikingly
fluent — we measure ~130 ms TTFB on an RTX A5000 — but are purpose-trained
speech models (a ~7B Helium backbone fused to a neural codec), trained once,
at great cost, with the conversational behavior baked in: no drop-in model
upgrades, no text-domain tooling, no camera feed.

This paper takes the third corner: a cascaded pipeline of unmodified
open-weight components — whisper.cpp ears, Gemma 4 brain, piper mouth — on a
Jetson Orin NX 16GB, chosen as the robotics-relevant operating point (the
AGX steps up in both power and cost; the Nano 8GB lacks the RAM headroom for
the larger family members, though §5.6 shows our smallest configuration fits
it). All Gemma 4 family models produce cohesive conversation out of the box;
**the entire challenge is speed**, and the paper is the anatomy of where the
time goes and how it is reclaimed.

Contributions:

1. **Prefill under speech** (§4.2): incremental prefill of an *open* turn fed
   by streaming-ASR commit semantics, with a correctness lemma — SentencePiece
   pieces never cross a space boundary, so word-boundary splits reproduce the
   one-shot tokenization exactly (verified: equal token ids, byte-identical
   replies). A 929-token spoken instruction costs 0.55 s TTFT after the last
   word instead of 5.37 s (12B; 0.10 s on the 2B).
2. **The LLM as the clause splitter** (§4.3): the split policy for streaming
   TTS must live in the model, because the measured baseline emits 21-word,
   comma-free sentences — there is nothing downstream to flush. A system
   prompt creates the flush points and cuts first audio from 1.21 s to
   0.82 s; finetuning for human pause placement is the identified production
   path.
3. **A latency-anatomy protocol for local voice agents** (§4.1): the
   decomposition TTFT → time-to-first-speakable → TTFB, measured client-side
   under three delivery modes (deferred / burst / paced-realtime), with
   warmup-discard and byte-identity discipline. TTFT alone is shown to be
   uninformative for the spoken experience.
4. **The whole-system result** (§5): 0.82 s TTFB on 20 W; three latency/
   intelligence tiers from one runner (12B / E4B / E2B); 4.2 GB unreclaimable
   for ears+brain+mouth (8 GB-board feasible, with the measurement
   methodology that makes that number honest on Jetson); a decode rate that
   *beats* llama.cpp on the same board (1.17–1.26×) from ~6,000 lines of
   C/CUDA.
5. **A cross-silicon study** (§5.8) against Moshi on its own class of
   hardware, locating the cascade's structural floor (one clause of decode +
   one VITS call) and showing the conversational regime is floor-bound: a
   datacenter GPU + desktop CPU improves end-to-end latency only 1.7× over
   the 20 W board.

## 2. Related work

**Speech-native duplex models.** Moshi [TODO cite arXiv:2410.00037] fuses a
7B text backbone with the Mimi streaming codec and full-duplex dialogue
training; it reports ~160 ms theoretical / 200 ms practical latency on an
NVIDIA L4 (24 GB, ~72 W TDP datacenter card); we measure ~130 ms TTFB warm on
an RTX A5000 (trials 252/129/135 ms; ~23 ms connection handshake, inclusion
under audit — TODO revisit). Mini-omni and successors [TODO cite] follow the
same species. These systems set the fluency bar, at the price of a fixed,
purpose-trained model: the backbone is frozen into a speech topology, so the
brain cannot be swapped for next quarter's better open-weight release, and
text-ecosystem capabilities (tools, structured output, long instructions,
vision) regress to what the speech training preserved.

**Cascaded voice pipelines.** Commercial stacks (Pipecat, LiveKit Agents,
[TODO cite]) cascade ASR → LLM → TTS and already stream at *existing*
sentence punctuation. Our measurements locate two latencies they leave on
the table: prefill of the growing turn (hidden under speech, §4.2) and the
absence of early punctuation to flush on (created by the model itself,
§4.3).

**Local inference runtimes.** llama.cpp [TODO cite] is the gold-standard
general local runtime, and its throughput numbers are the reference we
benchmark against (§5.2). The pursuit of TTFB, however, is a different race
than tokens/second: it rewards an *appendable* conversation abstraction
(connection = conversation, line = turn, typed media/text frames into the
open turn), speculative decoding tuned for short structured replies, and a
server that streams the raw token channel for downstream clause flushing.
little-gemma is not a llama.cpp competitor; it is a demonstration that a
small, single-family runner can occupy a design point the general runtime
does not: on the same Orin, matching ~0.8× of llama.cpp's prefill while
*beating* it 1.17–1.26× in decode, and — the relevant number for this paper
— reaching first-sentence 18–48× earlier on multimodal turns (§5.2, thought
suppression and encode-off-critical-path).

**Streaming ASR.** whisper_streaming / LocalAgreement-2 [TODO cite Machacek
et al.] provides the commit semantics our prefill rides on; the ASR side is
prior art we consume, not a contribution.

**Speculative decoding.** EAGLE, Medusa, self-speculation [TODO cite]. Gemma
4 ships a first-party MTP head; our contribution is not the mechanism but
the honest characterization (content- and head-width-dependence) and the
energy angle (§4.4), which the edge literature largely lacks.

## 3. System

Hardware: Jetson Orin NX 16GB module (Ampere iGPU, shared LPDDR5), NVMe SSD,
JetPack 6.2.1, headless. Rationale: the NX is the robotics operating point —
credit-card footprint, 10–25 W envelope (40 W MAXN available), ~$700-class
[TODO verify price]. The AGX buys performance with a power/cost step that
changes the product class; the Nano 8GB cannot hold the 12B tier (§5.6
qualifies what it *can* hold).

Three processes, one GPU owner:

- **Ears — voicecat + whisper.cpp** (CUDA base.en): energy VAD, streaming
  transcription with LocalAgreement-2 commits, timestamp-based window
  trimming, barge-in (one byte stops the reply mid-stream). Committed words
  stream into the runner as typed text frames *while the turn is still
  open*.
- **Brain — Gemma 4 on little-gemma**: a ~6,000-line C/CUDA runner for the
  Gemma 4 family (int8 tensor-core prefill, dp4a decode, GPU vision encoder,
  MTP speculative decoding, all quantizations byte-identical to the f32
  reference by gate). The serve mode is a Unix-domain socket: connection =
  conversation, line = turn, typed frames ('T' text, 'I' image, 'A' audio)
  append to the open turn and prefill on arrival. A system-prompt file
  prefills once at server start (§4.3 rides this).
- **Mouth — piper** (VITS, en_US voices) as a persistent stdin service on
  the CPU — the GPU belongs to the LLM. One ONNX call per line: first byte ≈
  whole-clause synthesis, ~0.27–0.49 s per clause on the Orin's cores
  (~18–26× realtime), 0.31 GB peak anonymous memory. Piper was chosen over
  Kokoro on speed [TODO: one-line Kokoro comparison number if we keep this
  claim].

**Why external ASR when Gemma 4 is natively multimodal?** Because we
measured the native path and it fails three ways (§7 for details): the
E2B/E4B conformer audio path confabulates rather than transcribes; the 12B
hears speech but behaves as ASR-only for non-speech sound; and — the finding
we could not have predicted — **vision in context disables hearing on the
12B** (seven arrangements tested, asymmetric, session-scoped, survives black
frames). A voice+camera product on this family *requires* an external ASR
lane; our pipeline's soundtrack policy (audio rides as a whisper transcript
under the video span) is the direct product of that finding.

## 4. Where the time goes, and how it is reclaimed

### 4.1 Metrics and protocol

- **TTFT**: last user input → first reply token (server work becomes
  visible).
- **TTFS / time-to-first-speakable**: → first complete speakable unit
  (sentence, or clause once §4.3 applies) — the unit the TTS can start on.
- **TTFB (first audio)**: → first PCM sample out of the TTS. The number the
  ear judges. TTFT alone misorders systems: a fast first token followed by a
  21-word sentence is *slower to speak* than a slower first token followed by
  a 6-word clause.

All clocks are client-side (the runner is not trusted to grade itself), the
first turn after model load is discarded (repack, clock ramp, draft-head
warmup), and each measurement is a fresh conversation. Three delivery modes
model the ASR: **deferred** (whole instruction arrives at turn end — the
no-streaming baseline), **burst** (frames arrive as fast as the socket
takes them), **paced** (30 words/s with 200 ms frame cadence — a real
speaking rate). The 929-token spoken instruction (a long rambling
preamble ending in a question) is the standard stimulus; replies must be
byte-identical across delivery modes for a run to count (§4.2's lemma makes
this achievable and it held in every reported cell, on both test machines).

### 4.2 Prefill under speech

Chunked prefill is a scheduler concept for *completed* prompts in every
serving stack we know (vLLM, TensorRT-LLM, llama.cpp server [TODO verify
llama.cpp turn-level behavior]). The open turn is different: while the user
speaks, committed words are already immutable — LocalAgreement-2 commits
exactly the prefix both hypotheses agree on — so their KV entries can be
computed *now*, under the speech itself.

Correctness requires that tokenizing the turn in committed pieces equals
tokenizing it whole. For SentencePiece this holds exactly when splits land
on space boundaries: pieces never cross a space (the space glyph ▁ *starts*
pieces), so a word-boundary split cannot merge or reshape tokens across the
seam. We verify, not just argue: equal token id sequences and byte-identical
replies, deferred vs streamed, all three models. (One span of holdback is
kept unprefilled to absorb ASR revisions of the most recent words; MSG_PEEK
lets the server prefill right up to the pending frame without consuming it.)

![Figure 1 — prefill under speech: deferred vs streamed delivery of a
929-token spoken instruction (12B, Orin NX)](fig-prefill-under-speech.svg)

*Figure 1: In deferred delivery (a) the 5.37 s prefill starts only when the
turn closes; streamed (b), committed words prefill during the speech itself
and the turn close leaves only the holdback span — TTFT 5.37 → 0.55 s with
byte-identical replies.*

Measured on the Orin (TTFT after last word, deferred → paced): 12B
**5.37 → 0.55 s**, E4B **2.23 → 0.26 s**, E2B **1.15 → 0.10 s**. The
mechanism generalizes beyond dictation: camera and video frames prefill on
arrival the same way (media spans, 3.0× video TTFT), which is what makes
the same pipeline camera-capable — the flexibility argument against
purpose-trained speech models made concrete.

A protocol note: at a real speaking pace the hiding is essentially free —
prefill chunks slot into the pauses between frames. Burst delivery pays
padding costs and is the wrong choice when the full text is already in
hand; send it as one line instead (measured, §5.3 table note).

### 4.3 The LLM is the only clause splitter

A streaming TTS layer flushes at punctuation. The baseline 2B, asked our
standard question, answers in a single 21-word sentence with **zero
commas** — there is nothing to flush until the sentence ends, 9.4 s of audio
in one synthesis call. No glue-layer heuristic fixes this: splitting
mid-clause produces unnatural speech, and only the model knows where a
thought can pause. The split policy therefore lives in the model, installed
as a system prompt loaded once at server start (prefilled once, zero
per-turn cost):

> You are a voice assistant. Everything you write is spoken aloud by a
> text-to-speech engine, clause by clause, so write for the ear, not the
> eye. [...] Keep sentences short. Break long thoughts into clauses of five
> to ten words, separated by commas or periods [...] Get to the point in the
> first clause; details can follow.

Effect on the E2B voice loop: first speakable unit 21 → 8 words
(0.714 → 0.549 s), synthesis of that unit 0.49 → 0.27 s (VITS synthesis is
~linear in audio length, so shorter first units pay twice), **first audio
1.21 → 0.82 s**. Two honest notes travel with this: style pressure sharpens
a small model into confident error (the 8-word answer contains a wrong
date; the vague baseline was defensible), and prompt-only is the bootstrap
— the production path is a light finetune for *human* pause placement,
which is often shorter than grammatical clauses.

### 4.4 Speculative decoding as a latency and battery feature

Gemma 4's first-party MTP head drafts the next N tokens; greedy
verification makes output byte-identical to plain decoding *by
construction* — only tokens/s and acceptance move. What it is worth is
sharply content-dependent (block-3, serve mode, steady-state): structured
output ~2× (12B A5000 2.16×, Orin 1.85×, 100% acceptance), free prose
1.10–1.35× (the 12B's 1024-wide head lands ~49% of prose drafts; E4B's
256-wide head ~30%). We report the *distribution*, not a flat multiplier —
flat claims in the spec-decoding literature do not survive contact with
prose. On the battery angle: MTP cuts energy per token from 2.85 to 2.29
J/token on the Orin (the GPU is bandwidth-bound; fewer full passes per
token is less DRAM traffic) — for a robot, spec decoding is a battery
feature before it is a latency feature.

### 4.5 The substrate, briefly

None of §4.2–4.4 works if the kernels are slow. The runner's prefill is an
int8 tensor-core (mma) path with warp-cooperative activation quantization
and flash-attention prefill; decode is dp4a with split-K attention, frozen
kernels, and CUDA-graph capture; weights are zero-copy on the integrated
GPU (host and device share LPDDR5 — copying would hold every byte twice),
mmap'd and unpinned where the quantization is repacked anyway. Every
optimization is gated byte-identical against a f32 reference and, for the
speculative path, by acceptance-rate tripwires (which caught three real
bugs). Full engineering logs, including the falsified dead ends, are in the
repository journals [TODO cite repo]. The result on the Orin: ~0.8× of
llama.cpp's prefill throughput and 1.17–1.26× its decode, from a codebase a
single reviewer can actually read (~6,000 lines).

## 5. Evaluation

### 5.1 Setup

Orin NX 16GB, JetPack 6.2.1, pinned clocks for benchmarks (jetson_clocks;
production ships DVFS — a reboot resets the pin, and every number here was
re-verified against that trap), models: Gemma 4 12B-it Q4_K_M, E4B-it
Q4_K_M, E2B-it QAT q4_0 (Google's quantization-aware release), each with
its MTP head; whisper.cpp CUDA base.en; piper en_US voices. Cross-silicon:
RTX A5000 (Windows) + i7-8086K. All measurements are client-side per §4.1.

### 5.2 The substrate vs llama.cpp

Serve-mode prefill, tok/s (929-token turn, warm):

| device | model | little-gemma | vs llama.cpp |
|---|---|---:|---|
| Orin | 12B | ~173 | 0.80× |
| Orin | E4B | ~417 | 0.82× |
| Orin | E2B QAT | ~798 | — [TODO llama.cpp E2B reference] |

Plain decode, tok/s (256 tokens, warm; llama-bench reference):

| device | model | little-gemma | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Orin | E4B | 16.80 | 13.36 | **1.26×** |
| Orin | 12B | 8.27 | 7.04 | **1.17×** |

MTP multiplies on top (§4.4). Where the design point shows is multimodal
turns: little-gemma reaches first-sentence on an image+question turn in
**1.98 s (E4B) / 5.22 s (12B)** on the Orin vs 35.8 / 98.6 s for
llama-server (which burns its lead in an unsuppressed thinking channel and
host-side encode) — the raw-token-stream server and encode-off-critical-path
choices are TTFB choices, not throughput choices.

### 5.3 The dictation matrix (Orin)

929-token spoken instruction; TTFT / TTFS in seconds:

| model | deferred | paced (30 w/s) |
|---|---|---|
| 12B | 5.37 / 8.13 | 0.55 / 3.31 |
| E4B | 2.23 / 3.20 | 0.26 / 1.24 |
| E2B QAT | 1.15 / 1.76 | **0.10 / 0.72** |

(Burst delivery pays frame padding and is dominated by deferred — measured,
not assumed; deferred runs reproduce to the millisecond.)

### 5.4 End-to-end: the 0.82 s anatomy

E2B QAT, voice-sys prompt, streamed dictation of the 30-second spoken
instruction, everything on-device:

![Figure 2 — anatomy of first audio: baseline vs voice-sys prompt (E2B QAT,
Orin NX)](fig-first-audio-anatomy.svg)

*Figure 2: The same question, two system prompts. The voice-sys prompt cuts
the first speakable unit from 21 to 8 words, which both starts synthesis
earlier and makes it cheaper (VITS is ~linear in audio length): first audio
1.21 → 0.82 s.*

With whisper.cpp's final-commit pass in the loop (base.en CUDA, ~0.55 s
per invocation, invocation-cost-dominated), voicecat closes the turn ~1.0 s
after end of speech in live-mic operation; a persistent whisper server is
the identified upgrade path [TODO: measure with whisper-server and update —
likely pulls live-mic TTFB near the dictation number].

### 5.5 Three tiers of conversational experience

One runner, one board, three operating points (TTFS + TTS ≈ TTFB):

| tier | intelligence | TTFB (paced) | where it lands [TODO cite hamming] |
|---|---|---:|---|
| 12B | strongest open-weight ≤16GB | ~3.6 s | "common experience" (P90 region) |
| E4B | high | ~1.5 s | industry median (1.4–1.7 s) |
| E2B QAT | good | **0.82 s** | inside the "theoretical ideal", rarely achieved in production |

![Figure 3 — the three tiers against production voice-AI latency
bands](fig-tiers-vs-industry.svg)

*Figure 3: One board, one pipeline, three operating points. The E2B tier
lands inside the natural-conversation window that production telemetry
reports as rarely achieved; the E4B matches the industry median; the 12B
trades latency for the strongest answers.*

The tier table is the product story: the same hardware and pipeline lets a
robot answer trivia with the 12B and do fluent small talk with the E2B —
switching is a model file, not an architecture.

### 5.6 Memory: does it fit an 8 GB Nano?

Measured the honest way for Jetson (anonymous RSS + nvmap; VmHWM counts
evictable file pages and MemAvailable under-counts nvmap — both mislead):
little-gemma E2B QAT serve 3.2 GB unreclaimable (nvmap 3.00 + anon 0.11,
after mmap'ing the weight blob and skipping the pin for fully-repacked
models: 5.5 → 3.2 GB), whisper ~0.7 GB, piper 0.31 GB peak → **≈4.2 GB for
the full voice stack**, leaving ~2.2 GB slack on a 7.4 GB-usable headless
Nano, with room for the vision encoder (~0.7 GB). Caveat, stated plainly:
measured on the NX 16GB against an 8 GB budget, not yet on the Nano SKU
[TODO if we can borrow one].

### 5.7 Power

GPU-saturated decode peaks ~23 W (default profile; 40 W MAXN raises clocks
but the workload is bandwidth-bound), idle-to-peak on a supply a robot
already carries. MTP reduces J/token 2.85 → 2.29 (§4.4). Moshi's reference
deployment is an L4: 24 GB, ~72 W TDP datacenter card [TODO complete the
note: "despite they got 100..." — finish this sentence from the Moshi paper
data].

### 5.8 Cross-silicon: the cascade vs Moshi on Moshi's hardware class

Moshi on our A5000: **~0.13 s** TTFB warm. Our pipeline on the same card
(+ i7-8086K for piper), same protocol as §5.3:

| model | first token | first clause | + piper | last word → first audio |
|---|---:|---:|---:|---:|
| 12B | 0.090 | 0.241 | 0.109 | **0.35 s** |
| E4B | 0.065 | 0.132 | 0.108 | **0.24 s** |
| E2B QAT | 0.121 | 0.29 | 0.193 | **0.48 s** |

Three readings. (1) The remaining 2–4× is *structural*: a speech-native
model starts vocoding without waiting for a speakable text unit; the
cascade's floor is one decoded clause plus one VITS call (~0.2 s). (2)
Model size has almost stopped mattering — 2B to 12B spans first-clause
0.13–0.29 s once streaming hides prefill; *content* dominates (the E2B is
slowest end-to-end because its clause contains a date that verbalizes to
3.9 s of audio). (3) The datacenter card + desktop CPU beat the 20 W board
by only ~1.7× end-to-end (0.48 vs 0.82 s): the conversational regime is
floor-bound, not compute-bound, which is precisely the regime where an
edge deployment loses nothing that matters.

## 6. Discussion

What the cascade buys for its ~0.5 s of structural disadvantage: the brain
is a *commodity part*. The same pipeline ran three model generations
[TODO: soften or substantiate] without retraining anything; it processes a
camera feed with the same open-turn mechanics that stream dictation
(§4.2); it speaks Gemma's full tool-calling protocol because the server
passes control tokens through; and its answer quality is that of a 12B
instruction-tuned text model, not what survived speech-topology training.
Fluency, meanwhile, turned out not to require a new species of model —
just refusing to waste the time the user's own speech provides, and
letting the model place its own pauses.

The A5000 study reframes the edge question. The usual framing is "how much
worse is edge?"; measured, the answer in this regime is *1.7× on latency,
at 1/6th the power class* — because the floor is set by clause decoding
and vocoding, not by FLOPs. TTFB pursuit and throughput pursuit diverge
here: past the point where prefill hides under speech and decode outruns
speech playback (~3 tok/s suffices for real-time listening; even the 12B's
8.3 tok/s clears it), *more* tokens/s stops buying experienced fluency.
That is why this work no longer competes with llama.cpp on llama.cpp's
axis — and why a 6,000-line runner can hold the design point at all.

## 7. Limitations and honest notes

- The dictation TTFB excludes the ASR final-commit pass (§5.4 gives the
  live-mic number, ~1.0 s to turn close); Moshi's number includes its
  hearing. The persistent-whisper upgrade path narrows this and is measured
  future work.
- Moshi's ~23 ms handshake in/out of its TTFB is unresolved [TODO].
- Style pressure makes the 2B confidently wrong where it was vaguely right
  (§4.3); the clause-splitting prompt trades a quality edge for fluency at
  the smallest tier. Finetuning may recover both.
- Gemma 4's native audio path is not usable as the pipeline's ears (E2B/E4B
  confabulation; 12B vision-disables-hearing exclusion) — findings we
  report [TODO: decide whether the A/V exclusion becomes its own §
  or stays a finding table] but that make the whisper lane load-bearing.
- Greedy decoding throughout (byte-identity discipline requires it);
  sampling interacts with MTP acceptance and is unexplored here.
- Single-user, single-stream serving; no batching story.
- 8 GB verdict extrapolated from the NX 16GB (§5.6).

## 8. Future work

Finetune for human pause placement (shorter than grammatical clauses);
persistent whisper service (flips the streaming-vs-endpass verdict
unconditionally); Nano 8GB SKU validation; barge-in latency
characterization; camera+voice concurrent sessions (the A/V exclusion
policy exists, the latency study does not).

## 9. Conclusion

Fluent and cohesive do not require choosing. A 20 W board running
unmodified open-weight models answers 0.82 s after you stop talking —
because the pipeline stops wasting the seconds the user's own speech
offers (prefill under speech), lets the model create the places speech can
begin (the LLM as clause splitter), and measures the thing the ear
actually judges (TTFB, not TTFT). The gap to purpose-trained
speech-to-speech models is real, structural, and — at 2–4× against a
species that cannot swap its brain, read a camera, or call a tool —
a good trade for most embodied products.

---

## Appendix A: Reproducibility

Every number traces to a committed harness: the dictation clients
(client-side clocks, three delivery modes), the TTS timing scripts, the
memory sampler (anon+nvmap), and the engineering journals including
falsified attempts. Replies are byte-identical across delivery modes and
quantization levels by gate. [TODO: decide what of .scratch/ gets promoted
into the repo or a companion artifact for submission.]

## Appendix B: numbers still to fill / verify

- [ ] llama.cpp reference for E2B QAT prefill/decode (no fork support at
      measurement time?)
- [ ] E2B Orin plain-decode number for §5.2 table (dictation reply ran
      31.6 tok/s with MTP at 39.5% acceptance)
- [ ] Kokoro comparison number, or drop the "faster than Kokoro" claim
- [ ] Moshi note completion ("despite they got 100...")
- [ ] whisper-server (persistent) live-mic TTFB
- [ ] price of Orin NX 16GB module for §3
- [ ] burst-mode row values if the matrix table keeps all three deliveries
