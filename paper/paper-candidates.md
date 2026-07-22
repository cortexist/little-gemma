# Paper candidates: what is actually novel in the pipeline

The headline result — end of speech to first audio in 0.82 s, ears/brain/voice
all on-device on a 20–25 W credit-card SBC, decode beating llama.cpp on the
same board — rests on many pieces. This list sorts them by *how defensible a
novelty claim is*, with prior art to search against for each, because the worst
thing a paper can do is claim novelty for something llama.cpp or vLLM already
ships. Numbers reference docs/prefill-performance-journal.md (2026-07-02
entries) and docs/dictation-timing.html.

## Tier A — likely novel, candidate headline claims

**1. Prefill hidden under human speech — incremental prefill of an OPEN turn
fed by streaming ASR.** Transcript commits stream into the live turn as 'T'
frames and prefill immediately: a 929-token instruction costs 0.10 s TTFT
after the last word instead of 1.15 s (5.37 s on the 12B). Chunked prefill
exists everywhere (vLLM, TensorRT-LLM) but as a scheduler concept for
completed prompts; prefilling *while the user is still speaking*, driven by
ASR commit semantics, is a different claim. Supporting lemma, crisp and
provable: space-boundary splits are EXACT for SentencePiece (pieces never
cross a space — proven by equal token ids and byte-identical replies), so
incremental prefill is correctness-preserving, not approximate.
*Search against:* OpenAI Realtime API (closed, no published mechanism),
Moshi / mini-omni (audio-native duplex models — different approach),
whisper_streaming (ASR side only), llama.cpp server.

**2. The LLM as the clause splitter — prompting/finetuning the model to
CREATE TTS flush points.** The measured kicker makes this a finding, not an
opinion: baseline E2B emits 21-word sentences with zero commas, so a
clause-flushing TTS layer has nothing to flush — the split policy must live
in the model. Voice-sys prompt (docs/voice-sys.txt): first audio
1.21 → 0.82 s. Forward-looking section: finetune for human pause placement,
often shorter than grammatical clauses.
*Search against:* streaming TTS pipelines (Pipecat, LiveKit agents — they
segment at *existing* punctuation), controllable-prosody literature.

**3. TTFS / first-audio as the metric, with the delivery-mode protocol.**
The decomposition TTFT → time-to-first-speakable → first audio, plus the
deferred/burst/paced delivery matrix, client-side clocks, warmup-turn
discard discipline. A rigorous, reproducible latency-anatomy protocol for
local voice agents.
*Search against:* Alexa/Google endpointing papers, ITU-T conversational
latency standards, recent full-duplex model evaluations.

**4. The whole-system result itself.** As a systems paper, the integration
and its measured anatomy is the contribution: 0.82 s first audio, 4.3 GB
unreclaimable (8 GB SBC feasible), 23 W peak, every number reproducible from
committed harnesses. The tiers below become its sections.

## Tier B — publishable empirical findings

**5. Vision-in-context disables hearing on Gemma 4 12B.** Seven
arrangements, asymmetric, session-scoped, survives black frames and earlier
turns; plus the whisper-transcript fallback policy that ships in mmcat.
*Search against:* multimodal interference / cross-modal suppression.

**6. Causal audio falsified.** Bidirectional attention is load-bearing for
the encoder-free audio path; 2x1.5 s chunk spans are the latency/quality
dial. Likewise video's bidirectional-within-frame-span requirement.

**7. MTP speculative decoding characterized honestly.** Acceptance is
content- and head-width-dependent (structured ~2x, prose 1.1–1.35x), and —
the fresh angle — MTP as a *battery* feature: 2.29 vs 2.85 J/token.
Energy-oriented speculative-decoding evaluation on edge hardware is nearly
absent.
*Search against:* EAGLE, Medusa, self-speculation; spec decoding + energy.

**8. Device-scoped kernel verdicts invert.** The same optimization wins on
A5000 and loses on Orin (and vice versa) with counter-level evidence:
staging, wide-N, 2-deep prefetch, stream-K all falsified per-device. An
empirical-methods section: never generalize across GPUs, with receipts.

## Tier C — methodology worth writing up, novelty moderate

**9. Byte-identity engineering discipline.** verify==decode by construction
for speculative decoding; acceptance% as the regression tripwire (caught
three real bugs); high-entropy-prompt gates because redundant prompts hide
corruption under argmax margins. Related to determinism-in-inference work;
the spec-verify angle looks unclaimed.

**10. Jetson memory accounting.** VmHWM and MemAvailable both lie; only
anonymous RSS + nvmap tells the truth. mmap + skip-pinning for
fully-repacked models: 5.5 → 3.2 GB.

## Tier D — good engineering, do NOT claim novelty

The q4_0-as-q4_K layout-contract repack, the own m16n8k32 kernel,
producer-epilogue activation quantization, flash K-staging, split-K decode,
Tegra zero-copy tricks, the streaming-whisper commit-ms >= 3x rule
(LocalAgreement-2 is Machacek et al.), piper-as-a-service. These are the
*how*, cited as implementation; llama.cpp/ggml practice is close prior art
throughout.

## Suggested framing

Tier A #4 is the thesis ("anatomy of a sub-second fully-local voice agent"),
#1–#3 its named techniques, Tier B the findings the build surfaced, Tier C
the discipline that made the numbers trustworthy. The strongest sentences
will be the falsifications — papers that show their dead ends are the ones
people believe.
