# Memory-consolidation finetune — sample design and dataset mix

*Companion to the dated ledger (`voicecat --memory`) and the age-graded
consolidation design in `.scratch/plan.md`. Canonical samples live in
`bench/consolidation_samples.jsonl`; this note explains what each sample
teaches and how a full dataset should be weighted.*

## The task family

One model, five compression levels, all phrased as maintenance turns the
pipeline already sends (or will):

| level | input | output budget | runs about |
|---|---|---|---|
| L0 session | live conversation + `(session maintenance)` ask | ≤ 120 words, dated | hourly (idle-compress) |
| L1 day | the day's ledger lines | ≤ 25 words | daily |
| L2 week | day notes | ≤ 18 words | weekly |
| L3 month | week notes | ≤ 14 words | monthly |
| L4 year | month notes | ONE sentence | yearly |

The compression *ratio* grows with age; so does the difficulty. The failure
that matters most at every level is **salience**, and it has a measured
baseline: prompt-only E4B kept "pasta, Ana, vegetarian" but dropped
"Friday" — the one token whose loss breaks the future conversation.

## The skills the samples encode

Every sample teaches the budget; each also carries one load-bearing skill:

- **Relative → absolute time.** "Friday" said on `[Tue 2026-07-07]` must be
  stored as "Friday, July 10" — deictic time is worthless a month later.
  (The session's FIRST clock marker carries the date for exactly this
  reason: `[Tue 2026-07-07 09:12]` — voicecat and voicedemo both send it.)
- **Open-loop carry-forward.** Unfinished business survives near-verbatim
  with its deadline ("send the report to Marta before Thursday, March 5");
  *resolved* loops are dropped at the next merge (the tomatoes vanish from
  the day note once watered). Salience is not importance alone — it is
  open-loop tracking.
- **Media becomes words.** The digest describes what was shown or heard,
  taking the description from the model's own in-conversation replies (the
  raw spans are gone by design — see the A/V exclusion rationale).
- **Corrections collapse.** Only the corrected fact survives ("Wednesday,
  not Thursday"), never the history of the mistake.
- **No confabulation.** Small talk compresses to "nothing to carry
  forward"; a quiet year is *honestly* quiet. At 1000:1 compression,
  inventing a past is the catastrophic failure mode, so degenerate inputs
  are a first-class sample type, not an afterthought.
- **Associative (flashbulb) salience.** Significance attaches to
  *episodes*, and details inherit their episode's weight: the car and the
  suit from graduation day outlive a sprained wrist two months later, even
  though on any ordinary day clothing is the first thing dropped. The
  structural enemy is a UNIFORM budget — "merge the year into one
  sentence" forces flat coverage (measured on E4B: "routine work,
  graduation, a sprained wrist, a new job, quiet months" — every event one
  clause, zero texture). The merge asks are therefore anomaly-shaped:
  *spend nearly all the words on the single defining event, keep its small
  details; everything else gets a clause or nothing*. Measured on E4B
  prompt-only, this ask alone recovers the human pattern ("Graduating was
  the big moment. You rode in Marta's red Civic. You wore the blue suit
  your dad bought." — stairs fall gone entirely), so the finetune's job is
  robustness and flagship-*judgment* in ambiguous periods, not the
  mechanism itself. The paired-contrast samples (flagship texture kept
  NEXT TO a notable-but-isolated event dropped, in the same target) are
  the teaching signal, and the uneventful-year sample guards the other
  edge: when no flagship exists, none may be manufactured.
- **Tag hygiene.** Conversation context contains `[[emotion:...]]` and
  thought spans (realistic); targets never do. (The client strips them
  mechanically too — belt and suspenders.)

## Dataset mix

For a LoRA-scale set (2–5k samples), weighted by a blend of runtime
frequency (L0 dominates a deployed day) and difficulty (L4's ratio is the
hardest, so it is oversampled ~10x its runtime share):

| slice | share | notes |
|---|---|---|
| L0 session digest | 40% | within: ~1/3 contain media, ~1/3 open loops, ~1/4 corrections, all with clock markers and relative dates to resolve |
| L1 day merge | 20% | resolved-loop dropping is the core skill |
| L2 week merge | 12% | life events over routine |
| L3 month merge | 8% | durable facts only |
| L4 year merge | 10% | anomaly-budget passage; includes uneventful years |
| degenerate / negative | 10% | small-talk sessions, trivia-only days, contradictory inputs, near-empty merges — spread across levels |

Within every merge level (L1–L4), roughly **a quarter of samples are
flashbulb-contrast pairs**: a flagship episode whose incidental texture
survives, next to a more notable-*sounding* isolated event that vanishes.
The remaining merges are ordinary periods, so the model also learns when
NOT to elevate anything.

A recommended companion slice (outside this mix, ~5% extra): **recall**
samples — a ledger seed in context, the user asks "when is Ana visiting?",
the model answers from the dated entry. Compression is only half the loop;
cheap to generate from the same material.

## Generating volume

The ten canonical samples are hand-built anchors. Scale comes from two
sources: (1) **teacher generation** — the 12B (or a cloud model) writes
candidate compressions over synthetic conversations templated from the
anchor cases, filtered by rule (budget respected, all dates absolute, no
tags, open loops present in input appear in output); (2) **harvested
ledgers** — every `--memory` file a real deployment writes is future
training material, already dated and already in production format.

## Evaluation

The probe harness closes the loop: run a scripted multi-day scenario
through compress cycles, then ask dated questions ("what day is Ana
coming?", "what did I show you yesterday?") and score answers against the
scenario ground truth. The E4B Friday-drop is the baseline to beat; the
metric is *open-loop survival rate* and *date accuracy* after N
consolidation levels, not summary fluency.

## Related work — what to take from each

Architectures. **Generative Agents** (Park et al. 2023, arXiv 2304.03442):
memory stream retrieved by recency x importance x relevance, and
*reflection* — periodic synthesis of raw memories into higher abstractions,
triggered when accumulated model-scored importance (1–10 "poignancy")
crosses a threshold. Take: the importance score is a cheap flagship-judgment
primitive, and consolidation can be importance-triggered, not only
time-triggered — a big day earns its merge early. **MemGPT/Letta** +
*sleep-time compute* (2023–25): memory work moved off the user-facing
critical path to idle agents — independent validation of the idle-cycle
trigger; also memory-pressure as a second trigger. **MemoryBank** (2023):
explicit Ebbinghaus curve with strength *refreshed on recall* — a ledger
entry the user actually asks about should decay slower (cheap: touch its
weight on use). **Mem0** (ECAI 2025): memory edits as structured
ADD/UPDATE/DELETE/NOOP decisions — the corrections-collapse discipline,
formalized; also the first ten-way comparison on LoCoMo. **Zep/Graphiti**:
bi-temporal bookkeeping — event time vs. record time — which our ledger
already has (digest anchors vs. line stamp); keep them distinct on purpose.
**TiMem** (2026, arXiv 2601.02845) "temporal-hierarchical memory
consolidation" and **MemoryOS**/**MemForest** (2025–26) are the closest
recent systems — read before building the consolidation job.

Datasets. **MSC** (Multi-Session Chat, Xu et al. 2021): multi-session
dialogs with gold summaries carried between sessions — reformat directly
into L0 samples. **DialogSum/SAMSum**: generic L0 volume. **LoCoMo**
(2024): steal the *generation method* — build a temporal event graph
first, render conversations from it, and eval ground truth comes free;
its five QA categories (single-hop / multi-hop / temporal / open-domain /
adversarial) shape the probe eval. **LongMemEval** (2024–25): timestamped
sessions, five abilities — information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, abstention — a near 1:1
match to this doc's skill list; adopt the taxonomy, and running the
benchmark itself through the compress pipeline is a publishable eval.
**BookSum**: paragraph->chapter->book targets, the hierarchical-ratio
precedent.

Cognitive science. Systems consolidation ("semanticization": episodic
detail fades to semantic gist) is the ledger's aging, and fuzzy-trace
theory (verbatim and gist as parallel traces, verbatim decaying faster)
is the today-verbatim/old-gist split. One warning worth engineering
around: flashbulb memories in humans are confidently WRONG in detail
(Neisser's Challenger studies) because each recall re-writes them. The
ledger avoids reconsolidation error structurally (append-only, recorded
once) — but only if merges QUOTE surviving details verbatim ("Marta's old
red Civic") instead of paraphrasing them; ten years of re-merges must not
be ten years of drift. Sample targets therefore carry flagship texture as
exact quotes across levels.
