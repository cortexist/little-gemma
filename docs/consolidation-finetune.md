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
| L4 year merge | 10% | one sentence; includes uneventful years |
| degenerate / negative | 10% | small-talk sessions, trivia-only days, contradictory inputs, near-empty merges — spread across levels |

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
