# Trimming the draft head's vocabulary

The MTP draft head is tied to `token_embd.weight`. For the 12B assistant that
tensor is **1024 × 262144**, and `mtp_up_h` (`src/cuda/mtp-kernel.cuh:200`)
uploads it as f16 — so **512 MiB of device memory, read in full on every draft
step**. It is 63.5% of the draft's 806 MiB of weight traffic; the four blocks
and the two projections are the other 36.5%.

Almost none of that vocabulary is ever used. In a 265k-token technical corpus
the Gemma tokenizer emitted only **11,137 distinct ids** out of 262,144. The
head carries a 262k-row table to answer a question whose answer is one of
~11k rows.

The fix is a **reduced-vocabulary draft head**: keep only K rows, plus a `d2t`
table (int32, length K) mapping draft row → target token id. This is the
mechanism llama.cpp already implements for EAGLE3 and DFlash
(`src/models/eagle3.cpp`, `src/models/dflash.cpp`), and which HauhauCS
patched into `qwen35.cpp` for their Qwen3.8 FastMTP release.

**Why it is safe to do aggressively, and safe to automate.** Verification is
greedy against the target's *full* vocabulary. A trimmed head can only
propose fewer candidates; a token outside the subset is simply a rejected
draft. **The byte-identical guarantee is untouched** — the only thing a bad
trim costs is acceptance, and the worst case for any bug in the whole scheme
is that we decode at the plain rate. That property is what makes unattended
on-device adaptation reasonable at all.

## The slice is exact and nearly free

A linear head's rows are independent: `logits[i] = W[i]·x`. Dropping rows does
not perturb the rows that remain, so **no retraining, no calibration, no
approximation** — it is an exact slice.

Better, it is a *byte* slice. `token_embd.weight` is Q4_0 with `ne[0]=1024`,
so each row is exactly 32 blocks × 18 B = **576 bytes**. Building a
16,384-row head is a **9 MiB row gather**. Milliseconds. No dequantize, no
requantize, no numerical change of any kind.

## What it buys (12B assistant, 262144 vocab)

| | full | K=16384 | saved |
|---|---|---|---|
| head resident on device (f16) | **512 MiB** | 32 MiB | **480 MiB** |
| transient host peak in `mtp_up_h` (f32+f16 staging) | **1.5 GiB** | 96 MiB | 1.4 GiB |
| head in the GGUF (Q4_0) | 144 MiB | 9 MiB | 135 MiB |
| `d2t` / id list added | — | 64 KiB | — |

On Orin NX that is unified memory: CPU *and* GPU headroom. The transient peak
matters too — `mtp_up_h` mallocs `n*4` floats plus `n*2` halves before freeing
both, so load-time peak drops by more than the resident saving.

Speed, **estimated** (assumed achieved bandwidth for the target pass, not
measured): a draft step becomes ~2.25× cheaper at K=32768, worth roughly 8%
end-to-end at one draft, ~12% at two, ~17% at three. Before writing any code,
measure it: `mtp-kernel.cuh:60` already prints `verify profile ... head %.1fms
ea` for the target; the same instrumentation around `mtp_draft_launches` gives
the draft head's real share on Orin in an afternoon.

Savings scale with head width, so the narrower E-series heads save
proportionally less in absolute terms — but the same *fraction* of the draft.

## Prior art: what HauhauCS actually shipped

Their `d2t` is readable, so the selection is inspectable. It is a **hybrid**:

- rows 0–1 are **pinned specials** (`<|endoftext|>`, `<|im_end|>`)
- **all 24,832 ids in `0..24831` are retained wholesale** — 76% of the 32,768
  budget spent as a blanket id-prefix
- the remaining **7,936 are corpus-selected** from higher ids, and are plainly
  real: `persistence`, `TLS`, `mitigation`, `escalate`, `hashes`, `incident`,
  `metrics`, `Solve`, `formulate`, `velte`, `/loading`, `.lower` — a
  security/ops/coding-assistant flavour, plus multilingual scraps

Coverage on our 250k-token technical corpus:

| subset | coverage of token occurrences |
|---|---|
| naive first-32768 ids | 92.61% |
| **HauhauCS `d2t`** | **95.28%** |

So the statistics are real, but three-quarters of the budget went to a prefix,
and it bought only +2.67 pp. 4.72% of occurrences still fall outside — about
1 token in 21 the draft can never propose — a hard ceiling on acceptance.

## How little we actually need

Same corpus, Gemma tokenizer — 265,574 tokens, **11,137 distinct**:

| K rows | coverage | head f16 on device |
|---|---|---|
| 4,096 | 95.50% | 8 MiB |
| 8,192 | 98.89% | 16 MiB |
| **16,384** | **100%** | **32 MiB** |
| full 262,144 | 100% | 512 MiB |

99% needs K=8,482; 99.9% needs K=10,872.

A **smaller domain-tuned trim beats a larger generic one on both axes**:
K=16,384 fits our traffic better than HauhauCS's 32,768 fits theirs, at 32 MiB
instead of 512.

## The adaptive scheme

Count vocabulary during normal use; rebuild the trimmed head when the device
is idle; the longer a unit runs in one environment, the faster it gets. Keep
the full head so a redeployed unit can reset.

Three refinements, each from a measurement below.

**1. Do not rewrite a GGUF. Persist the id list and gather at load.**
K × 4 B ≤ 64 KiB. `mtp_up_h` already walks the tensor to build the device
buffer; gathering a subset of rows is a small change to it. This removes the
trimming tool, the unattended file-replace on embedded storage, the corruption
window, and the second copy — the original GGUF is the only model file, always
intact. "Reset to full" becomes `rm idlist.bin`.

**2. Never turn the statistics off.** Split the corpus into observed /
held-out and measure coverage on the unseen remainder:

| observed tokens | shuffled (pure sample size) | sequential (with domain drift) |
|---|---|---|
| 10,000 | 86.1% | 72.4% |
| 25,000 | 91.6% | 79.7% |
| 50,000 | 94.6% | 82.4% |
| 100,000 | 96.7% | 88.7% |
| 200,000 | 98.1% | **83.2%** ↓ |

Coverage went *down* between 100k and 200k observed tokens once the domain
drifted. A rule that freezes the set after a goal is met would bake that in
permanently, with no signal left to detect it. Counting costs 1 MiB
(262,144 × u32) and one increment per emitted token. Cap the *aggressiveness*,
never the *measurement*.

**3. Union a generic base with local observations — do not replace it.**
Learned-local reaches 98.1% under stationary traffic but only ~83–89% under
drift — **worse than HauhauCS's generic 95.3%**. Ship a broad base set with
the model and add locally observed tokens on top. This is the argument for
HauhauCS's prefix-plus-corpus hybrid, which looks naive until you measure
drift.

**Add a miss-rate counter.** Per emitted token, is it in the current set? A
32 KiB bitmap, one lookup. It yields the acceptance ceiling being lost, a
drift alarm, and an automatic re-trim trigger — which makes manual "reset on
redeploy" unnecessary rather than something an operator must remember.

**Trigger on distinct-tokens-observed, not on K or a calendar.** In the table
above, K=8192, 16384 and 32768 give *identical* coverage until 200k observed
tokens: until more than K distinct tokens have been seen, "top-K" is just
"everything seen so far" and the budget is irrelevant. Count what should be
drafted — the target's *emitted* tokens — not prompt tokens.

## Implementation sketch

Estimated **~45–60 lines across three files** (from reading the code, not from
implementing it):

| file | change | lines |
|---|---|---|
| `src/mtp-internal.h` | `n_vocab_draft` + `const int32_t *d2t` in `struct mtp` | ~3 |
| `src/mtp.c` | optional `d2t` at open, validate rows vs head (`:171`); size `logits` malloc from it; CPU draft path (`:348-354`) over draft vocab, then `best = d2t[best]` | ~15 |
| `src/cuda/mtp-kernel.cuh` | `d2t` field + upload in `mtp_cuda_init`; four launch bounds (`:260,323,325,326`); a 3-line map kernel inside `mtp_draft_launches` so it stays in the captured graph; one `cudaFree` | ~28 |

Unchanged, and this is what keeps it small:

- `verify_head_spec` — verification runs on the target's full vocab
- `mtp_draft_chain_device` — re-embeds via the *target's* `token_embd`
  (`:385`), so once `d_tok` is a target id the chain works as-is
- the draft GGUF's `token_embd` is used **only** as the tied LM head; input
  embedding always comes from `m->ctx`. That is what makes trimming safe

Keep `d2t` optional (absent → `n_vocab_draft = n_vocab`, identity map) so
existing GGUFs keep working. Note the trimmed head changes `mtp_cuda` state and
the captured CUDA graph, so a hot swap must go through `mtp_free_device` and
re-init.

**Pin the specials** regardless of frequency — BOS/EOS/pad and the multimodal
sentinels. Gemma 4 has `[multimodal]` at id 5 and `suppress_tokens` at
258882–258883, which a pure top-K would drop.

## Caveat on these numbers

The corpus is `little-gemma/docs/*.md` + `src/*.c` + `src/cuda/*.cuh` +
llama.cpp's README — narrow, and the sequential split concatenates three
genuinely different domains back to back. Real traffic (dictation, image
description, multilingual, prose) will have more distinct tokens, so treat
11,137 as a floor and 83% as a pessimistic bound on drift. The direction is
what matters; the numbers to trust are the ones the instrumented server
produces.

## Open questions

- ~~The draft head's real ms share on Orin — measure before building.~~
  Measured below: ~56% of draft time, ~14% of a round.
- Whether the acceptance loss at a given K is worth the memory on E-series
  heads, where the head is a smaller absolute share.
- Whether the base set should ship per-locale.

---

# Phase-1 measurements (2026-08-19)

Everything above the line was estimated; this section is measured. Tools:
`little-gemma-tools/script/vocab-trim/` (`trim-head.py`, `vocab-stats.py`)
and `bench/mtp-vocab-ab.sh`. **Zero engine changes** — the trick is that a
*prefix* trim (keep rows `0..K-1`) makes draft row index == target token id,
the identity `d2t`, and `mtp_open` takes `n_vocab` from the head tensor's own
dims. Prefix trims turn out to be a measurement instrument rather than the
mechanism (first correction below), but they measure exactly the quantities
the scheme needs: draft cost vs K, byte-identity, memory, and the
acceptance-vs-coverage law.

Setup: 12B target + 12B assistant head, greedy, LG_MTP_N=3. Orin NX 16 GB
(`run-cuda-i8`, Q6_K head vintage, 840 B/row) and an RTX PRO 4500 dev box
(Q4_0 head vintage, 576 B/row). Three prompts (prose / C code / French),
3 turns each, first server turn discarded (graph warmup). The stats lines'
"ms ea" are per *draft* (2 drafts/round at block-3); per-round figures below
are 2×.

## Corrections to the estimates above

1. **Gemma's vocabulary is not frequency-ordered.** `.` is id 236761, `,` is
   236764. The "naive first-32768 = 92.61%" row above is a *Qwen* property
   (that is where HauhauCS's prefix hybrid comes from — and why it looked
   naive-but-workable). On this corpus, Gemma prefix-32768 covers **57.4%**.
   A Gemma trim lives or dies by the selected id list + `d2t`; there is no
   cheap prefix shortcut. (Top-K selection is unaffected: 100% at K=8,928 on
   the re-tokenized corpus, vs 11,137 in the table above — corpus drift since
   the doc was written, same shape.)
2. **The Q4_K_M assistant file's `token_embd` is Q6_K** (840 B/row), not
   Q4_0/576 B — that describes the QAT-vintage file. Still an exact
   row-aligned byte slice; `trim-head.py` derives row bytes from the actual
   type and byte-verifies every untouched tensor. Device f16 numbers are
   unchanged.
3. **The full 12B head does not arm on the Orin at all.** With the 12B target
   resident, `mtp_cuda_init` fails (`mtp: device upload failed, drafting
   disabled`) — the MemAvailable watermark shows the ~1.5 GiB `mtp_up_h`
   host staging spike before the failure. The server then silently pays the
   3-wide verify **for zero accepted drafts**: 7.49 tok/s, *slower than plain
   decode*. So on the 16 GB Orin the trim is not an optimization, it is the
   difference between 12B MTP existing and not — and the "gather at load"
   refinement above (which removes the staging and shrinks the resident head)
   is load-bearing, not hygiene.

## H1 — byte identity: PASS

Every reply, every variant, both machines, byte-identical to plain decode
(sha1 over 9 turns × 4 trims on Orin, 4 trims local). The safety argument
("a trimmed head can only lose acceptance") is now measured, not argued.

## H2 — cost vs K

Orin (per round = 2 drafts + one 3-wide verify; verify is flat ~132 ms
per round across all variants, as it must be — the trim never touches it):

| variant | draft ms/round | acc % | out tok/s | vs plain |
|---|---|---|---|---|
| plain decode | — | — | 8.26 | — |
| full head (failed arm, verify-only) | ~0 | 0.0 | 7.49 | **−9%** |
| K=8192 | 9.8 | 34.4 | 11.75 | +42% |
| K=16384 | 10.1 | 38.4 | 12.29 | +49% |
| K=32768 | 10.9 | 41.7 | 12.69 | +54% |
| K=65536 | 12.4 | 45.0 | 12.98 | +57% |

Draft cost is linear in K: ~1.3 ms/round per 57k rows, i.e. a full head
would draft at ~21.3 ms/round → the head is **~56% of draft time** (the
traffic model above said 63.5% at assumed bandwidth) and the whole draft is
~14% of a round. Extrapolated full-head-if-it-fit: ~14.4 tok/s.

Dev box (RTX PRO 4500), same prompt, generate mode: full head 81.2 tok/s,
K=32768 **109.3 tok/s (+35%)** despite acceptance dropping 65.9→45.1% — the
draft's warp-per-row head matvec is latency-bound there and cost MORE than
the 12-ms verify (16.6 ms/round), so trimming pays even at prefix coverage.
tok/s plateaus ~107–109 for K=8192..65536: below 64k rows the acceptance
loss eats exactly what the cheaper draft saves. Selected-K recovers it.

## H3 — acceptance = full-acceptance × coverage: CONFIRMED

Prefix-K coverage of the emitted streams and measured acceptance, Orin:

| K | stream coverage | acc measured | implied full-vocab acc |
|---|---|---|---|
| 8192 | 58.1% | 34.4% | 59.3% |
| 16384 | 64.1% | 38.4% | 59.9% |
| 32768 | 69.4% | 41.7% | 60.1% |
| 65536 | 74.1% | 45.0% | 60.7% |

The implied full-vocab acceptance is **constant to within 1.4 pp** — the
acceptance-ceiling model holds, and misses cost almost exactly their
frequency (no compounding penalty visible at block-3). Same result locally:
predicted 48.3% vs measured 45.1% at K=32768. Per-flavor coverage at
prefix-32768: prose 77.4%, C code 66.8%, French 64.8%.

## H4 — memory

Steady-state MemAvailable deltas match the f16 head sizes (k8192 vs k65536:
91 MiB observed vs 112 expected, page-cache noise); the full-head variant's
load watermark dips ~1 GiB below plain's before its arm fails, the staging
spike made visible. The doc's 480 MiB resident + 1.4 GiB transient savings
at K=16384 stand, with the addendum that on Orin the transient (or the
512 MiB device alloc it feeds) is what kills the full head.

## Drift table, reproduced

`vocab-stats.py heldout` on the re-tokenized corpus reproduces the *shape*
of the table above — sequential (drifting) coverage is 6–14 pp worse than
shuffled at every budget, and K is irrelevant until >K distinct ids are
seen — but not the 200k dip, which depended on that corpus build's domain
order. The direction (never freeze the statistics; union a base set) stands.

## Phase-2 go/no-go: GO — d2t selected-K is worth the ~50 engine lines

What prefix trims cannot give (Gemma correction 1), selected-K does. From
the corpus counts + pinned specials (`0-5,100,101,105,106,258880-258884`),
`vocab-stats.py select -k 16384`:

- **cold** (corpus-selected, out-of-domain traffic): 84.0% coverage of the
  Orin A/B's emitted streams — better than prefix-65536 at 1/8 the rows;
- **adapted** (re-selected after observing that traffic): 100% of it, i.e.
  the heldout table's 95–98% is the honest unseen-traffic estimate.

Projected on the measured round model (132 ms verify + 10.1 ms draft at
K=16384, acc ≈ 0.60 × 0.95..1.0): **~14.9–15.5 tok/s, ~1.8× plain decode
(8.26)** — beats the best prefix trim (12.98, +57%), beats the extrapolated
full head (14.4) while saving 480 MiB resident, and turns a configuration
that today runs *slower than plain* into the fastest one. The engine change
stays as sketched above (`idlist.bin` sidecar, gather in `mtp_up_h`,
`best = d2t[best]`); `vocab-stats.py select` already emits its input.

---

# Phase-2 measurements (2026-08-19): d2t selected-K, built

## The change came in under the sketch

Not ~45–60 lines across three files — **~40 lines in `mtp.c`, 6 in
`mtp-internal.h`, 2 in `mtp-kernel.cuh`**, because the gather moved to
`mtp_open`: `LG_MTP_IDS=idlist.bin` (u32 LE target ids, `vocab-stats.py
select` output) gathers the K head rows into a synthetic `gguf_tensor` and
swaps `t->head` to it, with `t->n_vocab = K`. Everything downstream — the
CPU draft, `mtp_up_h`'s upload *and its staging transient*, the logits
buffers, the launch bounds, the argmax — was already sized off those two, so
none of it changed; no map kernel, no CUDA-graph edits. The draft argmax now
returns a row index; both backends map it through `d2t` on the host at the
readback (`best >= 0 && t->d2t ? t->d2t[best] : best`). Absent `LG_MTP_IDS`,
nothing changes. A bad id list fails the open loudly. "Reset to full" =
unset the env var; the GGUF is never touched.

## 12B on Orin NX (same harness, same prompts, rebuilt binary, H1 PASS)

| variant | acc % | draft ms/round | out tok/s | vs plain |
|---|---|---|---|---|
| plain decode | — | — | 8.26 | — |
| prefix-16384 | 38.4 | 10.1 | 12.29 | +49% |
| selected-16384, cold corpus list | 56.7 | 10.1 | 14.85 | +80% |
| **selected-16384, adapted** | **70.0** | 10.1 | **16.70** | **+102%** |

Byte-identical output throughout, 32 MiB f16 head instead of a 512 MiB one
that cannot even load. The adapted list = corpus counts unioned with the
device's own observed replies, re-selected at the same K — i.e. one cycle of
the adaptive scheme above, run by hand. **2.0× plain decode.**

Two things the phase-1 model got instructively wrong, in opposite
directions, both worth keeping:

- **Full-vocab acceptance is ~70%, not ~60%.** The phase-1 "implied
  full-vocab acceptance" divided prefix-trim acceptance by frequency
  coverage — but prefix trims miss precisely the high-id punctuation
  (`.` `,` `\n`) that is *easiest* to draft, so frequency-weighting
  underestimates. It was a lower bound, and selected-K recovered the
  difference: misses aren't fungible, and a selection that keeps the easy
  frequent tokens is worth more than its coverage number says.
- The round model then lands exactly: 132 ms verify + 10.1 ms draft at
  acc 0.70 → 16.9 tok/s predicted, 16.70 measured.

Dev box, same change: full head 77.9 tok/s → cold 122.5 (acc 58.1%,
predicted 56.6%) → adapted **126.3 tok/s (+62%)** at acc 64.9% vs the full
head's 65.9%. The CPU backend (`run`) exercises the same gather + d2t map
and stays byte-identical.

## E-series on Orin NX — the models that matter

E2B and E4B assistant heads are 256-wide: 128 MiB f16 on device, and (per
the measured draft times) still ~55% of the draft's cost. Unlike the 12B,
their full heads DO arm on the Orin — so here the full head is a live
baseline, and the selected trim still beats it (H1 PASS on both, greedy,
byte-identical throughout):

**E2B** (the voice-pipeline model):

| variant | acc % | draft ms/round | out tok/s | vs plain | vs full head |
|---|---|---|---|---|---|
| plain decode | — | — | 34.57 | — | — |
| full head (128 MiB f16) | 50.2 | 6.6 | 50.20 | +45% | — |
| prefix-16384 | 30.5 | 3.0 | 44.42 | +29% | −12% |
| selected-16384, cold | 43.0 | 3.0 | 51.31 | +48% | +2% |
| **selected-16384, adapted** | **51.5** | 3.0 | **55.95** | **+62%** | **+11.5%** |

**E4B**:

| variant | acc % | draft ms/round | out tok/s | vs plain | vs full head |
|---|---|---|---|---|---|
| plain decode | — | — | 17.90 | — | — |
| full head (128 MiB f16) | 50.5 | 6.6 | 29.39 | +64% | — |
| prefix-16384 | 30.5 | 3.1 | 24.82 | +39% | −16% |
| selected-16384, cold | 44.3 | 3.1 | 29.11 | +63% | −1% |
| **selected-16384, adapted** | **51.7** | 3.1 | **31.40** | **+75%** | **+6.8%** |

Two results worth stating plainly:

- **The adapted trimmed head out-drafts the full head it was cut from**
  (E2B 51.5% vs 50.2% acceptance, E4B 51.7% vs 50.5%). This is not noise —
  the rounds are identical (byte-identical replies). When the full head's
  argmax is an out-of-domain token, it is almost always a *wrong* draft;
  restricting the argmax to a well-chosen subset sometimes corrects it. The
  id list acts as a free domain prior. So on the E-series, the trim is not
  a memory-for-acceptance trade: it is +6–12% decode speed AND 120 MiB back
  AND equal-or-better acceptance, per model.
- This answers the open question on E-series heads: yes, worth it — there
  is no acceptance loss to weigh once the list is adapted.

Prefix trims lose to the full head on the E-series (−12/−16%): with the
head only ~½ the draft and the draft only ~⅙ of a round, the acceptance
collapse (50→30%) dominates. Selection is the mechanism, exactly as the
Gemma-ordering correction predicted.

The E4B assistant GGUF, incidentally, carries upstream's own
`mtp.centroids.weight` and `mtp.token_ordering.weight` (I32, 262144) —
vestiges of a vocabulary-ordering mechanism little-gemma never reads. The
idea has upstream pedigree; the sidecar id list is just the deployable form.

## QAT pairs (~/gguf on the Orin): the full head arms, and still loses

The QAT targets are what actually ships, and the 12B QAT target is ~400 MB
smaller than the Q4_K_M — enough that **the full 512 MiB head now arms**.
That upgrade gives the comparison the Q4_K_M 12B couldn't: a live full-head
baseline. (Heads in `~/gguf` are the Q4_0 conversion vintage, `nextn.`
tensors; `mtp-e4b-patched.gguf` differs from the unpatched file only in the
arch-string separator, a workaround `mtp_open` no longer needs.)

**12B QAT** (H1 PASS; plain is +19% over Q4_K_M to begin with — the QAT win):

| variant | acc % | draft ms/round | out tok/s | vs plain | vs full head |
|---|---|---|---|---|---|
| plain decode | — | — | 9.80 | — | — |
| full head (512 MiB f16) | 71.3 | 21.3 | 16.05 | +64% | — |
| prefix-16384 | 40.2 | 10.0 | 12.89 | +32% | −20% |
| selected-16384, cold | 59.6 | 9.9 | 15.66 | +60% | −2% |
| **selected-16384, adapted** | **72.2** | 9.9 | **17.48** | **+78%** | **+8.9%** |

**E4B QAT**:

| variant | acc % | draft ms/round | out tok/s | vs plain | vs full head |
|---|---|---|---|---|---|
| plain decode | — | — | 20.70 | — | — |
| full head (128 MiB f16) | 54.4 | 6.6 | 33.16 | +60% | — |
| prefix-16384 | 30.8 | 3.1 | 27.19 | +31% | −18% |
| selected-16384, cold | 46.4 | 3.0 | 32.46 | +57% | −2% |
| **selected-16384, adapted** | **55.5** | 3.0 | **35.49** | **+71%** | **+7.0%** |

**E2B QAT** (measured 2026-09-02, Orin NX, clocks pinned; held-out prompt):

| variant | acc % | draft ms/round | out tok/s | vs plain | vs full head |
|---|---|---|---|---|---|
| plain decode | — | — | 34.5 | — | — |
| full head | 48.0 | 3.3 | 49.1 | +42% | — |
| selected-16384, cold | 39.6 | 1.5 | 49.3 | +43% | +0.4% |
| **selected-16384, adapted** | **48.2** | 1.5 | **54.1** | **+57%** | **+10.2%** |

E2B completes the QAT trio and shows the **largest** relative gain. Its draft
is already the cheapest (3.3 ms/round), so the cold list is a wash (+0.4%):
acceptance falls to 39.6% because a code/docs corpus covers only ~84% of what
E2B emits on the held-out domain. One adaptation cycle (base ∪ observed
traffic) restores coverage, acceptance returns to 48.2% — matching the live
full head's 48.0% — and the halved draft then lands **+10.2%** over it. Same
mechanism as 12B/E4B; the smaller the model, the more the draft-time saving
dominates the fixed verify cost.

Three loops close here:

- The 12B full head's steady-state draft is **21.3 ms/round — exactly** the
  phase-1 extrapolation from the prefix-K slope. And full-vocab acceptance
  is 71.3%, confirming the ~70% the adapted list implied. The round model
  then reproduces every row to two decimals (e.g. full: 129.8 ms verify +
  21.3 draft at acc 0.713 → 16.06 predicted, 16.05 measured).
- The adapted list beats the live full head on 12B too (+8.9%, acc 72.2 vs
  71.3) — now shown on all three models against a working full head, not
  just against a broken one.
- Memory watermarks: the full head costs ~1 GiB more headroom at load
  (4538 vs 5501 MiB available) — armed or not, that pressure is what the
  16 GB unit feels when the voice stack loads next to it. (Steady-state
  MemAvailable is page-cache-noisy across server restarts; the watermark is
  the honest column.)

## Block depth: the cheap draft moves the optimum (LG_MTP_N sweep)

Prior experience had prose peaking at N≈3 and code at N≈4. With the adapted
trimmed head the draft is ~2× cheaper and acceptance a little higher — both
push deeper — so N ∈ {2,3,4,5} was swept at sel16ka on all three models
(`-DLG_MTP_N` builds; H1 stayed byte-identical at every N, as greedy verify
must). Blended tok/s, with per-domain peaks:

| model | N=2 | N=3 | N=4 | N=5 | best |
|---|---|---|---|---|---|
| E2B | 53.1 | **56.0** | 54.0 | 47.9 | **3** (all domains) |
| E4B QAT | 32.3 | 35.5 | **36.8** | 32.9 | **4** (all domains; code +5% over N=3) |
| 12B QAT | 15.5 | 17.5 | **18.9** | 17.5 | **4** (all domains; code 19.9, +10%) |

So the hypothesis held for the two bigger models — and more than expected:
the cheap draft didn't just move code to 4, it *erased the prose/code split*
(prose joined code at N=4 on E4B and 12B). E2B stays at 3: its per-draft
acceptance is the lowest (p₁ ≈ 0.60–0.67), so the 3-deep chain product is
too small to pay even a cheap fourth slot. N=5 is past the peak everywhere —
the chain decays and the 5-wide verify takes a per-round jump (12B:
111/130/139/166 ms per round at N=2/3/4/5).

Chain probabilities solved from the sweep (acc₂ = p₁, then incrementally;
p₃/p₄ are subtractive estimates from 3-turn samples — read them as trend,
not precision): the chain decays *slowly* — p₂ ≈ p₁ − 0.05 or less (12B
code: p₁ 0.85, p₂ 0.87), which is why depth pays once drafting is cheap.
The re-embedding of each draft through the target's own `token_embd` keeps
the chained inputs honest.

New best configurations (adapted 16k list): **E2B N=3 56.0 · E4B-QAT N=4
36.8 · 12B-QAT N=4 18.9 tok/s** — the 12B now at 1.93× its plain rate and
+18% over its own full head at N=3.

## Where this leaves the adaptive scheme

Everything the scheme above needs is now measured or built: the counting
tools, the selection with pinned specials, the union-with-base behaviour
(cold list = the shipped base; adapted list = base ∪ observed, and one
adaptation cycle recovered full-head acceptance on every model), and a
hot-swap path (`mtp_free_device` + re-open, or just restart the server —
the gather is milliseconds). What remains is plumbing, not physics: count
emitted ids in `run.c`, re-run `select`, reload. The measured per-model
wins to defend: **12B 8.26→16.70 (2.02×), E4B 17.90→31.40 (1.75×),
E2B 34.57→55.95 (1.62×) tok/s**, all byte-identical, all with the head at
8–32 MiB instead of 128–512 — and on the shipping QAT pairs, **12B
9.80→17.48 and E4B 20.70→35.49**, beating even their working full heads
by 7–9%.
