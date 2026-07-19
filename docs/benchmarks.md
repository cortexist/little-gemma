# Benchmarks

Canonical performance numbers vs llama.cpp — one methodology, both devices,
all three models, measured in one sitting. **These tables supersede every
number published before 2026-07-16** (a reconciliation of the older,
mutually inconsistent figures is at the bottom).

- **Date / builds:** prefill 2026-07-16, decode 2026-07-17 (after the KV-split
  fix — prefill is untouched by it and was re-measured as the control:
  A5000 E2B 7,260 → 7,307, i.e. unchanged). **E4B/12B QAT rows and the MTP
  section: 2026-07-19** (the 2-row MTP verify kernel; llama.cpp fork
  `cd5ad883e` on the Orin, `74ade5274`/b9672 on the A5000 for those pairs).
  little-gemma `7980037`+ (`run-cuda-i8`);
  llama.cpp = the [Cortexist fork](https://github.com/cortexist/llama.cpp)
  (`83efbcc79` on the Orin, `10306b8fd` on the A5000), CUDA build.
- **Hardware:** Jetson Orin NX 16GB (Ampere sm_87, integrated, zero-copy
  weights, **pinned `jetson_clocks`, MAXN** — a reboot resets the pin; DVFS
  numbers are lower) and RTX A5000 24GB (Ampere sm_86, Windows/WDDM, stock
  clocks — the serve harness runs warm back-to-back turns; the clock probe
  shows sustained 1695 MHz boost during decode).
- **Models:** Gemma 4 **E4B and 12B QAT q4_0**
  (`gemma-4-{E4B,12B}-it-qat-UD-Q4_K_XL`, unsloth — **the defaults since
  2026-07-19**, with their matched MTP heads `mtp-gemma-4-{E4B,12B}-it.gguf`)
  and E2B QAT q4_0 (`gemma-4-E2B-it-qat-UD-Q4_K_XL`). The superseded
  Q4_K_M rows are kept for reference. The 12B QAT load on the 16GB Orin
  needs a page-cache drop first (the q4_0 repack's device copies race the
  blob's page cache for nvmap; the settle harness does it).

## Methodology

**little-gemma** is measured on its production path: the socket server
(`-s`), one connection per turn, first turn discarded (warmup), numbers read
from the per-turn `turn:` stderr stat. Prefill = six 929-token turns
(`bench/line929s.txt`). Decode = five turns of a prose question whose greedy
reply runs to the 1,024-token cap, i.e. **sustained decode averaged over
~0–1k context depth**. Harnesses: `bench/settle-bench.sh` (Orin),
`bench/settle-bench.ps1` (Windows). Greedy decoding; turn rates are
reproducible to ±1% (greedy replies are identical across turns).

**llama.cpp** is measured with `llama-bench`, best of `-fa 0/1`, `-r 3`:
prefill = `pp929`, decode = `tg128 -d 512` (generation at 512-token context
depth, matching the serve turns' average depth; on these models depth costs
llama only 1–4% because sliding-window attention caps most layers' KV).
The 12B on the Orin needs `drop_caches` + `-mmp 0` first.

Two deliberate asymmetries, both of which *favor llama.cpp*: the serve
numbers include little-gemma's per-token serving overhead (detokenize +
socket write; visible on Windows, negligible on Linux), while `llama-bench`
measures bare kernels with no serving stack at all; and llama gets
best-of-two attention configs. The ratios below are therefore conservative.

## Decode (tokens/s, batch 1)

Post the **2026-07-17 KV-split fix** (`SPLIT_KEYS 1024 → 64`; see
[upstream-llama-study.md §4](upstream-llama-study.md) and the journal):

| device | model | little-gemma (serve) | llama.cpp (tg128@d512) | ratio | was |
|--------|-------|---------------------:|-----------------------:|------:|----:|
| **Orin NX** | **E4B QAT** | **20.7** | 18.7 | **1.11×** | — |
| **Orin NX** | **12B QAT** | **9.8** | 9.05 | **1.08×** | — |
| **Orin NX** | E2B QAT | 34.5 | 37.4 | 0.92× | 0.80× |
| Orin NX | E4B Q4_K_M *(superseded)* | **17.9** | 14.1 | **1.27×** | 1.17× |
| Orin NX | 12B Q4_K_M *(superseded)* | **8.3** | 7.4 | **1.12×** | 1.08× |
| **A5000** | **E4B QAT** | 134 | 136.6 | 0.98× | — |
| **A5000** | **12B QAT** | 70.7 | 70.7 | **1.00×** | — |
| **A5000** | E2B QAT | **213.9** | 209.3 | **1.02×** | 0.70× |
| A5000 | E4B Q4_K_M *(superseded)* | **117.6** | 116.1 | **1.01×** | 0.81× |
| A5000 | 12B Q4_K_M *(superseded)* | 58.0 | 62.5 | 0.93× | 0.80× |

The QAT switch (2026-07-19) speeds up **both** stacks — Orin E4B 17.9 →
20.7 for us, 14.1 → 18.7 for llama; 12B 8.3 → 9.8 vs 7.4 → 9.05 — and
narrows the Orin ratios for the same reason E2B QAT sits at 0.92×: llama's
q4_0 matvec is stronger than its q4_K one, so q4_0 models play toward its
strength. Faster absolute tokens win over a prettier ratio; QAT is the
default. On the A5000 the switch **closes the last decode gap**: 12B goes
from 0.93× (Q4_K_M) to exact parity at 70.7 tok/s.

**On the edge device this project targets, decode leads llama.cpp on both
models that matter to it (E4B 1.27×, 12B 1.12×) — and desktop decode is now
at parity too** (E2B 1.02×, E4B 1.01×). Everything *around* the matmul —
launch overhead, syncs, norms, the PLE path, the captured CUDA graph — is
what a few thousand readable lines can do leanly.

The `was` column is the same harness one day earlier, and the delta is one
constant. The decode attention splits the KV walk across `n_split` blocks per
head; `SPLIT_KEYS` caps the keys **one block walks**, so it must be small
enough that per-block work stays flat as context grows and the *block count*
absorbs the depth. At 1024 it never did: below 1024 tokens `n_split == 1`, the
split degenerated to one block per head, and each block's walk grew linearly
with depth. Measured droop, shallow → 930-deep (A5000 E2B): **−14.3% before,
−0.6% after** (llama.cpp's own fattn loses ~1%, for exactly this reason — it
caps per-block work at `nbatch_fa`=128 keys and grows the grid instead).

The gain tracks attention's share of decode, which is why it is largest on the
smallest model (E2B +44.7% A5000) and smallest on the largest (12B +3.8%
Orin), and larger on the 64-SM A5000 (which the old constant left badly
under-filled) than the 8-SM Orin (where `n_head == 8 == SM count` filled the
board by accident, which is why the Orin lead existed at all and why the droop
went unnoticed).

The one decode gap left anywhere is the legacy quants: E2B/12B *Q4_K_M*-era
rows where llama's arch-tuned matvec leads — on the QAT defaults every
pair is at parity or ahead. [MTP](mtp.md) multiplies on top of these
numbers, byte-identical output — see the dedicated section below.

## Prefill (prompt tokens/s, 929-token turns)

| device | model | little-gemma | llama.cpp (pp929) | ratio |
|--------|-------|-------------:|------------------:|------:|
| Orin NX | **E4B QAT** | 474 | 553 | **0.86×** |
| Orin NX | **12B QAT** | 193 | 232 | 0.84× |
| Orin NX | E2B QAT | 834 | 1,020 | 0.82× |
| Orin NX | E4B Q4_K_M *(superseded)* | 426 | 524 | 0.81× |
| Orin NX | 12B Q4_K_M *(superseded)* | 174 | 217 | 0.80× |
| A5000 | **E4B QAT** | 4,335 | 5,254 | 0.83× |
| A5000 | **12B QAT** | 2,067 | 2,365 | **0.87×** |
| A5000 | E2B QAT | 7,222 | 8,785 | 0.82× |
| A5000 | E4B Q4_K_M *(superseded)* | 3,703 | 4,846 | 0.76× |
| A5000 | 12B Q4_K_M *(superseded)* | 1,782 | 2,207 | 0.81× |

**Prefill is 0.8× llama.cpp — 0.76–0.82 across six pairs on two devices,
one consistent number.** The 2026-07 prefill campaign took it from ~0.2× to
here (A5000 12B 533 → 1,782; Orin E4B 192 → 426) and measured the residual to its floor:
what remains is llama.cpp's ground-up MMQ instruction schedule and
flash-attention architecture, priced (wholesale kernel adoption) and
declined. The full campaign — every kept lever and every falsified one — is
[prefill-performance-journal.md](prefill-performance-journal.md);
in interactive serving the gap rarely shows (turns are short, `-sys` removes
the skills re-prefill, the GPU encoder removed the image one), but on very
long documents llama.cpp still wins the wait.

## MTP speculative decoding (2026-07-19, the 2-row verify kernel)

MTP verifies a block of drafted tokens as one small batch (B = `LG_MTP_N` =
2–5). That batch is **its own kernel regime** — too narrow for the
tensor-core prefill path (and barred from it: verify must argmax
bit-identically to decode), and priced per-column on the decode matvec. The
2026-07-19 **2-row verify kernel** (one warp owns two output rows; each
column's activation registers load once and dot against both rows' weights)
cut the Orin's verify cost to **1.01× / 1.13× / 1.24×** a decode step at
B=2/3/4 — llama.cpp's same-shape forwards measure 1.04× / 1.21× / 1.49×
(`llama-bench pp2..4` vs `tg`) — and took MTP from trailing `llama-server`'s
speculative decoding to **leading it on every content type measured**.
Output stays byte-identical to plain decoding (gated on both devices, text
and image turns; acceptance rates unchanged to the digit).

**Orin NX, E4B QAT, serve mode, same-day pairs** (plain decode 20.7; llama
plain 19.0). llama-server runs `--spec-type draft-mtp`; acceptance column is
little-gemma's block-3 chained rate:

| content | accept | little-gemma +MTP | llama-server +MTP |
|---------|-------:|------------------:|------------------:|
| free prose | 37% | **29.9 tok/s (1.40×)** | 24.5 (1.29×) |
| code | 76% | **40.7 (1.91×)** | 31.8 (n-max 1) / 34.1 (n-max 3) |
| counting | 99% | **48.6 (2.33×)** | — |
| image description (image-only turn) | 48% | **31.5 (1.52×)** | 28.8 (1.54×, plain 18.6) |

Block depth (`-DLG_MTP_N`, default 3) is a content dial (Orin tok/s):

| content \ block | 2 | 3 | 4 | 5 |
|-----------------|--:|--:|--:|--:|
| prose | 29.3 | **29.9** | 27.6 | 24.0 |
| code | 35.4 | 40.7 | **45.9** | — |
| counting | 38.1 | 48.6 | **57.8** | 57.3 |

Prose peaks at block 2–3, code at 4, counting at 4–5; the shipped 3 is a
good compromise everywhere (adaptive depth is the open lever). A widely
shared *"31 tok/s E4B on a Jetson"* llama-server figure reproduces here
exactly (31.8 — code content, `--spec-draft-n-max 1`); the shipped default
beats it by 28%, block-4 by 44%. MTP also survives vision context intact —
the image row is measured on a turn containing **only** an image, and the
claim that speculation stops paying there did not reproduce on either stack.

**12B QAT** (the 1024-wide draft head lands ~51% of prose drafts vs the
E4B 256-wide head's ~37%, so MTP helps chat *more* as the model grows):
Orin plain 9.8 → prose **14.5 (1.42×)**, counting **20.4 (2.08×)**;
A5000 plain 70.7 → prose **101.5 (1.44×)**, counting **143.7 (2.03×)**.
Byte-identity gated on both.

**A5000, E4B QAT:** plain 134 → prose **168.8 (1.26×)**, counting **282
(2.10×)**. (The 2-row kernel is gated to the integrated-GPU path: on the
A5000 it helped E4B but regressed 12B, so the discrete path keeps the
prior verify kernel — per-device verdicts, as ever.)

The kernel investigation — the ncu attribution and the two levers measured
and falsified on the way (occupancy forcing, shared-memory staging) — is in
the [performance journal](performance-journal.md); the MTP mechanism is
[mtp.md](mtp.md).

## Upstream cross-check (llama.cpp b10054, 2026-07-17)

The tables above use the Cortexist fork as the llama.cpp reference. Its
upstream merge-base is **2026-04-15** (`b3d758750`), so the obvious worry is
that we benchmark against a stale llama. Cross-checked against **current
upstream** — a source build (sm_87) on the Orin, release binaries on the
A5000, identical flags and GGUFs:

**Verdict: the fork is a current-strength reference. Every ratio above
stands.**

- **Orin (both built from source, same toolchain): parity.** Decode matches
  within ±0.6% on all three models (E4B 14.15 vs 14.06 at d512, 12B 7.45 vs
  7.41, E2B 37.18 vs 37.40); the fork's prefill is ~2% *ahead* (E4B 524 vs
  511, 12B 217 vs 212).
- **A5000: upstream's 3 months of code changes are performance-neutral
  within noise.** A first pass appeared to show upstream +2.3–4.4%, but that
  compared a locally-built fork against CI release binaries. Isolating the
  variables on E4B — same CUDA 12.4 CI toolchain, April code (b8833) vs July
  code (b10054) — gives tg128@d512 **117.5 vs 114.1** and pp929@d512 **4,887
  vs 4,786**, i.e. new code marginally *slower*; the CUDA 13.x pair gives
  +2.3%. The sign flips with the toolchain, so the effect is build
  configuration and CUDA version, not llama.cpp's code.

This is corroborated by upstream's own record. The one large MMQ change in
the window — PR #24127, "CUDA: refactor MMQ kernel configuration"
(2026-07-13) — introduced the per-architecture config tables but is a
**tunability** refactor whose author states *"I am seeing no changes to
performance beyond statistical fluctuations"*; the Ampere table encodes the
same values April hardcoded. The window's one real decode win, PDL
(#22522), is gated to CC ≥ 90 (Hopper+) and cannot touch an sm_86/sm_87
Ampere part. Nothing in the window targeted Tegra: upstream's MMQ/FA work
was measured on RTX 3090/4090, P40, RTX PRO 6000, DGX Spark and AMD parts,
and Jetson appears only as breakage — issue #24457 ("FA+MTP crash … on SM87
(Jetson Orin)"), caused by FA specializations being pruned for compile time,
which *"inadvertently broke Gemma 4 E4B MTP"* (fixed in #25148, 2026-06-30).

A source-level study of what upstream changed, and which mechanisms transfer
to little-gemma, is in
[upstream-llama-study.md](upstream-llama-study.md).

## TTFT / TTFS (2026-07-02 campaign)

Client-side, both stacks, same definition: last request byte sent → first
streamed token (TTFT) / first speakable sentence (TTFS), warm server, first
turn discarded, prompt caching off, same GGUFs, token parity checked,
medians of warm turns, vs `llama-server`. (Why prefill rate and TTFT are
deliberately not 1:1 — first-decode step, chunk padding, encoders, arrival
overlap — is walked through in [prefill-vs-ttft.md](prefill-vs-ttft.md).)

| device | model | turn | little-gemma ttft (s) / prefill (tok/s) | llama-server ttft (s) / prefill (tok/s) | input tokens (lg / llama) |
|--------|-----|------|----------------------------:|----------------------------:|:-------------------------:|
| A5000 | 12B | 929-tok text | **0.51** / 1,803 | 0.53 / 1,753 | 928 / 928 |
| A5000 | E4B | 929-tok text | 0.25 / 3,717 | **0.23** / 4,070 | 928 / 928 |
| A5000 | 12B | image+question | **0.20** / 827 | 0.29 / 715 | 150 / 157 |
| A5000 | E4B | image+question | **0.13** / 1,317 | 0.30 / 1,229 | 150 / 293 |
| Orin | 12B | 929-tok text | 5.37 / 173 | **4.64** / 200 | 929 / 928 |
| Orin | E4B | 929-tok text | 2.23 / 417 | **1.91** / 488 | 929 / 928 |
| Orin | 12B | image+question | **1.15** / 132 | 2.11 / 92 | 150 / 157 |
| Orin | E4B | image+question | **0.69** / 308 | 1.65 / 210 | 150 / 293 |

The prefill column is tok/s over the whole turn: ours from the serve stat
(everything after the burst, media included), llama's from its own
server-reported `prompt_n/prompt_ms` — which *excludes* its media encoder
(visible as ttft − prompt_ms ≈ 0.25–0.43 s on the Orin image rows). The
last column is the **number of input tokens** each stack processed for the
same turn: text rows differ only by BOS accounting; the E4B image rows
differ by upstream policy (the fork upscales every image to a 252-token
minimum; mmcat sends 624×480 natively as 130). Text turns track the kernel
ratio. **Media turns invert it** — 1.5–2.4× ahead, and the 12B image rows
are near-token-equal (both encode natively), so that win is not a policy
artifact. Paced arrival (video frames over a socket, spans prefilled while
the clip streams) has no llama-server counterpart at all — those numbers
(Orin 12B six-frame video ttft 8.79 → 2.34 s) are against our own deferred
baseline.

**TTFS** (time to first *speakable* sentence — when a sentence-chunking TTS
can start talking; thought channels are not speakable and delay it at
decode rate). Same clocks and discipline, prose-shaped questions so the
first sentence is a real one:

| device | turn | little-gemma ttfs | llama-server ttfs | first sentence |
|--------|------|------------------:|------------------:|:--------------:|
| A5000 | 12B, 929-tok text | **1.27 s** | 1.28 s | 29 tok, same words |
| A5000 | E4B, 929-tok text | **0.51 s** | 0.54 s | 19 tok, same words |
| A5000 | 12B, image+question | **0.78 s** | 8.9 s | thought: ~0 vs 528 tok |
| A5000 | E4B, image+question | **0.25 s** | 4.7 s | thought: ~0 vs 466 tok |
| Orin | 12B, 929-tok text | 9.23 s | **8.47 s** | 29 tok, same words |
| Orin | E4B, 929-tok text | 3.43 s | **3.21 s** | 19 tok, same words |
| Orin | 12B, image+question | **5.22 s** | 98.6 s | thought: short vs 717 tok |
| Orin | E4B, image+question | **1.98 s** | 35.8 s | thought: ~0 vs 481 tok |

Text rows are the clean kernel story: identical prompt bytes produce the
identical first sentence on both stacks, so TTFS = TTFT + the same tokens
at each stack's decode rate. The image rows are the product story, and two
effects compound: the media-TTFT levers above, and **the thought channel** —
llama-server's chat template elicits a 466–717-token "thinking process"
from these models before the first speakable word, paid at decode rate; our
serve template gets a terse thought from the same weights. Even subtracting
their thought entirely, our media TTFS still leads — but the lesson
generalizes: for voice, the prompt template's effect on thought length
dwarfs every kernel in the stack (see `-sys` in [serving.md](serving.md)).

**2026-07-19 addendum — the QAT voice loop.** E4B QAT + MTP + the voice
`-sys` prompt on the Orin (`bench/ttft_dictate.py`, client clocks, first
turn discarded): conversational turn TTFS **0.34 s** (plain decode: 0.44),
929-token *streamed* dictation TTFS 1.05 s. With the streaming TTS's
0.10 s first-PCM leg, first audio lands **≈0.44 s** after a short question
— under the old E2B 0.65 s headline, on the bigger model; with the
persistent-whisper ASR leg (~0.36 s) the full mic-to-audio loop stays
under a second.

## Reconciliation with previously published numbers

Numbers that circulated before this page, and why they differ:

- **"Orin E4B decode 30.0 / 27.8" (prefill journal scoreboard, 2026-07-02):
  wrong — those are E2B QAT values** (today's E2B: 27.8 tail / 30.0
  sustained, exactly). The scoreboard's decode column mislabeled the model.
  E4B serve decode was and is ~16.4 pinned.
- **README decode table "A5000 E4B 118.8 / 12B 62.3, ≈1.0× llama"
  (retired):** measured on the one-shot `-p` path (256 tokens), which on
  Windows reads ~20% higher than serve mode — the serve loop pays per-token
  detokenize + socket-write overhead that the one-shot path doesn't. The
  serve numbers (93.7 / 49.7) are the product truth and match the 12B
  plain-decode 49.9 recorded in the MTP campaign on 2026-07-02. Nothing
  regressed; the harnesses differed.
- **Cortexist llama.cpp fork README (Orin E4B baseline ~12.8–13.7):**
  reproduces today at 13.6–14.2 under pinned clocks (the fork bench ran
  `-fa off`, unpinned, via server timings). Consistent with this page.
- **"E2B 25% faster than llama.cpp" (old E2B claim):** measured on the
  retired Q3_K E2B model in 2026-06; it does not transfer to the QAT q4_0
  E2B. The current picture: near parity at shallow depth (0.96×,
  2026-07-07 pinned pair), 0.80× sustained to 1k — see the E2B rows above.
- The old scoreboard's **12B Orin decode 8.7** doesn't reproduce either
  (today: 8.0 sustained / 7.5 at 929-deep tail, pinned); treat 8.0 as
  canonical.
- **Pre-2026-07-19 MTP figures** ("E4B prose ~1.1×", "~2× structured",
  "MTP is noise on conversational prose"): measured before the 2-row verify
  kernel, whose B=2–4 verify now costs near one decode step. The MTP
  section above supersedes them; the mechanism was a kernel cost, not a
  property of speculation.

## Reproducing

```
# Orin (pin clocks first; a reboot resets them)
sudo jetson_clocks
bench/settle-bench.sh lg-e4b     # also: lg-12b, lg-e2b, ll-e4b, ll-12b, ll-e2b

# Windows / A5000
bench/settle-bench.ps1 -Model <path-to.gguf> -Tag e4b
llama-bench -m <model> -p 929 -n 128 -fa 0,1 -d 0,512 -r 3
```
