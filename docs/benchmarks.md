# Benchmarks

Canonical performance numbers vs llama.cpp — one methodology, both devices,
all three models, measured in one sitting. **These tables supersede every
number published before 2026-07-16** (a reconciliation of the older,
mutually inconsistent figures is at the bottom).

- **Date / builds:** prefill 2026-07-16, decode 2026-07-17 (after the KV-split
  fix — prefill is untouched by it and was re-measured as the control:
  A5000 E2B 7,260 → 7,307, i.e. unchanged). little-gemma `7980037`+ (`run-cuda-i8`);
  llama.cpp = the [Cortexist fork](https://github.com/cortexist/llama.cpp)
  (`83efbcc79` on the Orin, `10306b8fd` on the A5000), CUDA build.
- **Hardware:** Jetson Orin NX 16GB (Ampere sm_87, integrated, zero-copy
  weights, **pinned `jetson_clocks`, MAXN** — a reboot resets the pin; DVFS
  numbers are lower) and RTX A5000 24GB (Ampere sm_86, Windows/WDDM, stock
  clocks — the serve harness runs warm back-to-back turns; the clock probe
  shows sustained 1695 MHz boost during decode).
- **Models:** Gemma 4 E4B Q4_K_M, 12B Q4_K_M, E2B QAT q4_0
  (`gemma-4-E2B-it-qat-UD-Q4_K_XL`).

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
| **Orin NX** | E4B | **17.9** | 14.1 | **1.27×** | 1.17× |
| **Orin NX** | 12B | **8.3** | 7.4 | **1.12×** | 1.08× |
| **Orin NX** | E2B QAT | 34.5 | 37.4 | 0.92× | 0.80× |
| **A5000** | E4B | **117.6** | 116.1 | **1.01×** | 0.81× |
| A5000 | 12B | 58.0 | 62.5 | 0.93× | 0.80× |
| **A5000** | E2B QAT | **213.9** | 209.3 | **1.02×** | 0.70× |

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

The one remaining decode gap is **A5000 12B at 0.93×** — the largest model on
the widest device, where decode is most weight-bandwidth-bound and llama's
arch-tuned matvec still leads. [MTP](mtp.md) multiplies on top of these
numbers (byte-identical output): ~1.85–2.2× structured, 1.1–1.35× prose,
content-dependent (E4B prose measures **1.28×** on the A5000 post-fix, 150.1
vs 117.6).

## Prefill (prompt tokens/s, 929-token turns)

| device | model | little-gemma | llama.cpp (pp929) | ratio |
|--------|-------|-------------:|------------------:|------:|
| Orin NX | E4B | 426 | 524 | 0.81× |
| Orin NX | 12B | 174 | 217 | 0.80× |
| Orin NX | E2B QAT | 834 | 1,020 | 0.82× |
| A5000 | E4B | 3,703 | 4,846 | 0.76× |
| A5000 | 12B | 1,782 | 2,207 | 0.81× |
| A5000 | E2B QAT | 7,222 | 8,785 | 0.82× |

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

## Reproducing

```
# Orin (pin clocks first; a reboot resets them)
sudo jetson_clocks
bench/settle-bench.sh lg-e4b     # also: lg-12b, lg-e2b, ll-e4b, ll-12b, ll-e2b

# Windows / A5000
bench/settle-bench.ps1 -Model <path-to.gguf> -Tag e4b
llama-bench -m <model> -p 929 -n 128 -fa 0,1 -d 0,512 -r 3
```
