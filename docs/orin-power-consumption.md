# Orin power consumption under little-gemma

Measured 2026-07-02 on the Jetson Orin NX 16GB, MAXN (40 W) profile, asking
one question: **does little-gemma ever push the power envelope?** It does
not — and the way it doesn't is the interesting part.

## Method

`tegrastats` at 500 ms intervals, each line timestamped, workload phases
marked into a separate log and joined afterwards (harness:
`.scratch/power_run.sh`, `.scratch/power_run2.sh`,
`.scratch/parse_power.py`). Rails reported:

- **VDD_IN** — total module input power: what a battery would see (the
  carrier board's own conversion losses sit outside it).
- **VDD_CPU_GPU_CV** — the compute rail.
- **VDD_SOC** — includes the memory controller (EMC), which is where a
  bandwidth-bound story shows itself.
- **GR3D_FREQ %** — GPU busy cycles (utilization, *not* power).

Clocks were pinned (`jetson_clocks`, the bench discipline) for every load
phase; one idle cell re-applied `nvpmodel -m 0` to restore DVFS for
comparison, and clocks were re-pinned afterwards. Loads: sustained prefill
(five 928-token turns back-to-back), sustained decode (one turn run to the
1,024-token `SERVE_GEN` cap), the same decode under MTP, and camera bursts
(624×480 frame + question, ×6) as the production duty cycle. Models:
gemma-4 12B and E4B, Q4_K_M, the shipped `run-cuda-i8` serve path.

## Results

| phase | sec | VDD_IN avg | peak | CPU_GPU_CV | SOC | GPU% |
|---|---:|---:|---:|---:|---:|---:|
| idle, pinned clocks, no server | 20 | 7.26 W | 7.33 | 1.95 | 2.40 | 0 |
| idle, pinned, 12B resident | 20 | 7.23 W | 7.37 | 1.95 | 2.36 | 0 |
| prefill 12B (5×928 tok) | 30 | 19.48 W | 20.79 | 10.10 | 4.20 | 99 |
| decode 12B (1,024 tok) | 132 | 22.22 W | **23.00** | 10.75 | 5.46 | 99 |
| camera bursts 12B (×6) | 16 | 21.34 W | 23.00 | 10.66 | 4.97 | 98 |
| decode 12B + MTP (1,024 tok) | 114 | 20.67 W | 21.14 | 10.52 | 4.70 | 99 |
| prefill E4B (5×928) | 12 | 18.69 W | 19.38 | 9.55 | 4.08 | 97 |
| decode E4B (1,024 tok) | 65 | 20.04 W | 21.07 | 9.47 | 4.95 | 99 |
| idle, DVFS restored | 20 | **6.15 W** | 6.30 | 0.79 | 2.42 | 0 |
| idle, re-pinned | 8 | 7.45 W | 7.64 | 2.08 | 2.40 | 0 |

## Full utilization at half the power budget

Peak draw is **23.0 W against a 40 W profile, with GR3D at 99%** across
every load phase. That combination is not a contradiction — it is the
signature of a **memory-bandwidth-bound** workload. GR3D counts busy
cycles, and little-gemma's SMs spend those cycles waiting on LPDDR reads
(the weights live in zero-copy host memory), which occupies the GPU without
lighting up its ALU silicon.

The rails agree: decode — the pure weight-streaming phase, every parameter
read once per token — is the *hungriest* phase and posts the highest SOC
rail (5.46 W, the memory controller working hardest), while compute-denser
prefill is ~2.7 W **cheaper** at the same 99% utilization. Power here
follows bytes moved, not FLOPs.

A useful corollary: none of the performance numbers in this repo are power-
or thermally-throttled. The machine runs out of memory bandwidth long
before it runs out of watts.

## Energy per token — the battery metric

VDD_IN × time ÷ output tokens:

| workload | J / output token |
|---|---:|
| 12B decode | 2.85 |
| 12B decode + MTP | **2.29** |
| E4B decode | 1.27 |

**MTP is a battery feature, not just a latency one**: at 45.7% prose
acceptance it delivered 9.0 vs 7.8 tok/s (1.16×) *and* 20% less energy per
token, because accepted draft tokens share a weight pass — fewer trips over
the memory bus per token, and the bus is what costs watts. (On structured
output, where acceptance is ~100%, both factors roughly double.)

Prefill: ~113 J per 1,000 prompt tokens (12B), ~45 J (E4B). A camera burst
turn (130-token frame + short reply, 12B) costs ~56 J.

For a 100 Wh battery, ignoring everything but the module: ~16 h idle with
the model resident (DVFS), ~4.5 h of continuous 12B conversation (~5 h with
MTP), ~7.9 h E4B, and a camera assistant at 10% duty (one burst every ~10 s
over DVFS idle) lands around ~9 h.

## Deployment notes

- **`jetson_clocks` costs ~1.1 W at idle** (7.26 vs 6.15 W; the compute
  rail alone drops 1.95 → 0.79 W). It is a bench discipline — it removes
  DVFS ramp jitter from measurements — not a deployment setting. Ship with
  DVFS: under sustained load it reaches max clocks anyway. Restore DVFS
  with `sudo nvpmodel -m 0` (re-applying the mode resets the pins).
- The idle floor is ~6.2 W even under DVFS with the server resident —
  carrier board plus SOC baseline; no deeper sleep states were explored
  here. Loading the 12B changes idle draw by nothing measurable: resident
  weights are free until read.
- **CPU sits at ~0% in every phase** — the runner idle-waits on the GPU by
  design. Whisper (voicecat's streaming transcripts) will run on entirely
  free CPU headroom without touching the GPU power budget.
