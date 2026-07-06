#!/usr/bin/env python3
"""Generate fig-ttfb-vs-length.svg in the paper's house style."""

# Measured on Orin NX (clocks pinned, serve/app isolated per leg):
# durations = actual piper-synthesized spoken length of 6 coherent questions.
dur      = [3.84, 11.83, 13.47, 18.26, 23.87, 32.12]
asr_base = [1.196, 1.575, 1.742, 2.165, 3.356, 3.519]   # faster-whisper base int8, temp0
# ours: stream_ttft (prefill-under-speech, flat) measured at the same questions
ttft     = [0.090, 0.120, 0.120, 0.092, 0.122, 0.123]

# --- HF perfect-VAD floor = ASR(serial, after end of speech) + downstream D ---
# D = prefill + first-sentence decode + first MMS-TTS call, anchored to HF's
# measured tuned operating point (0.9 s TTFB, tiny.en ASR) -> D ~ 0.45 s.
D = 0.45
hf = [a + D for a in asr_base]

# least-squares fit of asr_base vs duration (to draw a readable trend line)
n = len(dur); sx = sum(dur); sy = sum(asr_base)
sxy = sum(x*y for x, y in zip(dur, asr_base)); sxx = sum(x*x for x in dur)
b = (n*sxy - sx*sy) / (n*sxx - sx*sx)
a = (sy - b*sx) / n
print("asr_base fit: %.3f + %.4f*T ; HF floor: %.3f + %.4f*T" % (a, b, a+D, b))

# ours composed TTFB: flat. dictation headline 0.65; live-mic +ASR commit ~1.0
OURS_DICT = 0.65
OURS_LIVE = 1.00

# ---- geometry ----
W, H = 760, 400
X0, X1 = 70, 705        # T = 0 .. 33 s
Y0, Ymax = 330, 55      # v = 0 .. 4.5 s
TMAX, VMAX = 33.0, 4.5
def xpx(t): return X0 + t * (X1 - X0) / TMAX
def ypx(v): return Y0 - v * (Y0 - Ymax) / VMAX

def path(pts):
    return "M " + " L ".join("%.1f %.1f" % p for p in pts)

s = []
s.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="Helvetica, Arial, sans-serif">' % (W, H))
s.append('  <rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))

# conversation band (< 1 s)
s.append('  <rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#eef3ee"/>'
         % (X0, ypx(1.0), X1 - X0, Y0 - ypx(1.0)))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#5f8a66">conversational zone (&lt; 1 s)</text>'
         % (X0 + 6, ypx(1.0) - 6))

# axes
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555b61" stroke-width="1"/>' % (X0, Y0, X1, Y0))
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555b61" stroke-width="1"/>' % (X0, Y0, X0, Ymax))
# y ticks
for v in [1, 2, 3, 4]:
    s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e6e8eb" stroke-width="1"/>' % (X0, ypx(v), X1, ypx(v)))
    s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="end">%d s</text>' % (X0 - 8, ypx(v) + 3, v))
# x ticks
for t in [5, 10, 15, 20, 25, 30]:
    s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555b61" stroke-width="1"/>' % (xpx(t), Y0, xpx(t), Y0 + 4))
    s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="middle">%d</text>' % (xpx(t), Y0 + 16, t))
s.append('  <text x="%.1f" y="%.1f" font-size="11" fill="#1f2328" text-anchor="middle">length of spoken prompt (seconds)</text>' % ((X0 + X1) / 2, Y0 + 34))
s.append('  <text x="18" y="%.1f" font-size="11" fill="#1f2328" text-anchor="middle" transform="rotate(-90 18 %.1f)">time to first audio (TTFB)</text>' % ((Y0 + Ymax) / 2, (Y0 + Ymax) / 2))

# HF floor trend line + markers
tline = [(xpx(t), ypx(a + D + b * t)) for t in [1, 33]]
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c47a3d" stroke-width="2"/>' % (tline[0][0], tline[0][1], tline[1][0], tline[1][1]))
for t, v in zip(dur, hf):
    s.append('  <circle cx="%.1f" cy="%.1f" r="4" fill="#c47a3d"/>' % (xpx(t), ypx(v)))
# HF endpoint value + label
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#a5632c" text-anchor="middle">%.1f s</text>' % (xpx(dur[-1]), ypx(hf[-1]) + 16, hf[-1]))
s.append('  <text x="%.1f" y="%.1f" font-size="11" fill="#a5632c" font-weight="bold">HuggingFace speech-to-speech</text>' % (xpx(17.5), ypx(a + D + b * 17.5) - 36))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#a5632c">perfect-VAD floor — the serial ASR pass sets the slope</text>' % (xpx(17.5), ypx(a + D + b * 17.5) - 23))

# ours live-mic (dashed) and dictation (solid), both flat
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#3e6fbf" stroke-width="1.5" stroke-dasharray="5 3"/>' % (xpx(0), ypx(OURS_LIVE), xpx(TMAX), ypx(OURS_LIVE)))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#3e6fbf" text-anchor="end">live mic, + streaming-ASR commit  ≈ 1.0 s</text>' % (xpx(TMAX), ypx(OURS_LIVE) - 6))
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#3e6fbf" stroke-width="2.5"/>' % (xpx(0), ypx(OURS_DICT), xpx(TMAX), ypx(OURS_DICT)))
for t in dur:
    s.append('  <circle cx="%.1f" cy="%.1f" r="4" fill="#3e6fbf"/>' % (xpx(t), ypx(OURS_DICT)))
s.append('  <text x="%.1f" y="%.1f" font-size="11" fill="#3e6fbf" font-weight="bold">little-gemma (§5.4) — 0.65 s, flat</text>' % (xpx(0.4), ypx(OURS_DICT) + 17))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#3e6fbf">prefill hides under speech; ASR streams during it</text>' % (xpx(0.4), ypx(OURS_DICT) + 30))

# title / caption
s.append('  <text x="%.1f" y="26" font-size="13" font-weight="bold" fill="#1f2328" text-anchor="middle">Time to first audio vs. spoken prompt length — E2B QAT, one Orin NX</text>' % (W / 2))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="middle">Same board, same E2B weights. HF floor = measured base-int8 ASR + 0.45 s downstream — its BEST case; real long input replies into ongoing speech.</text>' % (W / 2, H - 8))

s.append('</svg>')
open(r"C:\Users\Zero\Cortexist\little-gemma\docs\fig-ttfb-vs-length.svg", "w", encoding="utf-8").write("\n".join(s))
print("wrote fig-ttfb-vs-length.svg")
