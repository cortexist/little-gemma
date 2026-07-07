#!/usr/bin/env python3
"""Generate fig-session-aging.svg in the paper's house style."""
import os
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                    "fig-session-aging.svg")

# Measured on Orin NX (MAXN, jetson_clocks pinned; voice stack stopped so the
# SoC was quiet), E2B QAT, plain serve (no MTP, no system prompt). ONE persistent
# connection, 42 fixed ~178-token turns forcing a fixed short reply; the server
# reported "context full" and closed after turn 42 (~7.5K tokens). Client-side
# TTFT / TTFS per turn (bench/session_age.py):
ttft = [0.246,0.261,0.278,0.286,0.291,0.293,0.298,0.302,0.307,0.311,0.316,0.318,
        0.323,0.327,0.332,0.335,0.340,0.341,0.347,0.352,0.356,0.361,0.365,0.372,
        0.376,0.381,0.385,0.390,0.392,0.417,0.401,0.405,0.407,0.414,0.417,0.421,
        0.426,0.430,0.435,0.438,0.442,0.446]
ttfs = [0.393,0.422,0.451,0.462,0.471,0.465,0.472,0.479,0.486,0.493,0.500,0.496,
        0.502,0.508,0.514,0.518,0.525,0.521,0.528,0.534,0.539,0.545,0.549,0.572,
        0.577,0.584,0.589,0.596,0.592,0.619,0.604,0.609,0.613,0.621,0.619,0.624,
        0.630,0.635,0.641,0.646,0.645,0.649]
N = len(ttft)
# TTFB (time to first audio) = measured TTFS + the streaming vocoder's first-PCM
# constant (0.10 s, Fig 2 / S 4.4) — flat in prompt length, so it just offsets.
VOC = 0.10
ttfb = [v + VOC for v in ttfs]

# context depth at the START of each turn (the KV the turn's prefill attends over):
# server-reported token counts, turn 1 = 170 in + 7 out, turns 2.. = 171 in + 7 out.
tin = [170] + [171] * (N - 1)
tout = [7] * N
ctx, cum = [], 0
for i in range(N):
    ctx.append(cum); cum += tin[i] + tout[i]
CTX_FULL = cum  # 7475

# ---- geometry ----
W, H = 760, 400
X0, X1 = 82, 700
Y0, Ymax = 322, 64
CMAX, VMAX = 8000.0, 1.0
def xpx(c): return X0 + c * (X1 - X0) / CMAX
def ypx(v): return Y0 - v * (Y0 - Ymax) / VMAX
def poly(xs, ys):
    return " ".join("%.1f,%.1f" % (xpx(x), ypx(y)) for x, y in zip(xs, ys))

BLUE, AMBER = "#3e6fbf", "#c47a3d"
s = []
s.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="Helvetica, Arial, sans-serif">' % (W, H))
s.append('  <rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
# conversational zone (< 1 s) fills the whole plot area
s.append('  <rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#eef3ee"/>' % (X0, ypx(1.0), X1 - X0, Y0 - ypx(1.0)))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#5f8a66">conversational zone (&lt; 1 s)</text>' % (X0 + 6, ypx(1.0) + 14))
# axes
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555b61" stroke-width="1"/>' % (X0, Y0, X1, Y0))
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555b61" stroke-width="1"/>' % (X0, Y0, X0, Ymax))
for v in [0.25, 0.5, 0.75, 1.0]:
    s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e6e8eb" stroke-width="1"/>' % (X0, ypx(v), X1, ypx(v)))
    s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="end">%.2f s</text>' % (X0 - 8, ypx(v) + 3, v))
for c in [0, 2000, 4000, 6000, 8000]:
    s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555b61" stroke-width="1"/>' % (xpx(c), Y0, xpx(c), Y0 + 4))
    s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="middle">%s</text>' % (xpx(c), Y0 + 16, "0" if c == 0 else "%dk" % (c // 1000)))
s.append('  <text x="%.1f" y="%.1f" font-size="11" fill="#1f2328" text-anchor="middle">conversation depth (context tokens; one turn ≈ 178)</text>' % ((X0 + X1) / 2, Y0 + 34))
s.append('  <text x="20" y="%.1f" font-size="11" fill="#1f2328" text-anchor="middle" transform="rotate(-90 20 %.1f)">seconds after end of speech</text>' % ((Y0 + Ymax) / 2, (Y0 + Ymax) / 2))
# context-full marker
s.append('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9aa0a6" stroke-width="1.5" stroke-dasharray="4 3"/>' % (xpx(CTX_FULL), Y0, xpx(CTX_FULL), ypx(0.86)))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="end">context full · 42 turns</text>' % (xpx(CTX_FULL) - 4, ypx(0.90)))
# series: TTFB (emphasized) then TTFT
s.append('  <polyline fill="none" stroke="%s" stroke-width="2.5" points="%s"/>' % (BLUE, poly(ctx, ttfb)))
s.append('  <polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (AMBER, poly(ctx, ttft)))
for i in range(0, N, 7):
    s.append('  <circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (xpx(ctx[i]), ypx(ttfb[i]), BLUE))
    s.append('  <circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (xpx(ctx[i]), ypx(ttft[i]), AMBER))
s.append('  <circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (xpx(ctx[-1]), ypx(ttfb[-1]), BLUE))
s.append('  <circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (xpx(ctx[-1]), ypx(ttft[-1]), AMBER))
# endpoint value labels
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="%s" text-anchor="end">%.2f s</text>' % (xpx(ctx[-1]) - 6, ypx(ttfb[-1]) - 6, BLUE, ttfb[-1]))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="%s" text-anchor="end">%.2f s</text>' % (xpx(ctx[-1]) - 6, ypx(ttft[-1]) - 6, AMBER, ttft[-1]))
# legend (upper-left open area)
s.append('  <rect x="%.1f" y="%.1f" width="16" height="3" fill="%s"/>' % (X0 + 8, ypx(0.90), BLUE))
s.append('  <text x="%.1f" y="%.1f" font-size="11" fill="#1f2328">first audio (TTFB) = TTFS + 0.10 s streaming vocoder</text>' % (X0 + 30, ypx(0.90) + 4))
s.append('  <rect x="%.1f" y="%.1f" width="16" height="3" fill="%s"/>' % (X0 + 8, ypx(0.82), AMBER))
s.append('  <text x="%.1f" y="%.1f" font-size="11" fill="#1f2328">TTFT (first token)</text>' % (X0 + 30, ypx(0.82) + 4))
# title / caption
s.append('  <text x="%.1f" y="26" font-size="13" font-weight="bold" fill="#1f2328" text-anchor="middle">Session aging: responsiveness as one conversation fills the 8K window — E2B QAT, one Orin NX</text>' % (W / 2))
s.append('  <text x="%.1f" y="%.1f" font-size="10" fill="#555b61" text-anchor="middle">Plain serve, pinned clocks. TTFT 0.25→0.45 s and first audio 0.49→0.75 s across a full session — both stay inside the conversational band. Decode 39.6→28.6 tok/s.</text>' % (W / 2, H - 8))
s.append('</svg>')
open(_OUT, "w", encoding="utf-8").write("\n".join(s))
print("wrote fig-session-aging.svg; context-full at %d tokens over %d turns" % (CTX_FULL, N))
