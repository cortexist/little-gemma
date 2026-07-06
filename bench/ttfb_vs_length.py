#!/usr/bin/env python3
# ttfb_vs_length.py — the "ours" leg of the TTFB-vs-prompt-length sweep
# (docs/fig-ttfb-vs-length.svg, paper §5.9). Six realistic spoken questions of
# increasing length are streamed to a running little-gemma serve two ways:
#   streamed (wpf=1 fps=2.26)  — dictation pace; prefill hides under speech.
#   deferred (wpf=0 fps=0)     — one plain line after end of speech, the shape
#                                a cascade with non-streaming ASR produces (the
#                                HuggingFace LLM leg: prefill + first-sentence
#                                decode on the SAME E2B weights).
# Reports median ttft (last input byte -> first reply byte) and ttfs (-> first
# speakable sentence). Result: streamed ttft is FLAT ~0.1 s across 5–30 s; the
# deferred prefill rises only mildly (LLM prefill is not the cascade's cost —
# the serial ASR pass is, measured separately in asr_leg_vs_length.py).
#
#   python3 bench/ttfb_vs_length.py /tmp/lg.sock
# Needs a warm serve (E2B QAT + MTP here). Wraps bench/ttft_dictate.py.
import os, re, statistics, subprocess, sys

SOCK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lg.sock"
DICT = os.path.join(os.path.dirname(__file__), "ttft_dictate.py")
WPS = 2.26                                 # a natural dictation rate (words/sec)

# 6 coherent questions; their synthesized spoken length is ~4–32 s (the true
# x-axis — see asr_leg_vs_length.py, which measures the durations).
Q = {
 5:  "Can you tell me a short and funny joke to cheer me up right now?",
 10: "I am planning a small dinner party for six close friends this weekend, and I would love to cook something that feels impressive but is not too difficult, so what would you suggest?",
 15: "I have been trying to get back into running after a long break of almost two years, and my knees tend to ache afterward, so could you walk me through how to build up my distance safely without hurting myself again?",
 20: "My grandmother is turning ninety years old next month and the whole family is coming together to celebrate, and I want to give a short heartfelt toast that captures how much she has meant to all of us over the decades, so how should I begin writing something like that, and what should I avoid saying?",
 25: "I am a high school science teacher and I have a class of about thirty students who are starting to lose interest in physics, especially the quieter ones near the back, and I have been assigned to teach them the basics of electricity and magnetism over the next three weeks, so can you help me design a few hands on demonstrations that would genuinely surprise them and pull them back into the subject?",
 30: "Over the past several months I have slowly been teaching myself how to cook proper meals instead of ordering takeout every single night, and while I have gotten reasonably comfortable with pasta dishes and simple stir fries, I still feel completely lost whenever a recipe asks me to work with fresh fish or any kind of seafood, so could you explain, as if you were standing right next to me in the kitchen, how I should pick, prepare, and cook a nice piece of salmon for the very first time?",
}


def once(f, wpf, fps):
    r = subprocess.run(["python3", DICT, SOCK, f, str(wpf), str(fps)],
                       capture_output=True, text=True, timeout=180).stdout
    m = re.search(r"ttft ([\d.]+)s \| ttfs (n/a|[\d.]+s)", r)
    return float(m.group(1)), (None if m.group(2) == "n/a" else float(m.group(2).rstrip("s")))


def med(f, wpf, fps, k=3):
    tf, ts = [], []
    for _ in range(k):
        a, b = once(f, wpf, fps)
        tf.append(a); ts.append(b if b else 99)
    return statistics.median(tf), statistics.median(ts)


def linefile(t, txt):
    p = "/tmp/q_%d.txt" % t
    open(p, "w").write(txt + "\n")
    return p


once(linefile(5, Q[5]), 1, WPS)            # warmup: clear the MTP draft-warmup
print("len_s\twords\tstream_ttft\tstream_ttfs\tdefer_ttft\tdefer_ttfs", flush=True)
for t in sorted(Q):
    f = linefile(t, Q[t])
    sf, ss = med(f, 1, WPS)
    df, ds = med(f, 0, 0)
    print("%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f" % (t, len(Q[t].split()), sf, ss, df, ds), flush=True)
