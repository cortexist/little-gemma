#!/usr/bin/env python3
# asr_leg_vs_length.py — the HuggingFace-cascade ASR leg for §5.9 / Figure 4.
# Synthesizes the six questions from ttfb_vs_length.py with piper, measures each
# clip's REAL spoken duration, and times the faster-whisper base-int8 AND
# tiny.en transcription (temperature=0, no fallback — the fallback machinery
# otherwise retries pathologically on synthetic speech and 5x-inflates the
# time). base is the accuracy-matched leg the paper composes into the HF floor;
# tiny.en is what their default config used and anchors the downstream constant.
#
# Run with the little-gemma serve STOPPED and jetson_clocks pinned — the Orin
# is a shared-memory SoC, so a GPU serve contends for LPDDR5 bandwidth and
# 3–4x's a CPU-bound int8 whisper pass (measurement-poisoning, learned the
# hard way). Result reproduces the journal's 3.9 s / 31.8 s base-int8 anchor.
#
#   (stop serve+app; sudo jetson_clocks) ; python3 bench/asr_leg_vs_length.py
import os, statistics, subprocess, time, wave
from faster_whisper import WhisperModel

PIPER = os.path.expanduser("~/repos/piper/.venv/bin/piper")
VOICE = os.path.expanduser("~/repos/piper/local/model/oz_girl_v6-thresh18500-step18500.onnx")
LENS = [5, 10, 15, 20, 25, 30]             # nominal; real durations are measured
KW = dict(language="en", temperature=[0.0], condition_on_previous_text=False)


def synth(t):
    txt = open("/tmp/q_%d.txt" % t).read().strip()   # written by ttfb_vs_length.py
    raw, wav = "/tmp/qs_%d.22k.wav" % t, "/tmp/qs_%d.wav" % t
    subprocess.run([PIPER, "-m", VOICE, "-f", raw], input=txt.encode(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", raw,
                    "-ar", "16000", "-ac", "1", wav], check=True)
    with wave.open(wav) as w:
        return wav, w.getnframes() / w.getframerate()


base = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=6)
tiny = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=6)


def t_asr(m, p):
    m.transcribe(p, **KW)                  # warmup
    list(m.transcribe(p, **KW)[0])
    ts = []
    for _ in range(2):
        s = time.monotonic(); list(m.transcribe(p, **KW)[0]); ts.append(time.monotonic() - s)
    return statistics.median(ts)


print("len_s\tdur_s\tasr_base\tasr_tiny", flush=True)
for t in LENS:
    wav, dur = synth(t)
    print("%d\t%.2f\t%.3f\t%.3f" % (t, dur, t_asr(base, wav), t_asr(tiny, wav)), flush=True)
