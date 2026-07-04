#!/usr/bin/env python3
# tts_time.py - piper first-sentence latency, cold vs service.
# Service mode = one persistent piper process, lines in, raw PCM out
# (--output-raw). Clocks: line written -> first audio byte, and -> synthesis
# quiet (all bytes of the sentence drained). VITS generates a sentence in one
# ONNX call, so first-byte ~= whole-sentence synthesis; both are reported.
import json, os, select, subprocess, sys, time

MODEL = os.path.expanduser("~/repos/piper/data/model/en_US-ozgirl_v6-step18500.onnx")
PIPER = os.path.expanduser("~/repos/piper/.venv/bin/piper")
SENT = ("Paris became the capital of France primarily due to its historical "
        "significance and its role as a major cultural and political center.")
WARM = "Hello there."

rate = json.load(open(MODEL + ".json"))["audio"]["sample_rate"]

# ---- cold one-shot (includes python + espeak + onnx load) -------------------
t0 = time.time()
subprocess.run([PIPER, "-m", MODEL, "-f", "/tmp/tts_cold.wav"],
               input=SENT.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
cold = time.time() - t0

# ---- service: persistent process, warmup line, then measured lines ----------
t0 = time.time()
p = subprocess.Popen([PIPER, "-m", MODEL, "--output-raw"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL)
os.set_blocking(p.stdout.fileno(), False)

def synth(text, quiet=0.35):
    """Write one line, return (first_byte_s, done_s, audio_s)."""
    t0 = time.time(); first = None; nbytes = 0; last = t0
    p.stdin.write((text + "\n").encode()); p.stdin.flush()
    while True:
        r, _, _ = select.select([p.stdout], [], [], 0.05)
        if r:
            d = p.stdout.read(1 << 20)
            if d:
                if first is None: first = time.time() - t0
                nbytes += len(d); last = time.time()
        if first is not None and time.time() - last > quiet:
            break
        if time.time() - t0 > 60:
            break
    return first, last - t0, nbytes / 2.0 / rate

wf, wd, wa = synth(WARM)
ready = time.time() - t0 - wd            # process start -> ready-for-first-line
print("cold one-shot (load + synth):   %.2f s" % cold)
print("service startup to ready:       ~%.2f s (load %.2fs + warmup synth)" % (time.time() - t0, ready))
runs = [synth(SENT) for _ in range(3)]
for i, (f, d, a) in enumerate(runs):
    print("run %d: first audio byte %.3f s | synth done %.3f s | audio %.2f s (rtf %.2fx realtime)"
          % (i + 1, f, d - 0.35, a, a / max(d - 0.35, 1e-9)))
rss = int(open("/proc/%d/status" % p.pid).read().split("VmRSS:")[1].split()[0]) // 1024
print("piper service RSS: %d MB | sample rate %d Hz" % (rss, rate))
p.stdin.close(); p.terminate()
