#!/usr/bin/env python3
# piper_mem.py - peak memory of the piper service, measured the Jetson way:
# smaps_rollup Anonymous (VmHWM/RSS count evictable file-backed pages, which
# don't compete for an 8GB Nano's headroom).  Feeds the voice-loop clauses
# plus the 21-word baseline sentence (9.4s of audio = the biggest VITS
# activation seen in the dictation study), sampling at 50ms during synthesis.
import os, subprocess, threading, time

MODEL = os.path.expanduser("~/repos/piper/data/model/en_US-ozgirl_v6-step18500.onnx")
PIPER = os.path.expanduser("~/repos/piper/.venv/bin/piper")
LINES = [
    "Hello there.",
    "Paris became the capital of France in 1806.",
    "Napoleon Bonaparte moved the capital there.",
    "He wanted a central location for his empire.",
    "Paris became the capital of France primarily due to its historical "
    "significance and its role as a major cultural and political center.",
]

p = subprocess.Popen([PIPER, "-m", MODEL, "--output-raw"],
                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

peak = {"anon": 0, "when": ""}
tag = ["load"]
def sample():
    try:
        with open("/proc/%d/smaps_rollup" % p.pid) as f:
            for line in f:
                if line.startswith("Anonymous:"):
                    a = int(line.split()[1])
                    if a > peak["anon"]:
                        peak["anon"], peak["when"] = a, tag[0]
    except (FileNotFoundError, ProcessLookupError):
        pass

stop = [False]
def sampler():
    while not stop[0]:
        sample()
        time.sleep(0.05)
t = threading.Thread(target=sampler)
t.start()

time.sleep(3)                                   # model + espeak load
for line in LINES:
    tag[0] = "'%s...'" % line[:24]
    p.stdin.write((line + "\n").encode()); p.stdin.flush()
    time.sleep(3)                               # ~0.5s synth on Orin; generous
tag[0] = "idle-after"
time.sleep(1)
stop[0] = True
t.join()

hwm = rss = 0
with open("/proc/%d/status" % p.pid) as f:
    for line in f:
        if line.startswith("VmHWM:"): hwm = int(line.split()[1])
        if line.startswith("VmRSS:"): rss = int(line.split()[1])
print("piper service peak Anonymous: %d MB (during %s)" % (peak["anon"] // 1024, peak["when"]))
print("VmHWM %d MB | VmRSS(end) %d MB  (delta vs anon = evictable file-backed)" % (hwm // 1024, rss // 1024))
p.stdin.close(); p.terminate()
