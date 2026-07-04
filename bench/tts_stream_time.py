#!/usr/bin/env python3
# tts_stream_time.py - end-to-end first-byte timing for piper_stream.py on the
# Orin, protocol-identical to the tts_time.py that timed stock piper: write a
# line to the service's stdin, clock the first PCM byte out (includes espeak
# phonemization, enc, first chunk decode, and pipe overhead — everything).
import os, select, subprocess, sys, time

PY = os.path.expanduser("~/repos/piper/.venv/bin/python")
HOME = os.path.expanduser("~")
CMD = [PY, HOME + "/piper_stream.py",
       HOME + "/repos/piper/local/model/en_US-ozgirl_v6-step18500.onnx",
       HOME + "/en_US-ozgirl_v6-step18500-enc.onnx",
       HOME + "/en_US-ozgirl_v6-step18500-dec.onnx"] + sys.argv[1:]

TEXTS = [
    "Paris became the capital of France in 1806.",
    "Paris became the capital of France primarily due to its historical "
    "significance and its role as a major cultural and political center.",
]

p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL)
os.set_blocking(p.stdout.fileno(), False)

def synth(text, quiet=0.35):
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
    return first, nbytes / 2.0 / 22050

t0 = time.time()
wf, wa = synth("Hello there.")
print("service ready + warmup: %.2f s (warmup first byte %.3f)" % (time.time() - t0, wf))
for text in TEXTS:
    runs = [synth(text) for _ in range(3)]
    print("%r" % text[:44])
    for i, (f, a) in enumerate(runs):
        print("  run %d: FIRST PCM BYTE %.3f s | audio %.2f s" % (i + 1, f, a))
p.stdin.close(); p.terminate()
