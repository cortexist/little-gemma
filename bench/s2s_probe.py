#!/usr/bin/env python3
# s2s_probe.py - TTFB probe for huggingface/speech-to-speech in socket mode.
# Streams [0.5s silence][spoken question][silence...] at real-time pace in
# 1024-byte chunks (512 samples / 32 ms at 16 kHz), like a live mic. Clocks:
# t_end = last speech chunk sent -> first reply audio byte received.
# All published numbers (paper §5.9, journal 2026-07-03/04) were taken
# against speech-to-speech v0.2.10 (PyPI) — pin it when reproducing; their
# codebase moves quickly.
import socket, sys, threading, time, wave

HOST = "127.0.0.1"
CH = 1024
w = wave.open(sys.argv[1], "rb")
assert w.getframerate() == 16000 and w.getnchannels() == 1
speech = w.readframes(w.getnframes())
speech += b"\0" * ((-len(speech)) % CH)
n_speech = len(speech) // CH
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

rx = socket.socket(); rx.connect((HOST, 12346))
tx = socket.socket(); tx.connect((HOST, 12345))
st = {"first": None, "last": None, "n": 0, "t0": []}
def reader():
    while True:
        d = rx.recv(8192)
        if not d: return
        now = time.monotonic()
        if st["first"] is None: st["first"] = now
        st["last"] = now; st["n"] += len(d)
threading.Thread(target=reader, daemon=True).start()

silence = b"\0" * CH
t0 = time.monotonic(); tick = 0
def send(chunk):
    global tick
    tx.sendall(chunk); tick += 1
    d = t0 + tick * 0.032 - time.monotonic()
    if d > 0: time.sleep(d)

for trial in range(TRIALS):
    st["first"] = None; st["last"] = None; st["n"] = 0
    for _ in range(16): send(silence)                 # lead-in
    for i in range(n_speech): send(speech[i*CH:(i+1)*CH])
    t_end = time.monotonic()
    # trailing silence while waiting for the reply to finish
    while True:
        send(silence)
        now = time.monotonic()
        if st["first"] and now - st["last"] > 2.0: break
        if now - t_end > 60: break
    if st["first"]:
        print("trial %d: TTFB %.2fs | reply %.1fs of audio (%d bytes)"
              % (trial, st["first"] - t_end, st["n"] / 2 / 16000, st["n"]), flush=True)
    else:
        print("trial %d: NO AUDIO within 60s" % trial, flush=True)
