#!/usr/bin/env python3
# Session-aging harness: how does responsiveness hold as one conversation fills
# the context window? Opens ONE persistent connection to a serve-mode socket and
# sends T turns down it, so the session KV accumulates (unlike the reproducibility
# benches, which use one connection per turn to reset the session each time).
#
# Each turn is a fixed padded prompt that forces a fixed short speakable reply, so
# every turn adds ~the same number of tokens and the context fills at a constant
# rate. Client-side clocks (per the paper's protocol, S 4.1):
#   TTFT = line sent -> first reply byte
#   TTFS = line sent -> first completed speakable sentence (thought-channel spans
#          and special tokens stripped; when a TTS could begin).
# The server logs the exact per-turn token counts ("turn: N in ..., M out ...,
# ttft ...") to its stderr, from which the cumulative context depth is recovered;
# the run ends when the server prints "context full" and closes the session.
#
# usage: session_age.py <sock> <turns> <pad_words>
#   e.g. run-cuda-i8 -m <E2B-QAT.gguf> -s /tmp/lg-age.sock   (plain, no MTP/-sys)
#        session_age.py /tmp/lg-age.sock 50 140
import re, socket, sys, time

sock, T, WORDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PAD = "word " * WORDS
Q = PAD + ("Ignore all the words above. Reply with exactly this sentence and "
           "nothing else: All systems are ready now.")
TERM = re.compile(r"[.!?](?=[\s\*\)\"'\]]|$)")

def speakable(raw):
    x = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", raw, flags=re.S)
    return re.sub(r"<[^<>]{1,24}>", "", x)

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock)
s.settimeout(30.0)
print("turn,ttft_s,ttfs_s,reply", flush=True)
for t in range(1, T + 1):
    t0 = time.time(); s.sendall(Q.encode() + b"\n")
    buf = b""; t1 = None; ts = None
    while b"<turn|>" not in buf:
        try:
            d = s.recv(8192)
        except socket.timeout:
            print("%d,TIMEOUT,,(runaway)" % t, flush=True); sys.exit(1)
        if not d:                       # server closed: "context full"
            print("%d,-1,,(context full)" % t, flush=True); sys.exit(0)
        if t1 is None:
            t1 = time.time()
        buf += d
        if ts is None and TERM.search(speakable(buf.decode(errors="replace"))):
            ts = time.time()
    rep = speakable(buf.decode(errors="replace")).strip().split("\n")[0][:40]
    print("%d,%.3f,%s,%s" % (t, (t1 - t0) if t1 else -1,
                             ("%.3f" % (ts - t0)) if ts else "", rep), flush=True)
