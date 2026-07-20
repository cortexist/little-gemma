#!/usr/bin/env python3
# Audio-turn TTFT/TTFS client, chunked + paced: streams raw 16 kHz mono s16
# PCM as one or many 'A' media frames (the shape a native-ears voicecat would
# produce), optionally at real-time pace (each chunk sent after its own
# duration has elapsed, like a live mic), then the turn-closing text line.
# Clocks: last byte sent -> first reply byte (TTFT) and -> first speakable
# sentence (TTFS).
#   ttfb_audio2.py <sock> <pcm_file> [chunk_s] [pace 0|1] ["question"]
#   chunk_s 0 = single frame (deferred).
import re, socket, struct, sys, time

sock_path, path = sys.argv[1], sys.argv[2]
chunk_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
pace    = len(sys.argv) > 4 and sys.argv[4] == "1"
question = sys.argv[5] if len(sys.argv) > 5 else ""
pcm = open(path, "rb").read()

TERM = re.compile(r"[.!?](?=[\s\*\)\"'\]]|$)")
def speakable(raw):
    s = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", raw, flags=re.S)
    return re.sub(r"<[^<>]{1,24}>", "", s)

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
t0 = time.time()
if chunk_s <= 0:
    s.sendall(b"\x01A" + struct.pack("<HHI", 0, 0, len(pcm)) + pcm)
else:
    step = int(chunk_s * 16000) * 2                       # bytes per chunk
    off, n_chunks = 0, 0
    while off < len(pcm):
        part = pcm[off:off + step]
        off += step
        n_chunks += 1
        if pace:                                          # live mic: audio exists
            due = t0 + off / 32000.0                      # only once it is spoken
            wait = due - time.time()
            if wait > 0: time.sleep(wait)
        s.sendall(b"\x01A" + struct.pack("<HHI", 0, 0, len(part)) + part)
s.sendall(question.encode() + b"\n")
t_line = time.time()
first = s.recv(4096)
t_first = time.time()
buf = first
t_sent = time.time() if TERM.search(speakable(buf.decode(errors="replace"))) else None
while b"<turn|>" not in buf:
    d = s.recv(4096)
    if not d:
        break
    buf += d
    if t_sent is None and TERM.search(speakable(buf.decode(errors="replace"))):
        t_sent = time.time()
print("ttft %.3fs | ttfs %s | %.1fs audio, chunk %.1fs, %s"
      % (t_first - t_line, "%.3fs" % (t_sent - t_line) if t_sent else "n/a",
         len(pcm) / 32000.0, chunk_s, "paced" if pace else "burst"))
print("reply: %s" % buf.decode(errors="replace").replace("\n", " ")[:200])
