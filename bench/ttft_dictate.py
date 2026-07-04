#!/usr/bin/env python3
# Dictation client: streams a long instruction as 'T' word-group frames (the
# shape voicecat's confirmed commits produce), then closes the turn with a
# bare newline. Clocks: last byte sent -> first reply byte (TTFT) and ->
# first speakable sentence (TTFS).
#   ttft_dictate.py <sock> <line_file> <words_per_frame> <frames_per_sec>
#   frames_per_sec 0 = burst (no pacing); words_per_frame 0 = single line mode
import re, socket, sys, time

sock_path, path = sys.argv[1], sys.argv[2]
wpf, fps = int(sys.argv[3]), float(sys.argv[4])
line = open(path, encoding="utf-8").read().strip()

TERM = re.compile(r"[.!?](?=[\s\*\)\"'\]]|$)")
def speakable(raw):
    s = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", raw, flags=re.S)
    return re.sub(r"<[^<>]{1,24}>", "", s)

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
t0 = time.time()
if wpf == 0:                                        # baseline: one plain line
    s.sendall(line.encode() + b"\n")
else:
    import struct
    words = line.split(" ")
    for i in range(0, len(words), wpf):
        grp = (" " if i else "") + " ".join(words[i:i + wpf])
        s.sendall(b"\x01T" + struct.pack("<HHI", 0, 0, len(grp.encode())) + grp.encode())
        if fps > 0:
            time.sleep(1.0 / fps)
    s.sendall(b"\n")
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
print("ttft %.3fs | ttfs %s | send window %.1fs"
      % (t_first - t_line, "%.3fs" % (t_sent - t_line) if t_sent else "n/a", t_line - t0))
print("reply: %s" % buf.decode(errors="replace").replace("\n", " ")[:160])
