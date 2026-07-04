#!/usr/bin/env python3
# clause_pipe.py - the glue between the serve reply stream and a line-based
# TTS: reads little-gemma's raw token stream (special tokens included) on
# stdin, strips the thought channel and scaffolding, and emits one line per
# CLAUSE (any of , ; : . ! ? followed by whitespace) so the TTS can start
# speaking at the first boundary instead of at the first newline. The
# voice-sys prompt (docs/voice-sys.txt) is what makes the model produce those
# boundaries early; this pipe just honors them.
#
#   voicecat /tmp/lg.sock ... | clause_pipe.py | piper -m voice.onnx \
#       --output-raw --stream | aplay -r 22050 -f S16_LE -t raw -c 1 -
import os, re, sys

CLAUSE = re.compile(r"[,;:.!?](?=[\s\*\)\"'\]])")
TAG = re.compile(r"<[^<>]{1,24}>")
CHANNEL = re.compile(r"<\|channel>.*?(<channel\|>|$)", re.S)

buf = ""       # raw tail that may still grow a tag or thought span
spoken = 0     # speakable chars already flushed


def speakable(raw):
    return TAG.sub("", CHANNEL.sub("", raw))


def flush(upto):
    """Write speakable[spoken:upto] as one line if it says anything."""
    global spoken
    line = speakable(buf)[spoken:upto].strip()
    spoken = upto
    if line:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


while True:
    data = os.read(0, 4096)
    if not data:
        break
    for ch in data.decode(errors="replace"):
        if ch == "\n":                       # voicecat marks end of turn
            flush(len(speakable(buf)))
            buf = ""
            spoken = 0
            continue
        buf += ch
        sp = speakable(buf)
        m = None
        for m in CLAUSE.finditer(sp, spoken):
            pass                             # last boundary in the new text
        if m is not None:
            flush(m.end())
flush(len(speakable(buf)))
