#!/usr/bin/env python3
# clause_pipe.py - the clause policy's python sandbox: a structural port of
# clausecat (little-gemma-tools) kept byte-identical by differential feeds.
# Try split-policy changes here first, then mirror them in the C.
#
# Raw reply stream on stdin -> one speakable clause per line on stdout.
# Drops <|channel>..<channel|> thought spans and <tokens>; flushes a line at
# each clause boundary [,;:.!?] followed by a space-like char; a bare newline
# (voicecat's end-of-turn mark) flushes the remainder.
#
#   --allow-control-token PATTERN   a <|tool_call>...<tool_call|> span matching
#       PATTERN ('*' = any run) passes through verbatim as its own line,
#       ordered against the clauses (e.g. for piper's set_voice — see the
#       piper fork's piper.control). Non-matching spans are dropped whole:
#       their payload is never spoken.
import os
import sys

PUNCT = ",;:.!?"
AFTER = " \t\n\r\f\v*)\"']"
TAG_MAX = 26
SPAN_MAX = 4096
TAGSPAN_MAX = 64

allow = None
route_emotion = False
args = sys.argv[1:]
while args:
    if args[0] == "--allow-control-token" and len(args) > 1:
        allow = args[1]
        args = args[2:]
    elif args[0] == "--route-emotion":
        route_emotion = True
        args = args[1:]
    else:
        sys.exit("usage: clause_pipe.py [--allow-control-token PATTERN] [--route-emotion]")

line = ""


def glob_match(s, p):
    """'*' = any run of characters; the same iterative matcher as clausecat
    ('?'/'[' are NOT special — parity with the C beats fnmatch)."""
    si = pi = 0
    star_p = star_s = -1
    while si < len(s):
        if pi < len(p) and p[pi] == "*":
            pi += 1
            star_p, star_s = pi, si
        elif pi < len(p) and p[pi] == s[si]:
            pi += 1
            si += 1
        elif star_p >= 0:
            star_s += 1
            pi, si = star_p, star_s
        else:
            return False
    while pi < len(p) and p[pi] == "*":
        pi += 1
    return pi == len(p)


def flush_line():
    global line
    out = line.strip(" \t")
    line = ""
    if out:
        sys.stdout.write(out + "\n")
        sys.stdout.flush()


def emit(c):
    global line
    if line and line[-1] in PUNCT and c in AFTER:
        flush_line()
    if len(line) < 8191:
        line += c


def main():
    tag = ""                     # pending '<...' that may become a token
    thought = False              # inside <|channel> ... <channel|>
    close, cn = "<channel|>", 0
    span = ""                    # a <|tool_call> span being captured
    tcall = False
    tclose, tcn = "<tool_call|>", 0
    tsp = ""                     # a [[key:value]] inline tag being captured
    brk = False                  # inside [[ ... ]]
    pb = False                   # a pending lone '[' (may open a tag)

    while True:
        b = os.read(0, 4096)
        if not b:
            break
        for c in b.decode("utf-8", errors="replace"):
            if c == "\n":        # end of turn: pending '<...'/'[' was literal
                if not thought and not tcall:   # text; unterminated [[ drops
                    if pb:
                        emit("[")
                    for t in tag:
                        emit(t)
                flush_line()
                tag, thought, cn, tcall, span, tcn = "", False, 0, False, "", 0
                pb, brk, tsp = False, False, ""
                continue
            if thought:          # discard until the exact close marker
                cn = cn + 1 if c == close[cn] else (1 if c == close[0] else 0)
                if cn == len(close):
                    thought, cn = False, 0
                continue
            if tcall:            # capture until the exact close marker
                if len(span) < SPAN_MAX - 2:
                    span += c
                tcn = tcn + 1 if c == tclose[tcn] else (1 if c == tclose[0] else 0)
                if tcn == len(tclose):
                    if allow and len(span) < SPAN_MAX - 2 and glob_match(span, allow):
                        flush_line()   # the span holds its place among the clauses
                        sys.stdout.write(span + "\n")
                        sys.stdout.flush()
                    tcall, span, tcn = False, "", 0
                continue
            if tag:              # inside a potential <token>
                if c == ">":
                    if len(tag) == 1:
                        emit("<"); emit(">")
                        tag = ""
                        continue
                    if tag == "<|channel":
                        thought = True
                    elif tag == "<|tool_call":
                        tcall, span, tcn = True, "<|tool_call>", 0
                    tag = ""
                    continue
                if c != "<" and len(tag) < TAG_MAX - 1:
                    tag += c
                    continue
                for t in tag:    # too long or '<': literal
                    emit(t)
                tag = ""
            if brk:              # capture a [[...]] inline tag until "]]"
                if c == "]" and tsp.endswith("]"):
                    key = tsp[:-1]
                    # [[emotion:X]] -> piper's voice-switch line (--route-emotion);
                    # every other tag drops whole - inline tags are control, not speech
                    if route_emotion and key.startswith("emotion:") and key[8:]:
                        flush_line()   # the switch holds its place among the clauses
                        sys.stdout.write('<|tool_call>call:set_voice{speaker_id:<|"|>%s<|"|>}<tool_call|>\n' % key[8:])
                        sys.stdout.flush()
                    brk, tsp = False, ""
                    continue
                if len(tsp) < TAGSPAN_MAX - 1:
                    tsp += c
                    continue
                emit("[")        # overflow: it was literal text after all
                emit("[")
                for t in tsp:
                    emit(t)
                brk, tsp = False, ""
            if pb:
                pb = False
                if c == "[":     # "[[" opens a tag span
                    brk, tsp = True, ""
                    continue
                emit("[")        # a lone '[' was literal text
            if c == "[":
                pb = True
                continue
            if c == "<":
                tag = "<"
                continue
            emit(c)
    if not thought and not tcall:
        if pb:
            emit("[")
        for t in tag:
            emit(t)
    flush_line()


main()
