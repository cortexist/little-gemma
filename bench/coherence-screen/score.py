#!/usr/bin/env python3
# score.py — coherence screen scorer. Compares a CANDIDATE kernel's greedy
# replies against a REFERENCE (current main). Heuristic only — the tier-3 final
# test uses a larger-model judge; this tier-2 screen just proves the changed
# outputs are still COHERENT (no degeneration) and that divergence stays at
# paraphrase level. stdlib only.
#
#   score.py ref.jsonl cand.jsonl        # each line: {"id","prompt","reply"}
#
# Exit 0 = PASS (no candidate output degenerates). Exit 1 = FAIL.

import json
import sys
from collections import Counter


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["id"]] = r
    return out


def degeneration(text):
    """Return a list of degeneration flags (empty list = coherent)."""
    flags = []
    t = text.strip()
    toks = t.split()
    n = len(toks)
    if n == 0:
        return ["empty"]
    # looping: a 3-gram repeated far more than chance
    if n >= 12:
        grams = Counter(tuple(toks[i:i + 3]) for i in range(n - 2))
        gram, cnt = grams.most_common(1)[0]
        if cnt >= max(4, int(n * 0.15)):
            flags.append("loop[%s x%d]" % (" ".join(gram), cnt))
    # runaway: long output that recycles a tiny vocabulary
    if n > 25 and len(set(toks)) / n < 0.35:
        flags.append("low-diversity[%d/%d]" % (len(set(toks)), n))
    # control-char garbage (tabs/newlines excluded)
    bad = sum(1 for c in t if ord(c) < 9 or 13 < ord(c) < 32)
    if bad > 2:
        flags.append("control-chars[%d]" % bad)
    # single token repeated to fill (e.g. "the the the ...")
    if n >= 8:
        common, c = Counter(toks).most_common(1)[0]
        if c / n > 0.4:
            flags.append("token-spam[%s x%d]" % (common, c))
    return flags


def sim(a, b):
    """Cheap char-shingle Jaccard — paraphrase check without difflib cost."""
    def sh(s):
        s = " ".join(s.split()).lower()
        return set(s[i:i + 4] for i in range(max(0, len(s) - 3))) or {s}
    A, B = sh(a), sh(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def main():
    ref = load(sys.argv[1])
    cand = load(sys.argv[2])
    ids = sorted(set(ref) & set(cand), key=lambda x: int(x))
    same = 0
    diverged = []
    degenerate = []
    print("%-4s %-9s %-7s %s" % ("id", "match", "sim", "flags / prompt"))
    for i in ids:
        rr, cc = ref[i]["reply"], cand[i]["reply"]
        flags = degeneration(cc)
        if flags:
            degenerate.append((i, flags))
        if rr.strip() == cc.strip():
            same += 1
            tag, s = "exact", 1.0
        else:
            s = sim(rr, cc)
            diverged.append((i, s))
            tag = "paraphrase" if s >= 0.35 else "DIVERGENT"
        note = ("!!" + ",".join(flags)) if flags else (ref[i]["prompt"][:46])
        print("%-4s %-9s %-7.2f %s" % (i, tag, s, note))
    n = len(ids)
    print("\n--- summary ---")
    print("prompts:            %d" % n)
    print("exact match:        %d (%.0f%%)" % (same, 100 * same / n))
    print("paraphrased:        %d" % len(diverged))
    if diverged:
        lo = min(diverged, key=lambda x: x[1])
        print("lowest similarity:  %.2f (id %s)" % (lo[1], lo[0]))
    hard = [d for d in diverged if d[1] < 0.35]
    print("hard divergences:   %d %s" % (len(hard), [d[0] for d in hard] or ""))
    print("degenerate outputs: %d %s" % (len(degenerate), [d[0] for d in degenerate] or ""))
    # PASS if no candidate output degenerated. Hard divergences are surfaced for
    # human/tier-3 review but do not auto-fail (a valid answer can restructure).
    ok = not degenerate
    print("\nSCREEN: %s" % ("PASS" if ok else "FAIL"))
    if hard:
        print("NOTE: %d hard divergences — eyeball these before merge / send to tier-3." % len(hard))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
