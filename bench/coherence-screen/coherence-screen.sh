#!/bin/bash
# coherence-screen.sh — tier-2 coherence screen (reusable). Runs a fixed prompt
# set through a REFERENCE kernel config and a CANDIDATE kernel config in serve
# mode (greedy), captures replies, and scores the candidate for degeneration +
# divergence (bench/coherence-screen/score.py). Bigger than the coding smoke
# test, smaller than the tier-3 final test (which adds a larger-model judge).
#
# Usage:
#   coherence-screen.sh <model.gguf> [candidate-env]
#     candidate-env: env assignments that switch the kernel on (default
#                    "LG_FLASH_REG=1"); reference is the same binary, no env.
# Env: BIN=path to run-cuda-i8 (default build/run-cuda-i8 next to this repo).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${1:?usage: coherence-screen.sh <model.gguf> [candidate-env]}
CAND_ENV=${2:-LG_FLASH_REG=1}
BIN=${BIN:-$HERE/../../build/run-cuda-i8}
PROMPTS=$HERE/prompts.txt
OUT=${OUT:-/tmp/cscreen}
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no run-cuda-i8 at $BIN (set BIN=...)"; exit 2; }

run_set() {  # $1=tag  $2=env-assignments
  local tag=$1 envs=$2 srv i=0 line reply
  local sock=/tmp/cscreen-$tag.sock
  rm -f "$sock" "$OUT/$tag.jsonl"
  env $envs "$BIN" -m "$MODEL" -s "$sock" >"$OUT/$tag.load" 2>"$OUT/$tag.err" &
  srv=$!
  for i in $(seq 1 240); do [ -e "$sock" ] && break; sleep 0.5; done
  [ -e "$sock" ] || { echo "server failed ($tag)"; tail -3 "$OUT/$tag.err"; kill $srv 2>/dev/null; return 1; }
  i=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    reply=$(printf '%s\n' "$line" | "$BIN" -c "$sock" 2>/dev/null)
    # JSON-encode via python (safe for quotes/unicode)
    python3 - "$i" "$line" "$reply" >>"$OUT/$tag.jsonl" <<'PY'
import json, sys
print(json.dumps({"id": sys.argv[1], "prompt": sys.argv[2], "reply": sys.argv[3]}, ensure_ascii=False))
PY
    i=$((i + 1))
  done < "$PROMPTS"
  kill $srv 2>/dev/null; sleep 1
  echo "$tag: $i prompts"
}

echo "=== coherence screen: $(basename "$MODEL") | candidate env: $CAND_ENV ==="
run_set ref  ""          || exit 1
run_set cand "$CAND_ENV" || exit 1
echo
python3 "$HERE/score.py" "$OUT/ref.jsonl" "$OUT/cand.jsonl"
rc=$?
echo
echo "side-by-side saved: $OUT/ref.jsonl vs $OUT/cand.jsonl"
exit $rc
