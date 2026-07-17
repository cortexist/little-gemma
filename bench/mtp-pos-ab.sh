#!/bin/bash
# mtp-pos-ab.sh — A/B the chained-draft POSITION for MTP acceptance (E4B, block-3).
# Hypothesis: upstream pins all Gemma4 chained drafts to the same n_past
# (common/speculative.cpp:1589, citing the HF gemma4_assistant doc); little-gemma
# advances (run.c:61 `pos + j - 1`). If wrong, draft 2 accepts ~never and the
# block-3 acceptance ~halves.
# Usage: mtp-pos-ab.sh {baseline|patched|restore}
set -u
LG=~/repos/cortexist/little-gemma
SCR=~/repos/cortexist/llama.cpp/.scratch
E4B=$SCR/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf
HEAD=$SCR/gemma-4-e4b/gemma-e4b-assistant-mtp-Q4_K_M.gguf
OUT=/tmp/mtppos
mkdir -p $OUT

# Prose question (the low-acceptance regime where the effect should be largest).
Q="Explain in detail how a refrigerator works, covering the compressor, the refrigerant cycle, and why the inside gets cold while the back gets warm."

run_bench() {  # $1 = tag
  local tag=$1 sock=/tmp/lg-mtp.sock srv
  rm -f $sock
  $LG/build/run-cuda-i8 -m "$E4B" -mtp "$HEAD" -s $sock 2>$OUT/$tag.err &
  srv=$!
  for i in $(seq 1 360); do [ -e $sock ] && break; sleep 0.5; done
  [ -e $sock ] || { echo "SERVER FAILED ($tag)"; kill $srv 2>/dev/null; return 1; }
  # 4 turns; first discarded (the ~3.6s draft graph warmup).
  for i in 1 2 3 4; do
    printf '%s\n' "$Q" | $LG/build/run-cuda-i8 -c $sock > $OUT/$tag-$i.out
  done
  kill $srv 2>/dev/null; sleep 2
  echo "=== $tag: acceptance per turn (discard turn 1) ==="
  grep "accepted" $OUT/$tag.err
  echo "=== $tag: decode tok/s per turn ==="
  grep "turn:" $OUT/$tag.err
  echo "=== $tag: reply sha (must be IDENTICAL across baseline/patched — greedy verify) ==="
  sha1sum $OUT/$tag-4.out | cut -c1-16
}

case "${1:-}" in
  baseline)
    cd $LG && git diff --quiet src/run.c || { echo "run.c already dirty — restore first"; exit 1; }
    cmake --build build --target run-cuda-i8 -j6 > $OUT/build-base.log 2>&1 || { echo BUILD-FAIL; tail -5 $OUT/build-base.log; exit 1; }
    run_bench baseline
    ;;
  patched)
    cd $LG
    # draft j chains at `pos` (upstream's shared-memory rule) instead of `pos + j - 1`
    sed -i 's/int dj = mtp_draft_chain(t, m, kv, toks\[j - 1\], pos + j - 1);/int dj = mtp_draft_chain(t, m, kv, toks[j - 1], pos);/' src/run.c
    grep -n "mtp_draft_chain(t, m, kv, toks" src/run.c
    cmake --build build --target run-cuda-i8 -j6 > $OUT/build-patch.log 2>&1 || { echo BUILD-FAIL; tail -5 $OUT/build-patch.log; exit 1; }
    run_bench patched
    ;;
  restore)
    cd $LG && git checkout src/run.c && cmake --build build --target run-cuda-i8 -j6 > $OUT/build-restore.log 2>&1
    git diff --stat src/run.c; echo restored
    ;;
  *) echo "usage: mtp-pos-ab.sh {baseline|patched|restore}"; exit 1 ;;
esac
