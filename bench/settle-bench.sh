#!/bin/bash
# settle-bench.sh — canonical decode+prefill bench (Orin), one stage per call.
# Produces the numbers in docs/benchmarks.md. Pin clocks first: sudo jetson_clocks
# (a reboot resets the pin). Stages: env, lg-e4b, lg-12b, lg-e2b, ll-e4b, ll-12b, ll-e2b.
# little-gemma: serve mode, 6x 929-token prefill turns + 5x max-length decode
# turns, one connection each, first of each discarded, rates from `turn:` stderr.
# llama.cpp: llama-bench pp929 + tg32/128, fa 0/1, r 3 (12B: drop_caches + -mmp 0).
set -u
LG=~/repos/cortexist/little-gemma
LL=~/repos/cortexist/llama.cpp
SCR=$LL/.scratch
OUT=/tmp/settle
mkdir -p $OUT

E4B=$SCR/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf
B12=$SCR/gemma-4-12b/gemma-4-12b-it-Q4_K_M.gguf
E2B=$SCR/gemma-4-e2b/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf

DECODE_Q="Explain in detail how a refrigerator works, covering the compressor, the refrigerant cycle, and why the inside gets cold while the back gets warm."

bench_lg() {  # $1=tag $2=model
  local tag=$1 model=$2 sock=/tmp/lg-bench.sock srv
  rm -f $sock
  $LG/build/run-cuda-i8 -m "$model" -s $sock 2>$OUT/lg-$tag.err &
  srv=$!
  for i in $(seq 1 360); do [ -e $sock ] && break; sleep 0.5; done
  [ -e $sock ] || { echo "SERVER FAILED $tag"; kill $srv 2>/dev/null; return 1; }
  for i in $(seq 1 6); do
    cat $LG/bench/line929s.txt | $LG/build/run-cuda-i8 -c $sock > $OUT/lg-$tag-pf$i.out
  done
  for i in $(seq 1 5); do
    printf '%s\n' "$DECODE_Q" | $LG/build/run-cuda-i8 -c $sock > $OUT/lg-$tag-dec$i.out
  done
  kill $srv 2>/dev/null
  sleep 2
  echo "=== lg-$tag turn lines (6 prefill then 5 decode; discard first of each) ==="
  grep "turn:" $OUT/lg-$tag.err
}

bench_ll() {  # $1=tag $2=model [extra args...]
  local tag=$1 model=$2; shift 2
  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo "=== llama-bench $tag ==="
  $LL/build/bin/llama-bench -m "$model" -p 929 -n 32,128 -fa 0,1 -r 3 "$@" 2>$OUT/ll-$tag.err
}

case "${1:-}" in
  env)
    date
    echo "--- lg commit:";   git -C $LG log --oneline -1
    echo "--- llama commit:"; git -C $LL log --oneline -1
    echo "--- clocks:"; sudo jetson_clocks --show | grep -E 'GPU|EMC|NV Power'
    echo "--- leftover procs:"; pgrep -ax run-cuda-i8; pgrep -ax llama-server; pgrep -ax llama-bench; true
    ;;
  lg-e4b) bench_lg e4b "$E4B" ;;
  lg-12b) bench_lg 12b "$B12" ;;
  lg-e2b) bench_lg e2b "$E2B" ;;
  ll-e4b) bench_ll e4b "$E4B" ;;
  ll-12b) bench_ll 12b "$B12" -mmp 0 ;;
  ll-e2b) bench_ll e2b "$E2B" ;;
  *) echo "usage: settle-bench.sh {env|lg-e4b|lg-12b|lg-e2b|ll-e4b|ll-12b|ll-e2b}"; exit 1 ;;
esac
