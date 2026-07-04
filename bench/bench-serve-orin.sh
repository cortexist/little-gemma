#!/bin/bash
# serve-mode prefill bench, E2B QAT — mirrors .scratch/bench-serve.ps1:
# N identical turns, one connection each, first discarded (repack warmup).
cd ~/repos/cortexist/little-gemma
LINE=$(python3 -c "print('word ' * 900 + 'Ignore all the words above. Reply with exactly one word: what is the capital of France?')")
SOCK=/tmp/lg-bench.sock
rm -f $SOCK /tmp/lg-bench.err
./build/run-cuda-i8 -m ~/repos/cortexist/llama.cpp/.scratch/gemma-4-e2b/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf -s $SOCK 2>/tmp/lg-bench.err &
SRV=$!
for i in $(seq 1 120); do [ -e $SOCK ] && break; sleep 0.5; done
for i in $(seq 1 6); do printf '%s\n' "$LINE" | ./build/run-cuda-i8 -c $SOCK > /tmp/lg-bench.out; done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
grep -c Paris /tmp/lg-bench.out
grep "turn:" /tmp/lg-bench.err
