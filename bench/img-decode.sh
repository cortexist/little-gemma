#!/bin/bash
# img-decode.sh - is the decode RATE with an image in context the same as text?
# Same server session for both, so weights/clocks/thermals are identical.
# Text control first, then the same question with an image span prepended.
set -u
LG=~/repos/cortexist/little-gemma
SCR=~/repos/cortexist/llama.cpp/.scratch
E2B=$SCR/gemma-4-e2b/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
MM=$SCR/gemma-4-e2b/mmproj-F16.gguf
MMCAT=~/repos/cortexist/little-gemma-tools/build/mmcat
OUT=/tmp/imgdec; mkdir -p $OUT
SOCK=/tmp/lg-img.sock
rm -f $SOCK

Q="Describe this in detail, covering every object, colour, texture and spatial relationship you can see, and then explain what is probably happening."

$LG/build/run-cuda-i8 -m "$E2B" -mm "$MM" -s $SOCK 2>$OUT/srv.err &
SRV=$!
for i in $(seq 1 360); do [ -e $SOCK ] && break; sleep 0.5; done
[ -e $SOCK ] || { echo "SERVER FAILED"; kill $SRV 2>/dev/null; exit 1; }

echo "### TEXT turns (control, no image)"
for i in 1 2 3; do printf '%s\n' "$Q" | $LG/build/run-cuda-i8 -c $SOCK > $OUT/text-$i.out; done

echo "### IMAGE turns (image span + same question)"
for i in 1 2 3; do $MMCAT $SOCK /tmp/cf_01.jpg "$Q" > $OUT/img-$i.out 2>$OUT/img-$i.err; done

kill $SRV 2>/dev/null; sleep 2
echo "=== turn lines: first 3 = text control, last 3 = image (discard each first) ==="
grep "turn:" $OUT/srv.err
echo "=== image reply sanity (did it actually see the picture?) ==="
head -c 220 $OUT/img-3.out; echo
