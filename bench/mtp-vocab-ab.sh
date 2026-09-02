#!/bin/bash
# mtp-vocab-ab.sh — A/B the MTP draft head's vocabulary size on the Orin
# (docs/mtp-vocab-trim.md, phase 1). Prefix-trimmed heads (rows 0..K-1, so
# draft row index == target token id) run on the UNMODIFIED engine; the
# trimmed files are built by little-gemma-tools' trim-head.py.
#
# Measures, per variant (full head + each K):
#   H1  reply sha1s — must be byte-identical to the full head (greedy verify)
#   H2  draft ms/round + decode tok/s (run.c already prints both)
#   H3  acceptance %, against predicted = full-acceptance × prefix-K coverage
#       of the emitted stream (llama-tokenize + vocab-stats.py missrate)
#   H4  MemAvailable: watermark during load+warmup, steady state after
#
# Usage: mtp-vocab-ab.sh [K ...]     (default: 8192 16384 32768 65536)
# Env overrides:
#   TARGET/HEAD/OUT       other model pairs (E2B, E4B) into their own OUT dir
#   BASE="plain full"     which un-trimmed baselines to run
#   KS=""                 skip the prefix-trim variants (or a K list)
#   SEL="tag=idlist ..."  selected-vocab variants: full HEAD + LG_MTP_IDS
#                         (phase 2 — needs the d2t engine build)
#   BIN=...               engine binary (e.g. an -DLG_MTP_N=4 build)
set -u
LG=~/repos/cortexist/little-gemma
TOOLS=~/repos/cortexist/little-gemma-tools/script/vocab-trim
SCR=~/repos/cortexist/llama.cpp/.scratch/gemma-4-12b
TARGET=${TARGET:-$SCR/gemma-4-12b-it-Q4_K_M.gguf}
HEAD=${HEAD:-$SCR/gemma-4-12b-it-assistant-Q4_K_M.gguf}
TOKENIZE=~/repos/cortexist/llama.cpp/build/bin/llama-tokenize
OUT=${OUT:-/tmp/mtpvocab}
BIN=${BIN:-$LG/build/run-cuda-i8}
TURNS=3                              # per prompt; server's first turn overall is graph warmup
KS=${KS-${*:-"8192 16384 32768 65536"}}
TRIMDIR=$(dirname "$HEAD")/vocab-trim
mkdir -p "$OUT" "$TRIMDIR"

# Three token-distribution flavors: prose, code (corpus-like), non-English.
PROMPTS=(
  "Explain in detail how a refrigerator works, covering the compressor, the refrigerant cycle, and why the inside gets cold while the back gets warm."
  "Write a C function that parses a decimal integer from a string without using the standard library, with overflow checking, and explain each step in comments."
  "Explique en français, étape par étape, comment fonctionne un moteur à quatre temps."
)

avail_kb() { awk '/MemAvailable/{print $2}' /proc/meminfo; }
sampler() { # $1 = outfile; min MemAvailable until killed
  local min=999999999 a
  while :; do
    a=$(avail_kb)
    [ "$a" -lt "$min" ] && { min=$a; echo "$min" > "$1"; }
    sleep 0.15
  done
}

run_variant() {  # $1 = tag, $2 = mtp gguf path ("" = plain), $3 = idlist ("" = full vocab)
  local tag=$1 mtp=$2 ids=$3 sock=/tmp/lg-vocab.sock srv smp
  rm -f "$sock"
  sampler "$OUT/$tag.watermark" & smp=$!
  env ${ids:+LG_MTP_IDS="$ids"} "$BIN" -m "$TARGET" ${mtp:+-mtp "$mtp"} \
      -s "$sock" >"$OUT/$tag.load" 2>"$OUT/$tag.err" &
  srv=$!
  for _ in $(seq 1 720); do [ -e "$sock" ] && break; sleep 0.5; done
  [ -e "$sock" ] || { echo "SERVER FAILED ($tag)"; kill $srv $smp 2>/dev/null; return 1; }
  local p t n=0
  for p in 0 1 2; do
    for t in $(seq 1 $TURNS); do
      printf '%s\n' "${PROMPTS[$p]}" | "$BIN" -c "$sock" > "$OUT/$tag-p$p-t$t.out"
      n=$((n + 1))
      [ $n -eq 1 ] && { kill $smp 2>/dev/null; avail_kb > "$OUT/$tag.steady"; }
    done
  done
  kill $srv $smp 2>/dev/null; wait $srv 2>/dev/null; sleep 2
}

# ---- assemble variants ---------------------------------------------------------
# "plain" = no -mtp (the honest baseline); "full" = the untrimmed head, which on
# a 16 GB Orin fails to arm for the 12B ("mtp: device upload failed" — the
# 1.5 GiB mtp_up_h staging + 512 MiB device head don't fit next to the target)
# and then runs verify-overhead-only. Both are measurements, not mistakes.
# kN = prefix-trimmed GGUF (zero engine changes); SEL entries = the full HEAD
# gguf + LG_MTP_IDS row gather at open (phase-2 d2t build).
declare -A MTPFILE=() IDSFILE=()
VARIANTS=()
for b in ${BASE:-plain full}; do
  case $b in plain) MTPFILE[plain]="";; full) MTPFILE[full]=$HEAD;; esac
  VARIANTS+=("$b")
done
for K in $KS; do
  f=$TRIMDIR/$(basename "${HEAD%.gguf}")-k$K.gguf
  [ -s "$f" ] || python3 "$TOOLS/trim-head.py" "$HEAD" "$f" "$K" || exit 1
  MTPFILE[k$K]=$f; VARIANTS+=("k$K")
done
for s in ${SEL:-}; do
  tag=${s%%=*}
  MTPFILE[$tag]=$HEAD; IDSFILE[$tag]=${s#*=}
  [ -s "${IDSFILE[$tag]}" ] || { echo "missing idlist ${IDSFILE[$tag]}"; exit 1; }
  VARIANTS+=("$tag")
done

# ---- run (idempotent: a variant with stats on disk is not re-run) ------------
for v in "${VARIANTS[@]}"; do
  if [ -s "$OUT/$v.err" ]; then echo "=== $v already run, skipping ==="; continue; fi
  echo "=== running $v (${MTPFILE[$v]:-plain decode}${IDSFILE[$v]:+ + ${IDSFILE[$v]}}) ==="
  run_variant "$v" "${MTPFILE[$v]}" "${IDSFILE[$v]:-}" || exit 1
done

# ---- H1: byte identity ---------------------------------------------------------
echo
echo "=== H1: replies must be byte-identical to plain decode, every prompt+turn ==="
fail=0
for p in 0 1 2; do
  for t in $(seq 1 $TURNS); do
    ref=$(sha1sum "$OUT/plain-p$p-t$t.out" | cut -d' ' -f1)
    for v in "${VARIANTS[@]:1}"; do
      s=$(sha1sum "$OUT/$v-p$p-t$t.out" | cut -d' ' -f1)
      [ "$s" = "$ref" ] || { echo "MISMATCH p$p t$t $v ($s vs $ref)"; fail=1; }
    done
  done
done
[ $fail -eq 0 ] && echo "H1 PASS: all replies byte-identical" || echo "H1 FAIL"

# ---- H2/H3 measured: per-variant stats (discard each server's first turn) ----
mean() { echo "$1" | awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else printf "-"}'; }
echo
echo "=== stats (mean over turns 2..N of each variant; first turn = warmup) ==="
printf '%-8s %10s %8s %10s %11s %10s\n' variant "out tok/s" "acc %" "draft ms" "verify ms" "ttft s"
for v in "${VARIANTS[@]}"; do
  toks=$(grep "^turn:" "$OUT/$v.err" | tail -n +2 | sed 's/.* out [0-9.]*s (\([0-9.]*\) tok\/s).*/\1/')
  ttft=$(grep "^turn:" "$OUT/$v.err" | tail -n +2 | sed 's/.*ttft \([0-9.]*\)s.*/\1/')
  acc=$(grep "^mtp: *accepted" "$OUT/$v.err" | tail -n +2 | sed 's/.*(\([0-9.]*\)%).*/\1/')
  dms=$(grep "^mtp: *accepted" "$OUT/$v.err" | tail -n +2 | sed 's/.*draft [0-9.]*s (\([0-9.]*\)ms ea).*/\1/')
  vms=$(grep "^mtp: *accepted" "$OUT/$v.err" | tail -n +2 | sed 's/.*verify [0-9.]*s (\([0-9.]*\)ms ea).*/\1/')
  printf '%-8s %10s %8s %10s %11s %10s\n' "$v" "$(mean "$toks")" "$(mean "$acc")" \
         "$(mean "$dms")" "$(mean "$vms")" "$(mean "$ttft")"
done

# ---- H3 predicted: prefix-K coverage of the full head's emitted stream --------
echo
echo "=== H3: prefix-K coverage of the emitted stream (plain-decode replies) ==="
cat "$OUT"/plain-p*-t$TURNS.out > "$OUT/full-replies.txt"
"$TOKENIZE" -m "$TARGET" -f "$OUT/full-replies.txt" --ids 2>/dev/null > "$OUT/full-replies.ids"
for K in $KS; do
  python3 "$TOOLS/vocab-stats.py" missrate --prefix "$K" "$OUT/full-replies.ids" | tail -1 |
    sed "s/^total vs /K=$K: /"
done
echo "(predicted acceptance at K ~= full acceptance × that coverage)"

# ---- H4: memory ----------------------------------------------------------------
echo
echo "=== H4: MemAvailable (MiB): watermark during load+warmup / steady after turn 1 ==="
for v in "${VARIANTS[@]}"; do
  w=$(cat "$OUT/$v.watermark" 2>/dev/null || echo 0)
  s=$(cat "$OUT/$v.steady" 2>/dev/null || echo 0)
  printf '%-8s watermark %6d  steady %6d\n' "$v" $((w / 1024)) $((s / 1024))
done
echo
echo "raw logs in $OUT"
