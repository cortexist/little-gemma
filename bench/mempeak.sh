#!/bin/bash
# mempeak.sh - peak memory study: little-gemma E2B QAT dictation + whisper.cpp
# Metrics: per-process VmHWM (host peak RSS), nvmap iovmm clients (GPU allocs
# on Tegra), and a MemAvailable sampler (system watermark; discounts
# reclaimable page cache, so pinned + nvmap pages count and evictable file
# pages do not - the honest "fits in N GB" number).
set -u
D=~/repos/cortexist/llama.cpp/.scratch/gemma-4-e2b
LG=~/repos/cortexist/little-gemma/build/run-cuda-i8
WC=~/repos/whisper.cpp/build/bin/whisper-cli
WM=~/repos/whisper.cpp/models/ggml-base.en.bin
WAV=~/audio/alice_00041.wav

avail_kb() { awk '/MemAvailable/{print $2}' /proc/meminfo; }
mb() { echo $(( $1 / 1024 )); }

sampler() { # $1 = outfile; tracks min MemAvailable until killed
  local min=999999999 a
  while :; do
    a=$(avail_kb)
    if [ "$a" -lt "$min" ]; then min=$a; echo "$min" > "$1"; fi
    sleep 0.15
  done
}

hello_turn() {
python3 - <<'PY'
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect("/tmp/lg.sock")
s.sendall(b"Hello!\n"); buf = b""
while b"<turn|>" not in buf:
    d = s.recv(4096)
    if not d: break
    buf += d
PY
}

BASE=$(avail_kb)
echo "baseline MemAvailable: $(mb $BASE) MB (of $(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo) MB)"

# ---- phase 1: little-gemma alone, full dictation sequence -------------------
rm -f /tmp/lg.sock /tmp/min1
"$LG" -m "$D/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf" -mtp "$D/mtp-gemma-4-E2B-it.gguf" -s /tmp/lg.sock > /tmp/mem_lg.log 2>&1 &
SRV=$!
for i in $(seq 120); do [ -S /tmp/lg.sock ] && break; sleep 0.5; done
sampler /tmp/min1 & SMP=$!
hello_turn
python3 "$(dirname "$0")/ttft_dictate.py" /tmp/lg.sock "$(dirname "$0")/line929s.txt" 0 0 > /dev/null
python3 "$(dirname "$0")/ttft_dictate.py" /tmp/lg.sock "$(dirname "$0")/line929s.txt" 6 5 > /dev/null
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
MIN1=$(cat /tmp/min1)
LGHWM=$(awk '/VmHWM/{print $2}' /proc/$SRV/status)
echo "--- little-gemma alone (hello + deferred + paced dictation) ---"
echo "  serve VmHWM (host peak RSS): $(mb $LGHWM) MB"
echo "  system watermark: min avail $(mb $MIN1) MB -> stack uses $(mb $((BASE - MIN1))) MB over baseline"
echo "  nvmap clients (GPU allocations):"
sudo -n cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | sed 's/^/    /'

# ---- phase 2: concurrent - paced dictation turn + whisper passes ------------
rm -f /tmp/min2 /tmp/whwm /tmp/nvmap_w
sampler /tmp/min2 & SMP=$!
( while [ ! -f /tmp/dict_done ]; do
    "$WC" -m "$WM" -f "$WAV" > /dev/null 2>&1 &
    WPID=$!
    while [ -d /proc/$WPID ]; do
      awk '/VmHWM/{print $2}' /proc/$WPID/status 2>/dev/null | tail -1 >> /tmp/whwm
      sudo -n cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | grep -i whisper >> /tmp/nvmap_w
      sleep 0.1
    done
  done ) &
WLOOP=$!
python3 "$(dirname "$0")/ttft_dictate.py" /tmp/lg.sock "$(dirname "$0")/line929s.txt" 6 5
touch /tmp/dict_done
wait $WLOOP 2>/dev/null
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
MIN2=$(cat /tmp/min2)
WHWM=$(sort -n /tmp/whwm 2>/dev/null | tail -1)
echo "--- concurrent: paced dictation + back-to-back whisper passes ---"
echo "  whisper-cli VmHWM (host peak RSS): $(mb ${WHWM:-0}) MB"
echo "  whisper nvmap peak: $(sort -k4 -n /tmp/nvmap_w 2>/dev/null | tail -1)"
echo "  serve VmHWM now: $(mb $(awk '/VmHWM/{print $2}' /proc/$SRV/status)) MB"
echo "  system watermark: min avail $(mb $MIN2) MB -> stack peak $(mb $((BASE - MIN2))) MB over baseline"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
rm -f /tmp/dict_done
echo "done"
