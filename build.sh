#!/usr/bin/env bash
# Build "13.8 Billion Years in 5 Minutes" from scratch.
#   ./build.sh            -> origins.mp4 (1920x1080, 30 fps, ~5:00)
#   JOBS=8 ./build.sh     -> render with 8 parallel workers
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
CRF=${CRF:-20}
mkdir -p build

echo "== soundtrack"
python3 -m origins.music build/soundtrack.wav

echo "== frames ($JOBS workers)"
TOTAL=$(python3 -c "from origins import timeline as T; print(int(T.DURATION*T.FPS))")
CHUNK=$(( (TOTAL + JOBS - 1) / JOBS ))
: > build/parts.txt
for ((i = 0; i < JOBS; i++)); do
  s=$((i * CHUNK)); e=$(( (i + 1) * CHUNK )); (( e > TOTAL )) && e=$TOTAL
  python3 -m origins.render --start $s --end $e --crf "$CRF" --out build/part$i.mp4 &
  echo "file 'part$i.mp4'" >> build/parts.txt
done
wait

echo "== mux"
ffmpeg -y -loglevel error -f concat -safe 0 -i build/parts.txt -i build/soundtrack.wav \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -movflags +faststart -shortest origins.mp4
echo "done: origins.mp4"
