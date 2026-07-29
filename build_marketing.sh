#!/usr/bin/env bash
# Build the marketing spot locally from the 31 default greeting clips + Maya VO.
# 1280x720 branded frame: agent clip left, name/title right; intro + CTA cards; VO over montage.
set -uo pipefail
SRC=${SRC:-/c/Users/Garoo/3d-office-mockups/assets/preview}
OUTDIR=${OUTDIR:-/c/Users/Garoo/AppData/Local/Temp/claude/C--Users-Garoo/91327859-1098-409a-8a08-f436ac4219ee/scratchpad/mkt}
VO=${VO:-/c/Users/Garoo/AppData/Local/Temp/claude/C--Users-Garoo/91327859-1098-409a-8a08-f436ac4219ee/scratchpad/mkt_vo.wav}
FONT="C\\:/Windows/Fonts/segoeuib.ttf"
BG=0x0b1f30; GOLD=0xc7a900; SEG=2.0
mkdir -p "$OUTDIR"
q(){ printf '%s' "$1" | sed "s/'/\\\\'/g"; }

# hero agents: id|Name|Role
HEROES=(
 "ceo|Victoria|Chief Executive Officer"
 "sales|Riley|Sales Producer"
 "assistant|Maya|Chief of Staff"
 "cfo|Diana|Chief Financial Officer"
 "compliance|Dana|Compliance Officer"
 "marketing|Sage|Marketing Strategist"
 "underwriting|Sterling|Chief Underwriter"
 "claims_intake|Remy|Claims Specialist"
 "tech_support|Bolt|Tech Support"
)

# intro card
ffmpeg -nostdin -y -loglevel error -f lavfi -i "color=c=$BG:s=1280x720:d=2.5" \
  -vf "drawtext=fontfile='$FONT':text='Meet Your AI Team':fontcolor=$GOLD:fontsize=76:x=(w-tw)/2:y=290:alpha='min(1,t*1.5)',drawtext=fontfile='$FONT':text='31 specialists. Around the clock. Yours.':fontcolor=white:fontsize=34:x=(w-tw)/2:y=395:alpha='min(1,max(0,(t-0.4)*1.5))'" \
  -t 2.5 -r 25 -pix_fmt yuv420p "$OUTDIR/00_intro.mp4"

i=1
for row in "${HEROES[@]}"; do
  id=${row%%|*}; rest=${row#*|}; name=${rest%%|*}; role=${rest#*|}
  clip="$SRC/greet_${id}_hallo.mp4"
  [ -s "$clip" ] || { echo "skip $id (no clip)"; continue; }
  out="$OUTDIR/$(printf '%02d' $i)_${id}.mp4"
  ffmpeg -nostdin -y -loglevel error -i "$clip" -f lavfi -i "color=c=$BG:s=1280x720:d=$SEG" \
    -filter_complex "[0:v]trim=start=0.6:duration=$SEG,setpts=PTS-STARTPTS,scale=600:600[a];\
[1:v][a]overlay=95:60[bg];\
[bg]drawtext=fontfile='$FONT':text='$(q "$name")':fontcolor=$GOLD:fontsize=64:x=760:y=285,\
drawtext=fontfile='$FONT':text='$(q "$role")':fontcolor=white:fontsize=32:x=760:y=372,\
drawtext=fontfile='$FONT':text='Mocombe Financial AI Team':fontcolor=0x6f8aa0:fontsize=22:x=760:y=430" \
    -an -t $SEG -r 25 -pix_fmt yuv420p "$out"
  echo "built $out"
  i=$((i+1))
done

# CTA card
ffmpeg -nostdin -y -loglevel error -f lavfi -i "color=c=$BG:s=1280x720:d=3.5" \
  -vf "drawtext=fontfile='$FONT':text='Your AI team is waiting.':fontcolor=white:fontsize=52:x=(w-tw)/2:y=250,drawtext=fontfile='$FONT':text='Free for 14 days':fontcolor=$GOLD:fontsize=64:x=(w-tw)/2:y=340,drawtext=fontfile='$FONT':text='crm.mocombefinancial.com':fontcolor=white:fontsize=36:x=(w-tw)/2:y=445" \
  -t 3.5 -r 25 -pix_fmt yuv420p "$OUTDIR/zz_cta.mp4"

# concat list (sorted) + mux VO
ls "$OUTDIR"/*.mp4 | sort | sed "s/.*/file '&'/" > "$OUTDIR/list.txt"
ffmpeg -nostdin -y -loglevel error -f concat -safe 0 -i "$OUTDIR/list.txt" -c copy "$OUTDIR/_montage.mp4"
# overlay VO starting at intro end (0.0 ok) ; pad audio to video length
ffmpeg -nostdin -y -loglevel error -i "$OUTDIR/_montage.mp4" -i "$VO" \
  -filter_complex "[1:a]adelay=2500|2500,apad[a]" -map 0:v -map "[a]" -shortest \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 160k "$OUTDIR/marketing_ai_team.mp4"
echo "=== DONE: $OUTDIR/marketing_ai_team.mp4 ==="
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTDIR/marketing_ai_team.mp4"
