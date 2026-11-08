#!/usr/bin/env bash
# Assembles the final Ramp Model Labs demo from the two Studio takes.
#
#   demo/.work/studio-output/takeA.mp4   hero + copilot (silent master from studio_finish)
#   demo/.work/studio-output/takeB.mp4   finished lab + playground + workflow + outro
#   demo/.work/studio-output/take{A,B}.__voice.wav   optional local voice-over tracks (VOICE=1 finishes)
#   demo/higgsfield/01-hook.mp4, 03-outro.mp4        optional Higgsfield b-roll (python3 demo/higgsfield_broll.py)
#   "Moves Like Jagger - Official Instrumental.mp3"  the music bed (repo root)
#
# Outputs demo/ramp-model-labs-demo.mp4 (music only) and demo/ramp-model-labs-demo-narrated.mp4 (voice + ducked music).
# env: DROP_AT=<seconds into the video where the beat should land> (default: the first cut after the opening card)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/demo/.work/studio-output}"          # where the Studio takes live
FINAL="${FINAL:-$ROOT/demo}"                           # where the finished videos go
MUSIC="$ROOT/Moves Like Jagger - Official Instrumental.mp3"
BROLL="${BROLL:-$ROOT/demo/higgsfield}"
TMP="$ROOT/demo/.work/assemble"; mkdir -p "$TMP" "$FINAL"
A="$OUT/takeA.mp4"; B="$OUT/takeB.mp4"
[ -f "$A" ] && [ -f "$B" ] || { echo "need $A and $B (run the two takes first)"; exit 1; }

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
DA=$(dur "$A"); DB=$(dur "$B")

# ---- 1. picture: [hook b-roll] + takeA + takeB + [outro b-roll], 1920x1080 25fps, short crossfades on the b-roll joins
inputs=(); filt=""; n=0; names=()
add() { inputs+=(-i "$1"); names+=("[v$n]"); filt+="[$n:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=25,format=yuv420p,setsar=1[v$n];"; n=$((n+1)); }
HOOK=""; OUTRO=""
[ -f "$BROLL/01-hook.mp4" ] && HOOK="$BROLL/01-hook.mp4"
[ -f "$BROLL/03-outro.mp4" ] && OUTRO="$BROLL/03-outro.mp4"
[ -n "$HOOK" ] && add "$HOOK"
add "$A"; add "$B"
[ -n "$OUTRO" ] && add "$OUTRO"
filt+="$(printf '%s' "${names[@]}")concat=n=$n:v=1:a=0[vout]"
ffmpeg -y -hide_banner -loglevel error "${inputs[@]}" -filter_complex "$filt" -map "[vout]" -c:v libx264 -preset slow -crf 19 -pix_fmt yuv420p -movflags +faststart "$TMP/picture.mp4"
TOTAL=$(dur "$TMP/picture.mp4")
LEAD=0; [ -n "$HOOK" ] && LEAD=$(dur "$HOOK")
echo "picture: ${TOTAL}s (lead ${LEAD}s, A ${DA}s, B ${DB}s)"

# ---- 2. music bed: the beat drops at 7.5 s into the song; land it on DROP_AT (default: first reveal after the title card)
DROP_SONG=7.5
DROP_AT="${DROP_AT:-$(python3 -c "print(round($LEAD + 3.6, 2))")}"
SS=$(python3 -c "print(max(0.0, round($DROP_SONG - $DROP_AT, 3)))")
PAD=$(python3 -c "print(max(0.0, round($DROP_AT - $DROP_SONG, 3)))")
FADE_OUT=$(python3 -c "print(round($TOTAL - 4.0, 3))")
ffmpeg -y -hide_banner -loglevel error -ss "$SS" -i "$MUSIC" -t "$TOTAL" \
  -af "adelay=$(python3 -c "print(int($PAD*1000))")|$(python3 -c "print(int($PAD*1000))"),afade=t=in:st=0:d=0.8,afade=t=out:st=$FADE_OUT:d=4,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000" \
  -ac 2 "$TMP/music.wav"

# ---- 3. music-only master
ffmpeg -y -hide_banner -loglevel error -i "$TMP/picture.mp4" -i "$TMP/music.wav" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest "$FINAL/ramp-model-labs-demo.mp4"
echo "wrote demo/ramp-model-labs-demo.mp4"

# ---- 4. narrated variant: voice tracks placed at their take offsets, music ducked under speech
VA="$OUT/takeA.__voice.wav"; VB="$OUT/takeB.__voice.wav"
if [ -f "$VA" ] || [ -f "$VB" ]; then
  OFF_A=$(python3 -c "print(int($LEAD*1000))"); OFF_B=$(python3 -c "print(int(($LEAD+$DA)*1000))")
  vin=(); vf=""; k=0; mix=""
  if [ -f "$VA" ]; then vin+=(-i "$VA"); vf+="[$k:a]aresample=48000,adelay=$OFF_A|$OFF_A[va];"; mix+="[va]"; k=$((k+1)); fi
  if [ -f "$VB" ]; then vin+=(-i "$VB"); vf+="[$k:a]aresample=48000,adelay=$OFF_B|$OFF_B[vb];"; mix+="[vb]"; k=$((k+1)); fi
  cnt=$(( ${#vin[@]} / 2 ))
  if [ "$cnt" -gt 1 ]; then vf+="${mix}amix=inputs=$cnt:normalize=0,apad[voice];"; else vf+="${mix}apad[voice];"; fi
  ffmpeg -y -hide_banner -loglevel error "${vin[@]}" -filter_complex "${vf%;}" -map "[voice]" -ac 2 -t "$TOTAL" "$TMP/voice.wav"
  ffmpeg -y -hide_banner -loglevel error -i "$TMP/picture.mp4" -i "$TMP/voice.wav" -i "$TMP/music.wav" -filter_complex \
    "[1:a]volume=1.0[v];[2:a][v]sidechaincompress=threshold=0.05:ratio=8:attack=40:release=600:makeup=1[duck];[duck][1:a]amix=inputs=2:normalize=0:weights=1 1.6[a]" \
    -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -shortest "$FINAL/ramp-model-labs-demo-narrated.mp4"
  echo "wrote demo/ramp-model-labs-demo-narrated.mp4"
fi

# ---- 5. captions: merge the per-take .srt files with the right offsets
python3 - "$OUT/takeA.srt" "$OUT/takeB.srt" "$LEAD" "$DA" "$FINAL/ramp-model-labs-demo.srt" <<'PY'
import re, sys
a, b, lead, da, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
def parse(p, off):
    try: txt = open(p, encoding="utf8").read()
    except FileNotFoundError: return []
    cues = []
    for blk in re.split(r"\n\s*\n", txt.strip()):
        lines = blk.strip().split("\n")
        if len(lines) < 3: continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m: continue
        g = list(map(int, m.groups()))
        s = g[0]*3600+g[1]*60+g[2]+g[3]/1000+off; e = g[4]*3600+g[5]*60+g[6]+g[7]/1000+off
        cues.append((s, e, "\n".join(lines[2:])))
    return cues
def fmt(t):
    h = int(t//3600); m = int(t%3600//60); s = int(t%60); ms = int(round((t-int(t))*1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
cues = parse(a, lead) + parse(b, lead+da)
with open(out, "w", encoding="utf8") as f:
    for i, (s, e, t) in enumerate(cues, 1): f.write(f"{i}\n{fmt(s)} --> {fmt(e)}\n{t}\n\n")
print(f"wrote {out} ({len(cues)} cues)")
PY
