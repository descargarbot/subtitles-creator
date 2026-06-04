#!/usr/bin/env bash
# run_pipeline.sh - Pipeline: audio -> VAD -> Whisper por chunks -> filtro -> diarización -> SRT final
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  ./run_pipeline.sh <video|audio> [lang_iso]

Variables opcionales:
  WHISPER_MODEL=medium
  PYANNOTE_ENV=diarization-env
  KEEP_WORK_WAV=1

Ejemplo:
  WHISPER_MODEL=medium ./run_pipeline.sh pelicula.mkv zh
EOF
  exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || usage
[[ -f "$1" ]] || { echo "❌ El archivo '$1' no existe"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYANNOTE_ENV="${PYANNOTE_ENV:-diarization-env}"
PYANNOTE_PY="${SCRIPT_DIR}/${PYANNOTE_ENV}/bin/python"

[[ -x "$PYANNOTE_PY" ]] || {
  echo "❌ No encuentro Python de pyannote en: $PYANNOTE_PY"
  echo "   Ajustá PYANNOTE_ENV o revisá tu entorno virtual."
  exit 1
}

INPUT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")"
INPUT_DIR="$(dirname "$INPUT")"
INPUT_NAME="$(basename "$INPUT")"
BASENAME="${INPUT_NAME%.*}"

OUT_BASE="${INPUT_DIR}/${BASENAME}"
WORK_WAV="${INPUT_DIR}/${BASENAME}__work_16k.wav"

WHISPER_MODEL="large-v3"
LANGUAGE="en"

cleanup() {
  if [[ "${KEEP_WORK_WAV:-0}" != "1" ]]; then
    rm -f "$WORK_WAV"
  fi
}
trap cleanup EXIT

timer() {
  python3 - <<'EOF'
import time
print(int(time.time() * 1000))
EOF
}

format_elapsed() {
  local total_ms=$1
  local total_sec=$((total_ms / 1000))
  local days=$((total_sec / 86400))
  local hours=$(( (total_sec % 86400) / 3600 ))
  local minutes=$(( (total_sec % 3600) / 60 ))
  local seconds=$((total_sec % 60))

  if (( days > 0 )); then
    printf "%02d:%02d:%02d:%02d" "$days" "$hours" "$minutes" "$seconds"
  else
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
  fi
}

echo "=============================================="
echo "  PIPELINE PROFESIONAL DE SUBTÍTULOS"
echo "----------------------------------------------"
echo "  Input        : $INPUT"
echo "  Base salida  : $OUT_BASE"
echo "  Whisper      : $WHISPER_MODEL"
echo "  Idioma       : $LANGUAGE"
echo "  Pyannote env : $PYANNOTE_ENV"
echo "  Inicio       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

T0=$(timer)

echo "➤ 0/4 Normalizando audio a WAV mono 16 kHz…"
T_STEP=$(timer)
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" -vn -ac 1 -ar 16000 -sample_fmt s16 \
  "$WORK_WAV"
T_END=$(timer)
echo "   ⏱ $(format_elapsed $((T_END - T_STEP)))"

echo "➤ 1/4 VAD + diarización con Pyannote…"
T_STEP=$(timer)
"$PYANNOTE_PY" "$SCRIPT_DIR/filter_with_vad.py" "$WORK_WAV" \
  --output-base "$OUT_BASE"
T_END=$(timer)
echo "   ⏱ $(format_elapsed $((T_END - T_STEP)))"

echo "➤ 2/4 Whisper solo sobre regiones VAD…"
T_STEP=$(timer)
"$PYTHON_BIN" "$SCRIPT_DIR/whisper_transcribe.py" "$WORK_WAV" \
  --output-base "$OUT_BASE" \
  --vad-json "${OUT_BASE}_vad_speech.json" \
  --model "$WHISPER_MODEL" \
  --language "$LANGUAGE"
T_END=$(timer)
echo "   ⏱ $(format_elapsed $((T_END - T_STEP)))"

echo "➤ 3/4 Generando SRT final estilo scene/Netflix…"
T_STEP=$(timer)
"$PYTHON_BIN" "$SCRIPT_DIR/generate_subs_final.py" "$OUT_BASE"
T_END=$(timer)
echo "   ⏱ $(format_elapsed $((T_END - T_STEP)))"

TF=$(timer)
ELAPSED=$(format_elapsed $((TF - T0)))

echo "=============================================="
echo "✅ PIPELINE COMPLETADO en ${ELAPSED}"
echo "----------------------------------------------"
echo "Archivos generados:"
echo "  • ${OUT_BASE}_vad_speech.json"
echo "  • ${OUT_BASE}_vad_debug.srt"
echo "  • ${OUT_BASE}_pyannote_speakers.json"
echo "  • ${OUT_BASE}_whisper_raw.json"
echo "  • ${OUT_BASE}_filtered.json"
echo "  • ${OUT_BASE}_filtered.srt"
echo "  • ${OUT_BASE}_final.srt"
echo "=============================================="
