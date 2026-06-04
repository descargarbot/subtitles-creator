#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
whisper_transcribe.py

Segunda etapa:
  WAV completo + <base>_vad_speech.json
  -> Whisper por chunks VAD
  -> filtro de confianza
  -> <base>_whisper_raw.json
  -> <base>_filtered.json
  -> <base>_filtered.srt
"""

import argparse
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000

CREDIT_RE = re.compile(
    r"(amara\.org|sottotitoli\s+creati|subtitles?\s+by|captioning\s+by|opensubtitles|www\.)",
    re.IGNORECASE,
)


def sec_to_ts(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    h = ms_total // 3_600_000
    ms_total %= 3_600_000
    m = ms_total // 60_000
    ms_total %= 60_000
    s = ms_total // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\.\s*\.\s*\.", "…", text)
    text = re.sub(r"\.{4,}", "…", text)

    # c 'è -> c'è / l 'anno -> l'anno
    text = re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+'\s*", r"\1'", text)

    # E' -> È
    text = re.sub(r"\b[Ee]'\s+", "È ", text)

    text = re.sub(r"!{4,}", "!!!", text)
    text = re.sub(r"\?{4,}", "???", text)

    return text.strip()


def visible_len(text: str) -> int:
    return len(text.replace("\n", " ").strip())


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0

    compressed = zlib.compress(raw)
    return len(raw) / max(1, len(compressed))


def parse_temperature(value: str):
    value = str(value).strip()
    if "," in value:
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return float(value)


def looks_like_long_laughter(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z]+", "", text).lower()

    if len(compact) < 24:
        return False

    if re.fullmatch(r"(ha|ah|haha|ahaha|eheh|hehe|jaja)+", compact):
        return True

    if re.search(r"(.)\1{20,}", compact):
        return True

    return False


def looks_like_repetition(text: str) -> bool:
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())

    if len(toks) < 10:
        return False

    uniq = set(toks)

    allowed = {"no", "si", "sì", "ciao", "ah", "eh", "oh"}

    if len(uniq) <= 2 and not uniq.issubset(allowed):
        return True

    return False


def is_bad_text(text: str, max_chars: int) -> tuple[bool, str]:
    text = normalize_text(text)

    if not text:
        return True, "empty"

    if CREDIT_RE.search(text):
        return True, "credit"

    if visible_len(text) > max_chars:
        return True, "too_long_text"

    if looks_like_long_laughter(text):
        return True, "long_laughter"

    if looks_like_repetition(text):
        return True, "repetition"

    return False, ""


def load_vad_regions(vad_json: Path, duration: float):
    if not vad_json.exists():
        print(f"⚠️ No existe {vad_json}, se usará el audio completo.")
        return [{"id": 1, "start": 0.0, "end": duration, "duration": duration}]

    data = json.loads(vad_json.read_text(encoding="utf-8"))
    regions = []

    for i, r in enumerate(data.get("regions", []), start=1):
        try:
            st = float(r["start"])
            en = float(r["end"])
        except Exception:
            continue

        st = max(0.0, st)
        en = min(duration, en) if duration > 0 else en

        if en - st >= 0.15:
            regions.append(
                {
                    "id": int(r.get("id", i)),
                    "start": st,
                    "end": en,
                    "duration": en - st,
                }
            )

    regions.sort(key=lambda x: (x["start"], x["end"]))
    return regions


def detect_language(model, whisper_module, audio_np, regions, max_seconds=30.0):
    chunks = []
    total = 0.0

    for r in regions:
        st = int(r["start"] * SAMPLE_RATE)
        en = int(r["end"] * SAMPLE_RATE)

        if en <= st:
            continue

        piece = audio_np[st:en]
        if piece.size == 0:
            continue

        remaining = max_seconds - total
        if remaining <= 0:
            break

        max_samples = int(remaining * SAMPLE_RATE)
        piece = piece[:max_samples]

        chunks.append(piece)
        total += piece.size / SAMPLE_RATE

        if total >= max_seconds:
            break

    if not chunks:
        return None, {}

    sample = np.concatenate(chunks)
    sample = whisper_module.pad_or_trim(sample)

    mel = whisper_module.log_mel_spectrogram(sample).to(model.device)
    _, probs = model.detect_language(mel)

    lang = max(probs, key=probs.get)
    return lang, probs


def segment_word_stats(seg):
    words = seg.get("words", [])
    probs = [float(w.get("prob", 1.0)) for w in words if w.get("prob") is not None]

    if probs:
        avg_word_prob = sum(probs) / len(probs)
        low_word_ratio = sum(1 for p in probs if p < 0.15) / len(probs)
    else:
        avg_word_prob = 0.0
        low_word_ratio = 1.0

    if words:
        zero_time_ratio = sum(
            1 for w in words if float(w.get("end", 0)) - float(w.get("start", 0)) <= 0.035
        ) / len(words)

        max_word_duration = max(
            float(w.get("end", 0)) - float(w.get("start", 0)) for w in words
        )
    else:
        zero_time_ratio = 1.0
        max_word_duration = 0.0

    dur = max(0.001, float(seg["end"]) - float(seg["start"]))
    cps = visible_len(seg.get("text", "")) / dur

    return {
        "avg_word_prob": avg_word_prob,
        "low_word_ratio": low_word_ratio,
        "zero_time_ratio": zero_time_ratio,
        "max_word_duration": max_word_duration,
        "cps": cps,
        "word_count": len(words),
    }


def is_good_segment(seg, args):
    text = normalize_text(seg.get("text", ""))

    bad, reason = is_bad_text(text, args.max_segment_chars)
    if bad:
        return False, reason

    start = float(seg.get("start", 0))
    end = float(seg.get("end", 0))
    dur = end - start

    if dur < args.min_segment_duration:
        return False, "too_short"

    avg_logprob = float(seg.get("avg_logprob", 0.0))
    no_speech_prob = float(seg.get("no_speech_prob", 0.0))
    comp = float(seg.get("compression_ratio", compression_ratio(text)))

    stats = segment_word_stats(seg)

    # Texto extremadamente comprimible: típico de repeticiones/alucinaciones.
    if comp > args.max_compression_ratio and visible_len(text) > 20:
        return False, "compression"

    # Logprob muy bajo. Excepción: interjecciones cortas, porque "No!", "Oh!" suelen tener baja confianza.
    if avg_logprob < args.min_avg_logprob:
        short_interjection = stats["word_count"] <= 2 and visible_len(text) <= 18 and dur <= 2.5
        if not short_interjection:
            return False, "avg_logprob"

    # Probabilidad de no habla alta + logprob flojo.
    if no_speech_prob > args.max_no_speech_prob and avg_logprob < args.nospeech_logprob_limit:
        return False, "no_speech"

    if stats["word_count"] >= 3 and stats["avg_word_prob"] < args.min_avg_word_prob:
        return False, "word_prob"

    if stats["word_count"] >= 4 and stats["zero_time_ratio"] > args.max_zero_time_ratio:
        return False, "zero_time_words"

    if (
        stats["word_count"] >= 6
        and stats["low_word_ratio"] > args.max_low_word_ratio
        and avg_logprob < -0.35
    ):
        return False, "low_word_ratio"

    if (
        stats["max_word_duration"] > args.max_word_duration
        and stats["zero_time_ratio"] > 0.25
    ):
        return False, "bad_word_timestamps"

    if stats["cps"] > args.max_raw_cps and avg_logprob < -0.35:
        return False, "raw_cps"

    return True, ""


def dedupe_repeated_short_segments(segments, max_repeats=3):
    out = []
    last_key = None
    last_start = None
    run_count = 0

    for seg in segments:
        key = normalize_text(seg["text"]).lower()
        short = visible_len(key) <= 22

        if (
            short
            and key == last_key
            and last_start is not None
            and seg["start"] - last_start <= 6.0
        ):
            run_count += 1
        else:
            run_count = 1
            last_key = key

        last_start = seg["start"]

        if short and run_count > max_repeats:
            continue

        out.append(seg)

    return out


def write_srt(segments, path: Path):
    segments = sorted(segments, key=lambda x: (x["start"], x["end"]))

    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            st = float(seg["start"])
            en = float(seg["end"])

            if en <= st:
                en = st + 0.5

            f.write(f"{i}\n")
            f.write(f"{sec_to_ts(st)} --> {sec_to_ts(en)}\n")
            f.write(f"{normalize_text(seg['text'])}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe con Whisper solo regiones detectadas por VAD."
    )

    parser.add_argument("audio", help="WAV 16 kHz mono.")
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--vad-json", default=None)

    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--task", default="transcribe")

    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", default="0")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--initial-prompt", default=None)

    parser.add_argument("--decode-logprob-threshold", type=float, default=-2.0)
    parser.add_argument("--decode-no-speech-threshold", type=float, default=0.60)
    parser.add_argument("--decode-compression-ratio-threshold", type=float, default=2.6)

    parser.add_argument("--min-segment-duration", type=float, default=0.16)
    parser.add_argument("--min-avg-logprob", type=float, default=-1.80)
    parser.add_argument("--max-no-speech-prob", type=float, default=0.88)
    parser.add_argument("--nospeech-logprob-limit", type=float, default=-0.45)

    parser.add_argument("--min-avg-word-prob", type=float, default=0.22)
    parser.add_argument("--max-zero-time-ratio", type=float, default=0.55)
    parser.add_argument("--max-low-word-ratio", type=float, default=0.55)
    parser.add_argument("--max-word-duration", type=float, default=8.0)

    parser.add_argument("--max-raw-cps", type=float, default=34.0)
    parser.add_argument("--max-compression-ratio", type=float, default=2.7)
    parser.add_argument("--max-segment-chars", type=int, default=360)
    parser.add_argument("--max-consecutive-repeats", type=int, default=3)

    args = parser.parse_args()

    try:
        import whisper
    except ImportError:
        sys.exit("❌ No se pudo importar whisper. Instalá openai-whisper.")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"❌ No existe el audio: {audio_path}")

    base = (
        args.output_base
        if args.output_base
        else str(audio_path.with_suffix(""))
    )

    vad_json = Path(args.vad_json) if args.vad_json else Path(f"{base}_vad_speech.json")

    raw_json = Path(f"{base}_whisper_raw.json")
    filtered_json = Path(f"{base}_filtered.json")
    filtered_srt = Path(f"{base}_filtered.srt")

    print("==============================================")
    print(" WHISPER POR REGIONES VAD")
    print("----------------------------------------------")
    print(f"Audio        : {audio_path}")
    print(f"Base salida  : {base}")
    print(f"Modelo       : {args.model}")
    print(f"VAD JSON     : {vad_json}")
    print("==============================================")

    t0 = time.time()

    print("⏳ Cargando audio…")
    audio_np = whisper.load_audio(str(audio_path))
    duration = len(audio_np) / SAMPLE_RATE

    regions = load_vad_regions(vad_json, duration)

    if not regions:
        sys.exit("❌ VAD no produjo regiones de voz. No hay nada para transcribir.")

    print(f"✅ Regiones a transcribir: {len(regions)}")

    print(f"⏳ Cargando Whisper: {args.model}")
    if args.device:
        model = whisper.load_model(args.model, device=args.device)
    else:
        model = whisper.load_model(args.model)

    language_arg = args.language.strip().lower()
    language = None if language_arg in {"", "auto", "none", "null"} else args.language

    detected_language = None
    language_probs = {}

    if language is None:
        try:
            detected_language, language_probs = detect_language(
                model, whisper, audio_np, regions
            )
            language = detected_language
            print(f"🌎 Idioma detectado: {language}")
        except Exception as e:
            print(f"⚠️ No pude detectar idioma, Whisper lo hará por chunk: {e}")
            language = None
    else:
        print(f"🌎 Idioma fijado: {language}")

    temperature = parse_temperature(args.temperature)

    raw_segments = []
    dropped = []

    global_id = 0

    for idx, region in enumerate(regions, start=1):
        r_start = float(region["start"])
        r_end = float(region["end"])

        start_sample = max(0, int(round(r_start * SAMPLE_RATE)))
        end_sample = min(len(audio_np), int(round(r_end * SAMPLE_RATE)))

        if end_sample <= start_sample:
            continue

        chunk = audio_np[start_sample:end_sample]
        chunk_duration = len(chunk) / SAMPLE_RATE
        offset = start_sample / SAMPLE_RATE

        if chunk_duration < 0.15:
            continue

        print(
            f"🎙 [{idx}/{len(regions)}] "
            f"{sec_to_ts(offset)} --> {sec_to_ts(offset + chunk_duration)} "
            f"({chunk_duration:.2f}s)"
        )

        try:
            result = model.transcribe(
                chunk,
                language=language,
                task=args.task,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=temperature,
                beam_size=args.beam_size,
                verbose=False,
                initial_prompt=args.initial_prompt,
                logprob_threshold=args.decode_logprob_threshold,
                no_speech_threshold=args.decode_no_speech_threshold,
                compression_ratio_threshold=args.decode_compression_ratio_threshold,
                fp16=(getattr(model, "device", None) is not None and model.device.type == "cuda"),
            )
        except Exception as e:
            print(f"⚠️ Error transcribiendo región {idx}: {e}")
            continue

        for s in result.get("segments", []):
            rel_start = float(s.get("start", 0.0))
            rel_end = float(s.get("end", 0.0))

            # Whisper a veces genera cosas sobre el padding interno.
            if rel_start > chunk_duration + 0.15:
                continue

            rel_start = max(0.0, min(rel_start, chunk_duration))
            rel_end = max(0.0, min(rel_end, chunk_duration))

            abs_start = offset + rel_start
            abs_end = offset + rel_end

            text = normalize_text(s.get("text", ""))

            words = []

            for w in s.get("words", []) or []:
                token = str(w.get("word", "")).strip()
                if not token:
                    continue

                try:
                    ws_rel = float(w.get("start", rel_start))
                    we_rel = float(w.get("end", rel_end))
                except Exception:
                    continue

                if ws_rel > chunk_duration + 0.15:
                    continue

                ws_rel = max(0.0, min(ws_rel, chunk_duration))
                we_rel = max(0.0, min(we_rel, chunk_duration))

                ws = offset + ws_rel
                we = offset + we_rel

                if we <= ws:
                    we = min(offset + chunk_duration, ws + 0.04)

                prob = float(w.get("probability", w.get("prob", 1.0)) or 1.0)

                words.append(
                    {
                        "word": token,
                        "start": round(ws, 3),
                        "end": round(we, 3),
                        "prob": prob,
                    }
                )

            if words:
                abs_start = min(abs_start, words[0]["start"])
                abs_end = max(abs_end, words[-1]["end"])

            global_id += 1

            seg = {
                "id": global_id,
                "chunk_id": idx,
                "vad_region": {
                    "id": region.get("id", idx),
                    "start": round(offset, 3),
                    "end": round(offset + chunk_duration, 3),
                },
                "start": round(abs_start, 3),
                "end": round(abs_end, 3),
                "text": text,
                "avg_logprob": float(s.get("avg_logprob", 0.0)),
                "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
                "compression_ratio": float(
                    s.get("compression_ratio", compression_ratio(text))
                ),
                "words": words,
            }

            raw_segments.append(seg)

    raw_segments.sort(key=lambda x: (x["start"], x["end"]))

    filtered = []

    for seg in raw_segments:
        ok, reason = is_good_segment(seg, args)
        if ok:
            filtered.append(seg)
        else:
            dropped.append(
                {
                    "id": seg.get("id"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "reason": reason,
                }
            )

    filtered = dedupe_repeated_short_segments(
        filtered,
        max_repeats=args.max_consecutive_repeats,
    )

    raw_payload = {
        "audio": str(audio_path),
        "duration": duration,
        "model": args.model,
        "language": language,
        "detected_language": detected_language,
        "language_probs": language_probs,
        "vad_json": str(vad_json),
        "segments": raw_segments,
    }

    filtered_payload = {
        "audio": str(audio_path),
        "duration": duration,
        "model": args.model,
        "language": language,
        "vad_json": str(vad_json),
        "segments": filtered,
        "dropped": dropped,
    }

    raw_json.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    filtered_json.write_text(
        json.dumps(filtered_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_srt(filtered, filtered_srt)

    elapsed = time.time() - t0

    print("==============================================")
    print("✅ Whisper terminado")
    print("----------------------------------------------")
    print(f"Tiempo        : {elapsed:.1f}s")
    print(f"Raw segmentos : {len(raw_segments)}")
    print(f"Filtrados     : {len(filtered)}")
    print(f"Descartados   : {len(dropped)}")
    print(f"💾 {raw_json}")
    print(f"💾 {filtered_json}")
    print(f"💾 {filtered_srt}")
    print("==============================================")


if __name__ == "__main__":
    main()
