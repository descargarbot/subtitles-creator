#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_with_vad.py

Primera etapa del pipeline:
  audio WAV 16 kHz mono
  -> Pyannote VAD
  -> regiones de voz
  -> diarización

Genera:
  <base>_vad_speech.json
  <base>_vad_debug.srt
  <base>_pyannote_speakers.json

Compatible con distintas versiones de pyannote.audio:
- Algunas aceptan onset/offset.
- Otras aceptan threshold.
- Otras solo aceptan min_duration_on/min_duration_off.
"""

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path


def sec_to_ts(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    h = ms_total // 3_600_000
    ms_total %= 3_600_000
    m = ms_total // 60_000
    ms_total %= 60_000
    s = ms_total // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return 0.0


def hf_token(args):
    return (
        args.hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("PYANNOTE_AUTH_TOKEN")
    )


def load_pyannote_model(model_name: str, token):
    from pyannote.audio import Model

    if token:
        try:
            return Model.from_pretrained(model_name, use_auth_token=token)
        except TypeError:
            return Model.from_pretrained(model_name, token=token)

    return Model.from_pretrained(model_name)


def load_pyannote_pipeline(model_name: str, token):
    from pyannote.audio import Pipeline

    if token:
        try:
            return Pipeline.from_pretrained(model_name, use_auth_token=token)
        except TypeError:
            return Pipeline.from_pretrained(model_name, token=token)

    return Pipeline.from_pretrained(model_name)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def move_to_device(obj, device: str):
    if device == "cpu":
        return obj

    try:
        import torch

        obj.to(torch.device(device))
    except Exception as e:
        print(f"⚠️ No pude mover a {device}, sigo como está: {e}")

    return obj


def flatten_parameter_keys(params, prefix=""):
    """
    Convierte parámetros posiblemente anidados de pyannote en un set de claves.
    Ej:
      {"onset": ..., "offset": ...}
      {"segmentation": {"threshold": ...}}
    """
    keys = set()

    if not isinstance(params, dict):
        return keys

    for k, v in params.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        keys.add(full)
        keys.add(str(k))

        if isinstance(v, dict):
            keys |= flatten_parameter_keys(v, full)

    return keys


def get_available_parameters(pipeline):
    try:
        params = pipeline.parameters(instantiated=False)
        return flatten_parameter_keys(params), params
    except Exception:
        return set(), {}


def instantiate_vad_compat(vad, args):
    """
    Compatibilidad entre versiones de pyannote:
    - Algunas VAD usan onset/offset.
    - Otras usan threshold.
    - Otras solo min_duration_on/min_duration_off.

    Probamos primero por introspección. Si falla, hacemos reintentos.
    """

    available_keys, raw_params = get_available_parameters(vad)

    print("ℹ️ Parámetros VAD disponibles según pyannote:")
    if raw_params:
        print(raw_params)
    else:
        print("   No se pudieron introspectar. Se intentará compatibilidad automática.")

    full_candidates = {
        "onset": args.vad_onset,
        "offset": args.vad_offset,
        "threshold": args.vad_threshold,
        "min_duration_on": args.min_duration_on,
        "min_duration_off": args.min_duration_off,
    }

    # Si pudimos introspectar, mandamos solo los parámetros soportados.
    if available_keys:
        params = {}

        for k, v in full_candidates.items():
            if k in available_keys:
                params[k] = v

        # Algunos pyannote exponen segmentation.threshold pero instantiate espera threshold.
        if "threshold" not in params and "segmentation.threshold" in available_keys:
            params["threshold"] = args.vad_threshold

        print(f"ℹ️ Instanciando VAD con parámetros compatibles: {params}")

        try:
            return vad.instantiate(params)
        except Exception as e:
            print(f"⚠️ Falló instantiate con parámetros introspectados: {e}")

    # Fallbacks por compatibilidad.
    attempts = [
        {
            "onset": args.vad_onset,
            "offset": args.vad_offset,
            "min_duration_on": args.min_duration_on,
            "min_duration_off": args.min_duration_off,
        },
        {
            "threshold": args.vad_threshold,
            "min_duration_on": args.min_duration_on,
            "min_duration_off": args.min_duration_off,
        },
        {
            "min_duration_on": args.min_duration_on,
            "min_duration_off": args.min_duration_off,
        },
        {},
    ]

    last_error = None

    for params in attempts:
        try:
            print(f"ℹ️ Intentando instantiate VAD con: {params}")
            return vad.instantiate(params)
        except Exception as e:
            last_error = e
            print(f"⚠️ No compatible: {e}")

    raise RuntimeError(f"No pude instanciar VAD. Último error: {last_error}")


def timeline_to_regions(vad_result):
    """
    Convierte salida de Pyannote a lista:
      [{"start": float, "end": float}, ...]
    """

    if hasattr(vad_result, "get_timeline"):
        timeline = vad_result.get_timeline().support()
    elif hasattr(vad_result, "support"):
        timeline = vad_result.support()
    else:
        timeline = vad_result

    regions = []

    if hasattr(timeline, "itersegments"):
        iterator = timeline.itersegments()
    else:
        iterator = timeline

    for seg in iterator:
        regions.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
            }
        )

    return regions


def merge_regions(regions, merge_gap: float):
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: (r["start"], r["end"]))
    merged = [dict(regions[0])]

    for r in regions[1:]:
        last = merged[-1]

        if r["start"] <= last["end"] + merge_gap:
            last["end"] = max(last["end"], r["end"])
        else:
            merged.append(dict(r))

    return merged


def split_long_regions(regions, max_region: float):
    if max_region <= 0:
        return regions

    out = []

    for r in regions:
        start = r["start"]
        end = r["end"]

        if end - start <= max_region:
            out.append(dict(r))
            continue

        cur = start

        while cur < end:
            piece_end = min(end, cur + max_region)

            if piece_end - cur >= 0.25:
                out.append(
                    {
                        "start": cur,
                        "end": piece_end,
                    }
                )

            cur = piece_end

    return out


def postprocess_vad_regions(raw_regions, duration, args):
    regions = []

    for r in raw_regions:
        st = max(0.0, float(r["start"]) - args.pad_start)
        en = float(r["end"]) + args.pad_end

        if duration > 0:
            en = min(duration, en)

        if en - st >= args.min_region:
            regions.append(
                {
                    "start": st,
                    "end": en,
                }
            )

    regions = merge_regions(regions, args.merge_gap)
    regions = split_long_regions(regions, args.max_region)

    final = []

    for i, r in enumerate(regions, start=1):
        if r["end"] - r["start"] >= args.min_region:
            final.append(
                {
                    "id": i,
                    "start": round(r["start"], 3),
                    "end": round(r["end"], 3),
                    "duration": round(r["end"] - r["start"], 3),
                }
            )

    return final


def write_vad_srt(path: Path, regions):
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(regions, start=1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_ts(r['start'])} --> {sec_to_ts(r['end'])}\n")
            f.write("[voice]\n\n")


def merge_same_speaker_turns(turns, gap=0.25):
    grouped = {}

    for t in turns:
        grouped.setdefault(t["speaker"], []).append(t)

    merged_all = []

    for speaker, items in grouped.items():
        items = sorted(items, key=lambda x: (x["start"], x["end"]))
        merged = []

        for t in items:
            if not merged:
                merged.append(dict(t))
                continue

            last = merged[-1]

            if t["start"] <= last["end"] + gap:
                last["end"] = max(last["end"], t["end"])
            else:
                merged.append(dict(t))

        merged_all.extend(merged)

    merged_all.sort(key=lambda x: (x["start"], x["end"]))
    return merged_all


def run_vad(audio: Path, args):
    from pyannote.audio.pipelines import VoiceActivityDetection

    token = hf_token(args)
    device = choose_device(args.device)

    print(f"⏳ Cargando VAD: {args.vad_model}")

    vad_model = load_pyannote_model(args.vad_model, token)

    try:
        vad = VoiceActivityDetection(segmentation=vad_model)
    except TypeError:
        vad = VoiceActivityDetection(vad_model)

    vad = instantiate_vad_compat(vad, args)
    vad = move_to_device(vad, device)

    print(f"🎚 Ejecutando VAD en {device}…")

    result = vad(str(audio))
    return timeline_to_regions(result)


def run_diarization(audio: Path, args):
    if args.skip_diarization:
        print("ℹ️ Diarización omitida por --skip-diarization.")
        return []

    token = hf_token(args)
    device = choose_device(args.device)

    print(f"⏳ Cargando diarización: {args.diar_model}")

    pipeline = load_pyannote_pipeline(args.diar_model, token)
    pipeline = move_to_device(pipeline, device)

    kwargs = {}

    if args.num_speakers is not None:
        kwargs["num_speakers"] = args.num_speakers

    if args.min_speakers is not None:
        kwargs["min_speakers"] = args.min_speakers

    if args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    print(f"🗣 Ejecutando diarización en {device}…")

    diarization = pipeline(str(audio), **kwargs)

    turns = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        st = float(turn.start)
        en = float(turn.end)

        if en - st < args.min_diar_turn:
            continue

        turns.append(
            {
                "start": round(st, 3),
                "end": round(en, 3),
                "speaker": str(speaker),
            }
        )

    turns = merge_same_speaker_turns(turns, gap=args.merge_speaker_gap)
    return turns


def main():
    parser = argparse.ArgumentParser(
        description="VAD inicial + diarización con Pyannote."
    )

    parser.add_argument("audio", help="WAV 16 kHz mono.")
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--vad-model", default="pyannote/segmentation-3.0")
    parser.add_argument("--diar-model", default="pyannote/speaker-diarization-3.1")

    parser.add_argument("--device", default="auto")

    # Se mantienen aunque tu versión no los use.
    parser.add_argument("--vad-onset", type=float, default=0.50)
    parser.add_argument("--vad-offset", type=float, default=0.50)

    # Para versiones que usan threshold en vez de onset/offset.
    parser.add_argument("--vad-threshold", type=float, default=0.50)

    parser.add_argument("--min-duration-on", type=float, default=0.15)
    parser.add_argument("--min-duration-off", type=float, default=0.35)

    parser.add_argument("--pad-start", type=float, default=0.18)
    parser.add_argument("--pad-end", type=float, default=0.30)
    parser.add_argument("--merge-gap", type=float, default=0.65)
    parser.add_argument("--min-region", type=float, default=0.25)
    parser.add_argument("--max-region", type=float, default=28.0)

    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--min-diar-turn", type=float, default=0.12)
    parser.add_argument("--merge-speaker-gap", type=float, default=0.20)

    args = parser.parse_args()

    audio = Path(args.audio)

    if not audio.exists():
        sys.exit(f"❌ No existe el audio: {audio}")

    base = (
         args.output_base
         if args.output_base
         else str(audio.with_suffix(""))
    )

    vad_json = Path(f"{base}_vad_speech.json")
    vad_srt = Path(f"{base}_vad_debug.srt")
    diar_json = Path(f"{base}_pyannote_speakers.json")

    duration = get_audio_duration(audio)

    print("==============================================")
    print(" VAD + DIARIZACIÓN")
    print("----------------------------------------------")
    print(f"Audio    : {audio}")
    print(f"Duración : {duration:.2f}s")
    print(f"Base     : {base}")
    print("==============================================")

    try:
        raw_vad = run_vad(audio, args)
    except Exception as e:
        sys.exit(f"❌ Error ejecutando VAD: {e}")

    regions = postprocess_vad_regions(raw_vad, duration, args)

    vad_payload = {
        "audio": str(audio),
        "duration": duration,
        "model": args.vad_model,
        "params": {
            "vad_onset": args.vad_onset,
            "vad_offset": args.vad_offset,
            "vad_threshold": args.vad_threshold,
            "min_duration_on": args.min_duration_on,
            "min_duration_off": args.min_duration_off,
            "pad_start": args.pad_start,
            "pad_end": args.pad_end,
            "merge_gap": args.merge_gap,
            "min_region": args.min_region,
            "max_region": args.max_region,
        },
        "regions": regions,
    }

    vad_json.write_text(
        json.dumps(vad_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_vad_srt(vad_srt, regions)

    print(f"✅ Regiones VAD: {len(regions)}")
    print(f"💾 {vad_json}")
    print(f"💾 {vad_srt}")

    try:
        turns = run_diarization(audio, args)
    except Exception as e:
        sys.exit(f"❌ Error ejecutando diarización: {e}")

    diar_payload = {
        "audio": str(audio),
        "model": None if args.skip_diarization else args.diar_model,
        "segments": turns,
    }

    diar_json.write_text(
        json.dumps(diar_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✅ Turnos diarización: {len(turns)}")
    print(f"💾 {diar_json}")
    print("✔︎ VAD + diarización completados.")


if __name__ == "__main__":
    main()
