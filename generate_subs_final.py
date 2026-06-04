#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_subs_final.py

Tercera etapa:
  <base>_filtered.json
  + <base>_pyannote_speakers.json
  -> <base>_final.srt

Reglas:
  - máximo 2 líneas
  - máximo 42 caracteres por línea
  - timing sin solapes
  - CPS controlado
  - diálogo con guiones cuando corresponde
"""

import argparse
import json
import re
import sys
from pathlib import Path


CREDIT_RE = re.compile(
    r"(amara\.org|sottotitoli\s+creati|subtitles?\s+by|captioning\s+by|opensubtitles|www\.)",
    re.IGNORECASE,
)

TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
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
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    text = re.sub(r"\.\s*\.\s*\.", "…", text)
    text = re.sub(r"\.{4,}", "…", text)

    text = re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+'\s*", r"\1'", text)
    text = re.sub(r"\b[Ee]'\s+", "È ", text)

    text = re.sub(r"!{4,}", "!!!", text)
    text = re.sub(r"\?{4,}", "???", text)

    return text.strip()


def visible_len(text: str) -> int:
    return len(text.replace("\n", " ").strip())


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def capitalize_first(text: str) -> str:
    if not text:
        return text

    def repl(m):
        return m.group(1) + m.group(2).upper()

    return re.sub(
        r"^([\"'¿¡(\[\- ]*)([a-zà-öø-ÿ])",
        repl,
        text,
        count=1,
    )


def clean_final_text(text: str) -> str:
    text = normalize_text(text)

    if "\n" not in text:
        return capitalize_first(text)

    out = []
    for line in text.splitlines():
        line = normalize_text(line)
        if line.startswith("- "):
            out.append("- " + capitalize_first(line[2:].strip()))
        else:
            out.append(capitalize_first(line))

    return "\n".join(out)


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


def is_bad_text(text: str, keep_laughter=False) -> bool:
    text = normalize_text(text)

    if not text:
        return True

    if CREDIT_RE.search(text):
        return True

    if visible_len(text) > 400:
        return True

    if not keep_laughter and looks_like_long_laughter(text):
        return True

    if looks_like_repetition(text):
        return True

    return False


def parse_srt(path: Path):
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []

    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        time_idx = None
        for i, line in enumerate(lines):
            if "-->" in line:
                time_idx = i
                break

        if time_idx is None:
            continue

        m = TIME_RE.search(lines[time_idx])
        if not m:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()

        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
        #text = normalize_text(" ".join(lines[time_idx + 1:]))
        text = "\n".join(normalize_text(l) for l in lines[time_idx + 1:] if l.strip())

        if end > start and text:
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": [],
                }
            )

    return segments


def load_filtered_segments(base: str):
    filtered_json = Path(f"{base}_filtered.json")
    filtered_srt = Path(f"{base}_filtered.srt")

    if filtered_json.exists():
        data = json.loads(filtered_json.read_text(encoding="utf-8"))
        return data.get("segments", []), "filtered.json"

    if filtered_srt.exists():
        return parse_srt(filtered_srt), "filtered.srt"

    sys.exit(f"❌ No encontré {filtered_json} ni {filtered_srt}")


def load_diarization(base: str):
    path = Path(f"{base}_pyannote_speakers.json")
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    turns = []

    for t in data.get("segments", []):
        try:
            st = float(t["start"])
            en = float(t["end"])
        except Exception:
            continue

        if en - st < 0.12:
            continue

        turns.append(
            {
                "start": st,
                "end": en,
                "speaker": t.get("speaker"),
            }
        )

    turns.sort(key=lambda x: (x["start"], x["end"]))
    return turns


def best_speaker(start, end, turns):
    if not turns:
        return None

    scores = {}

    for t in turns:
        ov = overlap(start, end, t["start"], t["end"])
        if ov <= 0:
            continue

        spk = t.get("speaker")
        scores[spk] = scores.get(spk, 0.0) + ov

    if not scores:
        return None

    spk, score = max(scores.items(), key=lambda kv: kv[1])

    if score < 0.05:
        return None

    return spk


def make_pseudo_words(text, start, end):
    text = normalize_text(text)
    toks = text.split()

    if not toks:
        return []

    dur = max(0.2, end - start)
    step = dur / len(toks)

    words = []

    for i, tok in enumerate(toks):
        words.append(
            {
                "word": tok,
                "start": start + i * step,
                "end": start + (i + 1) * step,
                "prob": 1.0,
            }
        )

    return words


def repair_words(seg):
    start = float(seg["start"])
    end = float(seg["end"])
    words = []

    for w in seg.get("words", []) or []:
        token = str(w.get("word", "")).strip()
        if not token:
            continue

        try:
            st = float(w.get("start"))
            en = float(w.get("end"))
        except Exception:
            continue

        if st < start - 0.5 or st > end + 0.5:
            continue

        if en <= st:
            en = min(end, st + 0.05)

        if en <= st:
            continue

        words.append(
            {
                "word": token,
                "start": st,
                "end": en,
                "prob": float(w.get("prob", w.get("probability", 1.0)) or 1.0),
            }
        )

    words.sort(key=lambda x: (x["start"], x["end"]))

    if not words:
        words = make_pseudo_words(seg.get("text", ""), start, end)

    # Evita pequeños solapes internos entre palabras.
    for i in range(len(words) - 1):
        if words[i]["end"] > words[i + 1]["start"]:
            words[i]["end"] = max(words[i]["start"] + 0.02, words[i + 1]["start"])

    return words


def words_to_text(words):
    return normalize_text(" ".join(w["word"] for w in words).strip())


def make_cue(words, speaker=None):
    if not words:
        return None

    text = words_to_text(words)
    if not text:
        return None

    return {
        "start": float(words[0]["start"]),
        "end": float(max(w["end"] for w in words)),
        "text": text,
        "speaker": speaker,
        "words": list(words),
    }


def split_segment(seg, args):
    text = normalize_text(seg.get("text", ""))

    if is_bad_text(text, keep_laughter=args.keep_laughter):
        return []

    words = repair_words(seg)

    if not words:
        return []

    max_chars = args.max_cpl * 2
    cues = []
    cur = []

    for w in words:
        if not cur:
            cur = [w]
            continue

        candidate = cur + [w]
        candidate_text = words_to_text(candidate)
        current_text = words_to_text(cur)

        gap = float(w["start"]) - float(cur[-1]["end"])
        duration = float(w["end"]) - float(cur[0]["start"])
        prev_token = cur[-1]["word"].strip()

        prev_sentence_end = bool(re.search(r"[.!?…]$", prev_token))

        must_break = False

        if gap > args.max_word_gap:
            must_break = True

        if visible_len(candidate_text) > max_chars:
            must_break = True

        if duration > args.max_dur:
            must_break = True

        if prev_sentence_end and gap > 0.22 and visible_len(current_text) >= 30:
            must_break = True

        if must_break:
            cue = make_cue(cur)
            if cue and not is_bad_text(cue["text"], args.keep_laughter):
                cues.append(cue)
            cur = [w]
        else:
            cur = candidate

    if cur:
        cue = make_cue(cur)
        if cue and not is_bad_text(cue["text"], args.keep_laughter):
            cues.append(cue)

    return cues


def split_overlong_cues(cues, args):
    out = []
    max_chars = args.max_cpl * 2

    for cue in cues:
        if "\n" in cue["text"] or visible_len(cue["text"]) <= max_chars:
            out.append(cue)
            continue

        words = cue.get("words") or make_pseudo_words(
            cue["text"], cue["start"], cue["end"]
        )

        cur = []

        for w in words:
            candidate = cur + [w]

            if cur and visible_len(words_to_text(candidate)) > max_chars:
                new_cue = make_cue(cur, cue.get("speaker"))
                if new_cue:
                    out.append(new_cue)
                cur = [w]
            else:
                cur = candidate

        if cur:
            new_cue = make_cue(cur, cue.get("speaker"))
            if new_cue:
                out.append(new_cue)

    return out


def merge_same_speaker(cues, args):
    if not cues:
        return []

    cues = sorted(cues, key=lambda x: (x["start"], x["end"]))
    max_chars = args.max_cpl * 2
    out = []

    i = 0

    while i < len(cues):
        cur = dict(cues[i])
        i += 1

        while i < len(cues):
            nxt = cues[i]
            gap = nxt["start"] - cur["end"]

            if cur.get("speaker") != nxt.get("speaker"):
                break

            if gap < -0.05 or gap > args.merge_same_speaker_gap:
                break

            combined = normalize_text(cur["text"] + " " + nxt["text"])
            duration = nxt["end"] - cur["start"]

            if visible_len(combined) > max_chars:
                break

            if duration > args.max_dur:
                break

            cps = visible_len(combined) / max(0.1, duration)
            if cps > args.max_cps + 4:
                break

            cur["text"] = combined
            cur["end"] = nxt["end"]
            cur["words"] = cur.get("words", []) + nxt.get("words", [])
            i += 1

        out.append(cur)

    return out


def wrap_single(text: str, max_cpl: int):
    text = clean_final_text(text)

    if len(text) <= max_cpl:
        return text

    if len(text) > max_cpl * 2:
        return text

    spaces = [m.start() for m in re.finditer(r" ", text)]
    best = None

    for pos in spaces:
        left = text[:pos].strip()
        right = text[pos + 1:].strip()

        if not left or not right:
            continue

        if len(left) <= max_cpl and len(right) <= max_cpl:
            score = abs(len(left) - len(right))

            if left[-1:] in ",;:":
                score -= 6
            elif left[-1:] in ".!?…":
                score -= 4

            if best is None or score < best[0]:
                best = (score, left, right)

    if best:
        return best[1] + "\n" + best[2]

    return text


def wrap_text(text: str, max_cpl: int):
    text = clean_final_text(text)

    if "\n" in text:
        lines = [normalize_text(l) for l in text.splitlines() if l.strip()]
        return "\n".join(lines[:2])

    return wrap_single(text, max_cpl)


def can_merge_dialogue(a, b, args):
    if args.no_dialogue_merge:
        return False

    if not a.get("speaker") or not b.get("speaker"):
        return False

    if a["speaker"] == b["speaker"]:
        return False

    gap = b["start"] - a["end"]

    if gap < -0.08 or gap > args.dialog_gap:
        return False

    ta = clean_final_text(a["text"])
    tb = clean_final_text(b["text"])

    if len("- " + ta) > args.max_cpl:
        return False

    if len("- " + tb) > args.max_cpl:
        return False

    if b["end"] - a["start"] > args.max_dur:
        return False

    return True


def merge_dialogues(cues, args):
    out = []
    i = 0

    while i < len(cues):
        cur = cues[i]

        if i + 1 < len(cues):
            nxt = cues[i + 1]

            if can_merge_dialogue(cur, nxt, args):
                out.append(
                    {
                        "start": cur["start"],
                        "end": nxt["end"],
                        "speaker": None,
                        "text": f"- {clean_final_text(cur['text'])}\n- {clean_final_text(nxt['text'])}",
                        "words": cur.get("words", []) + nxt.get("words", []),
                    }
                )
                i += 2
                continue

        out.append(cur)
        i += 1

    return out


def apply_timing(cues, args):
    if not cues:
        return []

    cues = sorted(cues, key=lambda x: (x["start"], x["end"]))

    timed = []

    for c in cues:
        c = dict(c)
        c["start"] = max(0.0, float(c["start"]) - args.pad_start)
        c["end"] = max(c["start"] + 0.25, float(c["end"]) + args.pad_end)
        timed.append(c)

    # Ajuste por CPS sin pisar al siguiente.
    for i, c in enumerate(timed):
        chars = max(1, visible_len(c["text"]))
        needed = chars / args.max_cps
        needed = max(args.min_dur, min(args.max_dur, needed))

        desired_end = max(c["end"], c["start"] + needed)
        desired_end = min(desired_end, c["start"] + args.max_dur)

        if i + 1 < len(timed):
            desired_end = min(desired_end, timed[i + 1]["start"] - args.min_gap)

        c["end"] = desired_end

    final = []
    prev_end = -1.0

    for c in timed:
        if c["start"] < prev_end + args.min_gap:
            c["start"] = prev_end + args.min_gap

        if c["end"] <= c["start"] + 0.25:
            continue

        c["text"] = wrap_text(c["text"], args.max_cpl)

        if c["text"].count("\n") >= 2:
            continue

        final.append(c)
        prev_end = c["end"]

    return final


def dedupe_cues(cues):
    out = []

    for c in cues:
        key = normalize_text(c["text"]).lower()

        duplicate = False

        for prev in out[-3:]:
            prev_key = normalize_text(prev["text"]).lower()

            if key == prev_key and abs(c["start"] - prev["start"]) < 0.30:
                duplicate = True
                break

        if not duplicate:
            out.append(c)

    return out


def validate_no_overlaps(cues, min_gap):
    fixed = []
    prev_end = -1.0

    for c in sorted(cues, key=lambda x: (x["start"], x["end"])):
        c = dict(c)

        if c["start"] < prev_end + min_gap:
            c["start"] = prev_end + min_gap

        if c["end"] <= c["start"] + 0.25:
            continue

        fixed.append(c)
        prev_end = c["end"]

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Genera SRT final estilo scene/Netflix.")

    parser.add_argument("base", help="Base sin extensión. Ej: /ruta/output_test")

    parser.add_argument("--max-cpl", type=int, default=42)
    parser.add_argument("--max-cps", type=float, default=17.0)
    parser.add_argument("--min-dur", type=float, default=0.85)
    parser.add_argument("--max-dur", type=float, default=6.8)
    parser.add_argument("--min-gap", type=float, default=0.08)

    parser.add_argument("--pad-start", type=float, default=0.04)
    parser.add_argument("--pad-end", type=float, default=0.12)

    parser.add_argument("--max-word-gap", type=float, default=0.65)
    parser.add_argument("--merge-same-speaker-gap", type=float, default=0.30)
    parser.add_argument("--dialog-gap", type=float, default=0.35)

    parser.add_argument("--keep-laughter", action="store_true")
    parser.add_argument("--no-dialogue-merge", action="store_true")

    args = parser.parse_args()

    base = args.base
    out_srt = Path(f"{base}_final.srt")

    segments, source_name = load_filtered_segments(base)
    diar_turns = load_diarization(base)

    cues = []
    dropped = 0

    for seg in sorted(segments, key=lambda x: float(x.get("start", 0))):
        new_cues = split_segment(seg, args)

        if not new_cues:
            dropped += 1
            continue

        cues.extend(new_cues)

    cues = split_overlong_cues(cues, args)

    # Speaker por cue, no por segmento completo.
    for c in cues:
        c["speaker"] = best_speaker(c["start"], c["end"], diar_turns)

    cues = merge_same_speaker(cues, args)
    cues = merge_dialogues(cues, args)
    cues = split_overlong_cues(cues, args)
    cues = apply_timing(cues, args)
    cues = dedupe_cues(cues)
    cues = validate_no_overlaps(cues, args.min_gap)

    with out_srt.open("w", encoding="utf-8") as f:
        for i, c in enumerate(cues, start=1):
            text = wrap_text(c["text"], args.max_cpl)

            if not text.strip():
                continue

            f.write(f"{i}\n")
            f.write(f"{sec_to_ts(c['start'])} --> {sec_to_ts(c['end'])}\n")
            f.write(f"{text}\n\n")

    print("==============================================")
    print("🎬 SRT FINAL GENERADO")
    print("----------------------------------------------")
    print(f"Fuente textual : {source_name}")
    print(f"Segmentos in   : {len(segments)}")
    print(f"Descartados    : {dropped}")
    print(f"Captions out   : {len(cues)}")
    print(f"Archivo        : {out_srt}")
    print("==============================================")


if __name__ == "__main__":
    main()
