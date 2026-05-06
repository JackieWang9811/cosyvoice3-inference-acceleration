#!/usr/bin/env python3
"""Prepare customer-service data for CosyVoice3 fine-tuning.

The CosyVoice3 training example expects Kaldi-style files first:
wav.scp, text, utt2spk, spk2utt and optional instruct. The repository's
tools/make_parquet_list.py then converts those files into parquet shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {"audio_path", "text"}
DEFAULT_SPEAKER = "customer_service"


@dataclass
class Record:
    utt_id: str
    audio_path: str
    text: str
    speaker_id: str
    scenario: str
    emotion: str
    language: str
    split: str
    duration: float | None


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("您好!", "您好！").replace("您好?", "您好？")
    text = text.replace(",", "，").replace("?", "？").replace("!", "！")
    if text and text[-1] not in "。！？；":
        text += "。"
    return text


def make_utt_id(audio_path: str, text: str) -> str:
    digest = hashlib.sha1(f"{audio_path}\n{text}".encode("utf-8")).hexdigest()[:12]
    stem = Path(audio_path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "utt"
    return f"{safe_stem}_{digest}"


def wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, OSError):
        return None


def read_rows(input_path: Path) -> list[dict[str, str]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_no} is not a JSON object")
                rows.append({str(k): "" if v is None else str(v) for k, v in obj.items()})
        return rows
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            return [{k: "" if v is None else v for k, v in row.items()} for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported input format: {input_path}. Use .csv or .jsonl")


def validate_fields(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Input manifest is empty")
    missing = REQUIRED_FIELDS - set(rows[0])
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}")


def build_records(
    rows: list[dict[str, str]],
    audio_root: Path,
    min_seconds: float,
    max_seconds: float,
) -> tuple[list[Record], list[str]]:
    records: list[Record] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for idx, row in enumerate(rows, 1):
        raw_audio = row.get("audio_path", "").strip()
        raw_text = row.get("text", "").strip()
        if not raw_audio or not raw_text:
            warnings.append(f"row {idx}: empty audio_path or text, skipped")
            continue

        audio_path = Path(raw_audio)
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_path
        if not audio_path.exists():
            warnings.append(f"row {idx}: audio not found: {audio_path}")
            continue

        text = normalize_text(raw_text)
        duration = wav_duration(audio_path)
        if duration is not None and duration < min_seconds:
            warnings.append(f"row {idx}: audio too short ({duration:.2f}s), skipped")
            continue
        if duration is not None and duration > max_seconds:
            warnings.append(f"row {idx}: audio too long ({duration:.2f}s), skipped")
            continue

        utt_id = row.get("utt_id", "").strip() or make_utt_id(str(audio_path), text)
        while utt_id in seen_ids:
            utt_id = f"{utt_id}_dup"
        seen_ids.add(utt_id)

        records.append(
            Record(
                utt_id=utt_id,
                audio_path=str(audio_path),
                text=text,
                speaker_id=row.get("speaker_id", "").strip() or DEFAULT_SPEAKER,
                scenario=row.get("scenario", "").strip() or "general",
                emotion=row.get("emotion", "").strip() or "polite",
                language=row.get("language", "").strip() or "zh",
                split=row.get("split", "").strip().lower(),
                duration=duration,
            )
        )

    return records, warnings


def assign_split(records: list[Record], dev_ratio: float, seed: int) -> tuple[list[Record], list[Record]]:
    explicit_train = [r for r in records if r.split == "train"]
    explicit_dev = [r for r in records if r.split in {"dev", "valid", "validation"}]
    unspecified = [r for r in records if not r.split]

    rng = random.Random(seed)
    rng.shuffle(unspecified)
    dev_count = max(1, int(round(len(unspecified) * dev_ratio))) if unspecified else 0
    auto_dev = unspecified[:dev_count]
    auto_train = unspecified[dev_count:]
    return explicit_train + auto_train, explicit_dev + auto_dev


def to_json(record: Record) -> dict[str, object]:
    return {
        "utt_id": record.utt_id,
        "audio_path": record.audio_path,
        "text": record.text,
        "speaker_id": record.speaker_id,
        "scenario": record.scenario,
        "emotion": record.emotion,
        "language": record.language,
        "duration": record.duration,
    }


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_json(record), ensure_ascii=False) + "\n")


def write_metadata(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r.utt_id}|{r.audio_path}|{r.speaker_id}|{r.text}|{r.emotion}|{r.scenario}\n")


def write_kaldi_dir(path: Path, records: list[Record], instruct: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    spk2utt: dict[str, list[str]] = {}
    with (path / "wav.scp").open("w", encoding="utf-8") as wav_scp, \
            (path / "text").open("w", encoding="utf-8") as text_f, \
            (path / "utt2spk").open("w", encoding="utf-8") as utt2spk_f, \
            (path / "instruct").open("w", encoding="utf-8") as instruct_f:
        for record in records:
            wav_scp.write(f"{record.utt_id} {record.audio_path}\n")
            text_f.write(f"{record.utt_id} {record.text}\n")
            utt2spk_f.write(f"{record.utt_id} {record.speaker_id}\n")
            instruct_f.write(f"{record.utt_id} {instruct}\n")
            spk2utt.setdefault(record.speaker_id, []).append(record.utt_id)

    with (path / "spk2utt").open("w", encoding="utf-8") as spk2utt_f:
        for speaker, utts in sorted(spk2utt.items()):
            spk2utt_f.write(f"{speaker} {' '.join(utts)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="CSV or JSONL raw manifest")
    parser.add_argument("--audio-root", required=True, type=Path, help="Base directory for relative audio paths")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-seconds", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument(
        "--instruct",
        default="You are a helpful customer service assistant. Speak in a professional, warm and clear Mandarin Chinese tone.<|endofprompt|>",
        help="CosyVoice3 instruct prefix written to train/dev instruct files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    validate_fields(rows)
    records, warnings = build_records(rows, args.audio_root, args.min_seconds, args.max_seconds)
    train, dev = assign_split(records, args.dev_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "dev.jsonl", dev)
    write_metadata(args.output_dir / "metadata.list", train + dev)
    write_kaldi_dir(args.output_dir / "train", train, args.instruct)
    write_kaldi_dir(args.output_dir / "dev", dev, args.instruct)

    stats = {
        "total": len(records),
        "train": len(train),
        "dev": len(dev),
        "train_dir": str(args.output_dir / "train"),
        "dev_dir": str(args.output_dir / "dev"),
        "warnings": warnings[:200],
        "warning_count": len(warnings),
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
