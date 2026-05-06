#!/usr/bin/env python3
"""Import MatrixStudio/TTS-SCCusSerFSC into the customer-service raw manifest.

This dataset is useful for research prototypes because the utterances are
customer-service style Mandarin TTS samples. Its published license is
CC-BY-NC-ND-4.0, so do not use the resulting fine-tuned checkpoint for
commercial deployment without replacing or relicensing the data.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


def infer_scenario(text: str, domain: str) -> str:
    if any(k in text for k in ("您好", "你好", "欢迎")):
        return "greeting"
    if any(k in text for k in ("抱歉", "不好意思", "对不起")):
        return "apology"
    if any(k in text for k in ("退款", "退费", "退货")):
        return "refund"
    if any(k in text for k in ("订单", "查询", "课程", "报名")):
        return "order_query"
    if any(k in text for k in ("转接", "人工", "客服")):
        return "handoff"
    return domain or "general"


def safe_text(text: str) -> str:
    text = re.sub(r"\s+", "", str(text).strip())
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--speaker-id", default="tts_sccusserfsc_female_001")
    parser.add_argument("--dev-ratio", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_dir = args.output_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    manifest_path = args.output_root / "raw_manifest.csv"
    dev_every = max(1, round(1.0 / args.dev_ratio)) if args.dev_ratio > 0 else 0

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path",
                "text",
                "speaker_id",
                "scenario",
                "emotion",
                "language",
                "split",
                "source",
                "source_license",
            ],
        )
        writer.writeheader()
        for idx, row in df.iterrows():
            audio = row["audio"]
            audio_bytes = audio["bytes"]
            original_name = Path(audio.get("path") or f"{idx:06d}.wav").name
            wav_name = f"{idx:06d}_{original_name}"
            wav_path = wav_dir / wav_name
            wav_path.write_bytes(audio_bytes)

            text = safe_text(row["text"])
            domain = safe_text(row.get("domain", ""))
            split = "dev" if dev_every and idx % dev_every == 0 else "train"
            writer.writerow(
                {
                    "audio_path": str(Path("wavs") / wav_name),
                    "text": text,
                    "speaker_id": args.speaker_id,
                    "scenario": infer_scenario(text, domain),
                    "emotion": "polite",
                    "language": "zh",
                    "split": split,
                    "source": "MatrixStudio/TTS-SCCusSerFSC",
                    "source_license": "CC-BY-NC-ND-4.0",
                }
            )

    print(f"rows={len(df)}")
    print(f"manifest={manifest_path}")
    print(f"wavs={wav_dir}")


if __name__ == "__main__":
    main()
