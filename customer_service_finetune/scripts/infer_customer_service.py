#!/usr/bin/env python3
"""Customer-service inference wrapper for a fine-tuned CosyVoice checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCENARIO_PREFIX = {
    "greeting": "您好，",
    "order_query": "请您稍等，我正在为您查询订单信息。",
    "refund": "关于退款问题，",
    "after_sales": "关于售后服务，",
    "appointment": "我来帮您确认预约信息。",
    "apology": "非常抱歉给您带来不便。",
    "handoff": "我将为您转接人工客服。",
}


def normalize_customer_service_text(text: str, scenario: str) -> str:
    text = text.strip()
    prefix = SCENARIO_PREFIX.get(scenario, "")
    if prefix and not text.startswith(prefix):
        text = prefix + text
    if text and text[-1] not in "。！？；":
        text += "。"
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-root", type=Path, default=Path("."))
    parser.add_argument("--infer-entry", default="cosyvoice/bin/inference.py")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--scenario", default="general")
    parser.add_argument("--speaker", default="cs_001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.cosyvoice_root.resolve()
    infer_entry = Path(args.infer_entry)
    if not infer_entry.is_absolute():
        infer_entry = root / infer_entry
    if not infer_entry.exists():
        raise FileNotFoundError(f"Inference entry not found: {infer_entry}")

    text = normalize_customer_service_text(args.text, args.scenario)
    payload = {
        "text": text,
        "speaker": args.speaker,
        "scenario": args.scenario,
        "output": str(args.output),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    command = [
        sys.executable,
        str(infer_entry),
        "--model_dir",
        args.model_dir,
        "--text",
        text,
        "--spk_id",
        args.speaker,
        "--output",
        str(args.output),
    ]
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if not args.dry_run:
        subprocess.run(command, cwd=str(root), check=True)


if __name__ == "__main__":
    main()
