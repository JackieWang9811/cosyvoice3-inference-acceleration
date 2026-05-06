#!/usr/bin/env python3
"""Launch a CosyVoice3 customer-service LLM fine-tuning job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_command(root: Path, config: dict, train_entry: Path) -> list[str]:
    training = config.get("training", {})
    model_conf = config["model"]
    data_conf = config["data"]
    launcher = config.get("launcher", {})

    output_dir = resolve_path(root, config["experiment"]["output_dir"])
    tensorboard_dir = resolve_path(root, config["experiment"]["tensorboard_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    cuda_devices = str(training.get("cuda_visible_devices", "0"))
    num_gpus = int(training.get("num_gpus") or len([x for x in cuda_devices.split(",") if x.strip()]))

    command = [
        training.get("torchrun", "torchrun"),
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        f"--rdzv_id={training.get('job_id', 20260421)}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={training.get('rdzv_endpoint', 'localhost:1234')}",
        str(train_entry),
        "--train_engine",
        training.get("train_engine", "torch_ddp"),
        "--config",
        str(resolve_path(root, model_conf["hyperpyyaml"])),
        "--train_data",
        str(resolve_path(root, data_conf["train_manifest"])),
        "--cv_data",
        str(resolve_path(root, data_conf["dev_manifest"])),
        "--qwen_pretrain_path",
        str(resolve_path(root, model_conf["qwen_pretrain_path"])),
        "--onnx_path",
        str(resolve_path(root, model_conf["onnx_path"])),
        "--model",
        model_conf.get("train_model", "llm"),
        "--checkpoint",
        str(resolve_path(root, model_conf["checkpoint"])),
        "--model_dir",
        str(output_dir),
        "--tensorboard_dir",
        str(tensorboard_dir),
        "--ddp.dist_backend",
        training.get("dist_backend", "nccl"),
        "--num_workers",
        str(training.get("num_workers", 4)),
        "--prefetch",
        str(training.get("prefetch", 100)),
    ]
    if training.get("pin_memory", True):
        command.append("--pin_memory")
    if training.get("use_amp", True):
        command.append("--use_amp")
    if training.get("train_engine") == "deepspeed":
        command.extend(["--deepspeed_config", str(resolve_path(root, training["deepspeed_config"]))])
    if training.get("deepspeed_save_states"):
        command.extend(["--deepspeed.save_states", training["deepspeed_save_states"]])
    command.extend(str(arg) for arg in launcher.get("extra_args", []))
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-entry", default=None, help="Override CosyVoice train entry path")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.cosyvoice_root.resolve()
    config_path = args.config.resolve()
    config = load_config(config_path)
    train_entry_value = args.train_entry or config.get("launcher", {}).get("train_entry", "cosyvoice/bin/train.py")
    train_entry = resolve_path(root, train_entry_value)
    if not train_entry.exists():
        raise FileNotFoundError(f"Train entry not found: {train_entry}")
    command = build_command(root, config, train_entry)

    print("Resolved train entry:", train_entry)
    print("Command:")
    prefix = f"CUDA_VISIBLE_DEVICES={config.get('training', {}).get('cuda_visible_devices', '0')}"
    print(prefix, " ".join(f'"{part}"' if " " in part else part for part in command))
    if args.dry_run:
        return

    env = dict(**os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("training", {}).get("cuda_visible_devices", "0"))
    python_paths = [str(root), str(root / "third_party" / "Matcha-TTS")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("PYTHONIOENCODING", "UTF-8")
    subprocess.run(command, cwd=str(root), env=env, check=True)


if __name__ == "__main__":
    main()
