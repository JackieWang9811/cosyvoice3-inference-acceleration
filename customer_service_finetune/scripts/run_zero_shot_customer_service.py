#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path("/home/b24zll/data/wjq_data/CosyVoice")
os.environ.setdefault("PYTHONIOENCODING", "UTF-8")
os.environ.setdefault("CUDA_HOME", str(ROOT / "customer_service_finetune/fake_cuda"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

BASE_MODEL = ROOT / "pretrained_models/Fun-CosyVoice3-0.5B"
CKPT_DIR = ROOT / "customer_service_finetune/exp/cosyvoice3/llm/torch_ddp"
WORK_DIR = ROOT / "customer_service_finetune/infer_models_zero_shot"
OUT_DIR = ROOT / "customer_service_finetune/infer_outputs/zero_shot_customer_service_utf8"
PROMPT_WAV = ROOT / "asset/zero_shot_prompt.wav"

# Follow example.py exactly: inference_zero_shot(target_text, prompt_text, prompt_wav).
PROMPT_TEXTS = {
    "example_prompt": "You are a helpful assistant.<|endofprompt|>希望你以后能够做得比我还好呦。",
    "customer_service_prompt": "请用专业、温和、清晰、耐心的中文客服语气说这句话，语速适中，吐字清楚。<|endofprompt|>希望你以后能够做得比我还好呦。",
}

CHECKPOINTS = {
    "baseline": None,
    "epoch2": CKPT_DIR / "epoch_2_whole.pt",
    "epoch19": CKPT_DIR / "epoch_19_whole.pt",
}

TEST_CASES = [
    (
        "greeting_long",
        "您好，欢迎致电智能客服中心。为了更快帮您处理问题，请您简单描述一下遇到的情况，比如订单查询、退款进度、售后维修，或者需要转接人工客服。",
    ),
    (
        "order_long",
        "请您稍等，我正在为您查询订单信息。系统显示您的订单已经完成支付，目前正在仓库打包，预计今天下午发出，物流单号生成后会通过短信通知您。",
    ),
    (
        "apology_long",
        "非常抱歉给您带来不便。我们已经记录了您的问题，并会优先为您核实处理。如果后续需要补充材料，客服专员会通过电话或者短信与您联系。",
    ),
]


def prepare_model(tag: str, checkpoint: Path | None) -> Path:
    if checkpoint is None:
        return BASE_MODEL
    model_dir = WORK_DIR / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "cosyvoice3.yaml",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        "configuration.json",
        ".msc",
        ".mv",
    ]:
        src = BASE_MODEL / name
        if src.exists():
            dst = model_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
    qwen_src = BASE_MODEL / "CosyVoice-BlankEN"
    qwen_dst = model_dir / "CosyVoice-BlankEN"
    if qwen_dst.exists() or qwen_dst.is_symlink():
        qwen_dst.unlink()
    qwen_dst.symlink_to(qwen_src, target_is_directory=True)

    llm_dst = model_dir / "llm.pt"
    if llm_dst.exists() or llm_dst.is_symlink():
        llm_dst.unlink()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if k not in {"epoch", "step"}}
    torch.save(state, llm_dst)
    return model_dir


def run_case(
    cosyvoice,
    tag: str,
    prompt_name: str,
    prompt_text: str,
    case_idx: int,
    scenario: str,
    target_text: str,
):
    set_all_random_seed(20260422 + case_idx)
    count = 0
    for part_idx, result in enumerate(
        cosyvoice.inference_zero_shot(
            target_text,
            prompt_text,
            str(PROMPT_WAV),
            stream=False,
        )
    ):
        samples = int(result["tts_speech"].shape[1])
        duration = samples / cosyvoice.sample_rate
        out = OUT_DIR / f"{tag}_{prompt_name}_{case_idx + 1:02d}_{scenario}_{part_idx}.wav"
        torchaudio.save(str(out), result["tts_speech"].cpu(), cosyvoice.sample_rate)
        print(f"Saved {out} duration={duration:.2f}s", flush=True)
        yield {
            "checkpoint": tag,
            "prompt": prompt_name,
            "scenario": scenario,
            "target_text": target_text,
            "prompt_text": prompt_text,
            "path": str(out),
            "sample_rate": cosyvoice.sample_rate,
            "duration_seconds": duration,
        }
        count += 1
    if count == 0:
        print(f"NO_OUTPUT {tag} {prompt_name} {scenario}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tag, ckpt in CHECKPOINTS.items():
        model_dir = prepare_model(tag, ckpt)
        print(f"Loading {tag}: {model_dir}", flush=True)
        cosyvoice = AutoModel(model_dir=str(model_dir), fp16=False)
        for prompt_name, prompt_text in PROMPT_TEXTS.items():
            for case_idx, (scenario, target_text) in enumerate(TEST_CASES):
                try:
                    manifest.extend(
                        run_case(cosyvoice, tag, prompt_name, prompt_text, case_idx, scenario, target_text)
                    )
                except Exception as exc:
                    print(f"ERROR {tag} {prompt_name} {scenario}: {exc}", flush=True)
                    traceback.print_exc()
        del cosyvoice
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
