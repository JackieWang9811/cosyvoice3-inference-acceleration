#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import shutil
import argparse
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional

import torch
import torch.nn as nn

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
from transformers import AutoTokenizer, Qwen2ForCausalLM


# =========================================================
# Config helpers
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="WANDA prune CosyVoice3 Qwen2 backbone, save as HF backbone, then export to vLLM dir."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to CosyVoice3 model dir, e.g. pretrained_models/Fun-CosyVoice3-0.5B",
    )

    parser.add_argument(
        "--backbone_out_dir",
        type=str,
        required=True,
        help="Output HF directory for pruned Qwen2 backbone",
    )

    parser.add_argument(
        "--vllm_out_dir",
        type=str,
        required=True,
        help="Output exported vLLM dir built from pruned backbone",
    )

    parser.add_argument(
        "--activate_vllm_dir",
        type=str,
        default="",
        help="Optional active vllm dir to overwrite, e.g. /path/to/model_dir/vllm",
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.30,
        help="Row-wise unstructured sparsity ratio for WANDA, e.g. 0.3 / 0.5",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=64,
        help="Max number of calibration samples",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length per calibration sample",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )

    parser.add_argument(
        "--prune_attn_only",
        action="store_true",
        help="Only prune q_proj/k_proj/v_proj/o_proj",
    )

    parser.add_argument(
        "--prune_mlp_only",
        action="store_true",
        help="Only prune gate_proj/up_proj/down_proj",
    )

    parser.add_argument(
        "--run_zero_shot_test",
        action="store_true",
        help="Whether to run a quick zero-shot inference test after backbone replacement",
    )

    parser.add_argument(
        "--prompt_wav",
        type=str,
        default="",
        help="Prompt wav path for quick zero-shot test",
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


# =========================================================
# Shared calibration texts
# 这一部分就是你想和 AWQ 尽量对齐的核心
# =========================================================

def build_shared_calib_texts() -> List[str]:
    """
    与 AWQ 尽量对齐的共享校准文本。
    当前策略：prompt_text + target_text
    """
    calib_pairs = [
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做得比我还好。",
            "target_text": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请自然地朗读下面这句话。",
            "target_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请用平静温和的语气朗读。",
            "target_text": "今天的天气很好，适合出门散步。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请用尽可能快的语速说一句话。",
            "target_text": "请尽快确认会议时间，并把最终安排发给大家。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请用清晰标准的普通话朗读。",
            "target_text": "人工智能正在改变我们的工作方式，也推动着语音技术不断进步。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Please read the following sentence in a natural and expressive style.",
            "target_text": "This is a test sentence for pruning the Qwen2 backbone in CosyVoice3.",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Please speak in a calm and gentle tone.",
            "target_text": "The quick brown fox jumps over the lazy dog.",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Please read this sentence a little faster.",
            "target_text": "We are evaluating whether calibration data closer to real inference inputs can improve pruning quality.",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请带一点播报感，自然朗读下面内容。",
            "target_text": "根据最新安排，项目测试将在明天下午三点正式开始，请提前做好准备。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请自然、流畅地朗读，不要过度夸张。",
            "target_text": "如果一切顺利，我们希望这一步能够成为当前剪枝方案的最后一次关键优化。",
        },
    ]

    calib_texts = [x["prompt_text"] + x["target_text"] for x in calib_pairs]
    return calib_texts


def build_calib_input_ids(
    tokenizer,
    max_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    texts = build_shared_calib_texts()[:max_samples]
    seqs = []

    for idx, text in enumerate(texts):
        try:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = tokenizer.encode(text)

            ids = ids[:max_length]
            if len(ids) == 0:
                print(f"[WARN] empty calibration ids at sample {idx}, skip")
                continue
            seqs.append(torch.tensor(ids, dtype=torch.long))
        except Exception as e:
            print(f"[WARN] failed to encode calib sample {idx}: {e}")

    return seqs


# =========================================================
# WANDA stats
# =========================================================

class LinearStats:
    """
    Approx second-moment accumulator for WANDA scoring.
    """

    def __init__(self, layer: nn.Linear):
        self.columns = layer.weight.data.shape[1]
        self.H = torch.zeros(
            (self.columns, self.columns),
            device=layer.weight.device,
            dtype=torch.float32,
        )
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """
        inp:
            (B, T, C) or (T, C) or (N, C)
        """
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # (1, T, C)

        bsz = inp.shape[0]

        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])  # (B*T, C)

        inp = inp.t().contiguous()  # (C, N)

        self.H *= self.nsamples / (self.nsamples + bsz + 1e-8)
        self.nsamples += bsz

        inp = (2.0 / max(self.nsamples, 1)) ** 0.5 * inp.float()
        self.H += inp @ inp.t()


@torch.no_grad()
def wanda_prune_linear(layer: nn.Linear, H: torch.Tensor, sparsity: float):
    """
    Row-wise unstructured WANDA:
        metric = |W| * sqrt(diag(H))
    """
    W = layer.weight.data
    diag_h = torch.diag(H).clamp_min(1e-8)
    diag = torch.sqrt(diag_h).to(W.dtype)

    metric = torch.abs(W) * diag.unsqueeze(0)

    k = int(metric.shape[1] * sparsity)
    if k <= 0:
        return

    sort_idx = torch.argsort(metric, dim=1, stable=True)[:, :k]
    mask = torch.zeros_like(metric, dtype=torch.bool)
    mask.scatter_(1, sort_idx, True)
    layer.weight.data[mask] = 0


def iter_linear_layers(module: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            yield name, m


def should_prune_layer(name: str, prune_attn_only: bool, prune_mlp_only: bool) -> bool:
    attn_keys = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_keys = ("gate_proj", "up_proj", "down_proj")

    if prune_attn_only and prune_mlp_only:
        return False
    if prune_attn_only:
        return any(k in name for k in attn_keys)
    if prune_mlp_only:
        return any(k in name for k in mlp_keys)

    return any(k in name for k in attn_keys + mlp_keys)


# =========================================================
# CosyVoice helpers
# =========================================================

def ensure_tokenizers_from_cosyvoice(cosyvoice, hf_model):
    """
    返回两个 tokenizer:
    1. runtime_tokenizer: CosyVoice 运行时 tokenizer，用于 encode 校准文本
    2. hf_tokenizer: Hugging Face tokenizer，用于 save_pretrained
    """
    runtime_tokenizer = getattr(cosyvoice, "tokenizer", None)
    if runtime_tokenizer is None:
        runtime_tokenizer = getattr(getattr(cosyvoice, "frontend", None), "tokenizer", None)
    if runtime_tokenizer is None:
        raise RuntimeError("Failed to locate runtime tokenizer from CosyVoice object")

    src_name_or_path = getattr(hf_model.config, "_name_or_path", None)
    print("[INFO] backbone config._name_or_path =", src_name_or_path)

    if src_name_or_path is None or len(str(src_name_or_path)) == 0:
        raise RuntimeError("Cannot locate tokenizer source path from hf_model.config._name_or_path")

    hf_tokenizer = AutoTokenizer.from_pretrained(
        src_name_or_path,
        trust_remote_code=True,
    )

    return runtime_tokenizer, hf_tokenizer


def get_core_and_backbone(model_dir: str, load_vllm: bool = False):
    cosyvoice = AutoModel(
        model_dir=model_dir,
        load_vllm=load_vllm,
    )
    core = getattr(cosyvoice, "model", cosyvoice)
    backbone = core.llm.llm.model
    return cosyvoice, core, backbone


def compute_weight_sparsity(module: nn.Module) -> float:
    total = 0
    zeros = 0
    for p in module.parameters():
        if p is None:
            continue
        total += p.numel()
        zeros += (p == 0).sum().item()
    if total == 0:
        return 0.0
    return zeros / total


# =========================================================
# Step 1: prune backbone in-place
# =========================================================

@torch.no_grad()
def prune_qwen2_backbone_with_wanda(
    hf_model: Qwen2ForCausalLM,
    tokenizer,
    device: str,
    forward_dtype: torch.dtype,
    sparsity: float,
    max_samples: int,
    max_length: int,
    prune_attn_only: bool,
    prune_mlp_only: bool,
):
    hf_model.eval().to(device)

    if forward_dtype != torch.float32:
        hf_model = hf_model.to(dtype=forward_dtype)

    embed_tokens = hf_model.model.embed_tokens.to(device)
    if embed_tokens.weight.dtype != next(hf_model.parameters()).dtype:
        embed_tokens = embed_tokens.to(dtype=next(hf_model.parameters()).dtype)

    stats: Dict[str, LinearStats] = {}
    handles = []

    def make_hook(layer_name):
        def hook(mod, inp, out):
            if layer_name in stats and len(inp) > 0 and torch.is_tensor(inp[0]):
                stats[layer_name].add_batch(inp[0].detach())
        return hook

    num_hooked = 0
    for name, lin in iter_linear_layers(hf_model.model):
        if not should_prune_layer(name, prune_attn_only, prune_mlp_only):
            continue
        stats[name] = LinearStats(lin)
        handles.append(lin.register_forward_hook(make_hook(name)))
        num_hooked += 1

    print(f"[INFO] Hooked {num_hooked} linear layers for WANDA")

    calib_input_ids = build_calib_input_ids(
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
    )
    print(f"[INFO] Using {len(calib_input_ids)} shared calibration samples")

    n_forward = 0
    embed_dtype = next(hf_model.parameters()).dtype

    for idx, input_ids in enumerate(calib_input_ids):
        try:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # (1, T)
            input_ids = input_ids.to(device)

            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            inputs_embeds = embed_tokens(input_ids)
            if inputs_embeds.dtype != embed_dtype:
                inputs_embeds = inputs_embeds.to(embed_dtype)

            hf_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            n_forward += 1
        except Exception as e:
            print(f"[WARN] calibration forward failed at sample {idx}: {e}")

    for h in handles:
        h.remove()

    print(f"[INFO] Finished {n_forward} calibration forwards")

    print("[INFO] Applying WANDA pruning ...")
    num_pruned = 0
    for name, lin in iter_linear_layers(hf_model.model):
        if name not in stats:
            continue
        wanda_prune_linear(lin, stats[name].H, sparsity)
        num_pruned += 1

    print(f"[INFO] Pruned {num_pruned} layers")

    return hf_model


# =========================================================
# Step 2: save pruned backbone as HF directory
# =========================================================

def save_pruned_backbone_as_hf_dir(
    hf_model: Qwen2ForCausalLM,
    tokenizer,
    backbone_out_dir: str,
):
    if os.path.exists(backbone_out_dir):
        print(f"[INFO] Remove old backbone dir: {backbone_out_dir}")
        shutil.rmtree(backbone_out_dir)

    os.makedirs(backbone_out_dir, exist_ok=True)
    hf_model.save_pretrained(backbone_out_dir)
    tokenizer.save_pretrained(backbone_out_dir)

    # 额外记录一个 metadata，方便你之后分辨
    metadata = {
        "method": "wanda_pruning",
        "note": "Dense-format HF backbone with many zero weights; not AWQ format.",
    }
    with open(os.path.join(backbone_out_dir, "wanda_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved pruned HF backbone to: {backbone_out_dir}")


# =========================================================
# Step 3: load pruned backbone and export vLLM dir
# =========================================================

def load_wanda_backbone_to_target_device(
    backbone_out_dir: str,
    target_device: str,
    torch_dtype: torch.dtype,
):
    print("[INFO] Loading WANDA-pruned HF backbone ...")
    model = Qwen2ForCausalLM.from_pretrained(
        backbone_out_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = model.eval().to(target_device)
    print("[OK] WANDA-pruned backbone loaded.")
    return model


def export_wanda_vllm_dir(
    model_dir: str,
    backbone_out_dir: str,
    vllm_out_dir: str,
    forward_dtype: torch.dtype,
):
    cosyvoice, core, old_backbone = get_core_and_backbone(model_dir=model_dir, load_vllm=False)

    target_device = next(old_backbone.parameters()).device
    print("[INFO] old backbone device:", target_device)
    print("[INFO] old backbone dtype :", next(old_backbone.parameters()).dtype)

    wanda_backbone = load_wanda_backbone_to_target_device(
        backbone_out_dir=backbone_out_dir,
        target_device=target_device,
        torch_dtype=forward_dtype,
    )

    core.llm.llm.model = wanda_backbone

    if os.path.exists(vllm_out_dir):
        print(f"[INFO] Remove old vLLM out dir: {vllm_out_dir}")
        shutil.rmtree(vllm_out_dir)

    from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
    export_cosyvoice2_vllm(core.llm, vllm_out_dir, target_device)

    print(f"[OK] Exported WANDA-based CosyVoice vLLM dir to: {vllm_out_dir}")


def activate_as_vllm_dir(vllm_out_dir: str, activate_vllm_dir: str):
    if not activate_vllm_dir:
        return

    if not os.path.exists(vllm_out_dir):
        raise RuntimeError(f"{vllm_out_dir} not found")

    if os.path.exists(activate_vllm_dir):
        print(f"[INFO] Remove existing active dir: {activate_vllm_dir}")
        shutil.rmtree(activate_vllm_dir)

    shutil.copytree(vllm_out_dir, activate_vllm_dir)
    print(f"[OK] Activated {vllm_out_dir} as {activate_vllm_dir}")


# =========================================================
# Optional quick zero-shot test
# 这里只做基础测试，不强行 patch forward
# 因为如果 save/load 的还是标准 Qwen2ForCausalLM，通常应能直接替换
# =========================================================

@torch.no_grad()
def run_zero_shot_test_with_replaced_backbone(
    model_dir: str,
    backbone_out_dir: str,
    prompt_wav: str,
    forward_dtype: torch.dtype,
):
    if not prompt_wav:
        print("[WARN] --prompt_wav not given, skip zero-shot test")
        return

    cosyvoice, core, old_backbone = get_core_and_backbone(model_dir=model_dir, load_vllm=False)

    target_device = next(old_backbone.parameters()).device
    wanda_backbone = load_wanda_backbone_to_target_device(
        backbone_out_dir=backbone_out_dir,
        target_device=target_device,
        torch_dtype=forward_dtype,
    )
    core.llm.llm.model = wanda_backbone

    print("[INFO] Running quick zero-shot test ...")
    try:
        import torchaudio
        for i, out in enumerate(
            cosyvoice.inference_zero_shot(
                "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
                "You are a helpful assistant.<|endofprompt|>希望你以后能够做得比我还好呦。",
                prompt_wav,
                stream=False,
            )
        ):
            out_wav = f"zero_shot_wanda_{i}.wav"
            torchaudio.save(out_wav, out["tts_speech"], cosyvoice.sample_rate)
            print("[OK] saved:", out_wav)
    except Exception as e:
        print("[WARN] zero-shot test failed:", e)
        print("[WARN] This does not necessarily mean export will fail. Check model replacement path carefully.")


# =========================================================
# Main
# =========================================================

@torch.no_grad()
def main():
    args = parse_args()

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0, 1)")

    if args.prune_attn_only and args.prune_mlp_only:
        raise ValueError("Cannot set both --prune_attn_only and --prune_mlp_only")

    forward_dtype = get_torch_dtype(args.dtype)

    print("[INFO] Loading CosyVoice model for pruning ...")
    cosyvoice, core, hf_model = get_core_and_backbone(model_dir=args.model_dir, load_vllm=False)
    runtime_tokenizer, hf_tokenizer = ensure_tokenizers_from_cosyvoice(
        cosyvoice,
        hf_model,
    )

    print("[INFO] Before pruning sparsity:", f"{compute_weight_sparsity(hf_model):.6f}")

    hf_model = prune_qwen2_backbone_with_wanda(
        hf_model=hf_model,
        tokenizer=runtime_tokenizer,
        device=args.device,
        forward_dtype=forward_dtype,
        sparsity=args.sparsity,
        max_samples=args.max_samples,
        max_length=args.max_length,
        prune_attn_only=args.prune_attn_only,
        prune_mlp_only=args.prune_mlp_only,
    )

    print("[INFO] After pruning sparsity:", f"{compute_weight_sparsity(hf_model):.6f}")

    save_pruned_backbone_as_hf_dir(
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        backbone_out_dir=args.backbone_out_dir,
    )

    print("[INFO] After pruning sparsity:", f"{compute_weight_sparsity(hf_model):.6f}")

    if args.run_zero_shot_test:
        run_zero_shot_test_with_replaced_backbone(
            model_dir=args.model_dir,
            backbone_out_dir=args.backbone_out_dir,
            prompt_wav=args.prompt_wav,
            forward_dtype=forward_dtype,
        )

    export_wanda_vllm_dir(
        model_dir=args.model_dir,
        backbone_out_dir=args.backbone_out_dir,
        vllm_out_dir=args.vllm_out_dir,
        forward_dtype=forward_dtype,
    )

    activate_as_vllm_dir(
        vllm_out_dir=args.vllm_out_dir,
        activate_vllm_dir=args.activate_vllm_dir,
    )

    print("[DONE] All finished.")


if __name__ == "__main__":
    main()