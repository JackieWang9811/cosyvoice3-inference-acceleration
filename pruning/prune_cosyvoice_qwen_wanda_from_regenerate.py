#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
from typing import Dict, Iterator, List, Tuple, Optional
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

import torch
import torch.nn as nn

from cosyvoice.cli.cosyvoice import AutoModel

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="WANDA prune CosyVoice3 embedded Qwen2 using train_regenerate.jsonl"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to CosyVoice3 model dir, e.g. pretrained_models/Fun-CosyVoice3-0.5B-2512",
    )
    parser.add_argument(
        "--calib_jsonl",
        type=str,
        required=True,
        help="Path to train_regenerate.jsonl",
    )
    parser.add_argument(
        "--out_llm_pt",
        type=str,
        required=True,
        help="Output path for pruned llm state_dict, e.g. llm_pruned.pt",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.3,
        help="Row-wise unstructured sparsity ratio, e.g. 0.3 / 0.5",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=256,
        help="Max number of calibration samples",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max input token length for one calibration sample",
    )
    parser.add_argument(
        "--audio_token_offset",
        type=int,
        default=0,
        help=(
            "Offset added to each audio token. "
            "Set to 0 if audio_tokens are already final vocab ids."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Whether to prepend tokenizer BOS token",
    )
    parser.add_argument(
        "--append_eos",
        action="store_true",
        help="Whether to append tokenizer EOS token",
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
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class LinearStats:
    """
    Approximate second-moment accumulator for WANDA/SparseGPT-style scoring.
    """

    def __init__(self, layer: nn.Linear):
        self.columns = layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=layer.weight.device, dtype=torch.float32)
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
        W_metric = |W| * sqrt(diag(H))
    """
    W = layer.weight.data
    diag_h = torch.diag(H).clamp_min(1e-8)
    diag = torch.sqrt(diag_h).to(W.dtype)

    # |W| @ diag(diag) == |W| * diag.unsqueeze(0)
    W_metric = torch.abs(W) * diag.unsqueeze(0)

    k = int(W_metric.shape[1] * sparsity)
    if k <= 0:
        return

    sort_idx = torch.argsort(W_metric, dim=1, stable=True)[:, :k]
    mask = torch.zeros_like(W_metric, dtype=torch.bool)
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


def encode_text(tokenizer, text: str) -> List[int]:
    if text is None:
        return []
    text = str(text).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_input_ids_for_sample(
    sample: Dict,
    tokenizer,
    audio_token_offset: int,
    prepend_bos: bool,
    append_eos: bool,
    max_length: int,
) -> torch.Tensor:
    """
    Simple calibration composition:
        [BOS?] instruct_ids + text_ids + shifted_audio_tokens + [EOS?]

    This is intentionally simple and robust.
    If your local training prompt format is stricter, only modify this function.
    """
    instruct = sample.get("instruct", "")
    text = sample.get("text", "")
    audio_tokens = sample.get("audio_tokens", [])

    instruct_ids = encode_text(tokenizer, instruct)
    text_ids = encode_text(tokenizer, text)

    if not isinstance(audio_tokens, list):
        raise TypeError("sample['audio_tokens'] must be a list")

    audio_ids = [int(x) + int(audio_token_offset) for x in audio_tokens]

    seq = []

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if prepend_bos and bos_id is not None:
        seq.append(int(bos_id))

    seq.extend(instruct_ids)
    seq.extend(text_ids)
    seq.extend(audio_ids)

    if append_eos and eos_id is not None:
        seq.append(int(eos_id))

    seq = seq[:max_length]

    if len(seq) == 0:
        raise ValueError("Empty calibration sequence built from sample")

    return torch.tensor(seq, dtype=torch.long)


def build_inputs_embeds_from_input_ids(
    input_ids: torch.Tensor,
    embed_tokens: nn.Embedding,
    device: str,
    embed_dtype: torch.dtype,
):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # (1, T)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    inputs_embeds = embed_tokens(input_ids)

    if inputs_embeds.dtype != embed_dtype:
        inputs_embeds = inputs_embeds.to(embed_dtype)

    return inputs_embeds, attention_mask


def iter_calibration_batches(
    samples: List[Dict],
    tokenizer,
    embed_tokens: nn.Embedding,
    device: str,
    embed_dtype: torch.dtype,
    audio_token_offset: int,
    prepend_bos: bool,
    append_eos: bool,
    max_length: int,
):
    for idx, sample in enumerate(samples):
        try:
            input_ids = build_input_ids_for_sample(
                sample=sample,
                tokenizer=tokenizer,
                audio_token_offset=audio_token_offset,
                prepend_bos=prepend_bos,
                append_eos=append_eos,
                max_length=max_length,
            )
            inputs_embeds, attention_mask = build_inputs_embeds_from_input_ids(
                input_ids=input_ids,
                embed_tokens=embed_tokens,
                device=device,
                embed_dtype=embed_dtype,
            )
            yield {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            print(f"[WARN] skip sample {idx}: {e}")


@torch.no_grad()
def main():
    args = parse_args()

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0, 1)")

    device = args.device
    forward_dtype = get_torch_dtype(args.dtype)

    print(f"[INFO] Loading CosyVoice3 from: {args.model_dir}")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    # CosyVoice3 embedded Qwen2
    cosyvoice3_model = cosyvoice.model
    hf_model = cosyvoice3_model.llm.llm.model  # Qwen2ForCausalLM
    hf_model.eval().to(device)

    tokenizer = getattr(cosyvoice, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(cosyvoice, "frontend", None), "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Failed to locate tokenizer from CosyVoice object")

    embed_tokens = hf_model.model.embed_tokens.to(device)

    if forward_dtype != torch.float32:
        hf_model = hf_model.to(dtype=forward_dtype)
        embed_tokens = embed_tokens.to(dtype=forward_dtype)

    print(f"[INFO] Reading calibration data: {args.calib_jsonl}")
    samples = read_jsonl(args.calib_jsonl, max_samples=args.max_samples)
    print(f"[INFO] Loaded {len(samples)} calibration samples")

    # Register hooks
    stats: Dict[str, LinearStats] = {}
    handles = []

    def make_hook(layer_name):
        def hook(mod, inp, out):
            if layer_name in stats and len(inp) > 0 and torch.is_tensor(inp[0]):
                stats[layer_name].add_batch(inp[0].detach())
        return hook

    num_hooked = 0
    for name, lin in iter_linear_layers(hf_model.model):
        if not should_prune_layer(name, args.prune_attn_only, args.prune_mlp_only):
            continue
        stats[name] = LinearStats(lin)
        handles.append(lin.register_forward_hook(make_hook(name)))
        num_hooked += 1

    print(f"[INFO] Hooked {num_hooked} linear layers")

    # Calibration forward
    n_forward = 0
    embed_dtype = next(hf_model.parameters()).dtype
    for batch in iter_calibration_batches(
        samples=samples,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        device=device,
        embed_dtype=embed_dtype,
        audio_token_offset=args.audio_token_offset,
        prepend_bos=args.prepend_bos,
        append_eos=args.append_eos,
        max_length=args.max_length,
    ):
        hf_model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        n_forward += 1
        if n_forward % 20 == 0:
            print(f"[INFO] Calibration forwards: {n_forward}")

    for h in handles:
        h.remove()

    print(f"[INFO] Finished {n_forward} calibration forwards")

    # Apply WANDA pruning
    print("[INFO] Applying WANDA pruning ...")
    num_pruned = 0
    for name, lin in iter_linear_layers(hf_model.model):
        if name not in stats:
            continue
        wanda_prune_linear(lin, stats[name].H, args.sparsity)
        num_pruned += 1

    print(f"[INFO] Pruned {num_pruned} layers")

    # Save full llm module state_dict
    out_dir = os.path.dirname(args.out_llm_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(cosyvoice3_model.llm.state_dict(), args.out_llm_pt)
    print(f"[INFO] Saved pruned llm weights to: {args.out_llm_pt}")


if __name__ == "__main__":
    main()