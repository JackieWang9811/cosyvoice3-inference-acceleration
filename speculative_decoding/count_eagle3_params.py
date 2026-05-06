import os
from pathlib import Path
import torch
import argparse
from collections import Counter
import sys
import torch
import sys
from pathlib import Path
import torch

# FILE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = FILE_DIR.parent
# sys.path.insert(0, str(PROJECT_ROOT))
# sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

# print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
# print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

# from cosyvoice.cli.cosyvoice import AutoModel

# ckpt_dir = Path("/home/b24zll/data/wjq_data/CosyVoice/eagle3_cosyvoice3/checkpoint-40")

# def format_size(num_bytes: int) -> str:
#     units = ["B", "KB", "MB", "GB", "TB"]
#     size = float(num_bytes)
#     for unit in units:
#         if size < 1024 or unit == units[-1]:
#             return f"{size:.3f} {unit}"
#         size /= 1024

# def format_params(n: int) -> str:
#     if n >= 1e9:
#         return f"{n / 1e9:.3f} B"
#     if n >= 1e6:
#         return f"{n / 1e6:.3f} M"
#     if n >= 1e3:
#         return f"{n / 1e3:.3f} K"
#     return str(n)

# print(f"Checkpoint dir: {ckpt_dir}")

# # 1) 统计目录下所有文件大小
# all_files = []
# total_dir_size = 0
# for p in ckpt_dir.rglob("*"):
#     if p.is_file():
#         size = p.stat().st_size
#         all_files.append((str(p), size))
#         total_dir_size += size

# print("\n[All files in checkpoint]")
# for path, size in sorted(all_files, key=lambda x: x[1], reverse=True):
#     print(f"{format_size(size):>10}  {path}")

# print(f"\n[Checkpoint total size] {format_size(total_dir_size)}")

# # 2) 只统计模型权重文件大小
# weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
# exclude_keywords = ["optimizer", "scheduler", "trainer_state", "training_args", "rng_state"]

# weight_files = []
# pure_weight_size = 0
# for p in ckpt_dir.rglob("*"):
#     if p.is_file() and p.suffix in weight_suffixes:
#         name = p.name.lower()
#         if not any(k in name for k in exclude_keywords):
#             weight_files.append((str(p), p.stat().st_size))
#             pure_weight_size += p.stat().st_size

# print("\n[Model weight files only]")
# for path, size in sorted(weight_files, key=lambda x: x[1], reverse=True):
#     print(f"{format_size(size):>10}  {path}")

# print(f"\n[Pure model weight size] {format_size(pure_weight_size)}")

# # 3) 加载模型并统计参数量
# # 依赖你的 CosyVoice / AngelSlim 自定义代码已经在 PYTHONPATH 中
# from transformers import AutoModelForCausalLM

# print("\n[Loading model...]")
# model = AutoModelForCausalLM.from_pretrained(
#     str(ckpt_dir),
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
# )

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print("\n[Parameter count]")
# print(f"Total params     : {total_params} ({format_params(total_params)})")
# print(f"Trainable params : {trainable_params} ({format_params(trainable_params)})")

# # 4) 按模块粗略统计
# print("\n[Top-level module parameter breakdown]")
# for name, module in model.named_children():
#     n = sum(p.numel() for p in module.parameters())
#     print(f"{name:30s} {format_params(n):>12}")

# # 5) 数据类型分布
# dtype_stats = {}
# for p in model.parameters():
#     dtype_stats[str(p.dtype)] = dtype_stats.get(str(p.dtype), 0) + p.numel()

# print("\n[Dtype breakdown]")
# for k, v in sorted(dtype_stats.items(), key=lambda x: x[1], reverse=True):
#     print(f"{k:20s} {format_params(v):>12}")



from pathlib import Path
from safetensors.torch import load_file
import re

ckpt = Path("/home/b24zll/data/wjq_data/CosyVoice/eagle3_cosyvoice3/checkpoint-40/model.safetensors")

state_dict = load_file(str(ckpt))

total_params = 0

layer_ids = set()
for k, v in state_dict.items():
    total_params += v.numel()

    print(f"{k:60s} shape={tuple(v.shape)} params={v.numel()}")

print("layer ids:", sorted(layer_ids))
print("num layers found:", len(layer_ids))
print("max layer id:", max(layer_ids) if layer_ids else None)

print("\n[Summary]")
print("Tensor count :", len(state_dict))
print("Total params :", total_params)
print("Total params (M):", total_params / 1e6)