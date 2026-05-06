# analyze_cosyvoice3_model.py
# 作用：加载 CosyVoice / CosyVoice2 / CosyVoice3，统计参数量与模块分布
# 用法：
#   python analyze_cosyvoice3_model.py --model_dir /path/to/model_dir

import argparse
from collections import Counter
import sys
import torch
import sys
from pathlib import Path
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel


def _load_model(model_dir: str):
    print(f"尝试加载模型，model_dir={model_dir}")
    try:
        return AutoModel(model_dir=model_dir)
    except Exception as e:
        raise RuntimeError(
            f"加载失败：{e}\n"
            "请先确认：\n"
            "1) 你的 cosyvoice 代码版本支持 AutoModel\n"
            "2) 模型目录中存在 cosyvoice.yaml / cosyvoice2.yaml / cosyvoice3.yaml 之一\n"
            "3) 先用 python example.py 验证基础推理能跑通"
        )


def _param_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f}{unit}"
        x /= 1024
    return f"{x:.2f}PB"


def _count_module_types(module: torch.nn.Module) -> Counter:
    c = Counter()
    for m in module.modules():
        c[type(m).__name__] += 1
    return c


def _safe_eval(obj, name="object"):
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        print(f"[OK] {name}.eval()")
    else:
        print(f"[SKIP] {name} 不是 torch.nn.Module，类型={type(obj).__name__}")


def _collect_named_nn_modules(obj):
    """
    从 cosyvoice wrapper / core 对象中，收集其直接属性里是 nn.Module 的成员。
    """
    found = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if isinstance(val, torch.nn.Module):
            found[name] = val
    return found


def _print_module_stats(module: torch.nn.Module, name: str):
    total_params = sum(p.numel() for p in module.parameters())
    total_bytes = _param_bytes(module)
    print(f"[{name}] class={type(module).__name__}, params={total_params:,}, bytes≈{_fmt_bytes(total_bytes)}")


def main():
    MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"

    cosyvoice = _load_model(MODEL_DIR)

    print("\n=== 顶层对象信息 ===")
    print(f"type(cosyvoice) = {type(cosyvoice).__name__}")

    # AutoModel 返回的对象可能本身就是 wrapper，也可能内部有 .model
    core = getattr(cosyvoice, "model", cosyvoice)

    print(f"type(core) = {type(core).__name__}")

    # 不再假设 core 一定有 eval()
    _safe_eval(core, "core")

    print("\n=== 扫描 core 下的 nn.Module 子模块 ===")
    named_modules = _collect_named_nn_modules(core)

    if not named_modules:
        print("未在 core 的直接属性中发现 nn.Module。")
        print("这说明你的版本可能使用更深层封装；可先打印 dir(core) 进一步定位。")
        print("\ncore 可用属性如下：")
        for x in dir(core):
            if not x.startswith("_"):
                print(" ", x)
        return

    for name, mod in named_modules.items():
        _safe_eval(mod, name)

    print("\n=== 重点子模块统计 ===")
    for key in ["llm", "flow", "hift"]:
        if key in named_modules:
            _print_module_stats(named_modules[key], key)
        elif hasattr(core, key):
            print(f"[{key}] 存在，但不是 torch.nn.Module，类型={type(getattr(core, key)).__name__}")
        else:
            print(f"[{key}] 未找到")

    print("\n=== 所有直接 nn.Module 子模块统计 ===")
    total_params_sum = 0
    total_bytes_sum = 0
    for name, mod in named_modules.items():
        params = sum(p.numel() for p in mod.parameters())
        bytes_ = _param_bytes(mod)
        total_params_sum += params
        total_bytes_sum += bytes_
        print(f"{name:>20}: class={type(mod).__name__:<30} params={params:,}  bytes≈{_fmt_bytes(bytes_)}")

    print("\n=== 汇总（直接子模块参数求和） ===")
    print(f"Total params (sum of direct nn.Module attrs): {total_params_sum:,}")
    print(f"Param bytes (approx): {_fmt_bytes(total_bytes_sum)}")

    print("\n=== 模块类型计数（按直接子模块分别统计） ===")
    grand_counter = Counter()
    for name, mod in named_modules.items():
        c = _count_module_types(mod)
        grand_counter += c

    key_types = [
        "Linear", "Embedding", "Conv1d", "Conv2d", "ConvTranspose1d",
        "LayerNorm", "RMSNorm", "LlamaDecoderLayer", "Qwen2DecoderLayer"
    ]

    found_any = False
    for k in key_types:
        if k in grand_counter:
            found_any = True
            print(f"{k:>20}: {grand_counter[k]:,}")
    if not found_any:
        print("未命中预设类型名，可能你的实现使用了自定义模块名。")

    print("\n=== Top 30 module types ===")
    for name, cnt in grand_counter.most_common(30):
        print(f"{name:>30}: {cnt:,}")

    print("\n=== 提示 ===")
    print("1) CosyVoice3 的顶层 core 往往不是 nn.Module，这属于正常封装。")
    print("2) 真正可量化/可统计的通常是 llm、flow、hift 等内部 nn.Module。")
    print("3) 后续若做 INT8/INT4，重点看 llm 中 Linear/Embedding 占比。")


if __name__ == "__main__":
    main()