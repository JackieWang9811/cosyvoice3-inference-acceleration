#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import os
import json
import time
import gc
import argparse
import torchaudio
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel


# =========================
# 默认路径
# =========================
DEFAULT_MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
DEFAULT_PRUNED_CKPT = "/home/b24zll/data/wjq_data/CosyVoice/prune_wanda/llm_pruned.pt"
DEFAULT_PROMPT_WAV = "/home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav"
DEFAULT_OUT_DIR = "./benchmark_fp32_vs_pruned_outputs"


# =========================
# 6个固定样例
# =========================
TEST_CASES = [
    {
        "name": "zero_shot_cn_tongue",
        "mode": "zero_shot",
        "tts_text": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "stream": False,
    },
    {
        "name": "zero_shot_cn_news",
        "mode": "zero_shot",
        "tts_text": "今天的天气很好，适合出门散步，也适合做一些量化前后的性能对比实验。",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>请自然地朗读下面这句话。",
        "stream": False,
    },
    {
        "name": "cross_lingual_control",
        "mode": "cross_lingual",
        "tts_text": "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]",
        "stream": False,
    },
    {
        "name": "instruct_cantonese",
        "mode": "instruct2",
        "tts_text": "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。",
        "prompt_text": "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
        "stream": False,
    },
    {
        "name": "instruct_fast",
        "mode": "instruct2",
        "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        "prompt_text": "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
        "stream": False,
    },
    {
        "name": "japanese_cross_lingual",
        "mode": "cross_lingual",
        "tts_text": "You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。",
        "stream": False,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs pruned CosyVoice3 LLM")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pruned_ckpt", type=str, default=DEFAULT_PRUNED_CKPT)
    parser.add_argument("--prompt_wav", type=str, default=DEFAULT_PROMPT_WAV)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,pruned",
        help="Comma-separated modes: baseline,pruned"
    )
    return parser.parse_args()


def sizeof_gb(path):
    return os.path.getsize(path) / (1024 ** 3)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_gpu_mem(prefix=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"{prefix} allocated={allocated:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB")


def reset_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def calc_zero_ratio(module):
    total_params = 0
    zero_params = 0
    layer_zero_ratios = []

    for name, p in module.named_parameters():
        if p.ndim >= 2 and "weight" in name:
            total_params += p.numel()
            z = (p == 0).sum().item()
            zero_params += z
            zr = z / max(p.numel(), 1)
            if zr > 0:
                layer_zero_ratios.append((name, zr))

    global_zero_ratio = zero_params / max(total_params, 1)
    layer_zero_ratios = sorted(layer_zero_ratios, key=lambda x: x[1], reverse=True)
    return global_zero_ratio, layer_zero_ratios


def load_pruned_llm(cosyvoice, llm_ckpt_path: str):
    if not os.path.exists(llm_ckpt_path):
        raise FileNotFoundError(f"Cannot find pruned ckpt: {llm_ckpt_path}")

    print(f"[INFO] Loading pruned LLM ckpt from: {llm_ckpt_path}")
    state = torch.load(llm_ckpt_path, map_location="cpu")

    missing, unexpected = cosyvoice.model.llm.load_state_dict(state, strict=False)

    print(f"[INFO] Missing keys   : {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[WARN] Missing keys examples:", missing[:10])
    if len(unexpected) > 0:
        print("[WARN] Unexpected keys examples:", unexpected[:10])

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def build_cosyvoice(model_dir, mode="baseline", pruned_ckpt=None):
    cosyvoice = AutoModel(model_dir=model_dir)
    core = getattr(cosyvoice, "model", cosyvoice)

    load_info = None
    if mode == "pruned":
        load_info = load_pruned_llm(cosyvoice, pruned_ckpt)

    return cosyvoice, core, load_info


@torch.no_grad()
def synthesize_one(cosyvoice, case, prompt_wav, save_prefix, out_dir):
    sr = cosyvoice.sample_rate

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        reset_cuda()

    t0 = time.perf_counter()

    audio_chunks = []
    if case["mode"] == "zero_shot":
        gen = cosyvoice.inference_zero_shot(
            case["tts_text"],
            case["prompt_text"],
            prompt_wav,
            stream=case["stream"]
        )
    elif case["mode"] == "cross_lingual":
        gen = cosyvoice.inference_cross_lingual(
            case["tts_text"],
            prompt_wav,
            stream=case["stream"]
        )
    elif case["mode"] == "instruct2":
        gen = cosyvoice.inference_instruct2(
            case["tts_text"],
            case["prompt_text"],
            prompt_wav,
            stream=case["stream"]
        )
    else:
        raise ValueError(f"Unknown mode: {case['mode']}")

    for item in gen:
        audio_chunks.append(item["tts_speech"].cpu())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    if len(audio_chunks) == 0:
        raise RuntimeError(f"No audio generated for case: {case['name']}")

    wav = torch.cat(audio_chunks, dim=-1)
    audio_sec = wav.shape[-1] / sr
    infer_sec = t1 - t0
    rtf = infer_sec / max(audio_sec, 1e-8)

    out_wav = os.path.join(out_dir, f"{save_prefix}_{case['name']}.wav")
    torchaudio.save(out_wav, wav, sr)

    peak_mb = 0.0
    alloc_mb = 0.0
    reserved_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    return {
        "case": case["name"],
        "audio_sec": audio_sec,
        "infer_sec": infer_sec,
        "rtf": rtf,
        "peak_mb": peak_mb,
        "alloc_mb": alloc_mb,
        "reserved_mb": reserved_mb,
        "wav_path": out_wav,
    }


def benchmark_mode(mode, model_dir, pruned_ckpt, prompt_wav, out_dir):
    print("\n" + "=" * 80)
    print(f"[Benchmark] mode = {mode}")
    print("=" * 80)

    reset_cuda()
    gc.collect()

    cosyvoice, core, load_info = build_cosyvoice(
        model_dir=model_dir,
        mode=mode,
        pruned_ckpt=pruned_ckpt,
    )

    llm_module = core.llm
    llm_backbone = core.llm.llm.model

    total_params, trainable_params = count_params(llm_module)
    backbone_total_params, backbone_trainable_params = count_params(llm_backbone)

    global_zero_ratio, layer_zero_ratios = calc_zero_ratio(llm_module)

    print(f"[{mode}] llm total params            : {total_params / 1e6:.3f} M")
    print(f"[{mode}] llm trainable params        : {trainable_params / 1e6:.3f} M")
    print(f"[{mode}] backbone total params       : {backbone_total_params / 1e6:.3f} M")
    print(f"[{mode}] backbone trainable params   : {backbone_trainable_params / 1e6:.3f} M")
    print(f"[{mode}] global zero ratio          : {global_zero_ratio:.4f}")
    print_gpu_mem(prefix=f"[{mode}] after load")

    print(f"\n[{mode}] Top zero-ratio params:")
    for name, zr in layer_zero_ratios[:20]:
        print(f"  {name}: {zr:.4f}")

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{mode}] Running case {i}/{len(TEST_CASES)}: {case['name']}")
        one = synthesize_one(
            cosyvoice=cosyvoice,
            case=case,
            prompt_wav=prompt_wav,
            save_prefix=mode,
            out_dir=out_dir,
        )
        results.append(one)
        print(
            f"[{mode}] {case['name']} | "
            f"infer={one['infer_sec']:.3f}s | "
            f"audio={one['audio_sec']:.3f}s | "
            f"rtf={one['rtf']:.4f} | "
            f"peak={one['peak_mb']:.1f}MB"
        )

    avg_rtf = sum(x["rtf"] for x in results) / len(results)
    avg_peak = sum(x["peak_mb"] for x in results) / len(results)

    summary = {
        "mode": mode,
        "llm_total_params": total_params,
        "llm_trainable_params": trainable_params,
        "backbone_total_params": backbone_total_params,
        "backbone_trainable_params": backbone_trainable_params,
        "global_zero_ratio": global_zero_ratio,
        "top_zero_ratio_params": layer_zero_ratios[:50],
        "avg_rtf": avg_rtf,
        "avg_peak_mb": avg_peak,
        "load_info": load_info,
        "results": results,
    }

    with open(os.path.join(out_dir, f"summary_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del cosyvoice, core, llm_module, llm_backbone
    gc.collect()
    reset_cuda()

    return summary


def print_ckpt_size(pruned_ckpt):
    ckpt_gb = sizeof_gb(pruned_ckpt)
    print("\n" + "=" * 80)
    print("[Pruned CKPT Size]")
    print("=" * 80)
    print(f"Pruned ckpt: {ckpt_gb:.3f} GB")
    return {"pruned_ckpt_gb": ckpt_gb}


def print_final_comparison(baseline_summary, pruned_summary, size_info):
    print("\n" + "#" * 80)
    print("# Final Comparison")
    print("#" * 80)

    print("\n[Checkpoint Size]")
    print(f"Pruned ckpt size         : {size_info['pruned_ckpt_gb']:.3f} GB")

    print("\n[LLM Parameter Count]")
    print(f"Baseline llm params      : {baseline_summary['llm_total_params'] / 1e6:.3f} M")
    print(f"Pruned   llm params      : {pruned_summary['llm_total_params'] / 1e6:.3f} M")
    print("Note: unstructured pruning usually does NOT change parameter count, only zero ratio.")

    print("\n[Global Zero Ratio]")
    print(f"Baseline zero ratio      : {baseline_summary['global_zero_ratio']:.4f}")
    print(f"Pruned   zero ratio      : {pruned_summary['global_zero_ratio']:.4f}")

    print("\n[Average RTF on 6 cases]")
    print(f"Baseline avg RTF         : {baseline_summary['avg_rtf']:.4f}")
    print(f"Pruned   avg RTF         : {pruned_summary['avg_rtf']:.4f}")
    print(f"Speedup (baseline/pruned): {baseline_summary['avg_rtf'] / max(pruned_summary['avg_rtf'], 1e-8):.4f}x")

    print("\n[Average Peak GPU Memory on 6 cases]")
    print(f"Baseline avg peak memory : {baseline_summary['avg_peak_mb']:.1f} MB")
    print(f"Pruned   avg peak memory : {pruned_summary['avg_peak_mb']:.1f} MB")

    print("\n[Load State Dict]")
    b_miss = 0 if baseline_summary["load_info"] is None else len(baseline_summary["load_info"]["missing_keys"])
    p_miss = 0 if pruned_summary["load_info"] is None else len(pruned_summary["load_info"]["missing_keys"])
    p_unexp = 0 if pruned_summary["load_info"] is None else len(pruned_summary["load_info"]["unexpected_keys"])
    print(f"Baseline missing/unexpected : {b_miss}/0")
    print(f"Pruned   missing/unexpected : {p_miss}/{p_unexp}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    summaries = {}

    if "baseline" in modes:
        summaries["baseline"] = benchmark_mode(
            mode="baseline",
            model_dir=args.model_dir,
            pruned_ckpt=args.pruned_ckpt,
            prompt_wav=args.prompt_wav,
            out_dir=args.out_dir,
        )

    if "pruned" in modes:
        summaries["pruned"] = benchmark_mode(
            mode="pruned",
            model_dir=args.model_dir,
            pruned_ckpt=args.pruned_ckpt,
            prompt_wav=args.prompt_wav,
            out_dir=args.out_dir,
        )

    size_info = print_ckpt_size(args.pruned_ckpt)

    if "baseline" in summaries and "pruned" in summaries:
        print_final_comparison(
            baseline_summary=summaries["baseline"],
            pruned_summary=summaries["pruned"],
            size_info=size_info,
        )

        final_summary = {
            "baseline": summaries["baseline"],
            "pruned": summaries["pruned"],
            "size_info": size_info,
        }
        with open(os.path.join(args.out_dir, "final_summary_fp32_vs_pruned.json"), "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()