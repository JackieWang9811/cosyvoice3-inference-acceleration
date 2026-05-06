import sys
import os
import time
import json
import gc
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

# vLLM custom model registration
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
PROMPT_WAV = "/home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav"
OUT_DIR = "./vllm_6cases_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

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


def reset_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def synthesize_one(cosyvoice, case, seed=0):
    set_all_random_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reset_cuda()
    gc.collect()

    sr = cosyvoice.sample_rate
    t0 = time.perf_counter()
    audio_chunks = []

    if case["mode"] == "zero_shot":
        gen = cosyvoice.inference_zero_shot(
            case["tts_text"],
            case["prompt_text"],
            PROMPT_WAV,
            stream=case["stream"],
        )
    elif case["mode"] == "cross_lingual":
        gen = cosyvoice.inference_cross_lingual(
            case["tts_text"],
            PROMPT_WAV,
            stream=case["stream"],
        )
    elif case["mode"] == "instruct2":
        gen = cosyvoice.inference_instruct2(
            case["tts_text"],
            case["prompt_text"],
            PROMPT_WAV,
            stream=case["stream"],
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

    out_wav = os.path.join(OUT_DIR, f"{case['name']}.wav")
    torchaudio.save(out_wav, wav, sr)

    peak_mb = 0.0
    alloc_mb = 0.0
    reserved_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    result = {
        "case": case["name"],
        "mode": case["mode"],
        "audio_sec": audio_sec,
        "infer_sec": infer_sec,
        "rtf": rtf,
        "peak_mb": peak_mb,
        "alloc_mb": alloc_mb,
        "reserved_mb": reserved_mb,
        "wav_path": out_wav,
    }
    return result


def main():
    print("=" * 80)
    print("[INFO] Loading CosyVoice3 with vLLM backend...")
    print("=" * 80)

    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        load_trt=True,
        load_vllm=True,
        fp16=False,
    )

    print(f"[OK] sample_rate = {cosyvoice.sample_rate}")
    print(f"[OK] output_dir   = {OUT_DIR}")

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print("\n" + "-" * 80)
        print(f"[{i}/{len(TEST_CASES)}] Running: {case['name']} ({case['mode']})")
        print("-" * 80)

        one = synthesize_one(cosyvoice, case, seed=i - 1)
        results.append(one)

        print(
            f"[DONE] {case['name']} | "
            f"infer={one['infer_sec']:.3f}s | "
            f"audio={one['audio_sec']:.3f}s | "
            f"rtf={one['rtf']:.4f} | "
            f"peak={one['peak_mb']:.1f}MB | "
            f"saved={one['wav_path']}"
        )

    avg_rtf = sum(x["rtf"] for x in results) / len(results)
    avg_peak = sum(x["peak_mb"] for x in results) / len(results)

    summary = {
        "model_dir": MODEL_DIR,
        "backend": "vllm_awq_replaced_vllm",
        "num_cases": len(results),
        "avg_rtf": avg_rtf,
        "avg_peak_mb": avg_peak,
        "results": results,
    }

    summary_path = os.path.join(OUT_DIR, "summary_vllm_awq_6cases.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 80)
    print("# Final Summary")
    print("#" * 80)
    print(f"Average RTF       : {avg_rtf:.4f}")
    print(f"Average Peak Mem  : {avg_peak:.1f} MB")
    print(f"Summary saved to  : {summary_path}")

    print("\n[Per-case Results]")
    for x in results:
        print(
            f"{x['case']:<24} | "
            f"mode={x['mode']:<14} | "
            f"rtf={x['rtf']:.4f} | "
            f"peak={x['peak_mb']:.1f}MB"
        )


if __name__ == "__main__":
    main()