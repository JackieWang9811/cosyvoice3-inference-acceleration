import sys
from pathlib import Path
import os
import json
import time
import gc
import torchaudio
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
BACKBONE_FP_DIR = "/home/b24zll/data/wjq_data/CosyVoice/quantization/quantization_awq/cv3_qwen2_backbone_fp32"
BACKBONE_AWQ_DIR = "/home/b24zll/data/wjq_data/CosyVoice/quantization/quantization_awq/cv3_qwen2_backbone_awq_v2"
PROMPT_WAV = "/home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav"
OUT_DIR = "./benchmark_fp32_vs_awq_outputs"
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


def sizeof_gb(path):
    return os.path.getsize(path) / (1024 ** 3)


def format_gb(x):
    return f"{x:.3f} GB"


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


def load_awq_backbone_to_target_device(target_device):
    print("[INFO] Loading quantized AWQ backbone...")
    awq_model = AutoAWQForCausalLM.from_quantized(
        BACKBONE_AWQ_DIR,
        fuse_layers=False,
        trust_remote_code=True,
    )
    awq_model = awq_model.eval()
    real_model = awq_model.model if hasattr(awq_model, "model") else awq_model
    real_model = real_model.to(target_device).eval()
    print("[OK] AWQ backbone loaded.")
    return real_model


def patch_qwen2encoder_forward(core):
    import types
    from cosyvoice.utils.mask import make_pad_mask

    def new_forward(self, xs, xs_lens):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def new_forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache

    core.llm.llm.forward = types.MethodType(new_forward, core.llm.llm)
    core.llm.llm.forward_one_step = types.MethodType(new_forward_one_step, core.llm.llm)


def build_cosyvoice(mode="fp32"):
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    core = getattr(cosyvoice, "model", cosyvoice)

    old_backbone = core.llm.llm.model
    target_device = next(old_backbone.parameters()).device
    print("old backbone dtype =", next(core.llm.llm.model.parameters()).dtype)

    if mode == "awq":
        awq_backbone = load_awq_backbone_to_target_device(target_device)
        core.llm.llm.model = awq_backbone
        patch_qwen2encoder_forward(core)

    return cosyvoice, core


@torch.no_grad()
def synthesize_one(cosyvoice, case, save_prefix):
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
            PROMPT_WAV,
            stream=case["stream"]
        )
    elif case["mode"] == "cross_lingual":
        gen = cosyvoice.inference_cross_lingual(
            case["tts_text"],
            PROMPT_WAV,
            stream=case["stream"]
        )
    elif case["mode"] == "instruct2":
        gen = cosyvoice.inference_instruct2(
            case["tts_text"],
            case["prompt_text"],
            PROMPT_WAV,
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

    out_wav = os.path.join(OUT_DIR, f"{save_prefix}_{case['name']}.wav")
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


def benchmark_mode(mode="fp32"):
    print("\n" + "=" * 80)
    print(f"[Benchmark] mode = {mode}")
    print("=" * 80)

    reset_cuda()
    gc.collect()

    cosyvoice, core = build_cosyvoice(mode=mode)
    backbone = core.llm.llm.model

    total_params, trainable_params = count_params(backbone)

    print(f"[{mode}] total params     : {total_params / 1e6:.3f} M")
    print(f"[{mode}] trainable params : {trainable_params / 1e6:.3f} M")
    print_gpu_mem(prefix=f"[{mode}] after load")

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{mode}] Running case {i}/{len(TEST_CASES)}: {case['name']}")
        one = synthesize_one(cosyvoice, case, save_prefix=mode)
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
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_rtf": avg_rtf,
        "avg_peak_mb": avg_peak,
        "results": results,
    }

    with open(os.path.join(OUT_DIR, f"summary_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def print_disk_size_comparison():
    fp_path = os.path.join(BACKBONE_FP_DIR, "model.safetensors")
    awq_path = os.path.join(BACKBONE_AWQ_DIR, "model.safetensors")

    fp_gb = sizeof_gb(fp_path)
    awq_gb = sizeof_gb(awq_path)
    ratio = fp_gb / awq_gb

    print("\n" + "=" * 80)
    print("[Disk Size Comparison]")
    print("=" * 80)
    print(f"FP32 model.safetensors : {fp_gb:.3f} GB")
    print(f"AWQ  model.safetensors : {awq_gb:.3f} GB")
    print(f"Compression ratio      : {ratio:.3f}x")
    print(f"Reduction              : {(1 - awq_gb / fp_gb) * 100:.2f}%")

    return {
        "fp32_gb": fp_gb,
        "awq_gb": awq_gb,
        "compression_ratio": ratio,
        "reduction_percent": (1 - awq_gb / fp_gb) * 100,
    }


def print_final_comparison(fp_summary, awq_summary, size_info):
    print("\n" + "#" * 80)
    print("# Final Comparison")
    print("#" * 80)

    print("\n[Model Size]")
    print(f"FP32 model.safetensors : {size_info['fp32_gb']:.3f} GB")
    print(f"AWQ  model.safetensors : {size_info['awq_gb']:.3f} GB")
    print(f"Compression ratio      : {size_info['compression_ratio']:.3f}x")

    print("\n[Parameter Count]")
    print(f"FP32 total params      : {fp_summary['total_params'] / 1e6:.3f} M")
    print(f"AWQ  total params      : {awq_summary['total_params'] / 1e6:.3f} M")

    print("\n[Average RTF on 6 cases]")
    print(f"FP32 avg RTF           : {fp_summary['avg_rtf']:.4f}")
    print(f"AWQ  avg RTF           : {awq_summary['avg_rtf']:.4f}")
    print(f"Speedup (FP32/AWQ)     : {fp_summary['avg_rtf'] / awq_summary['avg_rtf']:.4f}x")

    print("\n[Average Peak GPU Memory on 6 cases]")
    print(f"FP32 avg peak memory   : {fp_summary['avg_peak_mb']:.1f} MB")
    print(f"AWQ  avg peak memory   : {awq_summary['avg_peak_mb']:.1f} MB")

    print("\n[Per-case Results]")
    for f, a in zip(fp_summary["results"], awq_summary["results"]):
        print(
            f"{f['case']:<24} | "
            f"FP32 RTF={f['rtf']:.4f}, peak={f['peak_mb']:.1f}MB || "
            f"AWQ RTF={a['rtf']:.4f}, peak={a['peak_mb']:.1f}MB"
        )


def main():
    size_info = print_disk_size_comparison()
    fp_summary = benchmark_mode("fp32")
    awq_summary = benchmark_mode("awq")
    print_final_comparison(fp_summary, awq_summary, size_info)


if __name__ == "__main__":
    main()