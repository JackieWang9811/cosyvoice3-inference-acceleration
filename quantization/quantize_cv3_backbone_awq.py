import sys
from pathlib import Path
import os
import json
import torchaudio
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
from transformers import Qwen2ForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM


MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
BACKBONE_FP_DIR = "/home/b24zll/data/wjq_data/CosyVoice/quantization/quantization_awq/cv3_qwen2_backbone_fp32"
BACKBONE_AWQ_DIR = "/home/b24zll/data/wjq_data/CosyVoice/quantization/quantization_awq/cv3_qwen2_backbone_awq_v2"
PROMPT_WAV = "/home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav"


# def build_calib_texts():
    # 第一版：先用少量、覆盖中英和控制提示的文本
    calib_texts = [
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做得比我还好。",
        "You are a helpful assistant.<|endofprompt|>请自然地朗读下面这句话。",
        "You are a helpful assistant.<|endofprompt|>请用平静温和的语气朗读。",
        "You are a helpful assistant.<|endofprompt|>请用尽可能快的语速说一句话。",
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        "今天的天气很好，适合出门散步。",
        "This is a test sentence for quantizing the Qwen2 backbone in CosyVoice3.",
        "Please read the following sentence in a natural and expressive style.",
        "The quick brown fox jumps over the lazy dog.",
    ]
#     return calib_texts

def build_calib_texts():
    # 第二版：用更接近 CosyVoice3 zero-shot 真实输入分布的校准文本：
    # 每条样本都是 prompt_text + target_text
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
            "target_text": "This is a test sentence for quantizing the Qwen2 backbone in CosyVoice3.",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Please speak in a calm and gentle tone.",
            "target_text": "The quick brown fox jumps over the lazy dog.",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Please read this sentence a little faster.",
            "target_text": "We are evaluating whether calibration data closer to real inference inputs can improve AWQ quantization quality.",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请带一点播报感，自然朗读下面内容。",
            "target_text": "根据最新安排，项目测试将在明天下午三点正式开始，请提前做好准备。",
        },
        {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>请自然、流畅地朗读，不要过度夸张。",
            "target_text": "如果一切顺利，我们希望这一步能够成为当前量化方案的最后一次关键优化。",
        },
    ]

    calib_texts = [
        item["prompt_text"] + item["target_text"]
        for item in calib_pairs
    ]
    return calib_texts


def export_backbone():
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    core = getattr(cosyvoice, "model", cosyvoice)
    qwen_model = core.llm.llm.model

    os.makedirs(BACKBONE_FP_DIR, exist_ok=True)
    qwen_model.save_pretrained(BACKBONE_FP_DIR)
    print(f"[OK] Exported backbone to: {BACKBONE_FP_DIR}")


def ensure_tokenizer():
    # Qwen2 backbone 本身通常需要 tokenizer 目录
    # 若 save_pretrained(backbone) 后没有 tokenizer，则从原 backbone 的 pretrain_path 补一份
    try:
        tok = AutoTokenizer.from_pretrained(BACKBONE_FP_DIR, trust_remote_code=True)
        print("[OK] Tokenizer already exists in backbone dir.")
        return tok
    except Exception:
        pass

    # 兜底：直接从原 CosyVoice3 内部 backbone 对应的 HF 目录找 tokenizer
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    core = getattr(cosyvoice, "model", cosyvoice)
    # 这里假设 name_or_path 存在
    src_name_or_path = getattr(core.llm.llm.model.config, "_name_or_path", None)
    print("[INFO] backbone config._name_or_path =", src_name_or_path)

    if src_name_or_path is None or len(str(src_name_or_path)) == 0:
        raise RuntimeError("Cannot locate tokenizer source path from backbone config._name_or_path")

    tok = AutoTokenizer.from_pretrained(src_name_or_path, trust_remote_code=True)
    tok.save_pretrained(BACKBONE_FP_DIR)
    print(f"[OK] Saved tokenizer to: {BACKBONE_FP_DIR}")
    return tok


def quantize_backbone():
    tokenizer = ensure_tokenizer()

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "gemm",
    }

    calib_texts = build_calib_texts()

    print("[INFO] Loading backbone for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(
        BACKBONE_FP_DIR,
        low_cpu_mem_usage=True,
        use_cache=True,
    )

    print("[INFO] Start quantization...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_samples=len(calib_texts),
        max_calib_seq_len=128,          # 先从 64 开始
        n_parallel_calib_samples=1,    # 保守一点
    )
    os.makedirs(BACKBONE_AWQ_DIR, exist_ok=True)
    model.save_quantized(BACKBONE_AWQ_DIR)
    tokenizer.save_pretrained(BACKBONE_AWQ_DIR)

    with open(os.path.join(BACKBONE_AWQ_DIR, "awq_quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(quant_config, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved AWQ backbone to: {BACKBONE_AWQ_DIR}")


def load_awq_backbone_to_target_device(target_device):
    print("[INFO] Loading quantized AWQ backbone...")
    awq_model = AutoAWQForCausalLM.from_quantized(
        BACKBONE_AWQ_DIR,
        fuse_layers=False,      # 第一版保守一点
        trust_remote_code=True,
    )
    awq_model = awq_model.eval()

    # AutoAWQ 返回包装对象，真实 HF-like 模型通常在 .model
    real_model = awq_model.model if hasattr(awq_model, "model") else awq_model
    real_model = real_model.to(target_device).eval()

    print("[OK] AWQ backbone loaded.")
    return real_model


def patch_qwen2encoder_forward(core):
    import types

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

def run_inference_with_replaced_backbone():
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    core = getattr(cosyvoice, "model", cosyvoice)

    old_backbone = core.llm.llm.model
    target_device = next(old_backbone.parameters()).device

    print("[INFO] old backbone device:", target_device)
    print("[INFO] old backbone dtype :", next(old_backbone.parameters()).dtype)

    awq_backbone = load_awq_backbone_to_target_device(target_device)
    core.llm.llm.model = awq_backbone
    patch_qwen2encoder_forward(core)
    print("[INFO] new backbone device:", next(core.llm.llm.model.parameters()).device)
    print("[INFO] speech_embedding device:", core.llm.speech_embedding.weight.device)
    print("[INFO] decoder device:", core.llm.llm_decoder.weight.device)
    print("[INFO] embed_tokens device:", core.llm.llm.model.model.embed_tokens.weight.device)

    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
            "You are a helpful assistant.<|endofprompt|>希望你以后能够做得比我还好呦。",
            PROMPT_WAV,
            stream=False,
        )
    ):
        out_wav = f"zero_shot_awq_{i}.wav"
        torchaudio.save(out_wav, j["tts_speech"], cosyvoice.sample_rate)
        print("[OK] saved:", out_wav)


def main():
    # export_backbone()
    quantize_backbone()
    run_inference_with_replaced_backbone()


if __name__ == "__main__":
    main()