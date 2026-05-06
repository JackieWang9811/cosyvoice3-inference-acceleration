import os
import shutil
import torch
import sys
from pathlib import Path
import os
import json
import torchaudio
import torch

"""
这段代码具体在做：
1. 备份原始的 vLLM 目录
2. 加载 AWQ 量化的 backbone 并替换原始 backbone
3. 导出基于 AWQ 的 vLLM 目录
4. 可选地替换官方的 vLLM 目录为 AWQ 版本
"""

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Matcha-TTS"))

print("[Path] PROJECT_ROOT =", PROJECT_ROOT)
print("[Path] MATCHA_ROOT  =", PROJECT_ROOT / "third_party" / "Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel
from awq import AutoAWQForCausalLM

# MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
# BACKBONE_AWQ_DIR = "/home/b24zll/data/wjq_data/CosyVoice/quantization/quantization_awq/cv3_qwen2_backbone_awq_v2"

# VLLM_ORIG_DIR = os.path.join(MODEL_DIR, "vllm")
# VLLM_AWQ_DIR = os.path.join(MODEL_DIR, "vllm_awq_v2")
# VLLM_BACKUP_DIR = os.path.join(MODEL_DIR, "vllm_original")


# def load_awq_backbone_to_target_device(target_device):
#     print("[INFO] Loading quantized AWQ backbone...")
#     awq_model = AutoAWQForCausalLM.from_quantized(
#         BACKBONE_AWQ_DIR,
#         fuse_layers=False,
#         trust_remote_code=True,
#     )
#     awq_model = awq_model.eval()

#     # 真实模型通常在 .model
#     real_model = awq_model.model if hasattr(awq_model, "model") else awq_model
#     real_model = real_model.to(target_device).eval()

#     print("[OK] AWQ backbone loaded.")
#     return real_model


# def backup_original_vllm_dir():
#     if os.path.exists(VLLM_ORIG_DIR) and not os.path.exists(VLLM_BACKUP_DIR):
#         print(f"[INFO] Backup original vllm dir: {VLLM_ORIG_DIR} -> {VLLM_BACKUP_DIR}")
#         shutil.copytree(VLLM_ORIG_DIR, VLLM_BACKUP_DIR)
#         print("[OK] Backup done.")
#     else:
#         print("[INFO] Skip backup (either original missing or backup already exists).")


# def export_awq_vllm_dir():
#     # 1) 加载 CosyVoice3，但先不要 load_vllm
#     cosyvoice = AutoModel(
#         model_dir=MODEL_DIR,
#         load_vllm=False,
#     )
#     core = getattr(cosyvoice, "model", cosyvoice)

#     # 2) 拿到原始 backbone 的 device
#     old_backbone = core.llm.llm.model
#     target_device = next(old_backbone.parameters()).device
#     print("[INFO] old backbone device:", target_device)

#     # 3) 加载 AWQ backbone，并替换
#     awq_backbone = load_awq_backbone_to_target_device(target_device)
#     core.llm.llm.model = awq_backbone

#     # 4) 删除旧的导出目录，确保 export 不会直接 return
#     if os.path.exists(VLLM_AWQ_DIR):
#         print(f"[INFO] Remove old dir: {VLLM_AWQ_DIR}")
#         shutil.rmtree(VLLM_AWQ_DIR)

#     # 5) 直接调用导出函数
#     # 注意：这里复用 CosyVoice 内部的 export 逻辑，
#     # 它会把 speech_embedding / llm_decoder 写进导出目录
#     from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm

#     # cosyvoice 通过一个export_cosyvoice2_vllm逻辑到处vllm版本的权重，这样走vllm的时候，
#     export_cosyvoice2_vllm(core.llm, VLLM_AWQ_DIR, target_device)

#     print(f"[OK] Exported AWQ-based CosyVoice vLLM dir to: {VLLM_AWQ_DIR}")


# def replace_official_vllm_dir():
#     # 小心使用：只有在你已经验证 vllm_awq 可用时再替换
#     if not os.path.exists(VLLM_AWQ_DIR):
#         raise RuntimeError(f"{VLLM_AWQ_DIR} not found")

#     backup_original_vllm_dir()

#     if os.path.exists(VLLM_ORIG_DIR):
#         shutil.rmtree(VLLM_ORIG_DIR)

#     shutil.copytree(VLLM_AWQ_DIR, VLLM_ORIG_DIR)
#     print(f"[OK] Replaced {VLLM_ORIG_DIR} with AWQ version.")


# if __name__ == "__main__":
#     backup_original_vllm_dir()
#     export_awq_vllm_dir()
#     # replace_official_vllm_dir()   # 先别急着开，等验证通过再替换

MODEL_DIR = "/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
BACKBONE_AWQ_DIR = "/home/b24zll/data/wjq_data/CosyVoice/quantization/quantization_awq/cv3_qwen2_backbone_awq_v2"


VLLM_ACTIVE_DIR = os.path.join(MODEL_DIR, "vllm")               # 临时激活目录，可不存在
VLLM_ORIGINAL_DIR = os.path.join(MODEL_DIR, "vllm_original")    # 长期保留的官方版本
VLLM_AWQ_V2_DIR = os.path.join(MODEL_DIR, "vllm_awq_v2")        # 这次新导出的版本


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


def export_awq_vllm_dir():
    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        load_vllm=False,
    )
    core = getattr(cosyvoice, "model", cosyvoice)

    old_backbone = core.llm.llm.model
    target_device = next(old_backbone.parameters()).device
    print("[INFO] old backbone device:", target_device)

    awq_backbone = load_awq_backbone_to_target_device(target_device)
    core.llm.llm.model = awq_backbone

    if os.path.exists(VLLM_AWQ_V2_DIR):
        print(f"[INFO] Remove old dir: {VLLM_AWQ_V2_DIR}")
        shutil.rmtree(VLLM_AWQ_V2_DIR)

    from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
    export_cosyvoice2_vllm(core.llm, VLLM_AWQ_V2_DIR, target_device)

    print(f"[OK] Exported AWQ-based CosyVoice vLLM dir to: {VLLM_AWQ_V2_DIR}")


def activate_awq_v2_as_vllm():
    if not os.path.exists(VLLM_AWQ_V2_DIR):
        raise RuntimeError(f"{VLLM_AWQ_V2_DIR} not found")

    if os.path.exists(VLLM_ACTIVE_DIR):
        print(f"[INFO] Remove existing active dir: {VLLM_ACTIVE_DIR}")
        shutil.rmtree(VLLM_ACTIVE_DIR)

    shutil.copytree(VLLM_AWQ_V2_DIR, VLLM_ACTIVE_DIR)
    print(f"[OK] Activated {VLLM_AWQ_V2_DIR} as {VLLM_ACTIVE_DIR}.")


if __name__ == "__main__":
    export_awq_vllm_dir()
    # activate_awq_v2_as_vllm()   # 验证通过后再打开