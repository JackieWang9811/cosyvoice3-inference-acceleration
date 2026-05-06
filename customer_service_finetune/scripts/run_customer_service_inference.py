
#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path('/home/b24zll/data/wjq_data/CosyVoice')
os.environ.setdefault('PYTHONIOENCODING', 'UTF-8')
os.environ.setdefault('CUDA_HOME', str(ROOT / 'customer_service_finetune/fake_cuda'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/Matcha-TTS'))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

BASE_MODEL = ROOT / 'pretrained_models/Fun-CosyVoice3-0.5B'
CKPT_DIR = ROOT / 'customer_service_finetune/exp/cosyvoice3/llm/torch_ddp'
WORK_DIR = ROOT / 'customer_service_finetune/infer_models'
OUT_DIR = ROOT / 'customer_service_finetune/infer_outputs/customer_service_samples'
PROMPT_WAV = ROOT / 'asset/zero_shot_prompt.wav'
STYLE = 'You are a helpful customer service assistant. Speak in Mandarin Chinese with a professional, warm, patient and clear tone.<|endofprompt|>'

CHECKPOINTS = {
    'baseline': None,
    'epoch0': CKPT_DIR / 'epoch_0_whole.pt',
    'epoch2': CKPT_DIR / 'epoch_2_whole.pt',
    'epoch10': CKPT_DIR / 'epoch_10_whole.pt',
    'epoch19': CKPT_DIR / 'epoch_19_whole.pt',
}

TEST_CASES = [
    ('greeting', '??????????????????????????'),
    ('order_query', '???????????????????????????????'),
    ('refund', '??????????????????????????'),
    ('apology', '?????????????????????????????'),
    ('handoff', '????????????????????????????????'),
]


def prepare_model(tag: str, checkpoint: Path | None) -> Path:
    if checkpoint is None:
        return BASE_MODEL
    model_dir = WORK_DIR / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        'cosyvoice3.yaml', 'flow.pt', 'hift.pt', 'campplus.onnx',
        'speech_tokenizer_v3.onnx', 'configuration.json', '.msc', '.mv'
    ]:
        src = BASE_MODEL / name
        if src.exists():
            dst = model_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
    qwen_src = BASE_MODEL / 'CosyVoice-BlankEN'
    qwen_dst = model_dir / 'CosyVoice-BlankEN'
    if qwen_dst.exists() or qwen_dst.is_symlink():
        qwen_dst.unlink()
    qwen_dst.symlink_to(qwen_src, target_is_directory=True)

    llm_dst = model_dir / 'llm.pt'
    if llm_dst.exists() or llm_dst.is_symlink():
        llm_dst.unlink()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {k: v for k, v in state.items() if k not in {'epoch', 'step'}}
    torch.save(state, llm_dst)
    return model_dir


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ckpt_idx, (tag, ckpt) in enumerate(CHECKPOINTS.items()):
        model_dir = prepare_model(tag, ckpt)
        print(f'Loading {tag}: {model_dir}', flush=True)
        cosyvoice = AutoModel(model_dir=str(model_dir), fp16=False)
        for case_idx, (scenario, text) in enumerate(TEST_CASES):
            set_all_random_seed(20260422 + ckpt_idx * 100 + case_idx)
            full_text = STYLE + text
            try:
                count = 0
                for part_idx, result in enumerate(cosyvoice.inference_cross_lingual(full_text, str(PROMPT_WAV), stream=False, text_frontend=True)):
                    samples = int(result['tts_speech'].shape[1])
                    duration = samples / cosyvoice.sample_rate
                    out = OUT_DIR / f'{tag}_{case_idx+1:02d}_{scenario}_{part_idx}.wav'
                    torchaudio.save(str(out), result['tts_speech'].cpu(), cosyvoice.sample_rate)
                    manifest.append({
                        'checkpoint': tag,
                        'scenario': scenario,
                        'text': text,
                        'path': str(out),
                        'sample_rate': cosyvoice.sample_rate,
                        'duration_seconds': duration,
                    })
                    print(f'Saved {out} duration={duration:.2f}s', flush=True)
                    count += 1
                if count == 0:
                    print(f'NO_OUTPUT {tag} {scenario}', flush=True)
            except Exception as exc:
                print(f'ERROR {tag} {scenario}: {exc}', flush=True)
                traceback.print_exc()
        del cosyvoice
    manifest_path = OUT_DIR / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'MANIFEST {manifest_path}', flush=True)

if __name__ == '__main__':
    main()
