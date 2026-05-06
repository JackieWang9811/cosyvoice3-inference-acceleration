
#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path('/home/b24zll/data/wjq_data/CosyVoice')
SRC = ROOT / 'customer_service_finetune/infer_outputs/zero_shot_customer_service_utf8'
OUT = ROOT / 'customer_service_finetune/eval/zero_shot_customer_service_utf8'
PROMPT_WAV = ROOT / 'asset/zero_shot_prompt.wav'
PY = '/home/b24zll/software/anaconda3/envs/voice/bin/python'


def load_manifest():
    data = json.load(open(SRC / 'manifest.json', encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    wavs = OUT / 'wavs'
    wavs.mkdir(exist_ok=True)
    rows = []
    for i, r in enumerate(data):
        stem = f"{r['checkpoint']}__{r['prompt']}__{r['scenario']}"
        src = Path(r['path'])
        dst = wavs / f'{stem}.wav'
        shutil.copy2(src, dst)
        rows.append({**r, 'utt_id': stem, 'eval_wav': str(dst)})
    with (OUT / 'wav.scp').open('w', encoding='utf-8') as f, \
         (OUT / 'prompt_wav.scp').open('w', encoding='utf-8') as pf, \
         (OUT / 'meta.lst').open('w', encoding='utf-8') as mf:
        for r in rows:
            f.write(f"{r['utt_id']} {r['eval_wav']}\n")
            pf.write(f"{r['utt_id']} {PROMPT_WAV}\n")
            mf.write(f"{r['utt_id']} {r['target_text']}\n")
    json.dump(rows, open(OUT / 'manifest.eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return rows


def run(cmd, log_name, env=None):
    log_path = OUT / log_name
    print('RUN', ' '.join(map(str, cmd)))
    with log_path.open('w', encoding='utf-8') as log:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    print('EXIT', p.returncode, log_path)
    return p.returncode


def run_dnsmos():
    dnsmos = ROOT / 'utils/DNSMOS/dnsmos_local_wavscp.py'
    lab = ROOT / 'utils/DNSMOS'
    rc = run([PY, str(dnsmos), '-t', str(OUT / 'wav.scp'), '-e', str(lab), '-o', str(OUT / 'mos.csv')], 'dnsmos.log')
    if rc == 0 and (OUT / 'mos.csv').exists():
        rows = list(csv.DictReader(open(OUT / 'mos.csv', encoding='utf-8')))
        keys = rows[0].keys() if rows else []
        numeric = defaultdict(list)
        for row in rows:
            for k in keys:
                try:
                    numeric[k].append(float(row[k]))
                except Exception:
                    pass
        with (OUT / 'dnsmos_summary.json').open('w', encoding='utf-8') as f:
            json.dump({k: float(np.mean(v)) for k, v in numeric.items() if v}, f, ensure_ascii=False, indent=2)


def run_speaker_similarity():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'utils/3D-Speaker') + ':' + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    rc = run([
        PY, str(ROOT / 'utils/eval_speaker_similarity.py'),
        '--model_id', 'damo/speech_eres2net_sv_en_voxceleb_16k',
        '--local_model_dir', str(ROOT / 'utils/3D-Speaker/pretrained'),
        '--prompt_wavs', str(OUT / 'prompt_wav.scp'),
        '--hyp_wavs', str(OUT / 'wav.scp'),
        '--log_file', str(OUT / 'spk_simi_scores.txt'),
        '--devices', '0',
    ], 'speaker_similarity.log', env=env)
    return rc


def run_wer():
    rc = run(['bash', str(ROOT / 'utils/cal_wer.sh'), str(OUT / 'meta.lst'), str(OUT), 'zh', '1'], 'wer.log')
    return rc


def voiced_segments(y, sr, frame_length=1024, hop_length=256):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return np.array([], dtype=bool), rms
    threshold = max(float(np.percentile(rms, 35)) * 0.8, 1e-4)
    return rms > threshold, rms


def prosody_stats(rows):
    out_rows = []
    for r in rows:
        wav = r['eval_wav']
        y, sr = librosa.load(wav, sr=None, mono=True)
        duration = len(y) / sr if sr else 0.0
        mask, rms = voiced_segments(y, sr)
        silence_ratio = float(1.0 - mask.mean()) if len(mask) else 1.0
        try:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:
            voiced_f0 = np.array([])
        chars = len(r['target_text'])
        cps = chars / duration if duration > 0 else 0.0
        row = {
            'utt_id': r['utt_id'],
            'checkpoint': r['checkpoint'],
            'prompt': r['prompt'],
            'scenario': r['scenario'],
            'duration_sec': duration,
            'chars': chars,
            'chars_per_sec': cps,
            'silence_ratio': silence_ratio,
            'rms_mean': float(np.mean(rms)) if len(rms) else 0.0,
            'rms_std': float(np.std(rms)) if len(rms) else 0.0,
            'f0_mean': float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0,
            'f0_std': float(np.std(voiced_f0)) if len(voiced_f0) else 0.0,
            'f0_voiced_ratio': float(len(voiced_f0) / len(rms)) if len(rms) else 0.0,
        }
        out_rows.append(row)
    with (OUT / 'prosody_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    grouped = defaultdict(list)
    for row in out_rows:
        grouped[(row['checkpoint'], row['prompt'])].append(row)
    summary = {}
    metrics = ['duration_sec','chars_per_sec','silence_ratio','rms_mean','rms_std','f0_mean','f0_std','f0_voiced_ratio']
    for key, vals in grouped.items():
        summary['/'.join(key)] = {m: float(np.mean([v[m] for v in vals])) for m in metrics}
    json.dump(summary, open(OUT / 'prosody_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def main():
    rows = load_manifest()
    prosody_stats(rows)
    run_dnsmos()
    run_speaker_similarity()
    run_wer()
    print('EVAL_DIR', OUT)

if __name__ == '__main__':
    main()
