#!/bin/bash
set -euo pipefail

ROOT=/home/b24zll/data/wjq_data/CosyVoice
UTILS_DIR=${ROOT}/utils
NUM_JOB=1
LANG=en

# 结果目录：里面放生成好的 wav 和对应的 jsonl
DECODE_DIR=/home/b24zll/data/wjq_data/CosyVoice/speculative_decoding/benchmark_results_0.6/eagle

# benchmark 输出的 jsonl
JSONL_PATH=/home/b24zll/data/wjq_data/CosyVoice/speculative_decoding/benchmark_results_0.6/cosyvoice3-temperature-0.6-eagle.jsonl
# wav 所在目录；如果 wav 已经在根目录，会自动复制到 wavs/
WAV_DIR=${DECODE_DIR}/
META_LST=${DECODE_DIR}/meta.lst

echo "=========================================="
echo "Scoring WER for ${DECODE_DIR}"
echo "=========================================="

mkdir -p "${WAV_DIR}"

# 把根目录下 wav 同步到 wavs/
# find "${DECODE_DIR}" -maxdepth 1 -name "*.wav" -exec cp -f {} "${WAV_DIR}/" \;

# 从 jsonl 生成 meta.lst
python3 - <<PY
import json
from pathlib import Path

jsonl_path = Path("${JSONL_PATH}")
wav_dir = Path("${WAV_DIR}")
meta_path = Path("${META_LST}")

if not jsonl_path.exists():
    raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

missing = []
count = 0

with jsonl_path.open("r", encoding="utf-8") as f, meta_path.open("w", encoding="utf-8") as out:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        # 优先用 question_id，对应 eval_{question_id}.wav
        qid = obj.get("question_id", i)
        wav_stem = f"eval_{qid}"
        wav_path = wav_dir / f"{wav_stem}.wav"

        text = None
        choices = obj.get("choices", [])
        if choices and isinstance(choices, list):
            text = choices[0].get("tts_text")

        if text is None:
            raise ValueError(f"Cannot find choices[0]['tts_text'] in line {i}")

        if not wav_path.exists():
            missing.append(str(wav_path))

        # cal_wer.sh 需要：utt_id + ref_text
        out.write(f"{wav_stem} {text}\n")
        count += 1

print(f"Generated meta list: {meta_path}")
print(f"Total entries: {count}")

if missing:
    print("\\n[Warning] Missing wav files:")
    for p in missing[:20]:
        print(p)
    if len(missing) > 20:
        print(f"... and {len(missing)-20} more")
PY

echo
echo "Preview of meta.lst:"
head -n 5 "${META_LST}" || true
echo

bash "${UTILS_DIR}/cal_wer.sh" \
    "${META_LST}" \
    "${DECODE_DIR}" \
    "${LANG}" \
    "${NUM_JOB}"

echo
echo "WER summary for ${DECODE_DIR}:"
cat "${DECODE_DIR}/wav_res_ref_text.wer"
echo