#!/bin/bash
set -e

ROOT=/home/b24zll/data/wjq_data/CosyVoice
SPK_LAB=${ROOT}/utils/3D-Speaker
export PYTHONPATH=${SPK_LAB}:${PYTHONPATH}

PROMPT_WAV=${ROOT}/AngelSlim/dataset/tts_fake_data/zero_shot_prompt.wav
    # ${ROOT}/concurrency_testing/test_results \
    # ${ROOT}/concurrency_testing/test_results_awq_v2
    # ${ROOT}/concurrency_testing/test_results_wanda
    # ${ROOT}/speculative_decoding/benchmark_results_0.8/eagle
for decode_dir in \
    ${ROOT}/concurrency_testing/test_results_wanda
do
    echo "Scoring speaker similarity for ${decode_dir}"

    find "${decode_dir}" -name "*.wav" | sort | \
        awk -F '/' '{fname=$NF; sub(/\.wav$/, "", fname); print fname, $0}' \
        > "${decode_dir}/wav.scp"

    awk -v pwav="${PROMPT_WAV}" '{print $1, pwav}' \
        "${decode_dir}/wav.scp" \
        > "${decode_dir}/prompt_wav.scp"

    python "${ROOT}/utils/eval_speaker_similarity.py" \
        --model_id damo/speech_eres2net_sv_en_voxceleb_16k \
        --local_model_dir "${SPK_LAB}/pretrained" \
        --prompt_wavs "${decode_dir}/prompt_wav.scp" \
        --hyp_wavs "${decode_dir}/wav.scp" \
        --log_file "${decode_dir}/spk_simi_scores.txt" \
        --devices "0"

    echo "Speaker similarity saved to ${decode_dir}/spk_simi_scores.txt"
    echo
done