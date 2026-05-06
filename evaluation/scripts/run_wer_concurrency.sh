#!/bin/bash
set -e

ROOT=/home/b24zll/data/wjq_data/CosyVoice
UTILS_DIR=${ROOT}/utils
NUM_JOB=1
LANG=zh

# 单一参考文本模式时使用
REF_TEXT="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"

# 如果你有每条样本不同的参考文本文件，就填路径 CUSTOM_META_LST=/你的路径/meta.lst；否则留空
# 格式要求：
# result_0 文本A
# result_1 文本B
# ...
CUSTOM_META_LST=""

# ${ROOT}/concurrency_testing/test_results \
# ${ROOT}/concurrency_testing/test_results_awq_v2
for decode_dir in \
    ${ROOT}/concurrency_testing/test_results_wanda
do
    echo "=========================================="
    echo "Scoring WER for ${decode_dir}"
    echo "=========================================="

    mkdir -p "${decode_dir}/wavs"

    # 把根目录下 wav 同步到 wavs/
    find "${decode_dir}" -maxdepth 1 -name "*.wav" -exec cp {} "${decode_dir}/wavs/" \;

    META_LST="${decode_dir}/meta.lst"

    if [ -n "${CUSTOM_META_LST}" ] && [ -f "${CUSTOM_META_LST}" ]; then
        echo "Using custom meta list: ${CUSTOM_META_LST}"
        cp "${CUSTOM_META_LST}" "${META_LST}"
    else
        echo "Using single shared reference text"
        find "${decode_dir}/wavs" -maxdepth 1 -name "*.wav" | sort | \
            awk -F '/' '{fname=$NF; sub(/\.wav$/, "", fname); print fname}' | \
            awk -v txt="${REF_TEXT}" '{print $1, txt}' \
            > "${META_LST}"
    fi

    bash "${UTILS_DIR}/cal_wer.sh" \
        "${META_LST}" \
        "${decode_dir}" \
        "${LANG}" \
        "${NUM_JOB}"

    echo
    echo "WER summary for ${decode_dir}:"
    cat "${decode_dir}/wav_res_ref_text.wer"
    echo
done