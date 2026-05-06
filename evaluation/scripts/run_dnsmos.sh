#!/bin/bash
set -e

ROOT=/home/b24zll/data/wjq_data/CosyVoice
DNSMOS_LAB=${ROOT}/utils/DNSMOS
    # ${ROOT}/concurrency_testing/test_results \
    # ${ROOT}/concurrency_testing/test_results_awq_v2
    # ${ROOT}/concurrency_testing/test_results_wanda
    # ${ROOT}/speculative_decoding/benchmark_results_0.8/eagle
for decode_dir in \
    ${ROOT}/concurrency_testing/test_results_wanda
do
    echo "Scoring DNSMOS for ${decode_dir}"

    find "${decode_dir}" -name "*.wav" | awk -F '/' '{print $NF, $0}' | sed 's@\.wav @ @g' > "${decode_dir}/wav.scp"

    python "${DNSMOS_LAB}/dnsmos_local_wavscp.py" \
      -t "${decode_dir}/wav.scp" \
      -e "${DNSMOS_LAB}" \
      -o "${decode_dir}/mos.csv"

    cat "${decode_dir}/mos.csv" | sed '1d' | awk -F ',' '{sum += $NF; count++} END {if (count > 0) print sum / count}' > "${decode_dir}/dnsmos_mean.txt"

    echo "${decode_dir} mean DNSMOS:"
    cat "${decode_dir}/dnsmos_mean.txt"
    echo
done