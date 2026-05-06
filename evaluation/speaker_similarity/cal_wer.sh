#!/bin/bash
set -x
set -e

meta_lst=$1
output_dir=$2
lang=$3
num_job=$4

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ -d "$output_dir/wavs" ]; then
    wav_dir="$output_dir/wavs"
else
    wav_dir="$output_dir"
fi

python3 "${SCRIPT_DIR}/get_wav_res_ref_text.py" "$meta_lst" "$wav_dir" "$wav_wav_text"

timestamp=$(date +%s)
thread_dir=${output_dir}/tmp/thread_metas_$timestamp/
out_dir=${thread_dir}/results/

mkdir -p "$out_dir"

num=$(wc -l "$wav_wav_text" | awk '{print $1}')
num_per_thread=$(( num / num_job + 1 ))

split -l "$num_per_thread" --additional-suffix=.lst -d "$wav_wav_text" "${thread_dir}/thread-"

num_job_minus_1=$(( num_job - 1 ))
if [ "$num_job_minus_1" -ge 0 ]; then
    for rank in $(seq 0 $((num_job - 1))); do
        sub_score_file="${out_dir}/thread-0${rank}.wer.out"
        CUDA_VISIBLE_DEVICES=${rank} \
        python3 "${SCRIPT_DIR}/run_wer.py" "${thread_dir}/thread-0${rank}.lst" "$sub_score_file" "$lang" &
    done
fi
wait

cat "${out_dir}"/thread-0*.wer.out > "${out_dir}/merge.out"
python3 "${SCRIPT_DIR}/average_wer.py" "${out_dir}/merge.out" "$score_file"