#!/bin/bash

export PYTHONPATH=/home/b24zll/data/wjq_data/CosyVoice:/home/b24zll/data/wjq_data/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python3 /home/b24zll/data/wjq_data/CosyVoice/AngelSlim/tools/spec_benchmark.py \
    --base-model-path /home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
    --eagle-model-path /home/b24zll/data/wjq_data/CosyVoice/eagle3_cosyvoice3/checkpoint-40 \
    --model-id cosyvoice3 \
    --mode both \
    --deploy-backend pytorch \
    --is-tts \
    --bench-name tts_fake_data \
    --temperature 0.6 \
    --output-dir /home/b24zll/data/wjq_data/CosyVoice/speculative_decoding/benchmark_results \
    --generate-audio