python benchmark_fp32_vs_pruned.py \
  --model_dir /home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --pruned_ckpt /home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/llm_pruned.pt \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --out_dir ./benchmark_fp32_vs_pruned_outputs