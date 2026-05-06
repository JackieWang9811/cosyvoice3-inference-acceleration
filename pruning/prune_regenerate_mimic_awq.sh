MODEL_DIR=/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
BACKBONE_OUT=/home/b24zll/data/wjq_data/CosyVoice/pruning/wanda/cv3_qwen2_backbone_wanda_s30
VLLM_OUT=/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/vllm_wanda_s30
PROMPT_WAV=/home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav


python prune_cosyvoice_qwen_wanda_from_regenerate_mimic_awq.py \
  --model_dir ${MODEL_DIR} \
  --backbone_out_dir ${BACKBONE_OUT} \
  --vllm_out_dir ${VLLM_OUT} \
  --sparsity 0.3 \
  --max_samples 10 \
  --max_length 128 \
  --dtype fp16 \
  --device cuda \
  --run_zero_shot_test \
  --prompt_wav ${PROMPT_WAV}