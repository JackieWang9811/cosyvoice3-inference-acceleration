python prune_cosyvoice_qwen_wanda_from_regenerate.py \
    --model_dir /home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
    --calib_jsonl /home/b24zll/data/wjq_data/CosyVoice/AngelSlim/dataset/tts_fake_data/train_regenerate.jsonl \
    --out_llm_pt /home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/llm_pruned.pt \
    --sparsity 0.3 \
    --max_samples 256 \
    --max_length 2048 \
    --audio_token_offset 0