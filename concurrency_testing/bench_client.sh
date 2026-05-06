export COSYVOICE_ROOT=/home/b24zll/data/wjq_data/CosyVoice
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --spk_id bench_spk \
  --concurrency 1 \
  --num_requests 2 \
  --out_json l20_c1_awq.json


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --spk_id bench_spk \
  --concurrency 2 \
  --num_requests 10 \
  --out_json l20_c2_awq.json


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --spk_id bench_spk \
  --concurrency 4 \
  --num_requests 20 \
  --out_json l20_c4_awq.json


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --spk_id bench_spk \
  --concurrency 8 \
  --num_requests 40 \
  --out_json l20_c8_awq.json