export COSYVOICE_ROOT=/home/b24zll/data/wjq_data/CosyVoice
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client_github.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --concurrency 1 \
  --total_requests 5\
  --output_dir /home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results_wanda

export COSYVOICE_ROOT=/home/b24zll/data/wjq_data/CosyVoice
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client_github.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --concurrency 2 \
  --total_requests 10\
  --output_dir /home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results_wanda


export COSYVOICE_ROOT=/home/b24zll/data/wjq_data/CosyVoice
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client_github.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --concurrency 4 \
  --total_requests 20\
  --output_dir /home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results_wanda

export COSYVOICE_ROOT=/home/b24zll/data/wjq_data/CosyVoice
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client_github.py \
  --host 127.0.0.1 \
  --port 50000 \
  --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
  --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
  --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --concurrency 8 \
  --total_requests 40 \
  --output_dir /home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results_wanda

# export COSYVOICE_ROOT=/home/b24zll/data/wjq_data/CosyVoice
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python bench_client_github.py \
#   --host 127.0.0.1 \
#   --port 50000 \
#   --prompt_text "You are a helpful assistant. <|endofprompt|>希望你以后能够做的比我还好呦。" \
#   --prompt_wav /home/b24zll/data/wjq_data/CosyVoice/asset/zero_shot_prompt.wav \
#   --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
#   --concurrency 16 \
#   --total_requests 80