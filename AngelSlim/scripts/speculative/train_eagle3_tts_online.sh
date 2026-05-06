# #!/bin/bash

# export CONFIG_DIR=angelslim/compressor/speculative/train/configs
# export TARGET_MODEL_NAME_OR_PATH=
# export DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/cosyvoice3-llm-eagle3.json
# export TRAIN_DATA_PATH=
# export OUTPUT_DIR=
# export RUN_NAME=
# export MODEL_MAX_LENGTH=

# torchrun --nproc_per_node=8 tools/train_eagle3_online.py \
#     --modal_type TTS \
#     --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
#     --draft_model_config_path $DRAFT_MODEL_CONFIG_PATH \
#     --train_data_path $TRAIN_DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 20 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --learning_rate 1e-4 \
#     --weight_decay 0.0 \
#     --warmup_ratio 0.1 \
#     --lr_scheduler_type "constant" \
#     --logging_steps 20 \
#     --model_max_length $MODEL_MAX_LENGTH \
#     --training_time_test_length 4 \
#     --deepspeed $CONFIG_DIR/deepspeed_zero3.json \
#     --report_to wandb \
#     --run_name $RUN_NAME \


export CONFIG_DIR=/home/b24zll/data/wjq_data/CosyVoice/AngelSlim/angelslim/compressor/speculative/train/configs
export TARGET_MODEL_NAME_OR_PATH=/home/b24zll/data/wjq_data/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
export DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/cosyvoice3-llm-eagle3.json
export TRAIN_DATA_PATH=/home/b24zll/data/wjq_data/CosyVoice/AngelSlim/dataset/tts_fake_data/train_regenerate.jsonl
export OUTPUT_DIR=/home/b24zll/data/wjq_data/CosyVoice/eagle3_cosyvoice3
export RUN_NAME=cosyvoice3_eagle3_tts
export MODEL_MAX_LENGTH=2048
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 /home/b24zll/data/wjq_data/CosyVoice/AngelSlim/tools/train_eagle3_online.py \
    --modal_type TTS \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path $DRAFT_MODEL_CONFIG_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --model_max_length $MODEL_MAX_LENGTH \
    --training_time_test_length 4 \
    --deepspeed $CONFIG_DIR/deepspeed_zero3.json \
    --report_to none \
    --run_name $RUN_NAME
