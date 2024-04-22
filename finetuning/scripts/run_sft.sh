#! /bin/bash
export CUDA_VISIBLE_DEVICES='0'
export WANDB_PROJECT=WizardCoder
export WANDB_RUN_ID=7
export WANDB_RESUME=allow
model_name_or_path=/hy-tmp/WizardCoder-Python-7B-V1.0

train_file=/hy-tmp/pixiu_train_new/data/MID_train_EN_53K.json
validation_file=/hy-tmp/pixiu_train_new/data/MID_eval_EN_2K.json
output_dir="/hy-tmp/pixiu_train_new/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=512

#LoRA with 8bit
torchrun --nproc_per_node 1 /hy-tmp/pixiu_train_new/src/sft_train_new.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --use_lora \
    --use_int8_training \
    --lora_config /hy-tmp/pixiu_train_new/configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...

# LoRA without 8bit
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --deepspeed configs/deepspeed_config_stage3.json \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 3e-4 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
   # --use_flash_attention
   # --resume_from_checkpoint ...
