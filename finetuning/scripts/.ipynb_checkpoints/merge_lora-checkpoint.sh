#! /bin/bash

model_name_or_path=/hy-tmp/WizardCoder-Python-7B-V1.0
lora_path=/hy-tmp/pixiu_train_new/saved_models/WizardCoder_7/checkpoint-9936
output_path=/hy-tmp/pixiu_train_new/merge_models/pft_checkpoint-9936

CUDA_VISIBLE_DEVICES=0 python /hy-tmp/pixiu_train_new/src/merge_llama_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --output_path ${output_path} \
    --lora_path ${lora_path}