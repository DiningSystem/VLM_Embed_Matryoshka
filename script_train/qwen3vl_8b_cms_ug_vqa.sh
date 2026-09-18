#!/usr/bin/env bash
set -euo pipefail

# CMS-MRL UG run for Qwen/Qwen3-VL-Embedding-8B.
torchrun --standalone --nproc_per_node=1 --master_port=29500 train_ddp_one_model.py \
    --lora --lora_r 16 --lora_alpha 64 \
    --model_name Qwen/Qwen3-VL-Embedding-8B --model_backbone qwen3_vl \
    --bf16 --pooling eos --normalize True --temperature 0.02 \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W \
    --dataset_split original \
    --image_dir /workspace/ComfyUI/models/gligen/VLM_Embed/vlm2vec_train/MMEB-train \
    --output_dir training/CMSMRL_Qwen3VL_8B_vqa_ug \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine --learning_rate 5e-6 --num_train_epochs 1 \
    --save_total_limit 2 --logging_steps 1 --save_strategy epoch --seed 42 --weight_decay 0.01 \
    --kd_loss_type cms_ug --warmup_ratio 0.03 --image_resolution mid \
    --cms_num_groups 8 --cms_router_hidden_dim 256 --cms_utility_weight 1.0 --cms_cmi_weight 0.1 \
    --nested_dims 64 128 256 512 768 1024 2048 4096
