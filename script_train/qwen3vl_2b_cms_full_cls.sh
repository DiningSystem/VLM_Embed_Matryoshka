#!/usr/bin/env bash
set -euo pipefail

# CMS-MRL FULL classification run for Qwen/Qwen3-VL-Embedding-2B.
torchrun --standalone --nproc_per_node=1 --master_port=29500 train_ddp_one_model.py \
    --lora --lora_r 16 --lora_alpha 64 \
    --model_name Qwen/Qwen3-VL-Embedding-2B --model_backbone qwen3_vl \
    --bf16 --pooling eos --normalize True --temperature 0.02 \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name ImageNet_1K HatefulMemes SUN397 N24News VOC2007 Place365 ImageNet-A ImageNet-R ObjectNet Country211 \
    --dataset_split original \
    --image_dir /workspace/ComfyUI/models/gligen/VLM_Embed/vlm2vec_train/MMEB-train \
    --output_dir training/CMSMRL_Qwen3VL_2B_cls_full \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine --learning_rate 1e-5 --num_train_epochs 1 \
    --save_total_limit 2 --logging_steps 1 --save_strategy epoch --seed 42 --weight_decay 0.01 \
    --kd_loss_type cms_mrl --warmup_ratio 0.03 --image_resolution mid \
    --cms_num_groups 8 --cms_router_hidden_dim 256 --cms_utility_weight 1.0 --cms_cmi_weight 0.1 \
    --nested_dims 64 128 256 512 768 1024 2048
