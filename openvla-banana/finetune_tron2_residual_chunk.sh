#!/bin/bash
# bash finetune_tron2_residual_chunk.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_DETERMINISTIC_OPS=1

mkdir -p /root/autodl-tmp/openvla/output_tron2
mkdir -p /root/autodl-tmp/openvla/adapter-tmp_tron2

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/root/autodl-tmp/openvla/openvla-7b" \
  --data_root_dir "/root/autodl-tmp/tensorflow_datasets" \
  --dataset_name tron2_lerobot \
  --run_root_dir "/root/autodl-tmp/openvla/output_tron2" \
  --adapter_tmp_dir "/root/autodl-tmp/openvla/adapter-tmp_tron2" \
  --action_dim 16 \
  --camera_views "primary,secondary,wrist" \
  --use_proprio True \
  --proprio_dim 16 \
  --use_lora True \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 20000 \
  --save_steps 2000 \
  --save_latest_checkpoint_only False \
  --seed 7 \
  --deterministic_training True \
  --use_residual_action_head True \
  --residual_action_dims "0,1,2,3,4,5,6,8,9,10,11,12,13,14" \
  --residual_action_scale 0.25 \
  --residual_loss_weight 1.0 \
  --use_action_chunking True \
  --action_chunk_size 4 \
  --action_chunk_dims "0,1,2,3,4,5,6,8,9,10,11,12,13,14" \
  --action_chunk_scale 0.75 \
  --action_chunk_loss_weight 1.0 \
  --wandb_project "tron2-openvla-residual-chunk" \
  --wandb_entity "1469512941-" \
  2>&1 | tee /root/autodl-tmp/openvla/output_tron2/train_log_residual_chunk.txt
