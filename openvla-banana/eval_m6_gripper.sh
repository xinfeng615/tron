#!/bin/bash
#export PYTHONPATH=/home/guo/openvla-m6:$PYTHONPATH
#bash eval_m6.sh
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python experiments/robot/metaworld/run_m6_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /home/guo/openvla-m6/output_m6/openvla-7b+metaworld_m6_50e+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug/merged-step-20000 \
  --task_suite_name metaworld_m6_50e \
  --center_crop True \
  --num_trials_per_task 50 \
  --seed 7 \
  --binarize_gripper False\
  --use_wandb True \
  --wandb_project "m6-eval" \
  --wandb_entity "1469512941-"
