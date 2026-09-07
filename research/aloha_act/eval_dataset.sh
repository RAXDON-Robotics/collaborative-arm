#!/bin/bash
task_name="folded_orange_towel"
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 eval_dataset.py \
    --task_name ${task_name} \
    --ckpt_dir output/act_ckpt/act-${task_name}/ \
    --ckpt_name policy_best.ckpt \
    --policy_class ACT \
    --chunk_size 30 \
    --control_rate 30 \
    --seed 0 \
    --temporal_agg