#!/bin/bash
# 分割任务微调入口 (针对 ShapeNet Part)

export CUDA_VISIBLE_DEVICES=0

python train/finetune_seg.py \
    --config configs/finetune_seg.yaml
