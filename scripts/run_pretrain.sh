#!/bin/bash
# 自监督预训练入口 (针对 ShapeNet Part)

export CUDA_VISIBLE_DEVICES=0

python train/pretrain.py \
    --config configs/pretrain.yaml
