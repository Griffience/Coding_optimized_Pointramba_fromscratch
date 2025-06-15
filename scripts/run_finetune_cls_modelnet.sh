#!/bin/bash
# 微调分类任务 ModelNet40

export CUDA_VISIBLE_DEVICES=0

python train/finetune_cls.py \
    --config configs/finetune_cls_modelnet.yaml
