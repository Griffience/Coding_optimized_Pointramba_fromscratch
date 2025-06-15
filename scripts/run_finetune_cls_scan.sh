#!/bin/bash
# 微调分类任务 ScanObjectNN

export CUDA_VISIBLE_DEVICES=0

python train/finetune_cls.py \
    --config configs/finetune_cls_scanobjectnn.yaml
