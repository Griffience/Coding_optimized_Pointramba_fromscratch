# utils/data_transforms.py

import numpy as np
import torch
import random

class PointcloudRotate(object):
    """随机绕y轴旋转点云"""
    def __call__(self, pc):
        B = pc.size(0)
        for i in range(B):
            angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(angle)
            sinval = np.sin(angle)
            rotation_matrix = torch.tensor([
                [cosval, 0, sinval],
                [0,      1, 0     ],
                [-sinval,0, cosval]
            ], dtype=torch.float32, device=pc.device)
            pc[i] = pc[i] @ rotation_matrix
        return pc

class PointcloudScale(object):
    """随机缩放点云"""
    def __init__(self, scale_low=0.66, scale_high=1.5):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        B = pc.size(0)
        for i in range(B):
            scales = torch.from_numpy(
                np.random.uniform(self.scale_low, self.scale_high, size=[3])
            ).float().to(pc.device)
            pc[i, :, 0:3] *= scales
        return pc

class PointcloudTranslate(object):
    """随机平移点云"""
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        B = pc.size(0)
        for i in range(B):
            translation = torch.from_numpy(
                np.random.uniform(-self.translate_range, self.translate_range, size=[3])
            ).float().to(pc.device)
            pc[i, :, 0:3] += translation
        return pc

class PointcloudJitter(object):
    """对点云增加噪声抖动"""
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, pc):
        B = pc.size(0)
        noise = torch.clamp(
            torch.randn((B, pc.size(1), 3), device=pc.device) * self.std, 
            -self.clip, self.clip
        )
        pc[:, :, 0:3] += noise
        return pc

class PointcloudRandomInputDropout(object):
    """随机dropout一些点"""
    def __init__(self, max_dropout_ratio=0.5):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        B, N, _ = pc.shape
        for b in range(B):
            dropout_ratio = np.random.rand() * self.max_dropout_ratio
            drop_idx = np.where(np.random.rand(N) < dropout_ratio)[0]
            if len(drop_idx) > 0:
                pc[b, drop_idx, :] = pc[b, 0, :].unsqueeze(0)  # 把drop的点替换成第一个点
        return pc
