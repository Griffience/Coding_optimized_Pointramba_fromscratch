#路径修正
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.grouping import GroupDivider
from models.blocks import TransformerEncoder, MambaEncoder, BiCrossFusion
from models.heads.seg_head import SegmentationHead
from models.heads.cls_head import ClassificationHead

class ImportancePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class TrambaUltra(nn.Module):
    def __init__(self, config, task="segmentation"):
        """
        Args:
            config: dict from YAML
            task: 'segmentation' or 'classification'
        """
        super().__init__()
        self.task = task

        self.trans_dim = config["trans_dim"]
        self.depth = config["depth"]
        self.group_size = config["group_size"]
        self.num_group = config["num_group"]
        self.encoder_dims = config["encoder_dims"]

        if self.task == "segmentation":
            self.seg_num_all = config["seg_num_all"]
        elif self.task == "classification":
            self.cls_num = config["cls_num"]
        else:
            raise ValueError(f"Unknown task {self.task}")

        self.group_divider = GroupDivider(self.num_group, self.group_size)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.transformer = TransformerEncoder(self.trans_dim, self.depth)
        self.importance_predictor = ImportancePredictor(self.trans_dim)
        self.mamba = MambaEncoder(self.trans_dim, self.depth)

        self.bi_fusion = BiCrossFusion(self.trans_dim)
        self.norm = nn.LayerNorm(self.trans_dim)

        if self.task == "segmentation":
            self.head = SegmentationHead(self.trans_dim, self.seg_num_all)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.head = ClassificationHead(self.trans_dim, self.cls_num)

    def forward(self, pts):
        """
        Args:
            pts: (B, N, 3)
        Returns:
            If segmentation: (B, G, seg_num_all)
            If classification: (B, cls_num)
        """
        B, N, _ = pts.shape

        center, group_tokens = self.group_divider(pts)
        pos = self.pos_embed(center)

        x = group_tokens + pos
        x = self.transformer(x)

        importance_score = self.importance_predictor(x)

        idx_descend = torch.argsort(importance_score.squeeze(-1), dim=1, descending=True)
        idx_ascend = torch.argsort(importance_score.squeeze(-1), dim=1, descending=False)

        x_descend = torch.gather(x, 1, idx_descend.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x_ascend = torch.gather(x, 1, idx_ascend.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        x_descend = self.mamba(x_descend)
        x_ascend = self.mamba(x_ascend)

        x_fused = self.bi_fusion(x_descend, x_ascend)
        x = self.norm(x_fused)

        if self.task == "segmentation":
            x = x.permute(0, 2, 1)  # (B, D, G)
            seg_pred = self.head(x)  # (B, seg_num_all, G)
            return seg_pred.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)  # (B, D, G)
            x = self.global_pool(x).squeeze(-1)  # (B, D)
            cls_pred = self.head(x)
            return cls_pred
