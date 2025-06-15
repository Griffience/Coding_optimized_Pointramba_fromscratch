import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): 输入特征维度
            num_classes (int): 分割类别数 (ShapeNetPart总共有50类)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, G, D)
        Returns:
            seg_logits: (B, G, num_classes)
        """
        x = x.permute(0, 2, 1)    # (B, D, G)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)    # (B, G, num_classes)
        return x
