import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): 输入特征维度
            num_classes (int): 分类类别数 (例如ModelNet40是40，ScanObjectNN是15/40/NN)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, D) 全局特征
        Returns:
            cls_logits: (B, num_classes)
        """
        x = self.mlp(x)
        return x
