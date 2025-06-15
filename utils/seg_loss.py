# utils/seg_loss.py

import torch
import torch.nn as nn

class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred: (B*N, num_classes) logits
        target: (B*N) labels
        """
        loss = self.criterion(pred, target)
        return loss
