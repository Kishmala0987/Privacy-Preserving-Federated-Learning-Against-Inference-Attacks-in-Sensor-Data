"""
model.py
========
A simple MLP for UCI-HAR.

UCI-HAR input is a 561-dimensional feature vector (not an image),
so we use a fully connected network, not a CNN.

Architecture:
  FC(561 → 256) → ReLU → Dropout
  FC(256 → 128) → ReLU → Dropout
  FC(128 → 6)   → output (6 activity classes)
"""

import torch
import torch.nn as nn


class HARModel(nn.Module):
    def __init__(self, input_dim=561, num_classes=6, dropout=0.3):
        super(HARModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)   # raw logits
        )

    def forward(self, x):
        return self.network(x)


def get_model(device="cpu"):
    model = HARModel()
    return model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)