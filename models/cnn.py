import torch
import torch.nn as nn
import torch.nn.functional as F


class Task1CNN(nn.Module):
    """
    Temporal CNN baseline for Remaining Time Prediction (Task A)

    Input:
        x: [B, T, 3, H, W]
    Output:
        y: [B]  (remaining time in seconds)
    """

    def __init__(self):
        super().__init__()

        # ---------------- Feature extractor ----------------
        self.backbone = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/2

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/4

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/8

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/16
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- Regression head ----------------
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: [B, T, 3, H, W]
        """

        B, T, C, H, W = x.shape

        # ---- process all frames with shared CNN ----
        x = x.view(B * T, C, H, W)              # [B*T,3,H,W]
        feat = self.backbone(x)                 # [B*T,256,h,w]
        feat = self.gap(feat)                   # [B*T,256,1,1]
        feat = feat.view(B, T, -1)              # [B,T,256]

        # ---- temporal aggregation (baseline choice) ----
        feat = feat.mean(dim=1)                 # [B,256]

        # ---- regression ----
        out = self.regressor(feat)              # [B,1]

        return out.squeeze(1)