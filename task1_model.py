import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision.models import resnet34


class TaskA_CNN(nn.Module):
    """
    Multi-task:
      - Phase classification
      - Phase remaining time regression

    Input:
        frames: [B, T, 3, H, W]

    Output:
        phase_logits: [B,7]
        phase_remain: [B]
    """

    def __init__(self, num_phases=7, dropout=0.3):
        super().__init__()

        # ---------------- CNN Backbone ----------------

        self.backbone = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- Temporal Aggregation ----------------

        self.temporal_dropout = nn.Dropout(dropout)

        # ---------------- Shared Representation ----------------

        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # ---------------- Heads ----------------

        # phase classification
        self.phase_head = nn.Linear(128, num_phases)

        # phase remaining regression
        self.remain_head = nn.Linear(128, 1)

    def forward(self, frames):

        B, T, C, H, W = frames.shape

        # CNN per frame
        x = frames.view(B * T, C, H, W)

        feat = self.backbone(x)
        feat = self.gap(feat)                 # [B*T,256,1,1]
        feat = feat.view(B, T, 256)           # [B,T,256]

        # temporal pooling
        feat = feat.mean(dim=1)               # [B,256]
        feat = self.temporal_dropout(feat)

        shared = self.shared_fc(feat)         # [B,128]

        # heads
        phase_logits = self.phase_head(shared)     # [B,7]

        phase_remain = self.remain_head(shared)    # [B,1]
        phase_remain = F.relu(phase_remain)        # 防负时间

        return phase_logits, phase_remain.squeeze(1)

class TaskA_CNN_LSTM(nn.Module):
    """
    MPHY0043 Task A compliant model

    Input:
        frames: [B, T, 3, H, W]

    Output:
        remaining_time:      [B]
        future_start_times:  [B, N]
        future_end_times:    [B, N]
        future_phase_logits: [B, N, K]
    """

    def __init__(
        self,
        num_phase_types=7,
        max_future_events=10,
        lstm_hidden=256,
        dropout=0.3
    ):
        super().__init__()

        self.num_phase_types = num_phase_types
        self.max_future_events = max_future_events

        # ---------------- CNN Backbone ----------------

        self.backbone = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- Temporal Model ----------------

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # ---------------- Shared FC ----------------

        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # ---------------- Output Heads ----------------

        # Remaining time regression
        self.remaining_head = nn.Linear(128, 1)

        # Future phase event timeline regression
        self.future_start_head = nn.Linear(128, max_future_events)
        self.future_end_head   = nn.Linear(128, max_future_events)

        # Future phase classification
        self.future_phase_head = nn.Linear(
            128, max_future_events * num_phase_types
        )

    def forward(self, frames):
        """
        frames: [B, T, 3, H, W]
        """

        B, T, C, H, W = frames.shape

        # -------- CNN feature extraction per frame --------

        x = frames.view(B * T, C, H, W)

        feat = self.backbone(x)
        feat = self.gap(feat)                 # [B*T, 256, 1, 1]
        feat = feat.view(B, T, -1)            # [B, T, 256]

        # -------- Temporal modeling --------

        lstm_out, _ = self.lstm(feat)

        # Use last time step feature
        temporal_feat = lstm_out[:, -1, :]    # [B, H]

        # -------- Shared representation --------

        shared = self.shared_fc(temporal_feat)  # [B, 128]

        # -------- Outputs --------

        # Remaining phase time
        remaining_time = self.remaining_head(shared).squeeze(1)

        # Future phase timeline
        future_start = self.future_start_head(shared)
        future_end   = self.future_end_head(shared)

        # Future phase classification
        phase_logits = self.future_phase_head(shared)
        phase_logits = phase_logits.view(
            B, self.max_future_events, self.num_phase_types
        )

        return remaining_time, future_start, future_end, phase_logits
