import torch.nn as nn
import torch.nn.functional as F

from workflow_schema import NUM_PHASES


class TaskA_CNN(nn.Module):
    """
    Multi-task:
      - Phase classification
      - Phase remaining time regression

    Input:
        frames: [B, T, 3, H, W]

    Output:
        phase_logits: [B,num_phases+1] with class 0 reserved as ignore_index
        phase_remain: [B]
    """

    def __init__(self, num_phases=NUM_PHASES, dropout=0.3):
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

        # class 0 is kept as an ignored padding label for CrossEntropyLoss
        self.phase_head = nn.Linear(128, num_phases + 1)

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
        phase_logits = self.phase_head(shared)     # [B,num_phases+1]

        phase_remain = self.remain_head(shared)    # [B,1]
        phase_remain = F.relu(phase_remain)        # 防负时间

        return phase_logits, phase_remain.squeeze(1)


class TaskA_CNN_LSTM(nn.Module):
    """
    Multi-task:
      - Phase classification
      - Phase remaining time regression

    Input:
        frames: [B, T, 3, H, W]

    Output:
        phase_logits: [B,num_phases+1] with class 0 reserved as ignore_index
        phase_remain: [B]
    """

    def __init__(
        self,
        num_phases=NUM_PHASES,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.3
    ):
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

        # ---------------- Temporal Modeling ----------------

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.temporal_dropout = nn.Dropout(dropout)

        # ---------------- Shared Representation ----------------

        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # ---------------- Heads ----------------

        # phase classification
        self.phase_head = nn.Linear(128, num_phases + 1)

        # phase remaining regression
        self.remain_head = nn.Linear(128, 1)

    def forward(self, frames):

        B, T, C, H, W = frames.shape

        # ---------- CNN per frame ----------

        x = frames.view(B * T, C, H, W)

        feat = self.backbone(x)
        feat = self.gap(feat)                 # [B*T,256,1,1]
        feat = feat.view(B, T, 256)           # [B,T,256]

        # ---------- LSTM temporal modeling ----------

        lstm_out, _ = self.lstm(feat)         # [B,T,H]

        # use last timestep (causal, online friendly)
        temporal_feat = lstm_out[:, -1, :]    # [B,H]

        temporal_feat = self.temporal_dropout(temporal_feat)

        # ---------- Shared FC ----------

        shared = self.shared_fc(temporal_feat)   # [B,128]

        # ---------- Heads ----------

        phase_logits = self.phase_head(shared)   # [B,num_phases+1]

        phase_remain = self.remain_head(shared)  # [B,1]
        phase_remain = F.relu(phase_remain)      # avoid negative time

        return phase_logits, phase_remain.squeeze(1)
