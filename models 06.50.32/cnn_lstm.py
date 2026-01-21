import torch
import torch.nn as nn


class Task1CNNLSTM(nn.Module):
    """
    CNN + LSTM model for Remaining Time Prediction (Task 1)

    Input:
        x: [B, T, 3, H, W]
    Output:
        y: [B]  (normalized remaining time)
    """

    def __init__(
        self,
        lstm_hidden_size=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.3
    ):
        super().__init__()

        # ---------------- CNN Feature Extractor ----------------

        self.backbone = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        cnn_feature_dim = 256

        # ---------------- LSTM ----------------

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden_size
        if bidirectional:
            lstm_out_dim *= 2

        # ---------------- Regression Head ----------------

        self.regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: [B, T, 3, H, W]
        """

        B, T, C, H, W = x.shape

        # Merge batch and time for CNN
        x = x.view(B * T, C, H, W)

        feat = self.backbone(x)
        feat = self.gap(feat)            # [B*T,256,1,1]
        feat = feat.view(B, T, -1)       # [B,T,256]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(feat)    # [B,T,H]

        # Use last timestep
        last_feat = lstm_out[:, -1, :]   # [B,H]

        out = self.regressor(last_feat) # [B,1]

        return out.squeeze(1)
