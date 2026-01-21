import sys
import pytest

import torch
import pytest
import torch
import torch.nn as nn

from models.cnn import Task1CNN

from task1_train import parse_args,build_model
from models.cnn import Task1CNN


def test_required_args_missing():

    test_args = ["train_task1.py"]

    sys.argv = test_args

    with pytest.raises(SystemExit):
        parse_args()


def test_single_train_step():

    device = torch.device("cpu")

    model = Task1CNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.SmoothL1Loss()

    # Simulate batch from dataset:
    # frames: [B, T, C, H, W]
    batch_size = 2
    seq_len = 16

    frames = torch.randn(batch_size, seq_len, 3, 224, 224)
    remain_norm = torch.rand(batch_size)

    # CNN baseline uses last frame
    x = frames[:, -1]

    pred = model(x)

    loss = criterion(pred, remain_norm)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -------- Assertions --------

    assert pred.shape == (batch_size,)
    assert torch.isfinite(loss)

def test_build_cnn_model():

    model = build_model("cnn")

    assert isinstance(model, Task1CNN)


def test_build_lstm_not_implemented():

    with pytest.raises(NotImplementedError):
        build_model("cnn_lstm")