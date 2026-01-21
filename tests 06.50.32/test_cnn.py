import torch
from models.cnn import Task1CNN


def test_task1_cnn_forward():

    model = Task1CNN()

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    # Shape check
    assert y.shape == (2,)

    # Numerical sanity check
    assert torch.isfinite(y).all()
