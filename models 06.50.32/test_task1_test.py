import torch
import numpy as np
import cv2
import pytest

from task1_data import Cholec80RemainingFramesDataset
from task1_train import build_model, validate


# ----------------------------
# Helper: fake image writer
# ----------------------------

def write_dummy_image(path, h=64, w=64):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ----------------------------
# Fake dataset fixture
# ----------------------------

@pytest.fixture
def fake_test_dataset(tmp_path):
    """
    创建一个最小可推理测试集:
      - frames: video01_000001.jpg ...
      - phase: 从0开始
      - 抽帧25
    """

    root = tmp_path

    frames_dir = root / "frames" / "video01"
    phase_dir = root / "phase_annotations"

    frames_dir.mkdir(parents=True)
    phase_dir.mkdir(parents=True)

    # ---------- phase ----------
    phase_file = phase_dir / "video01-phase.txt"

    with open(phase_file, "w") as f:
        f.write("Frame Phase\n")
        for i in range(150):
            if i < 75:
                f.write(f"{i} Preparation\n")
            else:
                f.write(f"{i} ClippingCutting\n")

    # ---------- frames ----------
    # 每25帧抽1张 -> 6 张
    for i in range(1, 7):
        fname = f"video01_{i:06d}.jpg"
        write_dummy_image(frames_dir / fname)

    dataset = Cholec80RemainingFramesDataset(
        root_dir=root,
        seq_len=4,
        stride=2,
        transform=None,
        sample_every=25
    )

    return dataset


# ----------------------------
# Fake checkpoint fixture
# ----------------------------

@pytest.fixture
def fake_checkpoint(tmp_path):
    """
    构造一个可加载的checkpoint
    """

    model = build_model("cnn")

    ckpt_path = tmp_path / "best.pth"

    torch.save({
        "epoch": 1,
        "model": "Task1CNNBaseline",
        "model_state": model.state_dict(),
        "best_mae": 0.1
    }, ckpt_path)

    return ckpt_path


# ----------------------------
# Test: checkpoint loading
# ----------------------------

def test_checkpoint_load(fake_checkpoint):

    ckpt = torch.load(fake_checkpoint, map_location="cpu")

    assert "model_state" in ckpt
    assert "epoch" in ckpt
    assert "best_mae" in ckpt


# ----------------------------
# Test: inference pipeline
# ----------------------------

def test_inference_pipeline(fake_test_dataset, fake_checkpoint):

    device = "cpu"

    loader = torch.utils.data.DataLoader(
        fake_test_dataset,
        batch_size=2,
        shuffle=False
    )

    # -------- load model --------

    model = build_model("cnn")

    ckpt = torch.load(fake_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    model.to(device)
    model.eval()

    # -------- run validation(test) --------

    mae = validate(model, loader, device)

    # -------- assertions --------

    assert isinstance(mae, float)
    assert mae >= 0.0
