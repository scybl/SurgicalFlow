import os
import cv2
import numpy as np
import torch
import pytest

from task1_data import Cholec80RemainingFramesDataset
# ↑ 改成你Dataset所在py文件名


# ---------------------------
# Helper: fake image writer
# ---------------------------

def write_dummy_image(path, h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


# ---------------------------
# Pytest fixture
# ---------------------------

@pytest.fixture
def fake_cholec80(tmp_path):
    """
    构造最小Cholec80结构：
    - phase: 从0开始，逐帧
    - frames: 从1开始编号，每25帧一张
    """

    root = tmp_path

    frames_dir = root / "frames" / "video01"
    phase_dir = root / "phase_annotations"

    frames_dir.mkdir(parents=True)
    phase_dir.mkdir(parents=True)

    # -------- create phase file (0-based, per-frame) --------
    # 模拟 100 帧:
    # frame 0-49 -> phase 0
    # frame 50-99 -> phase 1

    phase_file = phase_dir / "video01-phase.txt"

    with open(phase_file, "w") as f:
        f.write("Frame Phase\n")
        for i in range(100):
            if i < 50:
                f.write(f"{i} Preparation\n")
            else:
                f.write(f"{i} ClippingCutting\n")

    # -------- create sampled frames --------
    # 每25帧一张 → 100帧 => 4张
    #
    # image 1 -> frame 0
    # image 2 -> frame 25
    # image 3 -> frame 50
    # image 4 -> frame 75

    for i in range(1, 5):
        name = f"video01_{i:06d}.jpg"
        write_dummy_image(frames_dir / name)

    return root


# ---------------------------
# Test: dataset indexing
# ---------------------------

def test_phase_alignment(fake_cholec80):

    dataset = Cholec80RemainingFramesDataset(
        root_dir=fake_cholec80,
        seq_len=2,
        stride=1,
        sample_every=25
    )

    # 至少应产生1个sample
    assert len(dataset) > 0

    frames, remain_norm, remain_sec, phase_id = dataset[0]

    # -------- shape check --------

    assert isinstance(frames, torch.Tensor)
    assert frames.shape == (2, 3, 64, 64)

    # -------- phase alignment check --------
    #
    # clip最后一帧：
    # start=0, seq_len=2
    #
    # 对应抽帧index=1
    # 原始frame = 1*25 = 25
    #
    # frame25 < 50 → phase=Preparation → id=0

    assert phase_id.item() == 0


# ---------------------------
# Test: second window crosses phase boundary
# ---------------------------

def test_phase_transition(fake_cholec80):

    dataset = Cholec80RemainingFramesDataset(
        root_dir=fake_cholec80,
        seq_len=2,
        stride=1,
        sample_every=25
    )

    # 第二个窗口
    frames, _, _, phase_id = dataset[1]

    # window start=1:
    #
    # 抽帧index=2
    # 原始frame=50
    #
    # phase 应该变为 ClippingCutting → id=2

    assert phase_id.item() == 2
