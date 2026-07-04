import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from taskA_data_loader import Cholec80DatasetTaskA
from model_backbone import TaskA_CNN, TaskA_CNN_LSTM


# ============================================================
# ================ MODEL CONFIG AREA (EDIT HERE) ============
# ============================================================

MODELS = [

    {
        "name": "CNN Baseline",
        "exp": "backbone_cnn",
        "model": "cnn",
        "seq_len": 16,
        "stride": 8,
    },

    {
        "name": "CNN-LSTM (N=16)",
        "exp": "backbone_cnn_lstm_16",
        "model": "cnn_lstm",
        "seq_len": 16,
        "stride": 8,
    },

    {
        "name": "CNN-LSTM (N=32)",
        "exp": "backbone_cnn_lstm_32",
        "model": "cnn_lstm",
        "seq_len": 32,
        "stride": 16,
    },

]

# ============================================================
# ================= VISUAL WINDOW CONFIG ====================
# ============================================================

START_OFFSET = 300     # 跳过前面不稳定段（关键）
VIS_WINDOW = 720      # 可视化长度


# ============================================================
# ====================== ARG PARSER =========================
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/cholec80")
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="outputs/remaining_time_comparison.png")

    return parser.parse_args()


# ============================================================
# ======================= BUILD MODEL =======================
# ============================================================

def build_model(name):

    if name == "cnn":
        return TaskA_CNN()

    elif name == "cnn_lstm":
        return TaskA_CNN_LSTM()

    else:
        raise ValueError(name)


# ============================================================
# ====================== LOAD MODEL =========================
# ============================================================

def load_model(cfg, save_dir, device):

    model = build_model(cfg["model"]).to(device)

    ckpt_path = os.path.join(save_dir, cfg["exp"], "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Loaded: {cfg['name']}")

    return model


# ============================================================
# ============================ MAIN =========================
# ============================================================

@torch.no_grad()
def main():

    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # ---------------- Transform ----------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------------- Dataset ----------------

    max_seq_len = max([m["seq_len"] for m in MODELS])
    max_stride = max([m["stride"] for m in MODELS])

    print("Max seq_len:", max_seq_len)
    print("Max stride:", max_stride)

    test_dataset = Cholec80DatasetTaskA(
        root_dir=args.data_root,
        mode="test",
        seq_len=max_seq_len,
        stride=max_stride,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print("Total test samples:", len(test_dataset))

    # ---------------- Load Models ----------------

    models = []

    for cfg in MODELS:
        model = load_model(cfg, args.save_dir, device)
        models.append(model)

    # ---------------- Buffers ----------------

    gt_buffer = []
    pred_buffers = [[] for _ in MODELS]

    # ---------------- Forward full test set ----------------

    for frames, stage_order, ratio_list, all_time in tqdm(test_loader):

        frames = frames.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)

        # locate current phase
        mask = (ratio_list > 0)
        cur_stage_idx = mask.float().argmax(dim=1)

        gt_ratio = ratio_list.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        phase_total_time = all_time.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        gt_time = gt_ratio * phase_total_time

        gt_buffer.append(gt_time.cpu().numpy())

        # model predictions
        for i, model in enumerate(models):

            model_frames = frames[:, -MODELS[i]["seq_len"]:]

            _, pred_ratio = model(model_frames)

            pred_ratio = torch.clamp(pred_ratio, 0.0, 1.0)

            pred_time = pred_ratio * phase_total_time

            pred_buffers[i].append(pred_time.cpu().numpy())

    # ---------------- Concat ----------------

    gt_all = np.concatenate(gt_buffer) / 60.0

    pred_all = []
    for buf in pred_buffers:
        pred_all.append(np.concatenate(buf) / 60.0)

    total_len = len(gt_all)

    print("Total samples:", total_len)

    # =====================================================
    # ============== STABLE WINDOW SELECTION ==============
    # =====================================================

    end_idx = min(START_OFFSET + VIS_WINDOW, total_len)

    gt_all = gt_all[START_OFFSET:end_idx]

    for i in range(len(pred_all)):
        pred_all[i] = pred_all[i][START_OFFSET:end_idx]

    print("Window range:", START_OFFSET, "->", end_idx)

    # ---------------- Compute MAE (window) ----------------

    maes = []

    for i, cfg in enumerate(MODELS):
        mae = np.mean(np.abs(pred_all[i] - gt_all))
        maes.append(mae)

        print(cfg["name"], "MAE(min):", round(mae, 3))

    # ---------------- Plot ----------------

    x = np.arange(len(gt_all))

    plt.figure(figsize=(12, 4))

    # Ground Truth (solid)
    plt.plot(
        x,
        gt_all,
        linewidth=2.0,
        label="Ground Truth"
    )

    # Predictions
    for i, cfg in enumerate(MODELS):

        plt.plot(
            x,
            pred_all[i],
            linewidth=1.5,
            label=f"{cfg['name']} (MAE={maes[i]:.2f})"
        )

    plt.xlabel("Test Sample Sequence")
    plt.ylabel("Remaining Time (min)")
    plt.title("Remaining Time Prediction Comparison (Stable Window)")

    plt.ylim(0, 30)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_name = args.output
    output_dir = os.path.dirname(save_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(save_name, dpi=300)
    plt.show()

    print("Saved:", save_name)


if __name__ == "__main__":
    main()
