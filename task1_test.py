import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import json

from task1_data import Cholec80DatasetTaskA
from task1_model import TaskA_CNN, TaskA_CNN_LSTM

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()



def build_model(model_name):

    if model_name == "cnn":
        return TaskA_CNN()

    elif model_name == "cnn_lstm":
        return TaskA_CNN_LSTM()



@torch.no_grad()
def run_test(model, loader, device):

    model.eval()

    gt_all = []
    pred_all = []

    total_correct = 0
    total_num = 0

    for frames, stage_order, time_list in tqdm(loader, desc="Test"):

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        time_list = time_list.to(device)

        # ---------- 当前阶段 ----------
        cur_stage_idx = (time_list > 0).float().argmax(dim=1)

        phase_gt = stage_order.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        phase_remain_gt = time_list.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        # ---------- Forward ----------
        pred_phase_logits, pred_phase_remain = model(frames)

        # ---------- Phase acc ----------
        pred_phase = torch.argmax(pred_phase_logits, dim=1)

        total_correct += (pred_phase == phase_gt).sum().item()
        total_num += phase_gt.size(0)

        # ---------- Collect ----------
        gt_all.append(phase_remain_gt.cpu().numpy())
        pred_all.append(pred_phase_remain.cpu().numpy())

    gt = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)

    err = pred - gt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))

    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    phase_acc = total_correct / total_num

    metrics = {
        "phase_acc": phase_acc,
        "remain_mae_sec": mae,
        "remain_rmse_sec": rmse,
        "remain_r2": r2,
        "remain_mae_min": mae / 60,
        "remain_rmse_min": rmse / 60,
        "n_samples": int(len(gt))
    }

    return gt, pred, metrics


# ------------------------
# Main
# ------------------------

def main():

    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    # ---------- Transform (必须和 train 一致) ----------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------- Dataset ----------

    test_dataset = Cholec80DatasetTaskA(
        root_dir=args.data_root,
        mode="test",
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print("Test samples:", len(test_dataset))

    # ---------- Load model ----------

    model = build_model(args.model).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    print("Loaded checkpoint:", args.ckpt)

    # ---------- Run test ----------

    gt, pred, metrics = run_test(model, test_loader, device)


    print("\n====== Test Results ======")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ---------- Save ----------

    save_path = os.path.join(
        os.path.dirname(args.ckpt),
        "test_metrics.json"
    )

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics to:", save_path)

    save_dir = os.path.dirname(args.ckpt)

    # ---------------- Scatter plot ----------------

    plt.figure(figsize=(6, 6))

    plt.scatter(gt, pred, s=8, alpha=0.5)

    max_val = max(gt.max(), pred.max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")  # y=x reference

    plt.xlabel("GT Remaining Time (sec)")
    plt.ylabel("Predicted Remaining Time (sec)")
    plt.title(f"Remaining Time Prediction\nR2={metrics['remain_r2']:.3f}")

    plt.grid(True)

    scatter_path = os.path.join(save_dir, "test_scatter.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved scatter plot:", scatter_path)


    # ---------------- Error histogram ----------------

    error = pred - gt

    plt.figure(figsize=(7, 4))

    plt.hist(error, bins=60)

    plt.xlabel("Prediction Error (sec)")
    plt.ylabel("Count")
    plt.title("Remaining Time Error Distribution")

    plt.grid(True)

    hist_path = os.path.join(save_dir, "test_error_hist.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved error histogram:", hist_path)


if __name__ == "__main__":
    main()
