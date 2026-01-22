import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from task1_data import Cholec80RemainingFramesDataset
from models import Task1CNN, Task1CNNLSTM

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    gt_all = []
    pred_all = []
    phase_all = []

    for frames, remain_sec, phase_id in tqdm(loader, desc="Testing", ncols=120):
        frames = frames.to(device, non_blocking=True)
        remain_sec = remain_sec.to(device, non_blocking=True)

        pred_sec = model(frames)

        gt_all.append(remain_sec.detach().cpu().numpy())
        pred_all.append(pred_sec.detach().cpu().numpy())
        phase_all.append(phase_id.detach().cpu().numpy())

    gt = np.concatenate(gt_all, axis=0).astype(np.float64)
    pred = np.concatenate(pred_all, axis=0).astype(np.float64)
    phase = np.concatenate(phase_all, axis=0)

    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # R^2 = 1 - SS_res/SS_tot
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    return gt, pred, phase, mae, rmse, r2


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--exp", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None,
                   help="optional: path to checkpoint .pth (overrides --exp)")
    p.add_argument("--data_root", type=str, default="data/cholec80")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="checkpoints")

    # Plot options
    p.add_argument("--plot_n", type=int, default=400,
                   help="how many samples to plot (in order)")
    return p.parse_args()


def build_model(config):
    # 这里按你训练脚本一致：cnn baseline
    if config.model == "cnn_lstm":
        return Task1CNNLSTM()
    elif config.model == "cnn":
        return Task1CNN()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    gt_all = []
    pred_all = []
    phase_all = []

    for frames, remain_sec, phase_id in tqdm(loader, desc="Testing", ncols=120):
        frames = frames.to(device, non_blocking=True)
        remain_sec = remain_sec.to(device, non_blocking=True)

        pred_sec = model(frames)

        gt_all.append(remain_sec.detach().cpu().numpy())
        pred_all.append(pred_sec.detach().cpu().numpy())
        phase_all.append(phase_id.detach().cpu().numpy())

    gt = np.concatenate(gt_all, axis=0)
    pred = np.concatenate(pred_all, axis=0)
    phase = np.concatenate(phase_all, axis=0)

    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    return gt, pred, phase, mae, rmse


def save_metrics(out_dir, mae, rmse, r2):
    metrics = {
        "mae_sec": mae,
        "rmse_sec": rmse,
        "r2": r2,
        "mae_min": mae / 60.0,
        "rmse_min": rmse / 60.0,
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics


def plot_curve(gt, pred, out_dir, n=400):
    n = min(n, len(gt))
    x = np.arange(n)

    plt.figure(figsize=(12, 4))
    plt.plot(x, gt[:n], label="Ground Truth", linewidth=2)
    plt.plot(x, pred[:n], label="Prediction", linewidth=1, alpha=0.85)
    plt.xlabel("Sample index (test set order)")
    plt.ylabel("Remaining time (sec)")
    plt.title("Task1: Remaining Phase Time (GT vs Pred)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "gt_vs_pred_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    args = parse_args()

    exp_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_path = args.ckpt if args.ckpt is not None else os.path.join(exp_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", ckpt_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = Cholec80RemainingFramesDataset(
        root_dir=args.data_root,
        mode="test",           # video 51–80
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容你保存的字典格式
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("Loaded model_state from checkpoint dict.")
    else:
        model.load_state_dict(ckpt, strict=True)
        print("Loaded raw state_dict checkpoint.")

    gt, pred, phase, mae, rmse, r2 = evaluate(model, test_loader, device)
    metrics = save_metrics(exp_dir, mae, rmse, r2)

    metrics = save_metrics(exp_dir, mae, rmse)
    print("Test metrics:", metrics)

    # 保存所有预测，便于复现/写报告
    np.savez(
        os.path.join(exp_dir, "test_predictions.npz"),
        gt_sec=gt,
        pred_sec=pred,
        phase_id=phase,
    )
    print("Saved predictions to:", os.path.join(exp_dir, "test_predictions.npz"))

    fig_path = plot_curve(gt, pred, exp_dir, n=args.plot_n)
    print("Saved curve to:", fig_path)


if __name__ == "__main__":
    main()
