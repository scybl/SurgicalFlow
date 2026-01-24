import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json

from task1_data import Cholec80RemainingFramesDataset
from models import TaskA_CNN, TaskA_CNN_LSTM


# -------------------------------------------------
# Argument Parser (match your train.py style)
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)   # experiment folder name
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default="cuda")

    # plot controls
    parser.add_argument("--start", type=int, default=0)          # sample index start
    parser.add_argument("--length", type=int, default=1200)      # how many points to plot
    parser.add_argument("--to_min", action="store_true")         # plot in minutes
    parser.add_argument("--plot_name", type=str, default="gt_vs_pred_test.png")

    return parser.parse_args()


def build_model(model_name):
    if model_name == "cnn":
        return TaskA_CNN()
    elif model_name == "cnn_lstm":
        return TaskA_CNN_LSTM()
    else:
        raise ValueError("Unknown model type")


@torch.no_grad()
def run_test(model, loader, device, ignore_index=-1):
    model.eval()

    # ---------- accumulators ----------
    # Remaining
    gt_remain_all = []
    pred_remain_all = []

    # Future time (masked)
    sum_abs_start = 0.0
    sum_abs_end = 0.0
    sum_sq_start = 0.0
    sum_sq_end = 0.0
    count_future = 0.0  # number of valid future events across all samples

    # Future phase (masked)
    correct_phase = 0.0
    count_phase = 0.0
    ce_loss_sum = 0.0

    # CE for phase (ignore padding)
    ce_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    for batch in tqdm(loader, desc="Test", ncols=120):
        (
            frames,
            remain_sec,
            future_start,
            future_end,
            future_phase,
            future_mask
        ) = batch

        frames = frames.to(device, non_blocking=True)
        remain_sec = remain_sec.to(device, non_blocking=True)

        future_start = future_start.to(device, non_blocking=True)
        future_end   = future_end.to(device, non_blocking=True)
        future_phase = future_phase.to(device, non_blocking=True)
        future_mask  = future_mask.to(device, non_blocking=True)  # float 0/1

        # -------- forward --------
        pred_remain, pred_fstart, pred_fend, pred_phase_logits = model(frames)

        # =========================================================
        # 1) Remaining metrics
        # =========================================================
        gt_remain_all.append(remain_sec.detach().cpu().numpy())
        pred_remain_all.append(pred_remain.detach().cpu().numpy())

        # =========================================================
        # 2) Future timeline metrics (mask)
        # =========================================================
        # shapes: [B,N]
        abs_start = torch.abs(pred_fstart - future_start)
        abs_end   = torch.abs(pred_fend   - future_end)

        sq_start = (pred_fstart - future_start) ** 2
        sq_end   = (pred_fend   - future_end) ** 2

        # mask
        sum_abs_start += (abs_start * future_mask).sum().item()
        sum_abs_end   += (abs_end   * future_mask).sum().item()

        sum_sq_start  += (sq_start  * future_mask).sum().item()
        sum_sq_end    += (sq_end    * future_mask).sum().item()

        count_future  += future_mask.sum().item()

        # =========================================================
        # 3) Future phase metrics (mask)
        # =========================================================
        # logits: [B,N,K], gt: [B,N]
        B, N, K = pred_phase_logits.shape
        pred_phase = pred_phase_logits.argmax(dim=-1)  # [B,N]

        # accuracy (only valid events)
        correct_phase += ((pred_phase == future_phase).float() * future_mask).sum().item()
        count_phase   += future_mask.sum().item()

        # cross entropy (masked, ignore_index already handles -1 too)
        logits_flat = pred_phase_logits.view(B * N, K)
        gt_flat     = future_phase.view(B * N)
        mask_flat   = future_mask.view(B * N)

        ce_all = ce_fn(logits_flat, gt_flat)  # [B*N]
        ce_loss_sum += (ce_all * mask_flat).sum().item()

    # ---------- Remaining aggregate ----------
    gt = np.concatenate(gt_remain_all).astype(np.float64)
    pred = np.concatenate(pred_remain_all).astype(np.float64)

    err = pred - gt
    remain_mae = float(np.mean(np.abs(err)))
    remain_rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    remain_r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    # ---------- Future timeline aggregate ----------
    denom = max(count_future, 1e-6)
    future_start_mae = sum_abs_start / denom
    future_end_mae   = sum_abs_end   / denom

    future_start_rmse = float(np.sqrt(sum_sq_start / denom))
    future_end_rmse   = float(np.sqrt(sum_sq_end   / denom))

    # ---------- Future phase aggregate ----------
    denom_phase = max(count_phase, 1e-6)
    future_phase_acc = correct_phase / denom_phase
    future_phase_ce  = ce_loss_sum / denom_phase  # average CE over valid events

    metrics = {
        # Remaining
        "remain_mae_sec": remain_mae,
        "remain_rmse_sec": remain_rmse,
        "remain_r2": remain_r2,

        # Future timeline (per valid future event)
        "future_start_mae_sec": future_start_mae,
        "future_end_mae_sec": future_end_mae,
        "future_start_rmse_sec": future_start_rmse,
        "future_end_rmse_sec": future_end_rmse,

        # Future phase (per valid future event)
        "future_phase_acc": future_phase_acc,
        "future_phase_ce": future_phase_ce,

        # Counts
        "n_samples": int(len(gt)),
        "n_valid_future_events": float(count_future),
    }

    return gt, pred, metrics

def plot_curve(gt, pred, out_path, start=0, length=1200, to_min=False, title="GT vs Prediction"):
    s = max(0, start)
    e = min(len(gt), s + length)

    gt_seg = gt[s:e]
    pred_seg = pred[s:e]

    mae_sec = float(np.mean(np.abs(pred_seg - gt_seg)))
    mae_min = mae_sec / 60.0

    if to_min:
        gt_seg = gt_seg / 60.0
        pred_seg = pred_seg / 60.0
        ylab = "Remaining Time (min)"
    else:
        ylab = "Remaining Time (sec)"

    x = np.arange(s, e)

    plt.figure(figsize=(12, 5))
    plt.plot(x, gt_seg, label="Ground Truth", linewidth=2)
    plt.plot(x, pred_seg, label="Prediction", linewidth=1, alpha=0.85)
    plt.title(f"{title} (Segment MAE: {mae_min:.2f} min)")
    plt.xlabel("Frame Sequence (sample index)")
    plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    # ---------------- Paths ----------------
    exp_dir = os.path.join(args.save_dir, args.name)
    best_ckpt_path = os.path.join(exp_dir, "best.pth")

    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {best_ckpt_path}")

    # ---------------- Device ----------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Experiment:", args.name)
    print("Checkpoint:", best_ckpt_path)

    # ---------------- Transform (same as train) ----------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------------- Dataset (TEST split) ----------------
    test_dataset = Cholec80RemainingFramesDataset(
        root_dir=args.data_root,
        mode="test",                 # video 51–80
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

    # ---------------- Model ----------------
    model = build_model(args.model).to(device)

    ckpt = torch.load(best_ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    # ---------------- Run Test ----------------
    gt, pred, metrics = run_test(model, test_loader, device)
    print("\n=== Test Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    results = {
        "mae_sec": mae,
        "rmse_sec": rmse,
        "r2": r2,
        "mae_min": mae / 60.0,
        "rmse_min": rmse / 60.0,
        "n_samples": int(len(gt)),
    }

    print("\n=== Test Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # save metrics
    metrics_path = os.path.join(exp_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved metrics:", metrics_path)

    metrics_path = os.path.join(exp_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # plot
    plot_path = os.path.join(exp_dir, args.plot_name)
    plot_curve(
        gt, pred,
        out_path=plot_path,
        start=args.start,
        length=args.length,
        to_min=args.to_min,
        title=f"Test: {args.name}"
    )
    print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()