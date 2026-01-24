import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import logging

from task1_data import Cholec80RemainingFramesDataset
from models import TaskA_CNN, TaskA_CNN_LSTM


import numpy as np
import matplotlib.pyplot as plt
import json

# from models.cnn_lstm import 


# -------------------------------------------------
# Argument Parser
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str,required=True)
    parser.add_argument("--epochs", type=int,required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str,default="data/cholec80")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def save_config(args, save_path):
    with open(save_path, "w") as f:
        json.dump(vars(args), f, indent=4)

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir):

    log_file = os.path.join(log_dir, "train.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear duplicated handlers (important when re-running)
    if logger.hasHandlers():
        logger.handlers.clear()

    # -------- File Handler --------
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # -------- Console Handler --------
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def build_model(model_name):

    if model_name == "cnn":
        return TaskA_CNN()

    elif model_name == "cnn_lstm":
        return TaskA_CNN_LSTM()


# Validation function
def validate(model, loader, device):

    model.eval()

    mae_sum = 0.0
    count = 0

    all_gt = []
    all_pred = []

    with torch.no_grad():

        for batch in tqdm(loader, desc="Valid", leave=False):

            frames, remain_sec, *_ = batch

            frames = frames.to(device)
            remain_sec = remain_sec.to(device)

            # ---- unpack model output ----
            pred_remain, _, _, _ = model(frames)

            # ---- MAE ----
            mae_sum += torch.abs(pred_remain - remain_sec).sum().item()
            count += remain_sec.size(0)

            # ---- store for R2 ----
            all_gt.append(remain_sec.cpu())
            all_pred.append(pred_remain.cpu())

    gt = torch.cat(all_gt)
    pred = torch.cat(all_pred)

    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()

    r2 = 1 - ss_res / ss_tot

    return mae_sum / count, r2.item()

# Main training
def main():

    # ---------------- Parse args ----------------

    args = parse_args()
    set_seed(args.seed)

    print("Random seed:", args.seed)

    # ---------------- Experiment folder ----------------

    exp_dir = os.path.join("checkpoints", args.name)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)

    save_config(args, os.path.join(exp_dir, "config.json"))

    best_ckpt_path = os.path.join(exp_dir, "best.pth")

    # ---------------- Device ----------------

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Experiment: {args.name}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

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

    train_dataset = Cholec80RemainingFramesDataset(
        root_dir=args.data_root,
        mode="train",
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    val_dataset = Cholec80RemainingFramesDataset(
        root_dir=args.data_root,
        mode="val",
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    # ---------------- DataLoader ----------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # ---------------- Build Model ----------------

    model = build_model(args.model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # ---------- Loss functions ----------

    criterion_remain = nn.SmoothL1Loss()
    criterion_phase = nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=-1
    )

    # Loss weights (can report as hyper-parameters)
    lambda_remain = 1.0
    lambda_future = 0.5
    lambda_phase = 0.2

    best_mae = float("inf")

    train_loss_curve = []
    val_mae_curve = []
    val_r2_curve = []

    # ==========================================================
    #                      Training Loop
    # ==========================================================

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):

            (
                frames,
                remain_sec,
                future_start,
                future_end,
                future_phase,
                future_mask
            ) = batch

            # -------- Move to GPU --------

            frames = frames.to(device)
            remain_sec = remain_sec.to(device)

            future_start = future_start.to(device)
            future_end   = future_end.to(device)
            future_phase = future_phase.to(device)
            future_mask  = future_mask.to(device)

            # -------- Debug once --------

            if epoch == 0 and step == 0:
                logger.info(
                    f"Remain GT range: "
                    f"{remain_sec.min():.1f}s - {remain_sec.max():.1f}s"
                )

            # -------- Forward --------

            pred_remain, pred_fstart, pred_fend, pred_phase_logits = model(frames)

            # ==================================================
            #                  Loss computation
            # ==================================================

            # ---- Remaining time regression ----

            loss_remain = criterion_remain(
                pred_remain,
                remain_sec
            )

            # ---- Future timeline loss (masked L1) ----

            time_error = torch.abs(pred_fstart - future_start) \
                       + torch.abs(pred_fend - future_end)

            loss_future = (
                time_error * future_mask
            ).sum() / (future_mask.sum() + 1e-6)

            # ---- Phase classification loss (masked CE) ----

            B, N, K = pred_phase_logits.shape

            logits_flat = pred_phase_logits.view(B * N, K)
            phase_gt_flat = future_phase.view(B * N)
            mask_flat = future_mask.view(B * N)

            phase_loss_all = criterion_phase(
                logits_flat,
                phase_gt_flat
            )

            loss_phase = (
                phase_loss_all * mask_flat
            ).sum() / (mask_flat.sum() + 1e-6)

            # ---- Total loss ----

            total_loss = (
                lambda_remain * loss_remain +
                lambda_future * loss_future +
                lambda_phase * loss_phase
            )

            # ==================================================
            #                 Backprop
            # ==================================================

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            pbar.set_postfix(
                total=f"{total_loss.item():.3f}",
                remain=f"{loss_remain.item():.2f}",
                future=f"{loss_future.item():.2f}"
            )

        avg_train_loss = running_loss / len(train_loader)
        train_loss_curve.append(avg_train_loss)

        logger.info(
            f"Epoch {epoch+1} Train Loss: {avg_train_loss:.3f}"
        )

        # ==================================================
        #                    Validation
        # ==================================================

        val_mae, val_r2 = validate(model, val_loader, device)

        val_mae_curve.append(val_mae)
        val_r2_curve.append(val_r2)

        logger.info(
            f"Epoch {epoch+1} Val MAE: {val_mae:.2f}s | R2: {val_r2:.4f}"
        )

        # ==================================================
        #                  Save Best
        # ==================================================

        if val_mae < best_mae:

            best_mae = val_mae

            torch.save({
                "epoch": epoch + 1,
                "model": args.model,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mae": best_mae,
                "args": vars(args)
            }, best_ckpt_path)

            logger.info("Saved new best checkpoint")

    # ==========================================================
    #                  Training finished
    # ==========================================================

    logger.info("Training completed.")
    logger.info(f"Best Val MAE: {best_mae:.2f}s")

    # ---------------- Plot curves ----------------

    plt.figure(figsize=(8, 5))

    plt.plot(train_loss_curve, label="Train Loss")
    plt.plot(val_mae_curve, label="Val MAE")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(exp_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Curve saved to {curve_path}")


if __name__ == "__main__":
    main()
