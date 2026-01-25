import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import logging

from task1_data import Cholec80DatasetTaskA
from task1_model import TaskA_CNN, TaskA_CNN_LSTM
import numpy as np
from sklearn.metrics import r2_score

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

def validate_phase_and_remain(model, loader, device):

    model.eval()

    total_correct = 0
    total_num = 0

    preds_time = []
    gts_time = []

    with torch.no_grad():

        for frames, stage_order, time_list in loader:

            frames = frames.to(device)
            stage_order = stage_order.to(device)
            time_list = time_list.to(device)

            cur_stage_idx = (time_list > 0).float().argmax(dim=1)

            phase_gt = stage_order.gather(
                1,
                cur_stage_idx.unsqueeze(1)
            ).squeeze(1)

            phase_remain_gt = time_list.gather(
                1,
                cur_stage_idx.unsqueeze(1)
            ).squeeze(1)

            pred_phase_logits, pred_phase_remain = model(frames)

            pred_phase = torch.argmax(pred_phase_logits, dim=1)

            total_correct += (pred_phase == phase_gt).sum().item()
            total_num += phase_gt.size(0)

            preds_time.append(pred_phase_remain.cpu().numpy())
            gts_time.append(phase_remain_gt.cpu().numpy())

    acc = total_correct / total_num

    preds_time = np.concatenate(preds_time)
    gts_time = np.concatenate(gts_time)

    mae = np.mean(np.abs(preds_time - gts_time))
    r2 = r2_score(gts_time, preds_time)

    return acc, mae, r2

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

    train_dataset = Cholec80DatasetTaskA(
        root_dir=args.data_root,
        mode="train",
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    val_dataset = Cholec80DatasetTaskA(
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

    lambda_phase = 1.0
    lambda_remain = 0.5

    # ---------- Loss functions ----------
    criterion_phase = torch.nn.CrossEntropyLoss()
    criterion_remain = torch.nn.SmoothL1Loss()

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
        running_phase_acc = 0.0
        running_mae = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):

            # ==================================================
            #                Load batch
            # ==================================================

            frames, stage_order, time_list = batch

            frames = frames.to(device)
            stage_order = stage_order.to(device)
            time_list = time_list.to(device)

            # ==================================================
            #       Extract GT phase + phase remaining
            # ==================================================

            # 当前阶段 index
            cur_stage_idx = (time_list > 0).float().argmax(dim=1)

            # 当前阶段 GT label
            phase_gt = stage_order.gather(
                1,
                cur_stage_idx.unsqueeze(1)
            ).squeeze(1).long()

            # 当前阶段剩余时间 GT
            phase_remain_gt = time_list.gather(
                1,
                cur_stage_idx.unsqueeze(1)
            ).squeeze(1)

            # ==================================================
            #                 Forward
            # ==================================================

            pred_phase_logits, pred_phase_remain = model(frames)

            # ==================================================
            #                  Loss
            # ==================================================

            loss_phase = criterion_phase(pred_phase_logits, phase_gt)
            loss_remain = criterion_remain(pred_phase_remain, phase_remain_gt)

            total_loss = (
                lambda_phase * loss_phase +
                lambda_remain * loss_remain
            )

            # ==================================================
            #                Backprop
            # ==================================================

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ==================================================
            #                 Metrics
            # ==================================================

            with torch.no_grad():

                pred_phase = torch.argmax(pred_phase_logits, dim=1)

                phase_acc = (pred_phase == phase_gt).float().mean()

                mae = torch.abs(
                    pred_phase_remain - phase_remain_gt
                ).mean()

            running_loss += total_loss.item()
            running_phase_acc += phase_acc.item()
            running_mae += mae.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.3f}",
                acc=f"{phase_acc.item():.2f}",
                mae=f"{mae.item():.1f}s"
            )

        # ==================================================
        #              Epoch summary
        # ==================================================

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_phase_acc / len(train_loader)
        avg_mae = running_mae / len(train_loader)

        train_loss_curve.append(avg_loss)

        logger.info(
            f"Epoch {epoch+1} "
            f"Train Loss: {avg_loss:.3f} | "
            f"Phase Acc: {avg_acc:.3f} | "
            f"Remain MAE: {avg_mae:.2f}s"
        )

        # ==================================================
        #                Validation
        # ==================================================

        val_acc, val_mae, val_r2 = validate_phase_and_remain(
            model,
            val_loader,
            device
        )

        logger.info(
            f"Epoch {epoch+1} "
            f"Val Phase Acc: {val_acc:.3f} | "
            f"Remain MAE: {val_mae:.2f}s | "
            f"R2: {val_r2:.4f}"
        )

        # ==================================================
        #                Save Best
        # ==================================================

        if val_mae < best_mae:

            best_mae = val_mae

            torch.save({
                "epoch": epoch + 1,
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
