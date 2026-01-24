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
from task1_model import Res34, TaskA_CNN, TaskA_CNN_LSTM
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



def validate_phase_remain(model, loader, device):

    model.eval()

    preds = []
    gts = []

    with torch.no_grad():

        for frames, stage_order, time_list in loader:

            frames = frames.to(device)
            time_list = time_list.to(device)

            cur_stage_idx = (time_list > 0).float().argmax(dim=1)

            phase_remain_sec = time_list.gather(
                1,
                cur_stage_idx.unsqueeze(1)
            ).squeeze(1)

            pred = model(frames)

            preds.append(pred.cpu().numpy())
            gts.append(phase_remain_sec.cpu().numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    mae = np.mean(np.abs(preds - gts))
    r2 = r2_score(gts, preds)

    return mae, r2


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

    # ---------- Loss functions ----------
    criterion = nn.SmoothL1Loss()


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

            # ==================================================
            #                Load batch
            # ==================================================

            frames, stage_order, time_list = batch

            frames = frames.to(device)
            time_list = time_list.to(device)

            # ==================================================
            #      Extract current phase remaining time
            # ==================================================
            # time_list: [B, 7]
            # 当前阶段 = 第一个 remaining > 0 的阶段

            cur_stage_idx = (time_list > 0).float().argmax(dim=1)

            phase_remain_sec = time_list.gather(
                1,
                cur_stage_idx.unsqueeze(1)
            ).squeeze(1)   # [B]

            # ==================================================
            #               Debug (once)
            # ==================================================

            if epoch == 0 and step == 0:
                logger.info(
                    f"Phase Remain GT range: "
                    f"{phase_remain_sec.min():.1f}s - "
                    f"{phase_remain_sec.max():.1f}s"
                )

            # ==================================================
            #                 Forward
            # ==================================================

            pred_phase_remain = model(frames)   # [B]

            # ==================================================
            #                  Loss
            # ==================================================

            loss = criterion(pred_phase_remain, phase_remain_sec)

            # ==================================================
            #                Backprop
            # ==================================================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                gt_mean=f"{phase_remain_sec.mean():.1f}s"
            )

        avg_train_loss = running_loss / len(train_loader)
        train_loss_curve.append(avg_train_loss)

        logger.info(
            f"Epoch {epoch+1} Train Loss: {avg_train_loss:.3f}"
        )

        # ==================================================
        #                   Validation
        # ==================================================

        val_mae, val_r2 = validate_phase_remain(
            model,
            val_loader,
            device
        )

        val_mae_curve.append(val_mae)
        val_r2_curve.append(val_r2)

        logger.info(
            f"Epoch {epoch+1} Val MAE: {val_mae:.2f}s | R2: {val_r2:.4f}"
        )

        # ==================================================
        #                 Save Best
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

# Notice
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
