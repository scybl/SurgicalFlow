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
from model import Task1CNN
import numpy as np
import matplotlib.pyplot as plt
import json

# from models.cnn_lstm import 


# -------------------------------------------------
# Argument Parser
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,default="data/cholec80")
    parser.add_argument("--name", type=str,required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])
    parser.add_argument("--epochs", type=int,required=True)

    parser.add_argument("--batch_size", type=int, default=8)
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
        return Task1CNN()

    elif model_name == "cnn_lstm":
        raise NotImplementedError("cnn_lstm not implemented yet")

    else:
        raise ValueError("Unknown model type")


# Validation function
def validate(model, loader, device):

    model.eval()
    mae_sum = 0.0
    count = 0

    with torch.no_grad():
        for frames, remain_sec, _ in tqdm(loader, desc="Valid", leave=False):

            frames = frames.to(device)
            remain_sec = remain_sec.to(device)

            pred_sec = model(frames)

            mae_sum += torch.abs(pred_sec - remain_sec).sum().item()
            count += remain_sec.size(0)

    return mae_sum / count   # MAE in seconds

# Main training
def main():

    args = parse_args()

    set_seed(args.seed)
    print("Random seed:", args.seed)

    # ---------------- Paths ----------------

    exp_dir = os.path.join("checkpoints", args.name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(exp_dir)
    config_path = os.path.join(exp_dir, "config.json")
    save_config(args, config_path)

    best_ckpt_path = os.path.join(exp_dir, "best.pth")

    # ---------------- Device ----------------

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Experiment: {args.name}")
    logger.info(f"Saving to: {exp_dir}")
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Model: {args.model}")

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
        mode="train",          # video 1–40
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    val_dataset = Cholec80RemainingFramesDataset(
        root_dir=args.data_root,
        mode="val",            # video 41–50
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )

    # -------------------------
    # DataLoader
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,          # 训练集可以 shuffle（sample-level）
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,        # 可选，训练时通常开
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,         # 验证集绝对不要 shuffle
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------------- Model ----------------

    model = build_model(args.model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    criterion = nn.SmoothL1Loss()

    best_mae = float("inf")
    train_loss_history = []
    val_mae_history = []


    # ---------------- Training Loop ----------------

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, (frames, remain_sec, _) in enumerate(pbar):

                frames = frames.to(device)
                remain_sec = remain_sec.to(device)

                # 🔍 只在第一个 epoch 的第一个 batch 打印一次
                if epoch == 0 and i == 0:
                    print("GT min:", remain_sec.min().item())
                    print("GT max:", remain_sec.max().item())
                    print("GT mean:", remain_sec.mean().item())

                pred = model(frames)          # seconds

                loss = criterion(pred, remain_sec)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix(loss=f"{loss.item():.2f}s")

        avg_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        logger.info(
            f"Epoch {epoch+1} Train SmoothL1 Loss (sec): {avg_loss:.2f}"
        )

        # ---------------- Validation ----------------

        val_mae = validate(model, val_loader, device)
        val_mae_history.append(val_mae)

        logger.info(f"Epoch {epoch+1} Validation MAE (sec): {val_mae:.2f}")

        # ---------------- Save Best ----------------

        if val_mae < best_mae:

            best_mae = val_mae

            torch.save({
                "epoch": epoch + 1,
                "model": "Task1CNNBaseline",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_mae": best_mae,
                "args": vars(args)
            }, best_ckpt_path)

            logger.info("Saved best checkpoint → %s", best_ckpt_path)
            logger.info("Best Validation MAE: %.2f", best_mae)


    logger.info("\nTraining finished.")
    logger.info("Best Validation MAE: %.2f", best_mae)

    # ---------------- Plot Loss Curve ----------------

    plt.figure(figsize=(8, 5))

    plt.plot(train_loss_history, label="Train SmoothL1 Loss (sec)")
    plt.plot(val_mae_history, label="Val MAE (sec)")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Task1 Training Curve")

    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(exp_dir, "training_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Training curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
