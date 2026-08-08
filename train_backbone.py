import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import logging

from taskA_data_loader import Cholec80DatasetTaskA
from model_backbone import TaskA_CNN, TaskA_CNN_LSTM
from workflow_losses import (
    phase_class_weight_tensor,
    phase_group_loss,
    phase_order_loss,
)

import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json


# -------------------------------------------------
# Argument Parser
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)

    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase_loss_weight", type=float, default=1.0)
    parser.add_argument("--remain_loss_weight", type=float, default=0.2)
    parser.add_argument("--phase_group_loss_weight", type=float, default=0.25)
    parser.add_argument("--phase_order_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--disable_class_balance",
        action="store_true",
        help="Disable inverse-frequency phase class weights.",
    )

    return parser.parse_args()


# -------------------------------------------------
# Utils
# -------------------------------------------------

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir):

    log_file = os.path.join(log_dir, "train.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# -------------------------------------------------
# Build model
# -------------------------------------------------

def build_model(model_name):

    if model_name == "cnn":
        return TaskA_CNN()

    elif model_name == "cnn_lstm":
        return TaskA_CNN_LSTM()


# -------------------------------------------------
# Validation (ratio version + masked acc)
# -------------------------------------------------

@torch.no_grad()
def validate_phase_and_remain(model, loader, device):

    model.eval()

    total_correct = 0
    total_num = 0

    preds_ratio = []
    gts_ratio = []

    for frames, stage_order, ratio_list, all_time in loader:

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)

        # current stage index
        mask = (ratio_list > 0)
        cur_stage_idx = mask.float().argmax(dim=1)

        phase_gt = stage_order.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        phase_remain_gt = ratio_list.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        pred_phase_logits, pred_phase_remain = model(frames)

        pred_phase_remain = torch.clamp(pred_phase_remain, 0.0, 1.0)

        pred_phase = torch.argmax(pred_phase_logits, dim=1)

        valid = (phase_gt != 0)

        total_correct += ((pred_phase == phase_gt) & valid).sum().item()
        total_num += valid.sum().item()

        preds_ratio.append(pred_phase_remain.cpu().numpy())
        gts_ratio.append(phase_remain_gt.cpu().numpy())

    acc = total_correct / max(total_num, 1)

    preds_ratio = np.concatenate(preds_ratio)
    gts_ratio = np.concatenate(gts_ratio)

    mae_ratio = np.mean(np.abs(preds_ratio - gts_ratio))
    r2_ratio = r2_score(gts_ratio, preds_ratio)

    return acc, mae_ratio, r2_ratio


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    args = parse_args()
    set_seed(args.seed)

    print("Random seed:", args.seed)

    exp_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)
    save_config(args, os.path.join(exp_dir, "config.json"))

    best_ckpt_path = os.path.join(exp_dir, "best.pth")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # ---------------- Model ----------------

    model = build_model(args.model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # ---------------- Loss ----------------

    lambda_phase = args.phase_loss_weight
    lambda_remain = args.remain_loss_weight
    lambda_group = args.phase_group_loss_weight
    lambda_order = args.phase_order_loss_weight

    phase_weight = None
    if not args.disable_class_balance:
        phase_weight = phase_class_weight_tensor(train_dataset.samples, device)
        logger.info(f"Phase class weights: {phase_weight.detach().cpu().tolist()}")

    criterion_phase = torch.nn.CrossEntropyLoss(
        ignore_index=0,
        weight=phase_weight,
    )
    criterion_remain = torch.nn.SmoothL1Loss()

    best_mae = float("inf")

    train_loss_curve = []
    val_mae_curve = []

    # ==========================================================
    # Training Loop
    # ==========================================================

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0
        running_phase_acc = 0.0
        running_mae = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):

            frames, stage_order, ratio_list, all_time = batch

            frames = frames.to(device)
            stage_order = stage_order.to(device)
            ratio_list = ratio_list.to(device)

            # ---------------- current stage ----------------

            mask = (ratio_list > 0)
            cur_stage_idx = mask.float().argmax(dim=1)

            phase_gt = stage_order.gather(
                1, cur_stage_idx.unsqueeze(1)
            ).squeeze(1).long()

            phase_remain_gt = ratio_list.gather(
                1, cur_stage_idx.unsqueeze(1)
            ).squeeze(1)

            # ---------------- forward ----------------

            pred_phase_logits, pred_phase_remain = model(frames)

            pred_phase_remain = torch.clamp(pred_phase_remain, 0.0, 1.0)

            # ---------------- loss ----------------

            loss_phase = criterion_phase(pred_phase_logits, phase_gt)
            loss_group = phase_group_loss(pred_phase_logits, phase_gt)
            loss_order = phase_order_loss(pred_phase_logits, phase_gt)
            loss_remain = criterion_remain(pred_phase_remain, phase_remain_gt)

            total_loss = (
                lambda_phase * loss_phase +
                lambda_group * loss_group +
                lambda_order * loss_order +
                lambda_remain * loss_remain
            )

            # ---------------- backprop ----------------

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ---------------- metrics ----------------

            with torch.no_grad():

                pred_phase = torch.argmax(pred_phase_logits, dim=1)

                valid = (phase_gt != 0)

                phase_acc = ((pred_phase == phase_gt) & valid).float().sum() / valid.float().sum().clamp(min=1)

                mae = torch.abs(
                    pred_phase_remain - phase_remain_gt
                ).mean()

            running_loss += total_loss.item()
            running_phase_acc += phase_acc.item()
            running_mae += mae.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.3f}",
                acc=f"{phase_acc.item():.2f}",
                mae=f"{mae.item():.3f}",
                group=f"{loss_group.item():.3f}",
            )

        # ---------------- epoch summary ----------------

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_phase_acc / len(train_loader)
        avg_mae = running_mae / len(train_loader)

        train_loss_curve.append(avg_loss)

        logger.info(
            f"Epoch {epoch+1} "
            f"Train Loss: {avg_loss:.3f} | "
            f"Phase Acc: {avg_acc:.3f} | "
            f"Remain MAE (ratio): {avg_mae:.4f}"
        )

        # ---------------- validation ----------------

        val_acc, val_mae, val_r2 = validate_phase_and_remain(
            model,
            val_loader,
            device
        )

        val_mae_curve.append(val_mae)

        logger.info(
            f"Epoch {epoch+1} "
            f"Val Phase Acc: {val_acc:.3f} | "
            f"Remain MAE (ratio): {val_mae:.4f} | "
            f"R2: {val_r2:.4f}"
        )

        # ---------------- save best ----------------

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
    # Training finished
    # ==========================================================

    logger.info("Training completed.")
    logger.info(f"Best Val MAE (ratio): {best_mae:.4f}")

    # ---------------- plot curves ----------------

    plt.figure(figsize=(8, 5))

    plt.plot(train_loss_curve, label="Train Loss")
    plt.plot(val_mae_curve, label="Val MAE (ratio)")

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
