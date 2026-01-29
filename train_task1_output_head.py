import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import logging
import json
import numpy as np
import matplotlib.pyplot as plt

from task1_data import Cholec80DatasetTaskA
from output_head import FutureTimelineModel


NUM_PHASES = 7


# -------------------------------------------------
# Argument Parser
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)

    parser.add_argument("--data_root", type=str, default="data/cholec80")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

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

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# -------------------------------------------------
# Build GT timeline (NO constraint)
# -------------------------------------------------

def build_gt_future(cur_stage_idx, ratio_list, stage_order, all_time):

    """
    Build ground-truth cumulative future timeline

    Output:
        gt_future: [B,7]
    """

    B = cur_stage_idx.shape[0]

    gt_future = torch.zeros(
        (B, NUM_PHASES),
        device=cur_stage_idx.device
    )

    for b in range(B):

        cur = int(cur_stage_idx[b].item())

        if cur == 0:
            continue

        idx = cur - 1

        # current stage remaining
        remain = ratio_list[b, idx] * all_time[b, idx]

        acc = remain
        gt_future[b, idx] = acc

        # following stages
        for j in range(idx + 1, NUM_PHASES):

            if stage_order[b, j] == 0:
                continue

            acc = acc + all_time[b, j]
            gt_future[b, j] = acc

    return gt_future


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    args = parse_args()
    set_seed(args.seed)

    exp_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)
    save_config(args, os.path.join(exp_dir, "config.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    best_ckpt_path = os.path.join(exp_dir, "best.pth")

    logger.info(f"Experiment: {args.name}")
    logger.info(f"Device: {device}")
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
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # ---------------- Model ----------------

    model = FutureTimelineModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    criterion = nn.SmoothL1Loss()

    best_val = float("inf")

    train_curve = []
    val_curve = []

    # ==========================================================
    # Training Loop
    # ==========================================================

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:

            frames, stage_order, ratio_list, all_time = batch

            stage_order = stage_order.to(device)
            ratio_list = ratio_list.to(device)
            all_time = all_time.to(device)

            # ---------------- current stage ----------------

            mask = (ratio_list > 0)
            cur_stage_idx = mask.float().argmax(dim=1) + 1   # [B]

            ratio_input = ratio_list.gather(
                1, (cur_stage_idx - 1).unsqueeze(1)
            )

            # ---------------- forward ----------------

            raw_future = model(
                cur_stage_idx,
                ratio_input,
                stage_order,
                all_time
            )

            # ---------------- GT ----------------

            gt_future = build_gt_future(
                cur_stage_idx,
                ratio_list,
                stage_order,
                all_time
            )

            # ---------------- loss ----------------

            loss = criterion(raw_future, gt_future)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_train = running_loss / len(train_loader)
        train_curve.append(avg_train)

        logger.info(
            f"Epoch {epoch+1} Train Loss: {avg_train:.4f}"
        )

        # ---------------- Validation ----------------

        model.eval()
        val_running = 0.0

        with torch.no_grad():

            for batch in val_loader:

                frames, stage_order, ratio_list, all_time = batch

                stage_order = stage_order.to(device)
                ratio_list = ratio_list.to(device)
                all_time = all_time.to(device)

                mask = (ratio_list > 0)
                cur_stage_idx = mask.float().argmax(dim=1) + 1

                ratio_input = ratio_list.gather(
                    1, (cur_stage_idx - 1).unsqueeze(1)
                )

                raw_future = model(
                    cur_stage_idx,
                    ratio_input,
                    stage_order,
                    all_time
                )

                gt_future = build_gt_future(
                    cur_stage_idx,
                    ratio_list,
                    stage_order,
                    all_time
                )

                val_running += criterion(raw_future, gt_future).item()

        val_loss = val_running / len(val_loader)
        val_curve.append(val_loss)

        logger.info(
            f"Epoch {epoch+1} Val Loss: {val_loss:.4f}"
        )

        # ---------------- Save best ----------------

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_val,
                    "args": vars(args)
                },
                best_ckpt_path
            )

            logger.info("Saved new best checkpoint")

    # ==========================================================
    # Finished
    # ==========================================================

    logger.info(f"Training finished. Best Val Loss: {best_val:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(train_curve, label="Train")
    plt.plot(val_curve, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(exp_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Curve saved: {curve_path}")


if __name__ == "__main__":
    main()