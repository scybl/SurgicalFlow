import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import logging
from torch.utils.data import random_split

from task1_data import Cholec80RemainingFramesDataset
from models.cnn import Task1CNN
import numpy as np
import matplotlib.pyplot as plt
import json

# from models.cnn_lstm import 


# -------------------------------------------------
# Argument Parser
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,required=True)
    parser.add_argument("--name", type=str,required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])
    parser.add_argument("--epochs", type=int,required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=4)
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

        for frames, remain_norm, _, _ in tqdm(loader, desc="Valid", leave=True):

            frames = frames.to(device)
            remain_norm = remain_norm.to(device)

            x = frames[:, -1]

            pred_norm = model(x)

            mae_sum += torch.abs(pred_norm - remain_norm).sum().item()
            count += remain_norm.size(0)

    return mae_sum / count

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
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # ---------------- Dataset ----------------

    full_trainval_dataset = Cholec80RemainingFramesDataset(
        root_dir=args.data_root,
        mode="train",
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform,
    )




    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=(device.type == "cuda")
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=(device.type == "cuda")
    # )

    # # ---------------- Model ----------------

    # model = build_model(args.model).to(device)

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.lr
    # )

    # criterion = nn.SmoothL1Loss()

    # best_mae = float("inf")
    # train_loss_history = []
    # val_mae_history = []


    # # ---------------- Training Loop ----------------

    # for epoch in range(args.epochs):

    #     model.train()

    #     running_loss = 0.0

    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

    #     for frames, remain_norm, _, _ in pbar:

    #         frames = frames.to(device)
    #         remain_norm = remain_norm.to(device)

    #         # CNN baseline uses last frame
    #         x = frames[:, -1]

    #         pred = model(x)

    #         loss = criterion(pred, remain_norm)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         pbar.set_postfix(loss=f"{loss.item():.4f}")

    #     avg_loss = running_loss / len(train_loader)
    #     train_loss_history.append(avg_loss)

    #     logger.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

    #     # ---------------- Validation ----------------

    #     val_mae = validate(model, val_loader, device)
    #     val_mae_history.append(val_mae)

    #     logger.info(f"Epoch {epoch+1} Validation MAE (norm): {val_mae:.4f}")

    #     # ---------------- Save Best ----------------

    #     if val_mae < best_mae:

    #         best_mae = val_mae

    #         torch.save({
    #             "epoch": epoch + 1,
    #             "model": "Task1CNNBaseline",
    #             "model_state": model.state_dict(),
    #             "optimizer_state": optimizer.state_dict(),
    #             "best_mae": best_mae,
    #             "args": vars(args)
    #         }, best_ckpt_path)

    #         logger.info("Saved best checkpoint → %s", best_ckpt_path)
    #         logger.info("Best Validation MAE: %.2f", best_mae)


    # logger.info("\nTraining finished.")
    # logger.info("Best Validation MAE: %.2f", best_mae)

    # # ---------------- Plot Loss Curve ----------------

    # plt.figure(figsize=(8, 5))

    # plt.plot(train_loss_history, label="Train SmoothL1 Loss")
    # plt.plot(val_mae_history, label="Val MAE (sec)")

    # plt.xlabel("Epoch")
    # plt.ylabel("Value")
    # plt.title("Task1 Training Curve")

    # plt.legend()
    # plt.grid(True)

    # plot_path = os.path.join(exp_dir, "training_curve.png")
    # plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    # plt.close()

    # logger.info(f"Training curve saved to: {plot_path}")




if __name__ == "__main__":
    main()
