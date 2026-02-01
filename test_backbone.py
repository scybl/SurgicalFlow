import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import logging
import json
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

from taskA_data_loader import Cholec80DatasetTaskA
from model_backbone import TaskA_CNN, TaskA_CNN_LSTM


# -------------------------------------------------
# Argument Parser (IDENTICAL to train)
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)      # unused, kept for consistency

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# -------------------------------------------------
# Utils (SAME AS TRAIN)
# -------------------------------------------------

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

    log_file = os.path.join(log_dir, "test.log")

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
# Validation (IDENTICAL TO TRAIN)
# -------------------------------------------------

@torch.no_grad()
def validate_phase_and_remain(model, loader, device):

    model.eval()

    total_correct = 0
    total_num = 0

    preds_time = []
    gts_time = []

    for frames, stage_order, ratio_list, all_time in tqdm(loader):

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)

        # ----------------------------
        # locate current phase index
        # ----------------------------
        mask = (ratio_list > 0)
        cur_stage_idx = mask.float().argmax(dim=1)

        # phase gt
        phase_gt = stage_order.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        # ratio gt
        phase_remain_gt = ratio_list.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        # total time of current phase
        phase_total_time = all_time.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        # ----------------------------
        # model prediction
        # ----------------------------
        pred_phase_logits, pred_phase_remain = model(frames)

        pred_phase_remain = torch.clamp(pred_phase_remain, 0.0, 1.0)

        pred_phase = torch.argmax(pred_phase_logits, dim=1)

        valid = (phase_gt != 0)

        total_correct += ((pred_phase == phase_gt) & valid).sum().item()
        total_num += valid.sum().item()

        # ----------------------------
        # ratio -> time (seconds)
        # ----------------------------
        gt_time = phase_remain_gt * phase_total_time
        pred_time = pred_phase_remain * phase_total_time

        preds_time.append(pred_time[valid].cpu().numpy())
        gts_time.append(gt_time[valid].cpu().numpy())

    acc = total_correct / max(total_num, 1)

    preds_time = np.concatenate(preds_time)
    gts_time = np.concatenate(gts_time)

    mae_time = np.mean(np.abs(preds_time - gts_time))
    r2_time = r2_score(gts_time, preds_time)

    return acc, mae_time, r2_time


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    args = parse_args()
    set_seed(args.seed)

    exp_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("========== TEST MODE ==========")
    logger.info(f"Experiment: {args.name}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")

    # ---------------- Transform (SAME AS TRAIN) ----------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------------- Dataset ----------------

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
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # ---------------- Load model ----------------

    model = build_model(args.model).to(device)

    ckpt_path = os.path.join(exp_dir, "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])

    logger.info(f"Loaded checkpoint from: {ckpt_path}")
    logger.info(f"Best Val MAE (ratio): {checkpoint['best_mae']:.4f}")

    # ---------------- Test ----------------

    test_acc, test_mae, test_r2 = validate_phase_and_remain(
        model,
        test_loader,
        device
    )

    logger.info("========== TEST RESULT ==========")
    logger.info(f"Phase Acc: {test_acc:.4f}")
    logger.info(f"Remain MAE (ratio): {test_mae:.4f}")
    logger.info(f"R2: {test_r2:.4f}")

    # ---------------- Save result ----------------

    result_dict = {
        "phase_acc": float(test_acc),
        "mae_ratio": float(test_mae),
        "r2": float(test_r2)
    }

    result_path = os.path.join(exp_dir, "test_result.json")

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    logger.info(f"Test results saved to: {result_path}")


if __name__ == "__main__":
    main()