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
from model_out_head import FutureTimelineModel
import matplotlib.pyplot as plt

# -------------------------------------------------
# Argument Parser (IDENTICAL to train)
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    # ---------------- experiment names ----------------

    parser.add_argument("--backbone_name", type=str, required=True)
    parser.add_argument("--head_name", type=str, required=True)

    # ---------------- backbone config (MATCH TaskA) ----------------

    parser.add_argument("--backbone_model", type=str, required=True,
                        choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    # ---------------- training config ----------------

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--num_workers", type=int, default=8)

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

def build_gt_future(cur_stage_idx, ratio_list, stage_order, all_time):

    B = cur_stage_idx.shape[0]

    gt_future = torch.zeros((B, 7), device=cur_stage_idx.device)

    for b in range(B):

        cur = int(cur_stage_idx[b].item())

        if cur == 0:
            continue

        idx = cur - 1

        remain = ratio_list[b, idx] * all_time[b, idx]

        acc = remain
        gt_future[b, idx] = acc

        for j in range(idx + 1, 7):

            if stage_order[b, j] == 0:
                continue

            acc = acc + all_time[b, j]
            gt_future[b, j] = acc

    return gt_future

@torch.no_grad()
def validate_pipeline(backbone, head, loader, device):

    backbone.eval()
    head.eval()

    total_correct = 0
    total_num = 0

    preds_future_all = []
    gts_future_all = []

    for frames, stage_order, ratio_list, all_time in tqdm(loader):

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)

        # ======================================================
        # Locate current stage (GT)
        # ======================================================

        mask = (ratio_list > 0)
        cur_stage_idx = mask.float().argmax(dim=1)   # [B] (0-based)

        phase_gt = stage_order.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)   # [B]

        # ======================================================
        # Backbone forward
        # ======================================================

        pred_phase_logits, pred_ratio = backbone(frames)

        pred_ratio = torch.clamp(pred_ratio, 0.0, 1.0)
        pred_phase = torch.argmax(pred_phase_logits, dim=1)

        valid = (phase_gt != 0)

        total_correct += ((pred_phase == phase_gt) & valid).sum().item()
        total_num += valid.sum().item()

        # ======================================================
        # Output head forward
        # ======================================================

        # train 时你是 ratio_input shape = [B,1]
        ratio_input = pred_ratio.unsqueeze(1)

        pred_future = head(
            pred_phase,
            ratio_input,
            stage_order,
            all_time
        )   # [B,7]

        # ======================================================
        # GT future timeline
        # ======================================================

        gt_future = build_gt_future(
            cur_stage_idx,
            ratio_list,
            stage_order,
            all_time
        )   # [B,7]

        # ======================================================
        # Keep only valid samples
        # ======================================================

        pred_future_valid = pred_future[valid]   # [Nv,7]
        gt_future_valid   = gt_future[valid]     # [Nv,7]

        preds_future_all.append(pred_future_valid.cpu().numpy())
        gts_future_all.append(gt_future_valid.cpu().numpy())

    # ======================================================
    # Merge all batches
    # ======================================================

    acc = total_correct / max(total_num, 1)

    preds_future = np.concatenate(preds_future_all, axis=0)   # [N,7]
    gts_future   = np.concatenate(gts_future_all, axis=0)     # [N,7]

    # ======================================================
    # END time MAE (future end timestamps)
    # ======================================================

    end_mask = (gts_future > 0)

    end_mae = np.mean(
        np.abs(preds_future[end_mask] - gts_future[end_mask])
    )

    # ======================================================
    # START time MAE
    # start(k) = end(k-1)
    # ======================================================

    pred_start = preds_future[:, 1:]   # [N,6]
    gt_start   = gts_future[:, :-1]    # [N,6]

    start_mask = (gt_start > 0)

    start_mae = np.mean(
        np.abs(pred_start[start_mask] - gt_start[start_mask])
    )

    # ======================================================
    # R2 (flatten valid timeline points)
    # ======================================================

    r2_time = r2_score(
        gts_future[end_mask],
        preds_future[end_mask]
    )

    return acc, start_mae, end_mae, r2_time, preds_future, gts_future
# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    args = parse_args()
    set_seed(args.seed)

    exp_dir = os.path.join("checkpoints", args.head_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("========== TEST MODE ==========")
    logger.info(f"Backbone: {args.backbone_name}")
    logger.info(f"Output head: {args.head_name}")
    logger.info(f"Backbone model: {args.backbone_model}")
    logger.info(f"Device: {device}")
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

    # ---------------- Load backbone ----------------

    backbone_dir = os.path.join("checkpoints", args.backbone_name)
    backbone_ckpt = os.path.join(backbone_dir, "best.pth")

    if not os.path.exists(backbone_ckpt):
        raise FileNotFoundError(f"Backbone ckpt not found: {backbone_ckpt}")

    backbone = build_model(args.backbone_model).to(device)
    backbone = backbone.to(device)

    ckpt_a = torch.load(
        backbone_ckpt,
        map_location=device,
        weights_only=False
    )

    backbone.load_state_dict(ckpt_a["state_dict"])

    backbone.eval()

    logger.info(f"Loaded backbone checkpoint: {backbone_ckpt}")

    # ---------------- Load head ----------------

    ckpt_path = os.path.join("checkpoints", args.head_name, "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_head = FutureTimelineModel().to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    out_head.load_state_dict(checkpoint["state_dict"])

    logger.info(f"Loaded output head checkpoint: {ckpt_path}")

    # ---------------- Test ----------------

    test_acc, test_start_mae, test_end_mae, r2_score, preds_future, gts_future = validate_pipeline(
        backbone,
        out_head,
        test_loader,
        device
    )


    logger.info("========== TEST RESULT ==========")
    logger.info(f"Phase Acc: {test_acc:.4f}")
    logger.info(f"Start MAE (s): {test_start_mae:.2f}")
    logger.info(f"End MAE (s): {test_end_mae:.2f}")
    # ---------------- Save result ----------------

    result_dict = {
        "phase_acc": float(test_acc),
        "start_mae": float(test_start_mae),
        "end_mae": float(test_end_mae),
        "r2_score": float(r2_score)
    }

    result_path = os.path.join(exp_dir, "test_result.json")

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    logger.info(f"Test results saved to: {result_path}")


    timeline_save_path = os.path.join(exp_dir, "future_timeline_data.npz")

    np.savez(
        timeline_save_path,
        preds_future=preds_future,
        gts_future=gts_future
    )

    logger.info(f"Future timeline data saved to: {timeline_save_path}")


if __name__ == "__main__":
    main()