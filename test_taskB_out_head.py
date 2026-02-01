import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import logging
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from model_backbone import TaskA_CNN, TaskA_CNN_LSTM
from taskB_data_loader import Cholec80DatasetTaskB
from model_out_head import ToolPredictionModel


NUM_PHASES = 7
NUM_TOOLS = 7


# -------------------------------------------------
# Argument Parser (MATCH TRAIN STYLE)
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone_name", type=str, required=True)
    parser.add_argument("--head_name", type=str, required=True)
    parser.add_argument("--backbone_model", type=str, required=True,
                        choices=["cnn", "cnn_lstm"])


    parser.add_argument("--data_root", type=str, default="data/cholec80")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_backbone(name):

    if name == "cnn":
        return TaskA_CNN()
    elif name == "cnn_lstm":
        return TaskA_CNN_LSTM()
    else:
        raise ValueError(name)

# -------------------------------------------------
# Utils (SAME STYLE AS TASK1)
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
# Validation (Task2)
# -------------------------------------------------

from sklearn.metrics import accuracy_score, f1_score

@torch.no_grad()
def validate_pipeline(backbone, head, loader, device, thr=0.5):

    backbone.eval()
    head.eval()

    preds_all = []
    gts_all = []

    for batch in tqdm(loader):

        # 你说 dataloader 已经可以返回
        frames, stage_order, ratio_list, all_time, tool_gt = batch

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)
        tool_gt = tool_gt.to(device)

        # ======================================================
        # Step 1: Backbone prediction
        # ======================================================

        pred_phase_logits, pred_ratio = backbone(frames)

        pred_phase = torch.argmax(pred_phase_logits, dim=1)   # [B]
        pred_ratio = torch.clamp(pred_ratio, 0.0, 1.0)        # [B]

        valid = (pred_phase != 0)

        # ======================================================
        # Step 2: Build predicted remaining time
        # ======================================================

        # phase id -> index
        pred_phase_idx = torch.clamp(pred_phase - 1, 0, 6)

        # total duration of predicted phase
        pred_total_time = all_time.gather(
            1,
            pred_phase_idx.unsqueeze(1)
        ).squeeze(1)

        pred_remain_time = pred_ratio * pred_total_time   # [B]

        # ======================================================
        # Step 3: Out head forward (tool prediction)
        # ======================================================

        tool_logits = head(
            pred_phase,
            pred_remain_time
        )                       # [B,7]

        tool_prob = torch.sigmoid(tool_logits)
        tool_pred = (tool_prob > thr).int()

        # ======================================================
        # Step 4: Collect valid samples
        # ======================================================

        tool_pred = tool_pred[valid]
        tool_gt_v = tool_gt[valid]

        preds_all.append(tool_pred.cpu().numpy())
        gts_all.append(tool_gt_v.cpu().numpy())

    # ======================================================
    # Merge
    # ======================================================

    preds = np.concatenate(preds_all, axis=0)
    gts   = np.concatenate(gts_all, axis=0)

    # ======================================================
    # Metrics
    # ======================================================

    tool_acc = accuracy_score(
        gts.reshape(-1),
        preds.reshape(-1)
    )

    micro_f1 = f1_score(
        gts.reshape(-1),
        preds.reshape(-1),
        average="micro"
    )

    macro_f1 = f1_score(
        gts,
        preds,
        average="macro"
    )

    return tool_acc, micro_f1, macro_f1
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

    logger.info("========== TASK2 TEST MODE ==========")
    logger.info(f"Output head: {args.head_name}")
    logger.info(f"Device: {device}")
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

    test_dataset = Cholec80DatasetTaskB(
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
        raise FileNotFoundError(backbone_ckpt)

    backbone = build_backbone(args.backbone_model).to(device)

    ckpt_a = torch.load(
        backbone_ckpt,
        map_location=device,
        weights_only=False
    )

    backbone.load_state_dict(ckpt_a["state_dict"])
    backbone.eval()

    logger.info(f"Loaded backbone: {backbone_ckpt}")

    # ---------------- Load head ----------------

    ckpt_path = os.path.join("checkpoints", args.head_name, "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_head = ToolPredictionModel(
        hidden_dim=128,
        num_phases=NUM_PHASES,
        num_tools=NUM_TOOLS
    ).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    out_head.load_state_dict(checkpoint["state_dict"])

    logger.info(f"Loaded output head: {ckpt_path}")

    # ---------------- Test ----------------

    test_acc, test_micro_f1, test_macro_f1 = validate_pipeline(
        backbone,
        out_head,
        test_loader,
        device
    )

    logger.info("========== TEST RESULT ==========")
    logger.info(f"Tool Acc: {test_acc:.4f}")
    logger.info(f"Micro F1: {test_micro_f1:.4f}")
    logger.info(f"Macro F1: {test_macro_f1:.4f}")

    # ---------------- Save result ----------------

    result_dict = {
        "tool_acc": float(test_acc),
        "micro_f1": float(test_micro_f1),
        "macro_f1": float(test_macro_f1)
    }

    result_path = os.path.join(exp_dir, "test_result.json")

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    logger.info(f"Test results saved to: {result_path}")


if __name__ == "__main__":
    main()