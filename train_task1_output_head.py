import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from task1_data import Cholec80DatasetTaskA
from task1_model import TaskA_CNN, TaskA_CNN_LSTM
from output_head import FutureTimelineModel


NUM_PHASES = 7


# -------------------------------------------------
# Argument Parser
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--taska_ckpt", type=str, required=True)
    parser.add_argument("--timeline_ckpt", type=str, required=True)

    parser.add_argument("--model", type=str, default="cnn_lstm", choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# -------------------------------------------------
# Build models
# -------------------------------------------------

def build_taska(name):

    if name == "cnn":
        return TaskA_CNN()

    else:
        return TaskA_CNN_LSTM()


# -------------------------------------------------
# Timeline post-processing (industry standard)
# -------------------------------------------------

def post_process_timeline(raw, cur_stage_idx, stage_order):

    # enforce monotonic
    out, _ = torch.cummax(raw, dim=1)

    # phase position mask
    idx = cur_stage_idx.clamp(min=1) - 1

    arange = torch.arange(out.shape[1], device=out.device).unsqueeze(0)

    valid_mask = (arange >= idx.unsqueeze(1)).float()

    out = out * valid_mask
    out = out * stage_order

    return out


# -------------------------------------------------
# Build GT timeline
# -------------------------------------------------

def build_gt_future(cur_stage_idx, ratio_list, stage_order, all_time):

    B = cur_stage_idx.shape[0]

    gt_future = torch.zeros((B, NUM_PHASES), device=cur_stage_idx.device)

    for b in range(B):

        cur = int(cur_stage_idx[b].item())

        if cur == 0:
            continue

        idx = cur - 1

        remain = ratio_list[b, idx] * all_time[b, idx]

        acc = remain
        gt_future[b, idx] = acc

        for j in range(idx + 1, NUM_PHASES):

            if stage_order[b, j] == 0:
                continue

            acc = acc + all_time[b, j]
            gt_future[b, j] = acc

    return gt_future


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

@torch.no_grad()
def evaluate(taska, timeline, loader, device):

    taska.eval()
    timeline.eval()

    total_correct = 0
    total_num = 0

    preds_remain = []
    gts_remain = []

    preds_start = []
    gts_start = []

    preds_end = []
    gts_end = []

    for batch in tqdm(loader):

        frames, stage_order, ratio_list, all_time = batch

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)

        # ---------------- TaskA inference ----------------

        phase_logits, pred_ratio = taska(frames)

        pred_phase = torch.argmax(phase_logits, dim=1) + 1
        pred_ratio = torch.clamp(pred_ratio, 0.0, 1.0)

        # ---------------- Phase accuracy ----------------

        mask = (ratio_list > 0)
        cur_stage_gt = mask.float().argmax(dim=1) + 1

        valid = (cur_stage_gt != 0)

        total_correct += ((pred_phase == cur_stage_gt) & valid).sum().item()
        total_num += valid.sum().item()

        # ---------------- Timeline head ----------------

        raw_future = timeline(
            pred_phase,
            pred_ratio.unsqueeze(1),
            stage_order,
            all_time
        )

        pred_future = post_process_timeline(
            raw_future,
            pred_phase,
            stage_order
        )

        # ---------------- GT timeline ----------------

        gt_future = build_gt_future(
            cur_stage_gt,
            ratio_list,
            stage_order,
            all_time
        )

        # ---------------- Metrics ----------------

        B = pred_phase.shape[0]
        batch_idx = torch.arange(B).to(device)

        pred_idx = pred_phase - 1
        gt_idx = cur_stage_gt - 1

        pred_remain = pred_future[batch_idx, pred_idx]
        gt_remain = gt_future[batch_idx, gt_idx]

        pred_start = pred_future[batch_idx, pred_idx] - pred_remain
        gt_start = gt_future[batch_idx, gt_idx] - gt_remain

        pred_end = pred_future[batch_idx, pred_idx]
        gt_end = gt_future[batch_idx, gt_idx]

        valid_mask = valid

        preds_remain.append(pred_remain[valid_mask].cpu().numpy())
        gts_remain.append(gt_remain[valid_mask].cpu().numpy())

        preds_start.append(pred_start[valid_mask].cpu().numpy())
        gts_start.append(gt_start[valid_mask].cpu().numpy())

        preds_end.append(pred_end[valid_mask].cpu().numpy())
        gts_end.append(gt_end[valid_mask].cpu().numpy())

    # ---------------- Aggregate ----------------

    acc = total_correct / max(total_num, 1)

    preds_remain = np.concatenate(preds_remain)
    gts_remain = np.concatenate(gts_remain)

    preds_start = np.concatenate(preds_start)
    gts_start = np.concatenate(gts_start)

    preds_end = np.concatenate(preds_end)
    gts_end = np.concatenate(gts_end)

    mae_remain = np.mean(np.abs(preds_remain - gts_remain))
    mae_start = np.mean(np.abs(preds_start - gts_start))
    mae_end = np.mean(np.abs(preds_end - gts_end))

    return acc, mae_remain, mae_start, mae_end


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------------- Load models ----------------

    taska = build_taska(args.model).to(device)
    timeline = FutureTimelineModel().to(device)

    print("Loading TaskA:", args.taska_ckpt)
    ckpt_a = torch.load(args.taska_ckpt, map_location=device)
    taska.load_state_dict(ckpt_a["state_dict"])

    print("Loading Timeline:", args.timeline_ckpt)
    ckpt_t = torch.load(args.timeline_ckpt, map_location=device)
    timeline.load_state_dict(ckpt_t["state_dict"])

    # ---------------- Dataset ----------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = Cholec80DatasetTaskA(
        root_dir=args.data_root,
        mode="test",
        seq_len=args.seq_len,
        stride=args.stride,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Test samples:", len(test_dataset))

    # ---------------- Evaluate ----------------

    acc, mae_remain, mae_start, mae_end = evaluate(
        taska,
        timeline,
        test_loader,
        device
    )

    # ---------------- Print ----------------

    print("\n====== Final Evaluation ======")
    print(f"Remain MAE (s): {mae_remain:.2f}")
    print(f"Start  MAE (s): {mae_start:.2f}")
    print(f"End    MAE (s): {mae_end:.2f}")
    print(f"Phase Acc (%): {acc * 100:.2f}")

    print("\nLaTeX row:")
    print(
        f"{mae_remain:.2f} & "
        f"{mae_start:.2f} & "
        f"{mae_end:.2f} & "
        f"{acc * 100:.2f}"
    )


if __name__ == "__main__":
    main()