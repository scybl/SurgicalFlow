import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from task1_data import Cholec80DatasetTaskA
from task1_model import TaskA_CNN, TaskA_CNN_LSTM


# =========================================================
# Phase mapping (IMPORTANT: index 1-7 valid, 0 invalid)
# =========================================================

PHASE_NAMES = [
    "INVALID",                     # 0 占位
    "Preparation",                 # 1
    "CalotTriangleDissection",     # 2
    "ClippingCutting",             # 3
    "GallbladderDissection",       # 4
    "GallbladderPackaging",        # 5
    "CleaningCoagulation",         # 6
    "GallbladderRetraction"        # 7
]

NUM_PHASES = 7


# =========================================================
# Argument Parser (aligned with train.py)
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# =========================================================
# Build model
# =========================================================

def build_model(model_name):

    if model_name == "cnn":
        return TaskA_CNN()
    else:
        return TaskA_CNN_LSTM()

def json_default(o):
    import numpy as np
    import torch

    if isinstance(o, (np.integer,)):
        return int(o)

    if isinstance(o, (np.floating,)):
        return float(o)

    if isinstance(o, (torch.Tensor,)):
        return o.detach().cpu().tolist()

    return str(o)
# =========================================================
# Compute phase duration prior from TRAIN set
# =========================================================
import json
import os
import numpy as np
from tqdm import tqdm


def compute_phase_prior_on_the_fly(data_root, seq_len, stride):

    prior_path = os.path.join(data_root, "phase_prior.json")

    # =================================================
    # Load if exists
    # =================================================

    if os.path.exists(prior_path):

        print("\nLoading phase duration prior from file...")
        print("Path:", prior_path)

        with open(prior_path, "r") as f:
            prior = json.load(f)

        print("Phase prior loaded.\n")

        return prior

    # =================================================
    # Compute if not exists
    # =================================================

    print("\nComputing phase duration prior from TRAIN set...")

    dataset = Cholec80DatasetTaskA(
        root_dir=data_root,
        mode="train",
        seq_len=seq_len,
        stride=stride,
        transform=None
    )

    phase_times = {i: [] for i in range(1, NUM_PHASES + 1)}

    pbar = tqdm(
        range(len(dataset)),
        desc="Computing phase prior",
        ncols=100
    )

    for i in pbar:

        _, stage_order, ratio_list, all_time = dataset[i]

        for idx, phase_id in enumerate(stage_order):

            phase_id = int(phase_id)

            # Skip invalid stage
            if phase_id == 0:
                continue

            phase_times[phase_id].append(float(all_time[idx]))

    # =================================================
    # Statistics
    # =================================================

    prior = {}

    print("\nPhase duration statistics:")

    for k in phase_times:

        arr = np.array(phase_times[k])

        mean_val = float(arr.mean())
        std_val = float(arr.std())

        prior[PHASE_NAMES[k]] = {
            "mean": mean_val,
            "std": std_val
        }

        print(
            f"{PHASE_NAMES[k]:25s} "
            f"mean={mean_val:.1f}s "
            f"std={std_val:.1f}s"
        )

    # =================================================
    # Save to file
    # =================================================

    with open(prior_path, "w") as f:
        json.dump(prior, f, indent=4)

    print("\nPhase prior saved to:", prior_path)
    print("Phase prior ready.\n")

    return prior
# =========================================================
# Core prediction
# =========================================================

@torch.no_grad()
def evaluate_timeline_metrics(model, loader, device):

    model.eval()

    total_correct = 0
    total_num = 0

    preds_remain = []
    gts_remain = []

    preds_start = []
    gts_start = []

    preds_end = []
    gts_end = []

    for batch in loader:

        frames, stage_order, ratio_list, all_time = batch

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)

        # ---------------- locate current phase ----------------

        mask = (ratio_list > 0)
        cur_stage_idx = mask.float().argmax(dim=1)

        phase_gt = stage_order.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        ratio_gt = ratio_list.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        phase_total_time = all_time.gather(
            1, cur_stage_idx.unsqueeze(1)
        ).squeeze(1)

        # ---------------- model inference ----------------

        pred_phase_logits, pred_ratio = model(frames)

        pred_ratio = torch.clamp(pred_ratio, 0.0, 1.0)
        pred_phase = torch.argmax(pred_phase_logits, dim=1)

        valid = (phase_gt != 0)

        total_correct += ((pred_phase == phase_gt) & valid).sum().item()
        total_num += valid.sum().item()

        # ---------------- time reconstruction ----------------

        # -------- Current phase remain --------
        gt_remain = ratio_gt * phase_total_time
        pred_remain = pred_ratio * phase_total_time

        # -------- Current phase start (elapsed) --------
        gt_start = (1.0 - ratio_gt) * phase_total_time
        pred_start = (1.0 - pred_ratio) * phase_total_time

        # -------- Surgery remaining time (NEW End definition) --------
        # GT: sum of remaining time of all phases
        gt_end = torch.sum(ratio_list * all_time, dim=1)

        # Pred:
        # = predicted current phase remain
        # + full duration of all future phases
        pred_end = pred_remain.clone()

        for b in range(pred_end.shape[0]):

            idx = int(cur_stage_idx[b])

            # add future phases full duration
            if idx + 1 < all_time.shape[1]:
                pred_end[b] += torch.sum(all_time[b, idx + 1:])

        preds_remain.append(pred_remain[valid].cpu().numpy())
        gts_remain.append(gt_remain[valid].cpu().numpy())

        preds_start.append(pred_start[valid].cpu().numpy())
        gts_start.append(gt_start[valid].cpu().numpy())

        preds_end.append(pred_end[valid].cpu().numpy())
        gts_end.append(gt_end[valid].cpu().numpy())

    # ---------------- metrics ----------------

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
# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join("checkpoints", args.name)
    ckpt_path = os.path.join(exp_dir, "best.pth")

    print("Loading checkpoint:", ckpt_path)

    # ---------------- Load model ----------------

    model = build_model(args.model).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["state_dict"])

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Best val MAE:", checkpoint["best_mae"])

    # ---------------- Transform (same as train) ----------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------------- Test dataset ----------------

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

    # ---------------- Compute phase prior ----------------

    prior = compute_phase_prior_on_the_fly(
        args.data_root,
        args.seq_len,
        args.stride
    )

    # ---------------- Predict ----------------

    acc, mae_remain, mae_start, mae_end = evaluate_timeline_metrics(
        model,
        test_loader,
        device
    )

    # ---------------- Save ----------------

    save_path = os.path.join(exp_dir, "future_phase_prediction.json")

    results = {
        "phase_acc": acc,
        "mae_remain": mae_remain,
        "mae_start": mae_start,
        "mae_end": mae_end
    }

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=json_default)

    print("\n======= Evaluation Result =======")

    print(f"Remain MAE (s): {mae_remain:.2f}")
    print(f"Start  MAE (s): {mae_start:.2f}")
    print(f"End    MAE (s): {mae_end:.2f}")
    print(f"Phase Acc (%): {acc * 100:.2f}")

    print("\nLaTeX row:")

    print(
        f"{mae_remain:.2f} & "
        f"{mae_start:.2f} & "
        f"{mae_end:.2f} & "
        f"{acc*100:.2f}"
    )


if __name__ == "__main__":
    main()