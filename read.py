import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from task1_data_loader import Cholec80DatasetTaskA
from model_backbone import TaskA_CNN, TaskA_CNN_LSTM
from model_out_head import FutureTimelineModel

from types import SimpleNamespace


# =========================================================
# ===== 手动定义三组实验参数（命名保持原 args 风格）=====
# =========================================================

ARGS_LIST = [

    SimpleNamespace(
        backbone_name="TaskA_CNN_LSTM_8",
        head_name="TimelineHead_8",

        backbone_model="cnn_lstm",
        data_root="data/cholec80",

        seq_len=8,
        stride=4,

        batch_size=16,
        lr=1e-4,
        epochs=20,

        num_workers=8,
        device="cuda",
        seed=42
    ),

    SimpleNamespace(
        backbone_name="TaskA_CNN_LSTM_16",
        head_name="TimelineHead_16",

        backbone_model="cnn_lstm",
        data_root="data/cholec80",

        seq_len=16,
        stride=8,

        batch_size=16,
        lr=1e-4,
        epochs=20,

        num_workers=8,
        device="cuda",
        seed=42
    ),

    SimpleNamespace(
        backbone_name="TaskA_CNN_LSTM_32",
        head_name="TimelineHead_32",

        backbone_model="cnn_lstm",
        data_root="data/cholec80",

        seq_len=32,
        stride=16,

        batch_size=8,
        lr=1e-4,
        epochs=20,

        num_workers=8,
        device="cuda",
        seed=42
    )

]

SAVE_DIR = "checkpoints/temporal_window_curve"


# =========================================================
# Utils
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Build backbone
# =========================================================

def build_backbone(name):

    if name == "cnn":
        return TaskA_CNN()
    else:
        return TaskA_CNN_LSTM()


# =========================================================
# GT future builder
# =========================================================

def build_gt_future(cur_stage_idx, ratio_list, stage_order, all_time):

    B = cur_stage_idx.shape[0]
    gt_future = torch.zeros((B, 7), device=cur_stage_idx.device)

    for b in range(B):

        cur = int(cur_stage_idx[b])

        if cur == 0:
            continue

        idx = cur - 1

        remain = ratio_list[b, idx] * all_time[b, idx]

        acc = remain
        gt_future[b, idx] = acc

        for j in range(idx + 1, 7):

            if stage_order[b, j] == 0:
                continue

            acc += all_time[b, j]
            gt_future[b, j] = acc

    return gt_future


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate_pipeline(backbone, head, loader, device):

    backbone.eval()
    head.eval()

    preds_all = []
    gts_all = []

    for frames, stage_order, ratio_list, all_time in tqdm(loader):

        frames = frames.to(device)
        stage_order = stage_order.to(device)
        ratio_list = ratio_list.to(device)
        all_time = all_time.to(device)

        mask = (ratio_list > 0)
        cur_stage_idx = mask.float().argmax(dim=1)

        pred_phase_logits, pred_ratio = backbone(frames)

        pred_phase = torch.argmax(pred_phase_logits, dim=1)
        pred_ratio = torch.clamp(pred_ratio, 0, 1)

        ratio_input = pred_ratio.unsqueeze(1)

        pred_future = head(
            pred_phase,
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

        valid = (gt_future > 0)

        preds_all.append(pred_future[valid].cpu().numpy())
        gts_all.append(gt_future[valid].cpu().numpy())

    preds_all = np.concatenate(preds_all)
    gts_all = np.concatenate(gts_all)

    mae = np.mean(np.abs(preds_all - gts_all))

    return mae


# =========================================================
# Main
# =========================================================

def main():

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("========== Temporal Window Ablation ==========")

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    window_list = []
    mae_list = []

    # -------------------------------------------------
    # Loop experiments
    # -------------------------------------------------

    for args in ARGS_LIST:

        print("\n--------------------------------------")
        print("Running:", args.backbone_name)
        print("--------------------------------------")

        set_seed(args.seed)

        device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )

        # ---------------- Dataset ----------------

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
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )

        print("Test samples:", len(test_dataset))

        # ---------------- Load models ----------------

        backbone_ckpt = os.path.join(
            "checkpoints", args.backbone_name, "best.pth"
        )

        head_ckpt = os.path.join(
            "checkpoints", args.head_name, "best.pth"
        )

        backbone = build_backbone(args.backbone_model).to(device)
        head = FutureTimelineModel().to(device)

        backbone.load_state_dict(
            torch.load(backbone_ckpt, map_location=device)["state_dict"]
        )

        head.load_state_dict(
            torch.load(head_ckpt, map_location=device)["state_dict"]
        )

        # ---------------- Test ----------------

        mae = validate_pipeline(
            backbone,
            head,
            test_loader,
            device
        )

        print(f"Temporal Window = {args.seq_len}")
        print(f"MAE = {mae:.2f} s")

        window_list.append(args.seq_len)
        mae_list.append(mae)

    # -------------------------------------------------
    # Save numeric results
    # -------------------------------------------------

    import json

    json_path = os.path.join(SAVE_DIR, "temporal_window_mae.json")

    with open(json_path, "w") as f:
        json.dump(
            dict(zip(window_list, mae_list)),
            f,
            indent=4
        )

    print("\nSaved results:", json_path)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plt.figure(figsize=(6, 4))
    plt.plot(window_list, mae_list, marker="o")
    plt.xlabel("Temporal Window Length")
    plt.ylabel("MAE (seconds)")
    plt.title("Temporal Window vs MAE")
    plt.grid(True)

    fig_path = os.path.join(SAVE_DIR, "temporal_window_vs_mae.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved figure:", fig_path)


if __name__ == "__main__":
    main()
