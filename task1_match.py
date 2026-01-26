import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from task1_data import Cholec80DatasetTaskA
from task1_model import TaskA_CNN, TaskA_CNN_LSTM


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction"
]


# -------------------------------------------------
# Argument Parser (style aligned with train.py)
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)   # experiment name
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm"])

    parser.add_argument("--data_root", type=str, default="data/cholec80")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--prior_path", type=str, default="phase_duration_prior.json")

    return parser.parse_args()


# -------------------------------------------------
# Build model (reuse train.py logic)
# -------------------------------------------------

def build_model(model_name):

    if model_name == "cnn":
        return TaskA_CNN()
    else:
        return TaskA_CNN_LSTM()


# -------------------------------------------------
# Load phase prior
# -------------------------------------------------

def load_phase_prior(path):

    with open(path) as f:
        prior = json.load(f)

    return prior


# -------------------------------------------------
# Core prediction function
# -------------------------------------------------

@torch.no_grad()
def predict_future_timeline(model, loader, prior, device):

    model.eval()

    all_results = []

    for idx, batch in enumerate(loader):

        frames, stage_order, ratio_list, all_time = batch

        frames = frames.to(device)

        # ---------------- model inference ----------------

        pred_phase_logits, pred_ratio = model(frames)

        pred_ratio = torch.clamp(pred_ratio, 0.0, 1.0)

        pred_phase = torch.argmax(pred_phase_logits, dim=1).item()
        pred_ratio = pred_ratio.item()

        # ---------------- math fitting ----------------

        cur_phase_name = PHASE_NAMES[pred_phase]

        cur_phase_mean_time = prior[cur_phase_name]["mean"]

        remain_seconds = pred_ratio * cur_phase_mean_time

        # ---------------- build timeline ----------------

        timeline = {}

        current_time = 0.0   # 相对时间锚点（可替换为真实视频时间戳）

        cur_end = current_time + remain_seconds

        timeline[cur_phase_name] = {
            "start": None,
            "end": round(cur_end, 2)
        }

        prev_end = cur_end

        for next_id in range(pred_phase + 1, 7):

            name = PHASE_NAMES[next_id]
            dur = prior[name]["mean"]

            start = prev_end
            end = start + dur

            timeline[name] = {
                "start": round(start, 2),
                "end": round(end, 2)
            }

            prev_end = end

        all_results.append({
            "sample_id": idx,
            "pred_phase": cur_phase_name,
            "pred_ratio": round(pred_ratio, 4),
            "timeline": timeline
        })

    return all_results


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join("checkpoints", args.name)
    ckpt_path = os.path.join(exp_dir, "best.pth")

    print("Loading checkpoint:", ckpt_path)

    # ---------------- model ----------------

    model = build_model(args.model).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Best val MAE:", checkpoint["best_mae"])

    # ---------------- transform (same as train) ----------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------------- dataset ----------------

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

    # ---------------- load prior ----------------

    prior = load_phase_prior(args.prior_path)

    # ---------------- predict ----------------

    results = predict_future_timeline(
        model,
        test_loader,
        prior,
        device
    )

    # ---------------- save ----------------

    save_path = os.path.join(exp_dir, "future_phase_prediction.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Saved prediction:", save_path)


if __name__ == "__main__":
    main()
