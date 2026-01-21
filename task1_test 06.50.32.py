import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from task1_data import Cholec80RemainingFramesDataset
from models.cnn import Task1CNN

EXP_DIR = "checkpoints/test"
# --------------------
# Load config
# --------------------

def load_config(exp_dir):
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg


# --------------------
# Build model
# --------------------

def build_model(model_name):

    if model_name == "cnn":
        return Task1CNN()

    else:
        raise ValueError("Unsupported model type")


# --------------------
# Test function
# --------------------

def test(model, loader, device):

    model.eval()

    mae_sum = 0.0
    count = 0

    with torch.no_grad():

        for frames, remain_norm, _, _ in tqdm(loader, desc="Testing"):

            frames = frames.to(device)
            remain_norm = remain_norm.to(device)

            # CNN baseline uses last frame
            x = frames[:, -1]

            pred = model(x)

            mae_sum += torch.abs(pred - remain_norm).sum().item()
            count += remain_norm.size(0)

    mae = mae_sum / count

    return mae


# --------------------
# Main
# --------------------

def main():

    cfg = load_config(EXP_DIR)

    print("Loaded config from:", EXP_DIR)
    print("Model:", cfg["model"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # ---------------- Transform ----------------

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # ---------------- Dataset ----------------

    full_dataset = Cholec80RemainingFramesDataset(
        root_dir=cfg["data_root"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        transform=transform
    )

    dataset_size = len(full_dataset)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    _, _, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda")
    )

    print("Test samples:", len(test_set))

    # ---------------- Load model ----------------

    model = build_model(cfg["model"]).to(device)

    ckpt_path = os.path.join(EXP_DIR, "best.pth")

    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    print("Loaded checkpoint:", ckpt_path)

    # ---------------- Run test ----------------

    test_mae = test(model, test_loader, device)

    print("\n========== Test Result ==========")
    print(f"Test MAE (normalized): {test_mae:.4f}")
    print("=================================\n")


if __name__ == "__main__":
    main()
