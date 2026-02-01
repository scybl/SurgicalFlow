import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

DATA_ROOT = "data/cholec80"
PHASE_DIR = "phase_annotations"

FPS_ORI = 25   # original fps -> downsample to 1fps

PHASE2ID = {
    "Preparation": 1,
    "CalotTriangleDissection": 2,
    "ClippingCutting": 3,
    "GallbladderDissection": 4,
    "GallbladderPackaging": 5,
    "CleaningCoagulation": 6,
    "GallbladderRetraction": 7
}

ID2PHASE = {v: k for k, v in PHASE2ID.items()}


# -----------------------------
# Dataset split
# -----------------------------

def split_videos(video_list, mode):

    video_list = sorted(video_list)

    if mode == "train":
        return video_list[:40]
    elif mode == "val":
        return video_list[40:50]
    elif mode == "test":
        return video_list[50:80]
    else:
        raise ValueError(mode)


# -----------------------------
# Phase duration parser
# -----------------------------

def extract_phase_duration(phase_file):

    with open(phase_file, "r") as f:
        lines = f.readlines()

    # skip header + downsample to 1fps
    lines = lines[1::FPS_ORI]

    prev = None
    cur_len = 0

    phase_ids = []
    durations = []

    for line in lines:

        phase_name = line.split("\t")[1].strip()
        pid = PHASE2ID[phase_name]

        if pid != prev:

            if prev is not None:
                phase_ids.append(prev)
                durations.append(cur_len)

            prev = pid
            cur_len = 1

        else:
            cur_len += 1

    phase_ids.append(prev)
    durations.append(cur_len)

    return phase_ids, durations


# -----------------------------
# Statistics
# -----------------------------

def compute_split_stats(mode):

    phase_root = os.path.join(DATA_ROOT, PHASE_DIR)
    all_files = os.listdir(phase_root)

    used_files = split_videos(all_files, mode)

    print(f"[{mode}] videos:", len(used_files))

    pool = {i: [] for i in range(1, 8)}

    for fname in used_files:

        path = os.path.join(phase_root, fname)

        phase_ids, durations = extract_phase_duration(path)

        for pid, dur in zip(phase_ids, durations):
            pool[pid].append(dur)

    means = []

    for pid in range(1, 8):
        means.append(np.mean(pool[pid]))

    return np.array(means)


# -----------------------------
# Plot
# -----------------------------

def plot_compare(train_mean, val_mean, test_mean):

    phases = [ID2PHASE[i] for i in range(1, 8)]

    x = np.arange(len(phases))
    width = 0.25

    plt.figure(figsize=(9, 4.5))

    plt.bar(x - width, train_mean, width, label="Train")
    plt.bar(x, val_mean, width, label="Val")
    plt.bar(x + width, test_mean, width, label="Test")

    plt.ylabel("Duration (seconds)")
    plt.title("Phase Duration Comparison (Train / Val / Test)")
    plt.xticks(x, phases, rotation=20)
    plt.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------
# Entry
# -----------------------------

if __name__ == "__main__":

    train_mean = compute_split_stats("train")
    val_mean = compute_split_stats("val")
    test_mean = compute_split_stats("test")

    print("\n===== Mean Phase Duration (seconds) =====")

    for i in range(7):
        print(
            f"{ID2PHASE[i+1]:25s} "
            f"Train={train_mean[i]:.1f}  "
            f"Val={val_mean[i]:.1f}  "
            f"Test={test_mean[i]:.1f}"
        )

    plot_compare(train_mean, val_mean, test_mean)