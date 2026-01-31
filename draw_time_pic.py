import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

DATA_ROOT = "data/cholec80"
PHASE_DIR = "phase_annotations"
FPS_ORI = 25          # 原始视频 fps
DOWNSAMPLE_FPS = 1    # 统计用 1fps

MODE = "train"        # train / val / test


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
# Train/Val/Test split (Cholec80 official style)
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
        raise ValueError("Unknown mode")


# -----------------------------
# Phase duration extractor
# -----------------------------

def extract_phase_duration(phase_file):

    with open(phase_file, "r") as f:
        lines = f.readlines()

    # skip header + downsample to 1 fps
    lines = lines[1::FPS_ORI]

    durations = []
    phase_ids = []

    prev = None
    cur_len = 0

    for line in lines:

        phase_name = line.split("\t")[1].strip()
        pid = PHASE2ID[phase_name]

        if pid != prev:
            if prev is not None:
                durations.append(cur_len)
                phase_ids.append(prev)

            cur_len = 1
            prev = pid
        else:
            cur_len += 1

    durations.append(cur_len)
    phase_ids.append(prev)

    return phase_ids, durations


# -----------------------------
# Main statistics pipeline
# -----------------------------

def compute_all_phase_stats():

    phase_root = os.path.join(DATA_ROOT, PHASE_DIR)

    phase_files = os.listdir(phase_root)

    used_files = split_videos(phase_files, MODE)

    print(f"[INFO] Using {MODE} set: {len(used_files)} videos")

    phase_pool = {i: [] for i in range(1, 8)}

    for fname in used_files:

        phase_path = os.path.join(phase_root, fname)

        phase_ids, durations = extract_phase_duration(phase_path)

        for pid, dur in zip(phase_ids, durations):
            phase_pool[pid].append(dur)

    return phase_pool


# -----------------------------
# Plot
# -----------------------------

def plot_bar(phase_pool):

    names = []
    means = []

    print("\n===== Phase Statistics =====")

    for pid in sorted(phase_pool.keys()):

        arr = np.array(phase_pool[pid])

        mean_val = arr.mean()

        phase_name = ID2PHASE[pid]

        print(f"{phase_name:25s} mean = {mean_val:.2f} s")

        names.append(phase_name)
        means.append(mean_val)

    plt.figure(figsize=(10, 5))
    plt.bar(names, means)
    plt.ylabel("Duration (seconds)")
    plt.title(f"Average Phase Duration ({MODE} set)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Entry
# -----------------------------

if __name__ == "__main__":

    phase_pool = compute_all_phase_stats()

    plot_bar(phase_pool)
