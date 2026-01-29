import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

FPS_ORI = 25

PHASE2ID = {
    "Preparation": 1,
    "CalotTriangleDissection": 2,
    "ClippingCutting": 3,
    "GallbladderDissection": 4,
    "GallbladderPackaging": 5,
    "CleaningCoagulation": 6,
    "GallbladderRetraction": 7
}

NUM_TOOLS = 7


class Cholec80DatasetTaskA(Dataset):
    def __init__(
        self,
        root_dir,
        mode,
        seq_len=16,
        stride=8,
        transform=None,
        frames_dirname="frames",
        phase_dirname="phase_annotations",
        tool_dirname="tool_annotations"
    ):

        self.root_dir = root_dir
        self.frames_root = os.path.join(root_dir, frames_dirname)
        self.phase_dir = os.path.join(root_dir, phase_dirname)
        self.tool_dir = os.path.join(root_dir, tool_dirname)

        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.samples = []

        self.mode = mode
        self._build_index()

    # -------------------------------------------------

    def _build_index(self):

        all_videos = sorted(os.listdir(self.frames_root))

        if self.mode == "train":
            used_videos = all_videos[:40]
        elif self.mode == "val":
            used_videos = all_videos[40:50]
        elif self.mode == "test":
            used_videos = all_videos[50:80]
        else:
            raise ValueError(self.mode)

        print(f"[{self.mode}] videos:", len(used_videos))

        for video_name in used_videos:

            # ---------------- load phase ----------------

            phase_path = os.path.join(self.phase_dir, f"{video_name}-phase.txt")

            with open(phase_path, "r") as f:
                phase_lines = f.readlines()[1::FPS_ORI]

            phase_ids = []
            stage_order = []
            stage_durations = []

            prev = None
            cur_len = 0

            for line in phase_lines:

                pname = line.split("\t")[1].strip()

                if pname not in PHASE2ID:
                    raise ValueError(f"Unknown phase: {pname}")

                pid = PHASE2ID[pname]
                phase_ids.append(pid)

                if pid != prev:
                    if prev is not None:
                        stage_durations.append(cur_len)
                    stage_order.append(pid)
                    cur_len = 1
                    prev = pid
                else:
                    cur_len += 1

            stage_durations.append(cur_len)

            while len(stage_order) < 7:
                stage_order.append(0)

            # ---------------- load frames ----------------

            video_folder = os.path.join(self.frames_root, video_name)

            frame_names = sorted(os.listdir(video_folder))
            frame_names = [f for f in frame_names if f.lower().endswith(".png")]

            frame_names.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )

            frame_paths = [os.path.join(video_folder, f) for f in frame_names]

            # ---------------- load tools ----------------

            tool_path = os.path.join(self.tool_dir, f"{video_name}-tool.txt")

            with open(tool_path, "r") as f:
                tool_lines = f.readlines()[1::FPS_ORI]

            tool_labels = []

            for line in tool_lines:
                parts = line.strip().split()
                tools = list(map(int, parts[1:]))
                tool_labels.append(tools)

            tool_labels = torch.tensor(tool_labels, dtype=torch.float32)  # [N,7]

            # ---------------- align length ----------------

            usable_len = min(
                len(frame_paths),
                len(phase_ids),
                len(tool_labels)
            )

            frame_paths = frame_paths[:usable_len]
            phase_ids = phase_ids[:usable_len]
            tool_labels = tool_labels[:usable_len]

            N = usable_len

            # ---------------- sliding window ----------------

            for start in range(0, N - self.seq_len + 1, self.stride):

                end_idx = start + self.seq_len - 1

                clip_frame_paths = frame_paths[start:end_idx + 1]

                # ---------- TaskA timing target ----------

                cur_phase = phase_ids[end_idx]

                if cur_phase not in stage_order:
                    continue

                stage_pos = stage_order.index(cur_phase)

                phase_start_idx = end_idx
                while phase_start_idx > 0 and phase_ids[phase_start_idx - 1] == cur_phase:
                    phase_start_idx -= 1

                offset_in_phase = end_idx - phase_start_idx

                current_time = sum(stage_durations[:stage_pos]) + offset_in_phase

                remain = current_time
                time_list = []

                for dur in stage_durations:
                    if remain >= dur:
                        time_list.append(0)
                        remain -= dur
                    else:
                        time_list.append(dur - remain)
                        remain = 0

                # ---------- TaskB tool label (anchor frame) ----------

                cur_tool = tool_labels[end_idx]   # [7]

                # ---------- save sample ----------

                self.samples.append({
                    "frames": clip_frame_paths,
                    "stage_order": stage_order,
                    "time": time_list,
                    "all_time": stage_durations,
                    "tool": cur_tool
                })

        print(f"[{self.mode}] samples:", len(self.samples))

    # -------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        frame_paths = sample["frames"]
        stage_order = sample["stage_order"]
        time_list = sample["time"]
        all_time = sample["all_time"]
        tool_label = sample["tool"]

        frames = []

        for p in frame_paths:

            img = Image.open(p).convert("RGB")

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(
                    np.array(img, dtype=np.float32)
                ).permute(2, 0, 1) / 255.0

            frames.append(img)

        frames = torch.stack(frames, dim=0)

        ratio_list = []

        for r, t in zip(time_list, all_time):
            ratio = r / t
            ratio = min(max(ratio, 0.0), 1.0)
            ratio_list.append(ratio)

        if len(stage_order) == 6:
            stage_order.append(0)

        if len(ratio_list) == 6:
            ratio_list.append(0.0)

        if len(all_time) == 6:
            all_time.append(0.0)

        return (
            frames,
            torch.tensor(stage_order, dtype=torch.long),
            torch.tensor(ratio_list, dtype=torch.float32),
            torch.tensor(all_time, dtype=torch.float32),
            tool_label
        )



if __name__ == "__main__":

    dataset = Cholec80DatasetTaskA(
        root_dir="data/cholec80",
        mode="train"
    )

    frames, stage_order, ratio_list, all_time, tool = dataset[3]

    print("Frames:", frames.shape)
    print("Stage order:", stage_order)
    print("Ratio list:", ratio_list)
    print("All time:", all_time)
    print("Tool:", tool.shape)
