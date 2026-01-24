import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

FPS_ORI = 25

PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6
}


class Cholec80DatasetTaskA(Dataset):
    def __init__(
        self,
        root_dir,
        mode,
        seq_len=16,
        stride=8,
        transform=None,
        frames_dirname="frames",
        phase_dirname="phase_annotations"
    ):

        self.root_dir = root_dir
        self.frames_root = os.path.join(root_dir, frames_dirname)
        self.phase_dir = os.path.join(root_dir, phase_dirname)

        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.samples = []

        self.mode = mode
        self._build_index()


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
            # Notice: 先处理phase文件，得到阶段标签序列
            phase_paths = os.path.join(self.phase_dir, f"{video_name}-phase.txt")

            with open(phase_paths, "r") as f:
                phase_lines = f.readlines()[1::FPS_ORI]  # skip header + downsample

            phase_ids = []
            stage_order = []
            stage_durations = []   # 单位：秒

            prev = None
            cur_len = 0

            for line in phase_lines:

                parts_name = line.split("\t")[1].strip()

                if parts_name not in PHASE2ID:
                    raise ValueError(f"Unknown phase: {parts_name}")

                pid = PHASE2ID[parts_name]
                phase_ids.append(pid)

                # ---------- workflow order + duration ----------
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
                stage_order.append(-1)
            ## build samples
            video_folder = os.path.join(self.frames_root, video_name)

            # -------- load frames --------
            frame_names = sorted(os.listdir(video_folder))
            frame_names = [f for f in frame_names if f.lower().endswith(".png")]

            # 按 frame index 排序
            frame_names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

            frame_paths = [os.path.join(video_folder, f) for f in frame_names]

            # -------- 对齐 phase 与 frame --------
            usable_len = min(len(frame_paths), len(phase_ids))

            frame_paths = frame_paths[:usable_len]
            phase_ids = phase_ids[:usable_len]


            N = usable_len

            # -------- sliding window --------
            for start in range(0, N - self.seq_len + 1, self.stride):

                end_idx = start + self.seq_len - 1

                # 连续 frames
                clip_frame_paths = frame_paths[start:end_idx + 1]

                # 当前真实时间（秒）
                cur_phase = phase_ids[end_idx]

                if cur_phase not in stage_order:
                    continue

                stage_pos = stage_order.index(cur_phase)

                phase_start_idx = end_idx
                while phase_start_idx > 0 and phase_ids[phase_start_idx-1] == cur_phase:
                    phase_start_idx -= 1


                offset_in_phase = end_idx - phase_start_idx

                current_time = sum(stage_durations[:stage_pos]) + offset_in_phase


                # 构建各阶段 remaining 时间向量
                remain = current_time
                time_list = []

                for dur in stage_durations:
                    if remain >= dur:
                        time_list.append(0)
                        remain -= dur
                    else:
                        time_list.append(dur - remain)
                        remain = 0


                # 构建 sample
                self.samples.append({
                    "frames": clip_frame_paths,
                    "stage_order": stage_order,   # 已 padding 到 7
                    "time": time_list
                })
        print(f"[{self.mode}] samples:", len(self.samples))

    # --------------------------------------------------

    def _read_clip(self, frame_paths, start_idx):

        paths = frame_paths[start_idx:start_idx + self.seq_len]

        frames = []

        for p in paths:

            img = Image.open(p).convert("RGB")

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(
                    np.array(img, dtype=np.float32)
                ).permute(2, 0, 1) / 255.0

            frames.append(img)

        return torch.stack(frames)

    # --------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        frame_paths = sample["frames"]
        stage_order = sample["stage_order"]
        time_list = sample["time"]

        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [8,3,H,W]

        return (
            frames,
            torch.tensor(stage_order, dtype=torch.long),
            torch.tensor(time_list, dtype=torch.float32)
        )



if __name__ == "__main__":

    dataset = Cholec80DatasetTaskA(
        root_dir="data/cholec80",
        mode="train"
    )

    frames, stage_order, time_list = dataset[0]

    print("Image shape:", frames.shape)
    print("Label shape:", stage_order)
    print("Time list:", time_list)