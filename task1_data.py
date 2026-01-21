import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

FPS_ORI = 25.0  # 原视频fps（Cholec80）
PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6
}

def extract_index(name):
    base = os.path.splitext(name)[0]
    return int(base.split("_")[-1])


class Cholec80RemainingFramesDataset(Dataset):
    """
    从抽帧图片读取序列

    假设：
      - root_dir/frames/video_xx/ 下面是 1.jpg,2.jpg,...（或png）
      - 标注 phase_annotations/video_xx-phase.txt 仍是原始逐帧标注（frame_id phase）

    返回：
      frames: [T, C, H, W] float32 in [0,1]
      remaining_time_norm: float32 scalar
      remaining_time_sec: float32 scalar (基于抽帧后的“有效秒”)
      phase_id: int64 scalar
    """

    def __init__(
        self,
        root_dir,
        mode,
        seq_len=16,
        stride=8,
        transform=None,
        video_list=None,
        normalize=True,
        sample_every=25,          # 每隔多少“原始帧”保留一帧；你这里是 25
        frames_dirname="frames",  # 你的新目录名
        phase_dirname="phase_annotations",
        img_exts=(".jpg", ".jpeg", ".png")
    ):
        self.root_dir = root_dir
        self.frames_root = os.path.join(root_dir, frames_dirname)
        self.phase_dir = os.path.join(root_dir, phase_dirname)

        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        self.sample_every = sample_every
        self.fps_eff = FPS_ORI / float(sample_every)  # 抽帧后的有效fps；25/25=1fps

        self.img_exts = img_exts
        self.samples = []
        self.frame_cache = {}
        
        self.mode = mode  # "train", "test"
        self._build_index(video_list)
    

    def _load_phase_file(self, path):
        frames = []
        phases = []
        with open(path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0 and ("Frame" in line or "frame" in line):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                fid, phase = parts[0], parts[1]
                frames.append(int(fid))
                phases.append(PHASE2ID[phase])
        return frames, phases

    def _compute_remaining_time(self, phases, fps):
        # TODO: 这个要改
        # phases: list[int], 每个“抽帧点”的phase id
        remaining_sec = [0.0] * len(phases)
        phase_start = 0
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_end = i - 1
                for j in range(phase_start, phase_end + 1):
                    remaining_sec[j] = (phase_end - j) / fps
                phase_start = i

        phase_end = len(phases) - 1
        for j in range(phase_start, phase_end + 1):
            remaining_sec[j] = (phase_end - j) / fps

        return remaining_sec

    def _list_frame_files(self, video_folder):
        # cache: avoid listdir/sort every __getitem__
        if video_folder in self.frame_cache:
            return self.frame_cache[video_folder]

        files = []
        for fn in os.listdir(video_folder):
            if fn.lower().endswith(self.img_exts):
                files.append(fn)

        def extract_index(name):
            base = os.path.splitext(name)[0]
            idx = base.split("_")[-1]
            return int(idx)

        files.sort(key=extract_index)
        full_paths = [os.path.join(video_folder, f) for f in files]
        self.frame_cache[video_folder] = full_paths
        return full_paths


    def _build_index(self, video_list=None):
        # ---- step 1: collect & sort all videos ----
        all_videos = [
            v for v in sorted(os.listdir(self.frames_root))
            if os.path.isdir(os.path.join(self.frames_root, v))
        ]

        # sanity check
        if len(all_videos) < 50:
            raise RuntimeError(
                f"Expected at least 50 videos, but found {len(all_videos)}"
            )

        # ---- step 2: fixed split by mode (NO randomness) ----
        if self.mode == "train":
            used_videos = all_videos[:40]
        elif self.mode == "val":
            used_videos = all_videos[40:50]
        elif self.mode == "test":
            used_videos = all_videos[50:80]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        print(f"[{self.mode}] using videos: {used_videos[0]} ... {used_videos[-1]}")
        print(f"[{self.mode}] using {len(used_videos)} videos")

        # ---- step 3: build samples only from selected videos ----
        for video_name in used_videos:
            video_folder = os.path.join(self.frames_root, video_name)

            frame_paths = self._list_frame_files(video_folder)
            if len(frame_paths) < self.seq_len:
                continue

            phase_path = os.path.join(self.phase_dir, f"{video_name}-phase.txt")
            _, phases_full = self._load_phase_file(phase_path)

            # downsample phases to match sampled frames
            phases_sampled = []
            for k in range(len(frame_paths)):
                ori_idx = k * self.sample_every
                if ori_idx >= len(phases_full):
                    break
                phases_sampled.append(phases_full[ori_idx])

            usable_len = min(len(frame_paths), len(phases_sampled))
            frame_paths = frame_paths[:usable_len]
            phases_sampled = phases_sampled[:usable_len]

            remaining_sec = self._compute_remaining_time(
                phases_sampled, fps=self.fps_eff
            )
            max_phase_time = max(remaining_sec) + 1e-6

            for start in range(0, usable_len - self.seq_len, self.stride):
                end_idx = start + self.seq_len - 1

                remain_s = remaining_sec[end_idx]
                remain_norm = (
                    remain_s / max_phase_time if self.normalize else remain_s
                )
                phase_id = phases_sampled[end_idx]

                self.samples.append(
                    (frame_paths, start, remain_norm, remain_s, phase_id)
                )

        print(f"[{self.mode}] total samples: {len(self.samples)}")


    def _read_clip_from_paths(self, frame_paths, start_idx):
        paths = frame_paths[start_idx:start_idx + self.seq_len]
        frames = []

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                raise FileNotFoundError(f"Failed to read image: {p}")

            if self.transform:
                # torchvision transform usually expects PIL Image
                img = self.transform(img)
            else:
                # manually convert to tensor [C,H,W] in [0,1]
                img = torch.from_numpy(
                    np.array(img, dtype=np.float32)
                ).permute(2, 0, 1) / 255.0

            frames.append(img)

        return torch.stack(frames, dim=0)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, start, remain_norm, remain_sec, phase_id = self.samples[idx]
        frames = self._read_clip_from_paths(frame_paths, start)


        return (
            frames,
            torch.tensor(remain_norm, dtype=torch.float32),
            torch.tensor(remain_sec, dtype=torch.float32),
            torch.tensor(phase_id, dtype=torch.long),
        )
