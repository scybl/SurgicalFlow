import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

FPS_ORI = 25.0

PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6
}


class Cholec80RemainingFramesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode,
        seq_len=16,
        stride=8,
        transform=None,
        sample_every=25,
        max_future_events=10,
        frames_dirname="frames",
        phase_dirname="phase_annotations",
        img_exts=(".jpg", ".jpeg", ".png")
    ):

        self.root_dir = root_dir
        self.frames_root = os.path.join(root_dir, frames_dirname)
        self.phase_dir = os.path.join(root_dir, phase_dirname)

        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.sample_every = sample_every

        self.fps_eff = FPS_ORI / float(sample_every)

        self.max_future_events = max_future_events

        self.img_exts = img_exts
        self.samples = []
        self.frame_cache = {}

        self.mode = mode
        self._build_index()

    # --------------------------------------------------
    def _extract_phase_segments(self, phases):

        segments = []
        start = 0

        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                segments.append((phases[i - 1], start, i - 1))
                start = i

        segments.append((phases[-1], start, len(phases) - 1))

        return segments

    # --------------------------------------------------
    def _compute_future_events(self, phases, cur_idx):

        segments = self._extract_phase_segments(phases)

        cur_seg_id = None
        for i, (_, s, e) in enumerate(segments):
            if s <= cur_idx <= e:
                cur_seg_id = i
                break

        assert cur_seg_id is not None

        future_segments = segments[cur_seg_id + 1:]

        future_start = []
        future_end = []
        future_phase = []
        future_mask = []

        for pid, s, e in future_segments:

            # absolute timeline (seconds)
            start_t = s / self.fps_eff
            end_t   = e / self.fps_eff

            future_start.append(start_t)
            future_end.append(end_t)
            future_phase.append(pid)
            future_mask.append(1)

            if len(future_start) == self.max_future_events:
                break

        # padding
        while len(future_start) < self.max_future_events:
            future_start.append(0.)
            future_end.append(0.)
            future_phase.append(-1)
            future_mask.append(0)

        return future_start, future_end, future_phase, future_mask

    # --------------------------------------------------

    def _load_phase_file(self, path):

        phases = []

        with open(path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):

                if line_id == 0 and ("Frame" in line or "frame" in line):
                    continue

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                phases.append(PHASE2ID[parts[1]])

        return phases

    # --------------------------------------------------

    def _compute_remaining_time(self, phases):

        n = len(phases)
        remaining = [0.0] * n

        start = 0
        for i in range(1, n):
            if phases[i] != phases[i - 1]:
                end = i - 1
                for t in range(start, end + 1):
                    remaining[t] = (end - t) / self.fps_eff
                start = i

        end = n - 1
        for t in range(start, end + 1):
            remaining[t] = (end - t) / self.fps_eff

        return remaining

    # --------------------------------------------------

    def _list_frame_files(self, video_folder):

        if video_folder in self.frame_cache:
            return self.frame_cache[video_folder]

        files = [
            f for f in os.listdir(video_folder)
            if f.lower().endswith(self.img_exts)
        ]

        files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))

        full_paths = [os.path.join(video_folder, f) for f in files]

        self.frame_cache[video_folder] = full_paths

        return full_paths

    # --------------------------------------------------

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

            video_folder = os.path.join(self.frames_root, video_name)
            frame_paths = self._list_frame_files(video_folder)

            if len(frame_paths) < self.seq_len:
                continue

            phase_path = os.path.join(self.phase_dir, f"{video_name}-phase.txt")
            phases_full = self._load_phase_file(phase_path)

            # downsample phase labels
            phases_sampled = []
            for k in range(len(frame_paths)):
                idx = k * self.sample_every
                if idx >= len(phases_full):
                    break
                phases_sampled.append(phases_full[idx])

            usable_len = min(len(frame_paths), len(phases_sampled))

            frame_paths = frame_paths[:usable_len]
            phases_sampled = phases_sampled[:usable_len]

            remaining_sec = self._compute_remaining_time(phases_sampled)

            for start in range(0, usable_len - self.seq_len, self.stride):

                end_idx = start + self.seq_len - 1

                remain_time = remaining_sec[end_idx]

                future_start, future_end, future_phase, future_mask = \
                    self._compute_future_events(phases_sampled, end_idx)

                self.samples.append((
                    frame_paths,
                    start,
                    remain_time,
                    future_start,
                    future_end,
                    future_phase,
                    future_mask
                ))

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

        (
            frame_paths,
            start,
            remain_time,
            future_start,
            future_end,
            future_phase,
            future_mask
        ) = self.samples[idx]

        frames = self._read_clip(frame_paths, start)

        return (
            frames,
            torch.tensor(remain_time, dtype=torch.float32),

            torch.tensor(future_start, dtype=torch.float32),
            torch.tensor(future_end, dtype=torch.float32),

            torch.tensor(future_phase, dtype=torch.long),
            torch.tensor(future_mask, dtype=torch.float32),
        )
