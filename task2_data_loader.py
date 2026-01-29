import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from torchvision import transforms

# -----------------------------
# Constants
# -----------------------------

FPS_ORI = 25
JUMP_FPS = 1

PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6
}

NUM_PHASES = len(PHASE2ID)
NUM_TOOLS = 7



def extract_frame_idx(name):
    """
    frame_000025.jpg -> 25
    """
    base = os.path.splitext(name)[0]
    return int(base.split("_")[-1])


def load_tool_annotation(txt_path):
    """
    Tool annotation format:
    Frame Grasper Bipolar Hook Scissors Clipper Irrigator SpecimenBag
    0     1       0       0    0        0       0         0
    ...

    Returns:
        dict {frame_idx: [7-dim tool vector]}
    """
    tool_map = {}

    with open(txt_path, "r") as f:
        header = f.readline()  # skip header

        for line in f:
            parts = line.strip().split()

            frame_idx = int(parts[0])
            tools = list(map(int, parts[1:]))

            tool_map[frame_idx] = tools

    return tool_map


class Cholec80TaskBDataset(Dataset):
    """
    Task B Dataset (Joint supervision):

    Input:
        image (1 FPS, every 25 frames)

    Target:
        joint_label [14] = [ tool(7) , phase_onehot(7) ]

    Output:
        img: Tensor [3,H,W]
        joint_label: Tensor [14]
    """

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

        assert mode in ["train", "val", "test"]

        self.root_dir = root_dir
        self.mode = mode
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform


        self.frames_root = os.path.join(root_dir, frames_dirname)
        self.phase_root = os.path.join(root_dir, phase_dirname)
        self.tool_root = os.path.join(root_dir, tool_dirname)

        self.samples = []

        self._build_index()

    def _build_index(self):

        assert os.path.isdir(self.frames_root), f"frames_root not found: {self.frames_root}"
        assert os.path.isdir(self.phase_root),  f"phase_root not found: {self.phase_root}"
        assert os.path.isdir(self.tool_root),   f"tool_root not found: {self.tool_root}"

        video_list = sorted(os.listdir(self.frames_root))
        if self.mode == "train":
            video_list = video_list[:40]
        if self.mode == "val":
            video_list = video_list[40:50]
        if self.mode == "test":
            video_list = video_list[50:]

        print("[DBG] videos under frames_root:", len(video_list))

        for vid in video_list:
            print(f"[DBG] video: {vid}")

            frame_dir = os.path.join(self.frames_root, vid)

            # 允许 phase/tool 文件名可能是 "video01.txt" 或 "01.txt" 或 "video01_phase.txt" 这种
            phase_file = os.path.join(self.phase_root, f"{vid}-phase.txt")
            tool_file  = os.path.join(self.tool_root,  f"{vid}-tool.txt")

            frame_names = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith((".png"))])[::JUMP_FPS]


            with open(phase_file, "r") as f:
                phase_lines = f.readlines()[1::FPS_ORI * JUMP_FPS]  # 去表头 + 抽样

            phase_ids = []

            for line in phase_lines:
                _, phase_name = line.strip().split("\t")
                phase_id = PHASE2ID[phase_name]
                phase_ids.append(phase_id)

            # ---------- 循环结束后再转 tensor ----------

            phase_ids = torch.tensor(phase_ids, dtype=torch.long)   # [N]

            num_classes = len(PHASE2ID)
            phase_onehot = torch.nn.functional.one_hot(
                phase_ids,
                num_classes=num_classes
            ).float()   # [N, 7]

            with open(tool_file, "r") as f:
                tool_lines = f.readlines()[1::JUMP_FPS]  # 去表头 + 抽样

            tool_labels = []

            for line in tool_lines:
                parts = line.strip().split()

                # 去掉第一列 frame_id
                tool_vals = parts[1:]   # 长度应为 7

                # 转 int
                tool_vals = [int(x) for x in tool_vals]

                tool_labels.append(tool_vals)

            # 一次性转 tensor
            tool_labels = torch.tensor(tool_labels, dtype=torch.float32)   # [N, 7]

            print("[DBG] tool tensor shape:", tool_labels.shape)

            if len(frame_names) != len(phase_onehot) or len(frame_names) != len(tool_labels):
                min_len = min(len(frame_names), len(phase_onehot), len(tool_labels))
                frame_names = frame_names[:min_len]
                phase_onehot = phase_onehot[:min_len]
                tool_labels = tool_labels[:min_len]
                            
            seq_len = self.seq_len # 一个 sample 的长度 
            stride = self.stride # 采样步长

            N = min_len

            for start in range(0, N - seq_len + 1, stride):

                end = start + seq_len
                anchor = end - 1   # 当前时刻 t

                # -------- frames --------
                frame_seq = [
                    os.path.join(frame_dir, frame_names[i])
                    for i in range(start, end)
                ]   # length = 8

                # -------- tool labels --------
                tool_window = tool_labels[start:end]   # [8,7]

                tool_past = (tool_window[:4].sum(dim=0) > 0).float()
                tool_curr = tool_window[anchor - start]
                tool_fut  = (tool_window[4:].sum(dim=0) > 0).float()

                tool_3c = torch.stack([tool_past, tool_curr, tool_fut], dim=1)  # [7,3]

                # -------- phase labels --------
                phase_window = phase_onehot[start:end]  # [8,7]

                phase_curr_id = phase_window[anchor - start].argmax()

                phase_3c = torch.zeros(NUM_PHASES, 3)

                # past
                phase_3c[: ,0] = (phase_window[:4].sum(dim=0) > 0).float()

                # current (one-hot)
                phase_3c[phase_curr_id, 1] = 1.0

                # future
                phase_3c[:,2] = (phase_window[4:].sum(dim=0) > 0).float()

                # -------- append sample --------
                self.samples.append({
                    "frames": frame_seq,    # list of 8 paths
                    "tool": tool_3c,        # [7,3]
                    "phase": phase_3c       # [7,3]
                })

        print(f"[TaskB] {self.mode} loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        img_path = sample["frames"]
        tool = sample["tool"]
        phase = sample["phase"]

        frames = []

        for p in img_path:
            img = Image.open(p).convert("RGB")

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [8,3,H,W]

        return frames, tool, phase



if __name__ == "__main__":

    dataset = Cholec80TaskBDataset(
        root_dir="data/cholec80",
        mode="train"
    )

    img, tool, phase = dataset[50]

    print("Image shape:", img.shape)
    print("Label shape:", phase.shape)
    print("Tool part:", tool.shape)
