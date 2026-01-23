import os
from collections import defaultdict

PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6
}


def load_phase_file(path):

    phases = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):

            if line_id == 0 and ("Frame" in line or "frame" in line):
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            name = parts[1]

            if name not in PHASE2ID:
                continue

            phases.append(PHASE2ID[name])

    return phases


def compress_sequence(phases):
    """
    去除连续重复：
    [0,0,0,1,1,2,2] -> [0,1,2]
    """

    compressed = [phases[0]]

    for p in phases[1:]:
        if p != compressed[-1]:
            compressed.append(p)

    return tuple(compressed)


def analyze_each_video(phase_dir):

    video2pattern = {}
    pattern2videos = defaultdict(list)

    files = sorted([f for f in os.listdir(phase_dir) if f.endswith(".txt")])

    for fname in files:

        path = os.path.join(phase_dir, fname)

        phases = load_phase_file(path)

        pattern = compress_sequence(phases)

        video_name = fname.replace("-phase.txt", "")

        video2pattern[video_name] = pattern
        pattern2videos[pattern].append(video_name)

    return video2pattern, pattern2videos


if __name__ == "__main__":

    phase_annotation_dir = "data/cholec80/phase_annotations"   # 改成你的路径

    video2pattern, pattern2videos = analyze_each_video(phase_annotation_dir)

    print("\n========== Per-video Phase Pattern ==========")

    for v, p in video2pattern.items():
        print(v, "->", p)

    print("\n========== Pattern Groups ==========")

    for p, vids in pattern2videos.items():
        print(p, ":", len(vids), "videos")
        print("   ", vids)