import argparse
import os
from collections import defaultdict

from workflow_schema import ZERO_BASED_PHASE2ID as PHASE2ID


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
    if not phases:
        return tuple()

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Cholec80 phase-transition patterns."
    )
    parser.add_argument(
        "--phase_dir",
        default="data/cholec80/phase_annotations",
        help="Directory containing Cholec80 *-phase.txt files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving the distribution plot instead of showing it.",
    )
    return parser.parse_args()


def plot_pattern_distribution(pattern2videos, output=None):
    patterns = []
    counts = []

    for pattern, videos in pattern2videos.items():
        patterns.append(str(pattern))
        counts.append(len(videos))

    if not patterns:
        print("\nNo valid phase patterns found; skipping plot.")
        return False

    pairs = sorted(zip(patterns, counts), key=lambda x: x[1], reverse=True)
    patterns, counts = zip(*pairs)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    bars = plt.bar(range(len(counts)), counts)

    plt.xticks(
        ticks=range(len(patterns)),
        labels=patterns,
        rotation=30,
        fontsize=10
    )

    plt.xlabel("Phase Pattern")
    plt.ylabel("Number of Videos")
    plt.title("Distribution of Phase Transition Patterns (Cholec80)")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        print(f"\nSaved plot to {output}")
    else:
        plt.show()

    return True


def main():
    args = parse_args()
    video2pattern, pattern2videos = analyze_each_video(args.phase_dir)

    print("\n========== Per-video Phase Pattern ==========")

    for v, p in video2pattern.items():
        print(v, "->", p)

    print("\n========== Pattern Groups ==========")

    for p, vids in pattern2videos.items():
        print(p, ":", len(vids), "videos")
        print("   ", vids)

    plot_pattern_distribution(pattern2videos, args.output)


if __name__ == "__main__":
    main()
