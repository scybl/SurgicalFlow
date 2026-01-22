import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w=15):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="path to test_predictions.npz")
    ap.add_argument("--out", type=str, default="gt_vs_pred_like_paper.png")
    ap.add_argument("--title", type=str, default="Best Case: videoXX (MAE: ?? min)")
    ap.add_argument("--start", type=int, default=0, help="start index into arrays")
    ap.add_argument("--length", type=int, default=1200, help="how many points to plot")
    ap.add_argument("--to_min", action="store_true", help="convert seconds to minutes")
    ap.add_argument("--smooth", type=int, default=0, help="moving average window for prediction (0=no smooth)")
    args = ap.parse_args()

    data = np.load(args.npz)
    gt = data["gt_sec"].astype(np.float64)
    pred = data["pred_sec"].astype(np.float64)

    # slice
    s = max(0, args.start)
    e = min(len(gt), s + args.length)
    gt = gt[s:e]
    pred = pred[s:e]

    # optional smoothing (only for visual)
    pred_plot = moving_average(pred, args.smooth) if args.smooth > 1 else pred

    # MAE for this segment
    mae_sec = float(np.mean(np.abs(pred - gt)))
    mae_min = mae_sec / 60.0

    # unit conversion
    if args.to_min:
        gt_plot = gt / 60.0
        pred_plot = pred_plot / 60.0
        ylab = "Remaining Time (min)"
    else:
        gt_plot = gt
        pred_plot = pred_plot
        ylab = "Remaining Time (sec)"

    # x-axis as "Frame Sequence"-like index
    x = np.arange(s, e)

    plt.figure(figsize=(12, 5))
    plt.plot(x, gt_plot, label="Ground Truth", linewidth=2)
    plt.plot(x, pred_plot, label="Prediction", linewidth=1, alpha=0.85)

    # if user didn't fill MAE, we append it automatically
    title = args.title
    if "MAE" not in title:
        title = f"{title} (MAE: {mae_min:.2f} min)"
    else:
        # still useful to show computed MAE in console
        pass

    plt.title(title)
    plt.xlabel("Frame Sequence")
    plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved: {args.out}")
    print(f"Segment MAE: {mae_sec:.2f} sec = {mae_min:.2f} min")


if __name__ == "__main__":
    main()
