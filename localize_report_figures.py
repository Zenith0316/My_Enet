"""Regenerate report charts with Chinese labels without replacing source figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from show_analyze.evaluate_prediction import read_points, save_comparison, configure_plotting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="已生成 metrics.json 的评估结果目录")
    parser.add_argument("--difference-threshold", type=float, default=0.01)
    args = parser.parse_args()
    configure_plotting()
    root = Path(args.results).resolve()
    count = 0

    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        sample_dir = metrics_path.parent
        category = sample_dir.parent.name
        with metrics_path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        files = metrics["files"]
        partial = read_points(files["partial"])
        gt = read_points(files["gt"])
        prediction = read_points(files["prediction"])
        output = sample_dir / "report_figures_zh"
        output.mkdir(parents=True, exist_ok=True)
        save_comparison(partial, gt, prediction, output, difference_threshold=args.difference_threshold)

        curve = metrics.get("threshold_metrics", [])
        if curve:
            low = min(row["threshold"] for row in curve)
            high = max(row["threshold"] for row in curve)
            dense = np.linspace(low, high, 100)
            pred_dist = np.linalg.norm(prediction[:, None, :] - gt[None, :, :], axis=2)
            gt_dist = pred_dist.T
            precision = np.asarray([(pred_dist.min(axis=1) <= tau).mean() for tau in dense])
            recall = np.asarray([(gt_dist.min(axis=1) <= tau).mean() for tau in dense])
            fscore = np.divide(2 * precision * recall, precision + recall,
                               out=np.zeros_like(precision), where=(precision + recall) > 0)
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.plot(dense, fscore, linewidth=2, color="#8e44ad")
            axis.scatter([row["threshold"] for row in curve], [row["f_score"] for row in curve],
                         color="#e74c3c", zorder=3)
            axis.set_xlabel("距离阈值")
            axis.set_ylabel("F值")
            axis.set_title("F值-阈值曲线")
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(output / "05_fscore_curve.png", dpi=220)
            plt.close(figure)
        print("中文图表已生成：%s / %s" % (category, sample_dir.name))
        count += 1

    if count == 0:
        raise FileNotFoundError("No metrics.json found under %s/*/*/metrics.json" % root)
    print("共生成 %d 组中文报告图，原始评估图片未覆盖。" % count)


if __name__ == "__main__":
    main()
