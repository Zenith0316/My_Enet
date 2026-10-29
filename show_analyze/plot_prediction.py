"""Plot saved inference point clouds without running model inference or evaluation.

Use --input-dir for one sample, or --input-root for category/sample batches.
The sibling evaluate_prediction.py supplies the original plotting functions.
"""

import argparse
from pathlib import Path

import numpy as np
import evaluate_prediction as evaluation


DEFAULT_THRESHOLDS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]


def plot_sample(sample, output, thresholds, difference_threshold):
    partial, gt, prediction = [
        evaluation.read_points(sample / name)
        for name in ("partial.ply", "gt.ply", "prediction.ply")
    ]
    output.mkdir(parents=True, exist_ok=True)
    view = sample / "view.png"
    evaluation.save_comparison(
        partial, gt, prediction, output,
        view if view.is_file() else None, difference_threshold,
    )

    # Compute only the distances needed for drawing the original F-score curve.
    # Reuse them at all 100 thresholds rather than rebuilding nearest-neighbor trees.
    pred_dist = evaluation.nearest_distances(prediction, gt)
    gt_dist = evaluation.nearest_distances(gt, prediction)

    def f_score(threshold):
        precision = float(np.mean(pred_dist <= threshold))
        recall = float(np.mean(gt_dist <= threshold))
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    dense = np.linspace(min(thresholds), max(thresholds), 100)
    plt = evaluation.plt
    figure = plt.figure(figsize=(7, 5))
    plt.plot(dense, [f_score(t) for t in dense], linewidth=2, color="#8e44ad")
    plt.scatter(thresholds, [f_score(t) for t in thresholds], color="#e74c3c", zorder=3)
    plt.xlabel("距离阈值")
    plt.ylabel("F值")
    plt.title("F值-阈值曲线")
    plt.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "05_fscore_curve.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path, help="单样本文件夹，包含 partial.ply、gt.ply、prediction.ply，可含 view.png")
    inputs.add_argument("--input-root", type=Path, help="批量推理目录：root/category/sample")
    parser.add_argument("--output", type=Path, required=True, help="图片输出目录")
    parser.add_argument("--category", help="批量模式指定类别")
    parser.add_argument("--sample", help="批量模式指定样本编号，如 0000")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--difference-threshold", type=float, default=0.01)
    args = parser.parse_args()
    if args.input_dir and (args.category or args.sample):
        parser.error("--category 和 --sample 仅用于 --input-root 模式")
    if not all(np.isfinite(t) and t >= 0 for t in args.thresholds):
        parser.error("--thresholds 必须为有限非负数")
    if not np.isfinite(args.difference_threshold) or args.difference_threshold < 0:
        parser.error("--difference-threshold 必须为有限非负数")

    jobs = []
    if args.input_dir:
        jobs.append((args.input_dir.resolve(), args.output.resolve()))
    else:
        root = args.input_root.resolve()
        if not root.is_dir():
            parser.error("推理目录不存在：%s" % root)
        categories = [root / args.category] if args.category else sorted(p for p in root.iterdir() if p.is_dir())
        for category in categories:
            if not category.is_dir():
                parser.error("类别目录不存在：%s" % category)
            samples = [category / args.sample] if args.sample else sorted(p for p in category.iterdir() if p.is_dir())
            for sample in samples:
                required = [sample / name for name in ("partial.ply", "gt.ply", "prediction.ply")]
                if not all(p.is_file() for p in required):
                    if args.sample:
                        parser.error("指定样本缺少点云文件：%s" % sample)
                    print("skip incomplete sample: %s" % sample)
                    continue
                jobs.append((sample, args.output.resolve() / category.name / sample.name))
    if not jobs:
        parser.error("未找到可绘制的样本")
    evaluation.configure_plotting()
    for index, (sample, output) in enumerate(jobs, 1):
        plot_sample(sample, output, args.thresholds, args.difference_threshold)
        print("[%d/%d] Plotted %s -> %s" % (index, len(jobs), sample, output), flush=True)


if __name__ == "__main__":
    main()
