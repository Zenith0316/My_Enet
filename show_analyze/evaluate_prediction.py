"""Evaluate point clouds; optionally generate visualizations with --plot.

Examples:
    python evaluate_prediction.py --input-root ./result/inference
    python evaluate_prediction.py --input-root ./result/inference --category plane --sample 0000
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def configure_plotting():
    # Keep metrics-only runs free of matplotlib/Pillow imports and font setup.
    global plt, Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib import font_manager

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for font_path in (
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if os.path.isfile(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break


def read_points(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix in (".txt", ".xyz", ".pts", ".csv", ".dat"):
        points = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".ply":
        points = read_ascii_ply(path)
    elif suffix == ".pcd":
        points = read_ascii_pcd(path)
    else:
        raise ValueError("Unsupported point-cloud format: %s" % path)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 3:
        if points.shape[0] != 1:
            raise ValueError("Expected one point cloud, got shape %s" % (points.shape,))
        points = points[0]
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must have shape [N, 3+] but got %s" % (points.shape,))
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty: %s" % path)
    return points


def read_ascii_ply(path):
    with open(path, "r", encoding="ascii", errors="ignore") as file:
        line = file.readline().strip()
        if line != "ply":
            raise ValueError("Only ASCII PLY is supported: %s" % path)
        vertex_count = None
        format_name = None
        while True:
            line = file.readline().strip()
            if not line:
                raise ValueError("Invalid PLY header: %s" % path)
            if line.startswith("format"):
                format_name = line.split()[1]
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            if line == "end_header":
                break
        if format_name != "ascii" or vertex_count is None:
            raise ValueError("Only ASCII PLY with vertex data is supported: %s" % path)
        rows = [file.readline().split() for _ in range(vertex_count)]
    return np.asarray([[float(row[0]), float(row[1]), float(row[2])] for row in rows])


def read_ascii_pcd(path):
    with open(path, "r", encoding="ascii", errors="ignore") as file:
        lines = file.readlines()
    data_start = next((i for i, line in enumerate(lines) if line.strip().upper() == "DATA ASCII"), None)
    if data_start is None:
        raise ValueError("Only ASCII PCD is supported: %s" % path)
    rows = [line.split() for line in lines[data_start + 1:] if line.strip()]
    return np.asarray([[float(row[0]), float(row[1]), float(row[2])] for row in rows])


def nearest_distances(source, target):
    distances, _ = cKDTree(target).query(source, k=1)
    return distances


def cd_metrics(source, target):
    source_to_target = nearest_distances(source, target)
    target_to_source = nearest_distances(target, source)
    return {
        "source_to_target_mean": float(np.mean(source_to_target ** 2)),
        "target_to_source_mean": float(np.mean(target_to_source ** 2)),
        "cd": float(np.mean(source_to_target ** 2) + np.mean(target_to_source ** 2)),
        "source_to_target_rmse": float(np.sqrt(np.mean(source_to_target ** 2))),
        "target_to_source_rmse": float(np.sqrt(np.mean(target_to_source ** 2))),
    }


def sample_points(points, count, seed):
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), count, replace=False)]


def emd_approx(source, target, count=256, seed=0):
    source = sample_points(source, count, seed)
    target = sample_points(target, count, seed + 1)
    cost = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols]).mean()), len(rows)


def threshold_metrics(prediction, gt, threshold):
    pred_dist = nearest_distances(prediction, gt)
    gt_dist = nearest_distances(gt, prediction)
    precision = float(np.mean(pred_dist <= threshold))
    recall = float(np.mean(gt_dist <= threshold))
    f_score = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def set_equal_axes(axis, points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max((points.max(axis=0) - points.min(axis=0)).max() / 2.0, 1e-8)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()


def draw_cloud(axis, points, color, title, all_points):
    limit = min(len(points), 5000)
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    axis.scatter(points[indices, 0], points[indices, 1], points[indices, 2], s=1.2, c=color, depthshade=False)
    axis.set_title(title)
    set_equal_axes(axis, all_points)


def save_comparison(partial, gt, prediction, output, view=None, difference_threshold=0.01):
    all_points = np.concatenate([partial, gt, prediction], axis=0)
    figure = plt.figure(figsize=(16, 5))
    draw_cloud(figure.add_subplot(1, 3, 1, projection="3d"), partial, "#2f80ed", "初始点云", all_points)
    draw_cloud(figure.add_subplot(1, 3, 2, projection="3d"), gt, "#27ae60", "真值点云", all_points)
    draw_cloud(figure.add_subplot(1, 3, 3, projection="3d"), prediction, "#e74c3c", "补全预测", all_points)
    figure.tight_layout()
    figure.savefig(output / "01_three_column_comparison.png", dpi=220)
    plt.close(figure)

    pred_error = nearest_distances(prediction, gt)
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    limit = min(len(prediction), 8000)
    indices = np.linspace(0, len(prediction) - 1, limit, dtype=np.int64)
    scatter = axis.scatter(prediction[indices, 0], prediction[indices, 1], prediction[indices, 2],
                           c=pred_error[indices], cmap="turbo", s=2, depthshade=False)
    figure.colorbar(scatter, ax=axis, shrink=0.7, label="到真值最近邻距离")
    axis.set_title("补全误差热力图")
    set_equal_axes(axis, all_points)
    figure.tight_layout()
    figure.savefig(output / "02_prediction_error_heatmap.png", dpi=220)
    plt.close(figure)

    gt_error = nearest_distances(gt, prediction)
    combined_error = np.concatenate([pred_error, gt_error])
    zoom_radius = np.quantile(combined_error, 0.85)
    zoom_points = np.concatenate([prediction[pred_error >= zoom_radius], gt[gt_error >= zoom_radius]], axis=0)
    if len(zoom_points) == 0:
        zoom_points = prediction
    center = zoom_points.mean(axis=0)
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    pred_zoom = np.linalg.norm(prediction - center, axis=1) <= np.linalg.norm(zoom_points - center, axis=1).max()
    gt_zoom = np.linalg.norm(gt - center, axis=1) <= np.linalg.norm(zoom_points - center, axis=1).max()
    axis.scatter(gt[gt_zoom, 0], gt[gt_zoom, 1], gt[gt_zoom, 2], c="#27ae60", s=5, label="真值点", depthshade=False)
    axis.scatter(prediction[pred_zoom, 0], prediction[pred_zoom, 1], prediction[pred_zoom, 2], c="#e74c3c", s=5, label="预测点", depthshade=False)
    axis.set_title("高误差区域局部放大")
    set_equal_axes(axis, zoom_points)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "03_zoom_in.png", dpi=220)
    plt.close(figure)

    fp = prediction[pred_error > difference_threshold]
    missed = gt[gt_error > difference_threshold]
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    if len(fp):
        axis.scatter(fp[:, 0], fp[:, 1], fp[:, 2], c="#e74c3c", s=4, label="预测偏差")
    if len(missed):
        axis.scatter(missed[:, 0], missed[:, 1], missed[:, 2], c="#3498db", s=4, label="真值未覆盖区域")
    axis.set_title("差异点云，距离阈值 %.4f" % difference_threshold)
    set_equal_axes(axis, all_points)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "04_difference_cloud.png", dpi=220)
    plt.close(figure)

    if view is not None:
        with Image.open(view) as image:
            image.convert("RGB").save(output / "00_input_view.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", help="推理根目录，结构为 root/category/0000/{partial,gt,prediction,view}")
    parser.add_argument("--partial")
    parser.add_argument("--gt")
    parser.add_argument("--pred")
    parser.add_argument("--view", default=None)
    parser.add_argument("--plot", action="store_true", help="生成评估图片；默认只计算并保存指标，不绘图")
    parser.add_argument("--output", help="单样本评估输出目录；批量模式下默认为 input-root 的同级 evaluation 目录")
    parser.add_argument("--category", default=None, help="批量模式只评估一个类别")
    parser.add_argument("--sample", default=None, help="批量模式只评估一个样本编号")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01])
    parser.add_argument("--emd-points", type=int, default=256)
    parser.add_argument("--difference-threshold", type=float, default=0.01)
    args = parser.parse_args()

    if args.input_root:
        input_root = Path(args.input_root).resolve()
        output_root = Path(args.output).resolve() if args.output else input_root.parent / (input_root.name + "_evaluation")
        categories = [args.category] if args.category else sorted(p.name for p in input_root.iterdir() if p.is_dir())
        total = 0
        for category in categories:
            category_dir = input_root / category
            if not category_dir.is_dir():
                raise FileNotFoundError("category directory not found: %s" % category_dir)
            samples = [p for p in sorted(category_dir.iterdir()) if p.is_dir()]
            if args.sample:
                samples = [category_dir / args.sample]
            for sample_dir in samples:
                partial = sample_dir / "partial.ply"
                gt = sample_dir / "gt.ply"
                pred = sample_dir / "prediction.ply"
                view = sample_dir / "view.png"
                if not all(path.is_file() for path in (partial, gt, pred)):
                    print("skip incomplete sample: %s" % sample_dir)
                    continue
                output = output_root / category / sample_dir.name
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--partial", str(partial), "--gt", str(gt), "--pred", str(pred),
                           "--output", str(output),
                           "--emd-points", str(args.emd_points),
                           "--difference-threshold", str(args.difference_threshold),
                           "--thresholds", *map(str, args.thresholds)]
                if args.plot:
                    command += ["--plot"]
                if view.is_file():
                    command += ["--view", str(view)]
                subprocess.run(command, check=True)
                (output / "evaluation_complete.flag").touch()
                total += 1
        print("evaluated %d sample(s) -> %s" % (total, output_root))
        return

    missing = [name for name, value in (("--partial", args.partial), ("--gt", args.gt),
                                        ("--pred", args.pred), ("--output", args.output)) if not value]
    if missing:
        parser.error("单样本模式缺少参数: %s；或使用 --input-root 批量评估" % ", ".join(missing))

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    partial, gt, prediction = read_points(args.partial), read_points(args.gt), read_points(args.pred)

    partial_cd = cd_metrics(partial, gt)
    prediction_cd = cd_metrics(prediction, gt)
    emd, emd_points = emd_approx(prediction, gt, args.emd_points)
    curve = [threshold_metrics(prediction, gt, threshold) for threshold in args.thresholds]
    thresholds = np.asarray([item["threshold"] for item in curve])
    f_scores = np.asarray([item["f_score"] for item in curve])
    auc = float(np.trapz(f_scores, thresholds))
    improvement = 100.0 * (partial_cd["cd"] - prediction_cd["cd"]) / max(partial_cd["cd"], 1e-12)

    metrics = {
        "files": {"partial": str(Path(args.partial).resolve()), "gt": str(Path(args.gt).resolve()), "prediction": str(Path(args.pred).resolve())},
        "point_counts": {"partial": len(partial), "gt": len(gt), "prediction": len(prediction)},
        "partial_vs_gt": partial_cd,
        "prediction_vs_gt": prediction_cd,
        "cd_improvement_percent": float(improvement),
        "emd_approx_rmse": emd,
        "emd_sample_points": emd_points,
        "f_score_auc": auc,
        "threshold_metrics": curve,
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with open(output / "metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["threshold", "precision", "recall", "f_score"])
        for item in curve:
            writer.writerow([item["threshold"], item["precision"], item["recall"], item["f_score"]])
    with open(output / "metrics.txt", "w", encoding="utf-8") as file:
        file.write("Partial CD: %.8f\n" % partial_cd["cd"])
        file.write("Prediction CD: %.8f\n" % prediction_cd["cd"])
        file.write("CD improvement: %.3f%%\n" % improvement)
        file.write("Approximate EMD RMSE: %.8f (%d sampled points)\n" % (emd, emd_points))
        file.write("F-score AUC: %.8f\n\n" % auc)
        file.write("threshold\tprecision\trecall\tf_score\n")
        for item in curve:
            file.write("%.6f\t%.6f\t%.6f\t%.6f\n" % (item["threshold"], item["precision"], item["recall"], item["f_score"]))

    if args.plot:
        configure_plotting()
        save_comparison(partial, gt, prediction, output, args.view, args.difference_threshold)
        thresholds_dense = np.linspace(min(args.thresholds), max(args.thresholds), 100)
        dense_curve = [threshold_metrics(prediction, gt, threshold)["f_score"] for threshold in thresholds_dense]
        figure = plt.figure(figsize=(7, 5))
        plt.plot(thresholds_dense, dense_curve, linewidth=2, color="#8e44ad")
        plt.scatter(thresholds, f_scores, color="#e74c3c", zorder=3)
        plt.xlabel("距离阈值")
        plt.ylabel("F值")
        plt.title("F值-阈值曲线")
        plt.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "05_fscore_curve.png", dpi=220)
        plt.close(figure)
    print("Evaluation complete: %s" % output)
    print("Partial CD: %.8f" % partial_cd["cd"])
    print("Prediction CD: %.8f" % prediction_cd["cd"])
    print("CD improvement: %.3f%%" % improvement)
    print("Approximate EMD RMSE: %.8f" % emd)
    print("F-score AUC: %.8f" % auc)


if __name__ == "__main__":
    main()
