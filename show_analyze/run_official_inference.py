import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config_vipc import cfg
from models.EGIInet import EGIInet
from utils.ViPCdataloader import ViPCDataLoader
from utils.loss_utils import calc_cd, farthest_point_sample


def write_ply(path, points, color):
    points = np.asarray(points, dtype=np.float32)
    with open(path, "w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("element vertex %d\n" % len(points))
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point in points:
            file.write(
                "%.7f %.7f %.7f %d %d %d\n"
                % (point[0], point[1], point[2], color[0], color[1], color[2])
            )


def sample_for_plot(points, limit=4000):
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def save_sample(output_dir, category, index, view, partial_2048, gt_2048, prediction, cdl1, cdl2, f1):
    sample_dir = os.path.join(output_dir, category, "%04d" % index)
    os.makedirs(sample_dir, exist_ok=True)
    partial_np = partial_2048[0].cpu().numpy()
    gt_np = gt_2048[0].cpu().numpy()
    prediction_np = prediction[0].cpu().numpy()

    write_ply(os.path.join(sample_dir, "partial.ply"), partial_np, (40, 120, 255))
    write_ply(os.path.join(sample_dir, "gt.ply"), gt_np, (50, 190, 80))
    write_ply(os.path.join(sample_dir, "prediction.ply"), prediction_np, (230, 70, 60))

    image_np = view.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save(os.path.join(sample_dir, "view.png"))

    figure = plt.figure(figsize=(15, 5))
    point_sets = [
        (partial_np, "Partial input", (0.1, 0.4, 1.0)),
        (gt_np, "Ground truth", (0.1, 0.7, 0.2)),
        (prediction_np, "Prediction", (0.9, 0.1, 0.1)),
    ]
    for position, (points, title, color) in enumerate(point_sets, start=1):
        axis = figure.add_subplot(1, 3, position, projection="3d")
        points = sample_for_plot(points)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=[color], depthshade=False)
        axis.set_title(title)
        axis.set_box_aspect((1, 1, 1))
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(os.path.join(sample_dir, "comparison.png"), dpi=180)
    plt.close(figure)

    with open(os.path.join(sample_dir, "inference_metrics.txt"), "w", encoding="utf-8") as file:
        file.write("dataset_index=%d\n" % index)
        file.write("cd_l1_x1e3=%.6f\n" % (cdl1.mean().item() * 1e3))
        file.write("cd_l2_x1e3=%.6f\n" % (cdl2.mean().item() * 1e3))
        file.write("f1=%.6f\n" % f1.mean().item())
        file.write("partial_points=%d\n" % len(partial_np))
        file.write("gt_points=%d\n" % len(gt_np))
        file.write("prediction_points=%d\n" % len(prediction_np))


def main():
    categories = ("cabinet", "car", "chair", "couch", "lamp", "plane", "table", "watercraft")
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=categories + ("all",), default="plane")
    parser.add_argument("--weights", default=None, help="checkpoint file (official or self-trained); defaults to the selected category")
    parser.add_argument("--weights-dir", default=None, help="directory containing <category>-ckpt-best.pth (useful with --category all)")
    parser.add_argument("--test-list", default="./data_pick_list/test_list.txt")
    parser.add_argument("--index", type=int, default=None, help="save one sample")
    parser.add_argument("--count", type=int, default=12, help="save the first N samples")
    parser.add_argument("--all", action="store_true", help="save every sample")
    parser.add_argument("--output", default="./result/inference")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0")
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    selected_categories = categories if args.category == "all" else (args.category,)
    for category in selected_categories:
        weights = args.weights or os.path.join(
            args.weights_dir or "/mnt/d/Dataset/ShapeNetViPC/ShapeNetViPC/checkpoints_new",
            "%s-ckpt-best.pth" % category,
        )
        if not os.path.isfile(weights):
            raise FileNotFoundError("checkpoint not found: %s" % weights)
        dataset = ViPCDataLoader(
            args.test_list,
            data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status="test",
            view_align=False, category=category,
        )
        model = torch.nn.DataParallel(EGIInet()).to(device)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if args.index is not None:
            indices = [args.index]
        elif args.all:
            indices = list(range(len(dataset)))
        else:
            indices = list(range(min(max(args.count, 0), len(dataset))))
        for position, index in enumerate(indices, start=1):
            if index < 0 or index >= len(dataset):
                raise IndexError("sample index %d is outside [0, %d)" % (index, len(dataset)))
            view, gt_pc, part_pc = dataset[index]
            partial = part_pc.unsqueeze(0).to(device)
            gt = gt_pc.unsqueeze(0).to(device)
            with torch.no_grad():
                partial_n = farthest_point_sample(partial, cfg.DATASETS.SHAPENET.N_POINTS)
                gt_n = farthest_point_sample(gt, cfg.DATASETS.SHAPENET.N_POINTS)
                # 只使用预测点云，兼容旧版二项与当前三项模型返回值。
                prediction = model(partial_n, view.unsqueeze(0).to(device))[0]
                cdl1, cdl2, f1 = calc_cd(prediction, gt_n, calc_f1=True)
            save_sample(output_dir, category, index, view, partial_n, gt_n, prediction, cdl1, cdl2, f1)
            print("[%d/%d] %s/%04d" % (position, len(indices), category, index), flush=True)
        print("category=%s weights=%s samples=%d" % (category, weights, len(indices)))


if __name__ == "__main__":
    main()
