"""Verify, aggregate, and select representative samples from 1000-sample results."""

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path


CATEGORIES = ("cabinet", "car", "chair", "couch", "lamp", "plane", "table", "watercraft")
FIGURES = (
    "01_three_column_comparison.png",
    "02_prediction_error_heatmap.png",
    "03_zoom_in.png",
    "04_difference_cloud.png",
    "05_fscore_curve.png",
    "metrics.txt",
    "metrics.json",
)


def mean_std(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "count": len(values),
    }


def read_samples(root, category):
    category_root = root / category
    complete = sorted(category_root.glob("*/evaluation_complete.flag"))
    samples = []
    invalid = []
    for flag in complete:
        metrics_path = flag.parent / "metrics.json"
        if not metrics_path.is_file():
            invalid.append((flag.parent.name, "missing metrics.json"))
            continue
        try:
            with metrics_path.open(encoding="utf-8") as file:
                item = json.load(file)
            item["sample"] = flag.parent.name
            item["directory"] = str(flag.parent)
            samples.append(item)
        except Exception as error:
            invalid.append((flag.parent.name, str(error)))
    return samples, invalid


def value(item, *keys):
    current = item
    for key in keys:
        current = current[key]
    return float(current)


def summarize(samples):
    fields = {
        "partial_cd": ("partial_vs_gt", "cd"),
        "prediction_cd": ("prediction_vs_gt", "cd"),
        "cd_improvement_percent": ("cd_improvement_percent",),
        "prediction_rmse": ("prediction_vs_gt", "source_to_target_rmse"),
        "coverage_rmse": ("prediction_vs_gt", "target_to_source_rmse"),
        "approx_emd": ("emd_approx_rmse",),
        "f_score_auc": ("f_score_auc",),
    }
    result = {}
    for name, path in fields.items():
        result[name] = mean_std([value(item, *path) for item in samples])

    thresholds = {}
    for item in samples:
        for row in item.get("threshold_metrics", []):
            threshold = "%.6f" % float(row["threshold"])
            thresholds.setdefault(threshold, {"precision": [], "recall": [], "f_score": []})
            for metric in ("precision", "recall", "f_score"):
                thresholds[threshold][metric].append(float(row[metric]))
    result["threshold_metrics"] = {
        threshold: {metric: mean_std(values) for metric, values in values_by_metric.items()}
        for threshold, values_by_metric in thresholds.items()
    }
    return result


def choose_samples(samples):
    by_cd = lambda item: value(item, "prediction_vs_gt", "cd")
    by_coverage = lambda item: value(item, "prediction_vs_gt", "target_to_source_rmse")
    by_improvement = lambda item: value(item, "cd_improvement_percent")
    ordered_cd = sorted(samples, key=by_cd)
    ordered_coverage = sorted(samples, key=by_coverage, reverse=True)
    ordered_improvement = sorted(samples, key=by_improvement, reverse=True)
    return {
        "best_cd": ordered_cd[0],
        "worst_cd": ordered_cd[-1],
        "worst_coverage": ordered_coverage[0],
        "best_improvement": ordered_improvement[0],
        "median_cd": ordered_cd[len(ordered_cd) // 2],
    }


def copy_selected(selections, output):
    for category, selected in selections.items():
        for label, item in selected.items():
            destination = output / category / label / item["sample"]
            destination.mkdir(parents=True, exist_ok=True)
            source = Path(item["directory"])
            for figure in FIGURES:
                path = source / figure
                if path.is_file():
                    shutil.copy2(path, destination / figure)
            with (destination / "selection_reason.txt").open("w", encoding="utf-8") as file:
                file.write("类别：%s\n样例：%s\n选择类型：%s\n" % (category, item["sample"], label))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="评估结果根目录（包含各类别子目录）")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else root / "analysis_1000"
    output.mkdir(parents=True, exist_ok=True)

    all_samples = {}
    all_invalid = {}
    for category in CATEGORIES:
        all_samples[category], all_invalid[category] = read_samples(root, category)

    verification = {}
    for category in CATEGORIES:
        samples = all_samples[category]
        verification[category] = {
            "complete_samples": len(samples),
            "expected_samples": 1000,
            "is_complete": len(samples) == 1000 and not all_invalid[category],
            "invalid_samples": all_invalid[category],
        }

    with (output / "verification.json").open("w", encoding="utf-8") as file:
        json.dump(verification, file, ensure_ascii=False, indent=2)
    with (output / "verification.txt").open("w", encoding="utf-8") as file:
        file.write("1000样例结果完整性检查\n\n")
        for category in CATEGORIES:
            item = verification[category]
            status = "完整" if item["is_complete"] else "不完整"
            file.write("%-12s %4d/1000  %s\n" % (category, item["complete_samples"], status))
            for sample, reason in item["invalid_samples"]:
                file.write("  %s: %s\n" % (sample, reason))

    if not all(item["is_complete"] for item in verification.values()):
        print("完整性检查未通过，已写出：%s" % (output / "verification.txt"))
        for category, item in verification.items():
            print("%-12s %d/1000" % (category, item["complete_samples"]))
        raise SystemExit(2)

    summaries = {category: summarize(samples) for category, samples in all_samples.items()}
    with (output / "category_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

    rows = []
    for category in CATEGORIES:
        summary = summaries[category]
        rows.append({
            "category": category,
            "samples": 1000,
            "partial_cd_mean": summary["partial_cd"]["mean"],
            "prediction_cd_mean": summary["prediction_cd"]["mean"],
            "cd_improvement_mean_percent": summary["cd_improvement_percent"]["mean"],
            "coverage_rmse_mean": summary["coverage_rmse"]["mean"],
            "approx_emd_mean": summary["approx_emd"]["mean"],
            "f_score_auc_mean": summary["f_score_auc"]["mean"],
        })
    with (output / "category_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selections = {category: choose_samples(samples) for category, samples in all_samples.items()}
    selection_json = {
        category: {
            label: {
                "sample": item["sample"],
                "directory": item["directory"],
                "prediction_cd": value(item, "prediction_vs_gt", "cd"),
                "coverage_rmse": value(item, "prediction_vs_gt", "target_to_source_rmse"),
                "cd_improvement_percent": value(item, "cd_improvement_percent"),
            }
            for label, item in selected.items()
        }
        for category, selected in selections.items()
    }
    with (output / "selected_samples.json").open("w", encoding="utf-8") as file:
        json.dump(selection_json, file, ensure_ascii=False, indent=2)
    copy_selected(selections, output / "selected_samples")

    print("完整性检查通过：8 个类别均为 1000/1000")
    print("汇总结果：%s" % (output / "category_summary.csv"))
    print("特殊样例：%s" % (output / "selected_samples.json"))
    for category in CATEGORIES:
        selected = selection_json[category]
        print("%s: 缺失最明显=%s, 最差CD=%s, 最佳CD=%s" % (
            category, selected["worst_coverage"]["sample"], selected["worst_cd"]["sample"], selected["best_cd"]["sample"]))


if __name__ == "__main__":
    main()
