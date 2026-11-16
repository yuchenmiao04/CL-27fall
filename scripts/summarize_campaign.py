#!/usr/bin/env python3
"""Summarize explicitly selected, complete CL27 main experiments.

Only Python's standard library is needed. This tool never ranks candidate runs
by their test scores. A duplicate method/dataset requires an explicit selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


DATASETS = {"cifar224": ("CIFAR-100", 100, 10), "imagenetr": ("ImageNet-R", 200, 20)}
METHODS = (
    "shared_basis",
    "history_transport",
    "current_adapted",
    "plastic_shared_basis",
    "er",
    "joint",
)
UPSTREAM = "7a6e904c5bc5cb7a4e1823b3434020be27469b63"
METRICS = (
    "final_accuracy",
    "aia",
    "forgetting_previous",
    "forgetting_including_final",
)
NONFORMAL_PARTS = {"scripts", "smoke", "validation", "validation_only", "development", "dev"}


class SummaryError(ValueError):
    pass


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError) as exc:
        raise SummaryError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"Expected a JSON object: {path}")
    return value


def scalar(value, name, *, accuracy=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SummaryError(f"{name} must be finite and numeric")
    if accuracy and not 0 <= value <= 1:
        raise SummaryError(f"{name} must be a fraction in [0, 1]")
    return float(value)


def same(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise SummaryError(f"{name} differs from its complete stage records")


def validate_complete(config, result):
    """Reject complete-looking partial runs and incompatible main protocols."""
    dataset, method = config["dataset"], config["model_name"]
    _, classes, width = DATASETS[dataset]
    if result.get("method") != method:
        raise SummaryError("Result method and config model_name disagree")
    if config.get("init_cls") != width or config.get("increment") != width:
        raise SummaryError("Main experiments require the selected ten-stage class split")
    if config.get("backbone_type") != "vit_base_patch16_224" or config.get("pretrained") is not True:
        raise SummaryError("Main experiments require the selected pretrained ViT-B/16")
    if config.get("cl27_protocol") != "pilot_figure2" or result.get("upstream_commit") != UPSTREAM:
        raise SummaryError("Pinned PILOT protocol or upstream commit differs")
    if config.get("evaluation_head", "cnn") != "cnn" or result.get("evaluation_head") != "cnn":
        raise SummaryError("The six main methods require their preselected CNN head")
    if method == "shared_basis":
        if config.get("memory_size", 0) != 0 or config.get("memory_per_class", 0) != 0:
            raise SummaryError("Frozen shared basis must have zero image replay")
    elif method != "joint":
        if dataset == "cifar224":
            if config.get("fixed_memory") is not False or config.get("memory_size") != 2000:
                raise SummaryError("CIFAR-100 replay must have a fixed total of 2000")
        elif config.get("fixed_memory") is not True or config.get("memory_per_class") != 20:
            raise SummaryError("ImageNet-R replay must have 20 images per class")
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise SummaryError("Missing structured metrics")
    values = {"final_accuracy": scalar(metrics.get("final_accuracy"), "Final", accuracy=True)}
    stages = result.get("stages", [])
    if not isinstance(stages, list):
        raise SummaryError("Missing structured stages")
    if method == "joint":
        if (
            result.get("evaluation_mode") != "offline_joint_final"
            or result.get("selected_classes") != classes
            or result.get("dataset_classes") != classes
            or result.get("completed_fits") != 1
            or metrics.get("completed_fits") != 1
            or len(stages) != 1
            or stages[0].get("seen_classes") != classes
        ):
            raise SummaryError("Joint must be one completed full-class offline fit")
        if any(metrics.get(key) is not None for key in METRICS[1:]):
            raise SummaryError("Joint AIA and forgetting must be null, never zero or Final")
        same(values["final_accuracy"], scalar(stages[0].get("accuracy"), "Joint stage Final", accuracy=True), "Final")
        return {**values, **dict.fromkeys(METRICS[1:])}
    if metrics.get("completed_tasks") != 10 or len(stages) != 10:
        raise SummaryError("Continual main experiments require all ten completed stages")
    accuracies, matrix = [], []
    for task, stage in enumerate(stages):
        if (
            stage.get("task") != task
            or stage.get("seen_classes") != (task + 1) * width
            or stage.get("prediction_space") != "all_seen_classes_without_task_id"
        ):
            raise SummaryError(f"Stage {task} is not the selected all-seen Class-IL evaluation")
        row, counts = stage.get("task_accuracy", []), stage.get("task_counts", [])
        if len(row) != 10 or len(counts) != 10:
            raise SummaryError("Each stage must retain ten task accuracies and sample counts")
        for j in range(10):
            if not isinstance(counts[j], int) or isinstance(counts[j], bool):
                raise SummaryError("Task counts must be integers")
            if j <= task:
                scalar(row[j], "task accuracy", accuracy=True)
                if counts[j] <= 0:
                    raise SummaryError("Seen tasks must have positive evaluation sample counts")
            elif counts[j] != 0 or row[j] is not None:
                raise SummaryError("Future tasks must be absent from evaluation")
        accuracy = scalar(stage.get("accuracy"), "stage accuracy", accuracy=True)
        weighted = sum(row[j] * counts[j] for j in range(task + 1)) / sum(counts)
        same(accuracy, weighted, f"Stage {task} accuracy")
        accuracies.append(accuracy)
        matrix.append(row)
    derived = {
        "final_accuracy": accuracies[-1],
        "aia": sum(accuracies) / 10,
        "forgetting_previous": sum(max(matrix[t][j] for t in range(j, 9)) - matrix[9][j] for j in range(9)) / 9,
        "forgetting_including_final": sum(max(matrix[t][j] for t in range(j, 10)) - matrix[9][j] for j in range(9)) / 9,
    }
    for key in METRICS:
        values[key] = scalar(metrics.get(key), key, accuracy=key in {"final_accuracy", "aia"})
        same(values[key], derived[key], key)
    return values


def inspect_run(result_path, root):
    relative = result_path.relative_to(root)
    if NONFORMAL_PARTS.intersection(part.lower() for part in relative.parts[:-1]):
        return None, "nonformal directory"
    config_path = result_path.with_name("config.json")
    if not config_path.is_file():
        return None, "no sibling config.json"
    config, result = read_json(config_path), read_json(result_path)
    dataset, method = config.get("dataset"), config.get("model_name")
    if dataset not in DATASETS or method not in METHODS:
        return None, "outside the two-dataset/six-method scope"
    seed = config.get("seed")
    if seed != 1993 and seed != [1993]:
        return None, "seed is not 1993"
    if result.get("status") != "complete" or result.get("phase") != "final" or config.get("phase") != "final":
        return None, "not complete final-phase output"
    if (
        config.get("validation_only")
        or config.get("smoke")
        or config.get("max_tasks") is not None
        or config.get("joint_max_classes") is not None
        or result.get("performance_claim") is False
    ):
        return None, "validation/smoke/truncated run"
    try:
        metrics = validate_complete(config, result)
    except SummaryError as exc:
        raise SummaryError(f"Invalid complete final result {result_path}: {exc}") from exc
    final_diagnostic = (result.get("task_diagnostics") or [{}])[-1]
    resources = {
        "execution": result.get("execution", {}),
        "final_task_resources": {
            key: value for key, value in final_diagnostic.items()
            if key == "resources" or "bytes" in key or key in {"replay_images", "raw_replay_images"}
        },
        "configured_replay": {key: config.get(key) for key in ("memory_size", "memory_per_class", "fixed_memory")},
    }
    if method == "joint":
        permission = "offline / all training data"
    elif method == "shared_basis":
        permission = "0"
    else:
        permission = "2000 total" if dataset == "cifar224" else "20/class (4000 final)"
    return {
        "dataset": dataset,
        "method": method,
        "seed": 1993,
        "metrics": metrics,
        "data_permission": permission,
        "resources": resources,
        "source": {
            "directory": str(result_path.parent),
            "results": str(result_path),
            "config": str(config_path),
            "data_manifest": str(result_path.with_name("data_manifest.json")) if result_path.with_name("data_manifest.json").is_file() else None,
            "checkpoint": str(result_path.with_name("checkpoint.pt")) if result_path.with_name("checkpoint.pt").is_file() else None,
            "config_fingerprint": result.get("config_fingerprint"),
            "data_fingerprint": result.get("data_fingerprint"),
            "upstream_commit": result.get("upstream_commit"),
            "backbone_provenance": result.get("backbone_provenance"),
        },
    }, None


def summarize_campaign(runs_root, selection_path=None):
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise SummaryError(f"Runs root is not a directory: {root}")
    selection = read_json(selection_path) if selection_path else {}
    if set(selection) - set(DATASETS):
        raise SummaryError("Selection contains an unknown dataset")
    for dataset, methods in selection.items():
        if not isinstance(methods, dict) or set(methods) - set(METHODS):
            raise SummaryError(f"Invalid method selection for {dataset}")
    candidates = {(d, m): [] for d in DATASETS for m in METHODS}
    skipped = []
    for path in sorted(root.rglob("results.json")):
        record, reason = inspect_run(path, root)
        if record is None:
            skipped.append({"path": str(path), "reason": reason})
        else:
            candidates[record["dataset"], record["method"]].append(record)
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            matches = candidates[dataset, method]
            chosen = selection.get(dataset, {}).get(method)
            if chosen is not None:
                if not isinstance(chosen, str) or not chosen:
                    raise SummaryError(f"Selection path for {dataset}/{method} must be a nonempty string")
                path = Path(chosen)
                path = (root / path).resolve() if not path.is_absolute() else path.resolve()
                if path.name == "results.json":
                    path = path.parent
                matches = [record for record in matches if Path(record["source"]["directory"]) == path]
                if len(matches) != 1:
                    raise SummaryError(f"Explicit selection for {dataset}/{method} is not one eligible final run: {chosen}")
            elif len(matches) > 1:
                paths = ", ".join(record["source"]["directory"] for record in matches)
                raise SummaryError(f"Duplicate final runs for {dataset}/{method}; provide --selection instead of choosing by test score: {paths}")
            rows.append(matches[0] if matches else {"dataset": dataset, "method": method, "status": "missing", "metrics": dict.fromkeys(METRICS)})
    return {
        "schema_version": 1,
        "runs_root": str(root),
        "selection_file": str(Path(selection_path).resolve()) if selection_path else None,
        "selection_policy": "explicit path for duplicates; never optimize selection against reported scores",
        "seed": 1993,
        "expected_results": 12,
        "available_results": sum(row.get("status") != "missing" for row in rows),
        "metric_units": "fractions; Markdown multiplies accuracy by 100 and forgetting by 100 percentage points",
        "rows": rows,
        "eligible_candidates": {f"{d}/{m}": [r["source"]["directory"] for r in values] for (d, m), values in candidates.items()},
        "skipped_results": skipped,
    }


def markdown(summary):
    lines = [
        "# Campaign main results", "",
        f"Seed 1993; {summary['available_results']}/12 eligible final results. Missing results are —.",
        "Accuracy: %. Forgetting: percentage points. Joint is an offline reference; its AIA and forgetting are not applicable.",
        "Selections use explicit run paths when duplicates exist, never the highest reported test score.", "",
    ]
    for dataset, (title, _, _) in DATASETS.items():
        lines += [f"## {title}", "", "| Method | Historical image permission | Final ↑ | AIA ↑ | F_previous ↓ | F_including_final ↓ |", "| --- | --- | --- | --- | --- | --- |"]
        for row in summary["rows"]:
            if row["dataset"] != dataset:
                continue
            values = ["—" if row["metrics"][key] is None else f"{100 * row['metrics'][key]:.2f}" for key in METRICS]
            permission = row.get("data_permission", "—")
            lines.append("| " + " | ".join([row["method"], permission, *values]) + " |")
        lines.append("")
    lines += ["F_previous excludes the final evaluation when taking each old task's historical maximum; F_including_final includes it. Raw metrics, resource records, selected sources and excluded-run reasons are retained in summary.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--selection", type=Path, help="JSON: {dataset: {method: run directory relative to runs-root, or absolute path}}")
    parser.add_argument("--output-dir", type=Path, help="Default: runs-root/summary")
    args = parser.parse_args()
    try:
        summary = summarize_campaign(args.runs_root, args.selection)
    except SummaryError as exc:
        parser.exit(2, f"Error: {exc}\n")
    output = args.output_dir or args.runs_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(markdown(summary))
    print(f"{summary['available_results']}/12 final results; wrote {output / 'summary.md'} and {output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
