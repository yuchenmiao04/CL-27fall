#!/usr/bin/env python3
"""Short native pretrained-ViT pipeline verification; never a full benchmark.

Default workload per method: two CIFAR-100 class tasks, ten classes/task,
three fitting images/class plus three train-holdout images/class, one epoch.
The complete official training pickle supplies the split; the official test
pickle is never opened. No dataset download is attempted. PILOT/timm may fetch
the selected public pretrained weight if it is not cached.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


METHODS = (
    "shared_basis",
    "history_transport",
    "current_adapted",
    "plastic_shared_basis",
)
ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def configuration(method, options):
    config = {
        "model_name": method,
        "cl27_protocol": "pilot_figure2",
        "dataset": "cifar224",
        "data_root": str(Path(options.data_root).expanduser().resolve()),
        "download": False,
        "phase": "development",
        "validation_only": True,
        "smoke": False,
        "pretrained": True,
        "seed": [1993],
        "device": [options.device],
        "shuffle": True,
        "init_cls": 10,
        "increment": 10,
        "validation_fraction": 0.1,
        "split_seed": 0,
        "validation_max_per_class": options.max_per_class,
        "max_tasks": 2,
        "epochs": 1,
        "batch_size": options.batch_size,
        "replay_batch_size": 8,
        "feature_batch_size": options.feature_batch_size,
        "eval_batch_size": options.feature_batch_size,
        "num_workers": 0,
        "cpu_threads": options.cpu_threads,
        "backbone_type": "vit_base_patch16_224",
        "feature_normalization": "l2",
        "backbone_lr": 1e-5,
        "head_lr": 1e-3,
        "module_lr": 1e-4,
        "weight_decay": 1e-4,
        "memory_size": 0 if method == "shared_basis" else 2000,
        "memory_per_class": 0,
        "fixed_memory": False,
        "replay_selection": "herding",
        "module_layers": [7, 9],
        "modules_per_layer": 2,
        "module_rank": 2,
        "outer_updates": 1,
        "support_shots": [1],
        "background_shots": 1,
        "target_classes": 1,
        "query_per_class": 1,
        "basis_rank": 4,
        "basis_max_iter": 2,
        "transport_mode": "direct",
        "crossfit_folds": 2,
        "save_checkpoints": bool(options.save_checkpoints),
    }
    if method == "plastic_shared_basis":
        # A predeclared mechanics setting, not selected by validation accuracy.
        # With three rows/class and C=10 or 20, s²=1 provably cannot open a
        # nonzero basis from an isotropic reset (s² >= max(n_c)/C). Use a lower
        # working noise to exercise learned-basis prediction in this tiny check.
        # Formal configs retain the original frozen-route defaults.
        config.update(basis_noise_variance=0.01, basis_reset_each_task=True)
    config.update(options.overrides.get("common", {}))
    config.update(options.overrides.get(method, {}))
    # Overrides may adjust hyperparameters for a targeted check, but never turn
    # this verification tool into a full-stream or official-test run.
    required = {
        "cl27_protocol": "pilot_figure2",
        "dataset": "cifar224",
        "phase": "development",
        "validation_only": True,
        "smoke": False,
        "download": False,
        "pretrained": True,
        "max_tasks": 2,
        "epochs": 1,
        "init_cls": 10,
        "increment": 10,
        "backbone_type": "vit_base_patch16_224",
        "model_name": method,
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError(
            "Overrides cannot change the short training-only verification protocol"
        )
    if config.get("validation_max_per_class") not in (2, 3):
        raise ValueError("Verification is limited to two or three images per class")
    if method == "current_adapted" and int(config["batch_size"]) < 8:
        raise ValueError(
            "Current episodes require logical batch>=8; reduce feature_batch_size for memory"
        )
    config["output_dir"] = str(options.output_dir / method)
    return config


def validate_result(method, result, manifest, config):
    if (
        result["status"] != "validation_partial"
        or result["phase"] != "development"
        or result["performance_claim"] is not False
    ):
        raise ValueError("Run was not explicitly marked partial development validation")
    if (
        manifest["official_test_read"]
        or manifest["evaluation_source"] != "training_holdout"
    ):
        raise ValueError("Verification used an official test split")
    provenance = result.get("backbone_provenance") or {}
    if (
        provenance.get("kind") != "pretrained"
        or provenance.get("timm_version") != "0.6.12"
        or provenance.get("checkpoint_family") != "augreg_in21k_ft_in1k"
    ):
        raise ValueError("Missing genuine pinned-timm pretrained checkpoint provenance")
    for name in ("weight_file_sha256", "initial_state_sha256"):
        if len(provenance.get(name, "")) != 64:
            raise ValueError("Missing pretrained hash: " + name)
    stages = result["stages"]
    diagnostics = result["task_diagnostics"]
    per = config["validation_max_per_class"]
    if len(stages) != 2 or len(diagnostics) != 2:
        raise ValueError("Expected exactly two completed tasks")
    if result["metrics"]["completed_tasks"] != 2:
        raise ValueError("Summary task count differs")
    checks = []
    for task, (stage, diag) in enumerate(zip(stages, diagnostics)):
        seen = (task + 1) * 10
        arrivals = seen * per
        if stage["seen_classes"] != seen or stage["evaluation_images"] != seen * per:
            raise ValueError("Unexpected class or holdout count")
        if stage["prediction_space"] != "all_seen_classes_without_task_id":
            raise ValueError("Task-ID-free all-seen prediction was not used")
        if not math.isfinite(stage["accuracy"]) or not 0 <= stage["accuracy"] <= 1:
            raise ValueError("Nonfinite/out-of-range evaluation accuracy")
        if method == "shared_basis":
            count = diag["samples"]
            replay = diag["raw_replay_images"]
            if not diag["feature_map_frozen"] or replay != 0:
                raise ValueError("Shared route did not stay frozen/exemplar-free")
        elif method in {"history_transport", "plastic_shared_basis"}:
            count = diag["statistics"]["statistics_count"]
            replay = diag["replay_images"]
            if diag["training"]["current_training_visits"] != 10 * per:
                raise ValueError("Current task training visits differ")
            if not diag["anchor_frozen"] or not math.isfinite(
                diag["training"]["loss_mean"]
            ):
                raise ValueError("Invalid plastic training/anchor diagnostics")
            if method == "plastic_shared_basis":
                statistics = diag["statistics"]
                if statistics["shared_basis"]["samples"] != arrivals:
                    raise ValueError("Shared basis accumulated a full snapshot twice")
                if (
                    statistics.get("historical_objective")
                    != "transported_moment_surrogate"
                ):
                    raise ValueError("Transported basis objective is mislabeled")
                if statistics.get("basis_reset_each_task") is not True:
                    raise ValueError("Old-coordinate basis was silently retained")
        else:
            count = diag["history_count"]
            replay = diag["replay_images"]
            if diag["counters"]["logical_current_presentations"] != arrivals:
                raise ValueError("Current rows were not trained exactly one epoch")
        if count != arrivals:
            raise ValueError("History count is not unique task arrivals")
        if replay != (0 if method == "shared_basis" else arrivals):
            raise ValueError(
                "Unexpected replay count for this small fully-resident subset"
            )
        checks.append(
            {
                "task": task,
                "seen_classes": seen,
                "fitting_images_committed": int(count),
                "heldout_images": stage["evaluation_images"],
                "replay_images": replay,
                "body_changed": diag.get("backbone_changed", False),
                "modules_changed": diag.get("modules_changed", False),
            }
        )
        if method == "plastic_shared_basis":
            checks[-1]["basis_effective_rank"] = diag["statistics"]["shared_basis"][
                "effective_rank"
            ]
    if method != "shared_basis" and not any(row["body_changed"] for row in checks):
        raise ValueError(
            "No evidence of plastic native backbone updates in this short stream"
        )
    if method == "current_adapted" and not any(
        row["modules_changed"] for row in checks
    ):
        raise ValueError("No evidence of fixed-library learning in this short stream")
    if method == "plastic_shared_basis" and not any(
        row["basis_effective_rank"] > 0 for row in checks
    ):
        raise ValueError("This mechanics check did not exercise a nonzero shared basis")
    for name in ("final_accuracy", "aia"):
        if not math.isfinite(result["metrics"][name]):
            raise ValueError("Nonfinite summary metric")
    return {
        "method": method,
        "status": "verified",
        "checks": checks,
        "pretrained": provenance,
        "wall_seconds": result["execution"]["wall_seconds"],
        "accuracy_is_not_a_performance_claim": True,
        "transport_nonresident_branch_exercised": (
            False if method in {"history_transport", "plastic_shared_basis"} else None
        ),
        "limitation": "Protocol buffer2000 retains every old image in this small subset; missing-history transport is covered by separate core unit tests.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Existing directory containing cifar-100-python/train",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="A new directory for configs, logs and reports",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="-1 for CPU; otherwise a single CUDA index",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable with timm==0.6.12"
    )
    parser.add_argument("--max-per-class", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Logical current batch; current method requires >=8",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=2,
        help="Native feature/VJP microbatch size",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--overrides-json",
        type=Path,
        help="Optional object with common/method-specific hyperparameter overrides",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write bounded configs without importing torch/timm or running training",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save large genuine ViT task checkpoints in addition to JSON reports",
    )
    options = parser.parse_args(argv)
    options.output_dir = options.output_dir.expanduser().resolve()
    options.overrides = (
        json.loads(options.overrides_json.read_text()) if options.overrides_json else {}
    )
    if not isinstance(options.overrides, dict):
        raise ValueError("Override root must be an object")
    if (
        min(options.batch_size, options.feature_batch_size, options.cpu_threads) < 1
        or options.device < -1
    ):
        raise ValueError("Invalid device or execution budget")
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise FileExistsError("Verification requires a new output directory")
    configs = {
        method: configuration(method, options)
        for method in dict.fromkeys(options.methods)
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for method, config in configs.items():
        dump(options.output_dir / (method + ".json"), config)
    workload = {
        "kind": "genuine_pretrained_pipeline_correctness_only",
        "methods": list(configs),
        "tasks": 2,
        "new_classes_per_task": 10,
        "epochs": 1,
        "fit_images_per_class": {
            m: c["validation_max_per_class"] for m, c in configs.items()
        },
        "train_holdout_images_per_class": {
            m: c["validation_max_per_class"] for m, c in configs.items()
        },
        "official_test_access": False,
        "dataset_download": False,
        "weight_download_if_uncached": True,
        "full_experiment": False,
        "prepared_only": options.prepare_only,
        "note": "All retained replay is resident in this tiny subset; no missing-history transport test is claimed.",
    }
    dump(options.output_dir / "workload.json", workload)
    if options.prepare_only:
        print(
            json.dumps(
                {"status": "prepared", "methods": list(configs), "tasks_per_method": 2}
            )
        )
        return 0
    if not (
        Path(options.data_root).expanduser() / "cifar-100-python" / "train"
    ).is_file():
        raise FileNotFoundError(
            "Existing CIFAR-100 training pickle required; this script does not download datasets"
        )
    reports = []
    for method, config in configs.items():
        with (options.output_dir / (method + ".log")).open("w") as log:
            completed = subprocess.run(
                [
                    options.python,
                    str(ROOT / "main.py"),
                    "--config",
                    str(options.output_dir / (method + ".json")),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            dump(
                options.output_dir / "verification_report.json",
                {
                    "status": "failed",
                    "method": method,
                    "returncode": completed.returncode,
                    "completed": reports,
                    "log": method + ".log",
                },
            )
            raise RuntimeError(
                "Native verification failed for " + method + "; inspect its local log"
            )
        result = json.loads((options.output_dir / method / "results.json").read_text())
        manifest = json.loads(
            (options.output_dir / method / "data_manifest.json").read_text()
        )
        reports.append(validate_result(method, result, manifest, config))
        dump(
            options.output_dir / "verification_report.json",
            {"status": "running", "completed": reports},
        )
        print(
            json.dumps({"method": method, "status": "verified"}, allow_nan=False),
            flush=True,
        )
    dump(
        options.output_dir / "verification_report.json",
        {"status": "verified", "workload": workload, "completed": reports},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
