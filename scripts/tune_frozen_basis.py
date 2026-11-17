#!/usr/bin/env python3
"""Cache frozen training-only features and evaluate one development candidate.

`extract` reads the designated training split through ProtocolDataManager's
fixed 10% holdout. It never requests the official test split. `evaluate` is a
CPU-only single-configuration fit; it does not enumerate or rank candidates.
The cache is an offline tuning asset, not persistent memory supplied to the
formal exemplar-free learner. Future classes enter the numerical core only at
their scheduled arrival, even though their immutable features are precomputed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_VERSION = 1
DATASETS = {"cifar224": (100, 10), "imagenetr": (200, 20)}
ARRAY_NAMES = (
    "fit_features",
    "fit_labels",
    "fit_ids",
    "holdout_features",
    "holdout_labels",
    "holdout_ids",
)
IDENTITY_FIELDS = (
    "cache_version",
    "purpose",
    "protocol",
    "data_manifest",
    "feature_map",
    "arrays",
    "payload_sha256",
)
PARAMETERS = {
    "basis_rank": "rank",
    "basis_epsilon": "epsilon",
    "basis_noise_variance": "noise_variance",
    "basis_max_iter": "max_iter",
    "basis_tolerance": "tol",
}


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_record(value):
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def fresh_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Use a new empty directory: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def development_config(config):
    """Use only the already implemented fixed train holdout, without subsets."""
    config = copy.deepcopy(config)
    if (
        config.get("dataset") not in DATASETS
        or config.get("model_name") != "shared_basis"
    ):
        raise ValueError(
            "Extraction requires a shared_basis CIFAR-100/ImageNet-R config"
        )
    seed = config.get("seed")
    if seed != 1993 and seed != [1993]:
        raise ValueError("This campaign uses only seed 1993")
    classes, width = DATASETS[config["dataset"]]
    if (config.get("init_cls"), config.get("increment")) != (width, width):
        raise ValueError("This campaign uses ten equal class stages")
    if (
        config.get("backbone_type") != "vit_base_patch16_224"
        or config.get("pretrained") is not True
    ):
        raise ValueError("Use the pinned pretrained ViT-B/16")
    if (
        config.get("validation_max_per_class") is not None
        or config.get("max_tasks") is not None
    ):
        raise ValueError(
            "The reusable cache must contain the complete development split"
        )
    if config.get("shuffle", True) is not True:
        raise ValueError("The campaign requires the seeded class permutation")
    config.update(
        phase="development",
        validation_only=True,
        validation_fraction=0.1,
        split_seed=0,
        seed=1993,
        download=False,
        memory_size=0,
        memory_per_class=0,
        fixed_memory=False,
    )
    return config, classes, width


def save_cache(cache_dir, metadata, arrays):
    """Bind every cached row, its coordinate map and original split identity."""
    cache_dir = fresh_directory(cache_dir)
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Unexpected cache arrays")
    metadata = copy.deepcopy(metadata)
    payload = cache_dir / "features.npz"
    with payload.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, **arrays)
    payload.with_suffix(".tmp").replace(payload)
    metadata["arrays"] = {name: array_record(arrays[name]) for name in ARRAY_NAMES}
    metadata["payload_sha256"] = file_hash(payload)
    metadata["cache_identity_sha256"] = canonical_hash(
        {key: metadata[key] for key in IDENTITY_FIELDS}
    )
    write_json(cache_dir / "metadata.json", metadata)
    return metadata


def load_cache(cache_dir, expected_dataset):
    cache_dir = Path(cache_dir).expanduser().resolve()
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if (
        metadata.get("cache_version") != CACHE_VERSION
        or metadata.get("purpose") != "training_holdout_development_only"
    ):
        raise ValueError("This is not a development-only frozen-feature cache")
    if (
        expected_dataset not in DATASETS
        or metadata.get("protocol", {}).get("dataset") != expected_dataset
    ):
        raise ValueError("Cache dataset differs from the explicitly requested dataset")
    identity = canonical_hash({key: metadata[key] for key in IDENTITY_FIELDS})
    if identity != metadata.get("cache_identity_sha256"):
        raise ValueError("Cache identity differs; metadata was changed")
    protocol = metadata["protocol"]
    classes, width = DATASETS[expected_dataset]
    if (
        protocol.get("seed") != 1993
        or protocol.get("split_seed") != 0
        or protocol.get("validation_fraction") != 0.1
        or protocol.get("class_count") != classes
        or protocol.get("increment") != width
    ):
        raise ValueError("Cache uses a different development protocol")
    manifest = metadata["data_manifest"]
    if (
        manifest.get("phase") != "development"
        or manifest.get("official_test_read") is not False
        or manifest.get("evaluation_source") != "training_holdout"
    ):
        raise ValueError("Cache did not isolate the official test split")
    fingerprint = hashlib.sha256(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "fingerprint"}, sort_keys=True
        ).encode()
    ).hexdigest()
    if fingerprint != manifest.get("fingerprint"):
        raise ValueError("Cache data manifest fingerprint differs")
    if (
        manifest.get("class_order")
        != np.random.RandomState(1993).permutation(classes).tolist()
    ):
        raise ValueError("Cache class order differs from seed 1993")
    feature_map = metadata["feature_map"]
    if (
        feature_map.get("normalization") not in {"l2", "none"}
        or feature_map.get("backbone_type") != "vit_base_patch16_224"
    ):
        raise ValueError("Unsupported cached feature coordinate map")
    provenance = feature_map.get("backbone_provenance", {})
    if (
        provenance.get("kind") != "pretrained"
        or provenance.get("timm_version") != "0.6.12"
        or not provenance.get("weight_file_sha256")
    ):
        raise ValueError("Cache does not bind the pinned pretrained weights")
    if feature_map.get("frozen_state_sha256") != provenance.get("initial_state_sha256"):
        raise ValueError(
            "The cached backbone differs from its initial pretrained state"
        )
    payload = cache_dir / "features.npz"
    if file_hash(payload) != metadata["payload_sha256"]:
        raise ValueError("Cache payload checksum differs")
    with np.load(payload, allow_pickle=False) as loaded:
        if set(loaded.files) != set(ARRAY_NAMES):
            raise ValueError("Unexpected cache arrays")
        arrays = {name: loaded[name] for name in ARRAY_NAMES}
    for name, value in arrays.items():
        if array_record(value) != metadata["arrays"].get(name):
            raise ValueError("Cached array identity differs: " + name)
    dimension = int(feature_map["feature_dim"])
    for prefix, count_key, ids_key in (
        ("fit", "train_count", "fit_ids_sha256"),
        ("holdout", "evaluation_count", "evaluation_ids_sha256"),
    ):
        features, labels, ids = (
            arrays[prefix + suffix] for suffix in ("_features", "_labels", "_ids")
        )
        n = int(manifest[count_key])
        if (
            features.shape != (n, dimension)
            or features.dtype != np.float64
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid cached feature rows")
        if (
            labels.shape != (n,)
            or ids.shape != (n,)
            or labels.dtype != np.int64
            or ids.dtype != np.int64
        ):
            raise ValueError("Invalid cached labels or original IDs")
        if len(np.unique(ids)) != n or sorted(np.unique(labels).tolist()) != list(
            range(classes)
        ):
            raise ValueError("Cached rows omit classes or repeat original IDs")
        if hashlib.sha256(np.sort(ids).tobytes()).hexdigest() != manifest[ids_key]:
            raise ValueError("Cached original IDs differ from the fixed split")
    if np.intersect1d(arrays["fit_ids"], arrays["holdout_ids"]).size:
        raise ValueError("Cached fit and holdout original IDs overlap")
    all_ids = np.concatenate((arrays["fit_ids"], arrays["holdout_ids"]))
    total = 50000 if expected_dataset == "cifar224" else 24000
    if not np.array_equal(np.sort(all_ids), np.arange(total, dtype=np.int64)):
        raise ValueError("Cache omits or adds original training rows")
    fit_counts = np.bincount(arrays["fit_labels"], minlength=classes)
    holdout_counts = np.bincount(arrays["holdout_labels"], minlength=classes)
    expected_holdout = np.maximum(
        1,
        np.minimum(
            fit_counts + holdout_counts - 1,
            ((fit_counts + holdout_counts) * 0.1).astype(int),
        ),
    )
    if not np.array_equal(holdout_counts, expected_holdout):
        raise ValueError(
            "Cache class counts differ from the complete fixed 10% holdout"
        )
    return metadata, arrays


def extract(
    config_path,
    cache_dir,
    device_index=0,
    feature_batch_size=None,
    eval_batch_size=None,
    data_root=None,
):
    import torch
    from cl27.common import (
        build_backbone,
        extract_features,
        feature_normalization,
        seed_everything,
        state_hash,
    )
    from cl27.data import ProtocolDataManager

    config, classes, width = development_config(
        json.loads(Path(config_path).read_text())
    )
    if data_root is not None:
        config["data_root"] = str(Path(data_root).expanduser().resolve())
    normalization = feature_normalization(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    seed_everything(1993)
    device = torch.device("cpu" if device_index == -1 else "cuda:" + str(device_index))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.set_device(device)
    # Reject an occupied destination before encoding any data.
    cache_dir = fresh_directory(cache_dir)
    began = time.perf_counter()
    manager = ProtocolDataManager(config["dataset"], True, 1993, width, width, config)
    if manager.nb_tasks != 10 or manager.nb_classes != classes:
        raise ValueError("Unexpected development task structure")
    backbone = build_backbone(config).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    frozen_hash = state_hash(backbone)
    arrays, transforms = {}, {}
    batches = {
        "fit": int(
            feature_batch_size
            if feature_batch_size is not None
            else config.get("feature_batch_size", 16)
        ),
        "holdout": int(
            eval_batch_size
            if eval_batch_size is not None
            else config.get("eval_batch_size", 64)
        ),
    }
    if min(batches.values()) < 1:
        raise ValueError("Feature and evaluation batch sizes must be positive")
    for prefix, source in (("fit", "train"), ("holdout", "test")):
        dataset = manager.get_dataset(np.arange(classes), source=source, mode="test")
        transforms[prefix] = repr(dataset.trsf)
        print(
            json.dumps(
                {
                    "extract_split": prefix,
                    "images": len(dataset),
                    "official_test_read": False,
                }
            ),
            flush=True,
        )
        features, labels, ids = extract_features(
            backbone, dataset, device, batches[prefix], normalization
        )
        arrays.update(
            {
                prefix + "_features": features,
                prefix + "_labels": labels,
                prefix + "_ids": ids,
            }
        )
    if state_hash(backbone) != frozen_hash:
        raise RuntimeError("Feature extraction changed the frozen backbone")
    metadata = {
        "cache_version": CACHE_VERSION,
        "purpose": "training_holdout_development_only",
        "protocol": {
            "dataset": config["dataset"],
            "seed": 1993,
            "split_seed": 0,
            "validation_fraction": 0.1,
            "class_count": classes,
            "increment": width,
        },
        "data_manifest": manager.manifest,
        "feature_map": {
            "backbone_type": config["backbone_type"],
            "normalization": normalization,
            "feature_dim": int(backbone.out_dim),
            "frozen_state_sha256": frozen_hash,
            "backbone_provenance": copy.deepcopy(backbone.cl27_provenance),
            "deterministic_transforms": transforms,
            "encoding_batch_sizes": batches,
            "floating_policy": {
                "torch": str(torch.__version__),
                "autocast": False,
                "tf32": False,
            },
            "extractor_source_sha256": {
                str(p.relative_to(ROOT)): file_hash(p)
                for p in (
                    ROOT / "cl27/common.py",
                    ROOT / "cl27/data.py",
                    ROOT / "utils/data.py",
                    ROOT / "utils/inc_net.py",
                )
            },
        },
        "execution": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device": str(device),
            "feature_batch_size": batches["fit"],
            "eval_batch_size": batches["holdout"],
            "wall_seconds": time.perf_counter() - began,
        },
        "config_source": str(Path(config_path).resolve()),
        "not_formal_learner_memory": True,
    }
    metadata = save_cache(cache_dir, metadata, arrays)
    # Exercise the same complete identity checks that every later trial uses.
    load_cache(cache_dir, config["dataset"])
    print(
        json.dumps(
            {
                "cache_complete": True,
                "cache_identity_sha256": metadata["cache_identity_sha256"],
            }
        ),
        flush=True,
    )
    return metadata


def evaluate_arrays(arrays, *, classes, width, parameters, max_tasks=10, on_stage=None):
    """Replay the class stream over a fixed cache; only current rows reach fit."""
    import torch
    from cl27.cores.shared_basis import SharedPredictorBasis
    from cl27.runner import summarize

    if (
        not isinstance(max_tasks, int)
        or isinstance(max_tasks, bool)
        or not 1 <= max_tasks <= 10
    ):
        raise ValueError("max_tasks must be in [1, 10]")
    if classes != width * 10:
        raise ValueError("Expected ten equal class stages")
    unknown = set(parameters) - set(PARAMETERS)
    required = set(PARAMETERS) - {"basis_tolerance"}
    if unknown or not required.issubset(parameters):
        raise ValueError(
            "Supply exactly one explicit basis parameter object; missing or unknown keys"
        )
    options = {PARAMETERS[key]: value for key, value in parameters.items()}
    options.setdefault("tol", 1e-6)
    core = SharedPredictorBasis(
        feature_dim=arrays["fit_features"].shape[1], seed=1993, **options
    )
    stages, diagnostics = [], []
    for task in range(max_tasks):
        began = time.perf_counter()
        seen = (task + 1) * width
        fit = (arrays["fit_labels"] >= task * width) & (arrays["fit_labels"] < seen)
        diagnostic = core.observe(
            arrays["fit_features"][fit], arrays["fit_labels"][fit]
        )
        heldout = arrays["holdout_labels"] < seen
        labels = arrays["holdout_labels"][heldout]
        scores = core.predict_scores(arrays["holdout_features"][heldout])
        # Match the deployed analytic head's FP32 output and CPU top-k rule.
        predictions = torch.topk(
            torch.from_numpy(scores).float(), k=min(5, seen), dim=1
        )[1][:, 0].numpy()
        correct = predictions == labels
        task_accuracy, counts = [], []
        for j in range(10):
            mask = (labels >= j * width) & (labels < (j + 1) * width)
            counts.append(int(mask.sum()))
            task_accuracy.append(float(correct[mask].mean()) if mask.any() else None)
        old = labels < task * width
        row = {
            "task": task,
            "seen_classes": seen,
            "accuracy": float(correct.mean()),
            "task_accuracy": task_accuracy,
            "task_counts": counts,
            "evaluation_images": len(labels),
            "prediction_space": "all_seen_classes_without_task_id",
            "old_accuracy": float(correct[old].mean()) if old.any() else None,
            "current_accuracy": float(correct[~old].mean()),
            "wall_seconds": time.perf_counter() - began,
        }
        if core.n_samples != int((arrays["fit_labels"] < seen).sum()):
            raise RuntimeError(
                "Historical fitting counts differ from scheduled arrivals"
            )
        stages.append(row)
        diagnostics.append(diagnostic)
        if on_stage is not None:
            on_stage(stages, diagnostics, summarize(stages))
    return stages, diagnostics, summarize(stages)


def evaluate(
    cache_dir,
    dataset,
    parameters_path,
    output_dir,
    max_tasks=10,
    cpu_threads=4,
    reason="Explicit initial candidate",
):
    import torch

    if cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    torch.set_num_threads(cpu_threads)
    metadata, arrays = load_cache(cache_dir, dataset)
    parameters = json.loads(Path(parameters_path).read_text())
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be one JSON object, not a list/search space")
    output = fresh_directory(output_dir)
    write_json(output / "parameters.json", parameters)
    write_json(
        output / "cache_reference.json",
        {
            "directory": str(Path(cache_dir).resolve()),
            "cache_identity_sha256": metadata["cache_identity_sha256"],
            "data_fingerprint": metadata["data_manifest"]["fingerprint"],
            "feature_map": metadata["feature_map"],
        },
    )
    write_json(
        output / "decision.json",
        {
            "reason_recorded_before_evaluation": reason,
            "hyperparameter_source": "explicit_single_configuration",
            "official_test_used": False,
        },
    )
    began = time.perf_counter()

    def record(stages, diagnostics, metrics):
        write_json(
            output / "progress.json",
            {
                "status": "development_running",
                "phase": "development",
                "validation_only": True,
                "metrics": metrics,
                "stages": stages,
                "task_diagnostics": diagnostics,
            },
        )
        print(
            json.dumps(
                {
                    "task_complete": len(stages) - 1,
                    "phase": "development",
                    "accuracy": stages[-1]["accuracy"],
                    "effective_rank": diagnostics[-1]["effective_rank"],
                    "em_stop_reason": diagnostics[-1]["stop_reason"],
                }
            ),
            flush=True,
        )

    protocol = metadata["protocol"]
    stages, diagnostics, metrics = evaluate_arrays(
        arrays,
        classes=protocol["class_count"],
        width=protocol["increment"],
        parameters=parameters,
        max_tasks=max_tasks,
        on_stage=record,
    )
    result = {
        "status": "development_complete" if max_tasks == 10 else "development_partial",
        "phase": "development",
        "validation_only": True,
        "performance_claim": False,
        "method": "shared_basis",
        "dataset": dataset,
        "seed": 1993,
        "official_test_read": False,
        "evaluation_source": "training_holdout",
        "parameters": parameters,
        "metrics": metrics,
        "stages": stages,
        "task_diagnostics": diagnostics,
        "cache_identity_sha256": metadata["cache_identity_sha256"],
        "data_fingerprint": metadata["data_manifest"]["fingerprint"],
        "backbone_provenance": metadata["feature_map"]["backbone_provenance"],
        "execution": {
            "device": "cpu",
            "cpu_threads": cpu_threads,
            "wall_seconds": time.perf_counter() - began,
            "feature_extraction_shared_cost_seconds": metadata["execution"][
                "wall_seconds"
            ],
        },
        "boundary": "Offline development cache only; not a formal result or the deployed learner's image/feature memory",
        "score_convention": "FP32 analytic logits, CPU torch.topk; exact tie order may differ from CUDA topk",
        "code_sha256": {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in (Path(__file__), ROOT / "cl27/cores/shared_basis.py")
        },
    }
    write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser(
        "extract", help="Encode only the fixed train/holdout split once"
    )
    extraction.add_argument("--config", type=Path, required=True)
    extraction.add_argument("--cache-dir", type=Path, required=True)
    extraction.add_argument("--data-root", type=Path)
    extraction.add_argument(
        "--device", type=int, default=0, help="-1 for CPU or one CUDA index"
    )
    extraction.add_argument("--feature-batch-size", type=int)
    extraction.add_argument("--eval-batch-size", type=int)
    evaluation = commands.add_parser(
        "evaluate", help="CPU fit of one explicit candidate, never a grid"
    )
    evaluation.add_argument("--cache-dir", type=Path, required=True)
    evaluation.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    evaluation.add_argument("--parameters-json", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--max-tasks", type=int, default=10)
    evaluation.add_argument("--cpu-threads", type=int, default=4)
    evaluation.add_argument("--reason", default="Explicit initial candidate")
    options = parser.parse_args()
    if options.command == "extract":
        extract(
            options.config,
            options.cache_dir,
            options.device,
            options.feature_batch_size,
            options.eval_batch_size,
            options.data_root,
        )
    else:
        evaluate(
            options.cache_dir,
            options.dataset,
            options.parameters_json,
            options.output_dir,
            options.max_tasks,
            options.cpu_threads,
            options.reason,
        )


if __name__ == "__main__":
    main()
