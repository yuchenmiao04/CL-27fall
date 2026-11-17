#!/usr/bin/env python3
"""Coherent base/common/full diagnostics on an internal-training holdout.

This never trains a model or opens an official test split. For stage > 0 a
previous-stage checkpoint is required: the deployed residual head was fitted
before replay shrinkage, so the latest resident buffer alone is insufficient
to reconstruct its fitting pool. The common head is refitted on that same pool;
turning modules off while keeping the full head is not used as a comparison.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cl27.common import file_hash, jsonable, seed_everything, state_hash
from cl27.current.engine import class_weights
from cl27.current.readout import centered_residual_head
from cl27.data import ProtocolDataManager
from cl27.protocol import validate_protocol
from cl27.replay import ReplayMemory
from cl27.runner import configuration_fingerprint, validate_checkpoint_random_stream
from models.current_adapted import Learner


@contextmanager
def preserve_rng():
    """A read-only diagnostic must not advance the caller's training RNGs."""
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(str(tuple(value.shape)).encode())
    result.update(str(value.dtype).encode())
    result.update(value.numpy().tobytes())
    return result.hexdigest()


def readonly_state(model):
    core = model.core
    return {
        "anchor": state_hash(core.encoder.anchor),
        "main": state_hash(core.encoder.main),
        "libraries": state_hash(core.encoder.libraries),
        "G": tensor_digest(core.statistics.G),
        "C": tensor_digest(core.statistics.C),
        "W0": tensor_digest(core.W0),
        "V": tensor_digest(core.V),
        "counters": copy.deepcopy(core.counters),
        "updates": (core.updates, core.module_updates, core.backbone_updates),
        "episode_draws": core.episode_draws,
        "episode_rng": tensor_digest(core.rng.get_state()),
        "episode_kind_rng": tensor_digest(core.kind_rng.get_state()),
        "memory_ids": [record.sample_id for record in model.memory.records],
        "memory_rng": copy.deepcopy(model.memory.rng.bit_generator.state),
    }


def normalize_config(config, device=None):
    args = copy.deepcopy(config)
    if args.get("model_name") != "current_adapted":
        raise ValueError("This diagnostic is only for current_adapted")
    if args.get("phase") not in {"development", "smoke"} or not args.get(
        "validation_only", False
    ):
        raise ValueError("Use a validation-only internal-training-holdout checkpoint")
    if isinstance(args.get("seed"), list):
        if len(args["seed"]) != 1:
            raise ValueError("Exactly one training seed is required")
        args["seed"] = int(args["seed"][0])
    if device is not None:
        args["device"] = [int(device)]
    if len(args.get("device", [])) != 1:
        raise ValueError("Use one diagnostic device")
    validate_protocol(args)
    return args


def load_state(path):
    path = Path(path).expanduser().resolve()
    before = file_hash(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if file_hash(path) != before:
        raise ValueError("Checkpoint changed while it was being read")
    validate_checkpoint_random_stream(state)
    return state, before


def load_model_and_data(args, state):
    dm = ProtocolDataManager(
        args["dataset"],
        args.get("shuffle", True),
        int(args["seed"]),
        int(args["init_cls"]),
        int(args["increment"]),
        args,
    )
    if (
        dm.manifest["official_test_read"]
        or dm.manifest["evaluation_source"] != "training_holdout"
    ):
        raise ValueError(
            "Diagnostic evaluation must come only from the training holdout"
        )
    if state.get("config_fingerprint") != configuration_fingerprint(args):
        raise ValueError("Configuration differs from the checkpoint")
    if state.get("data_fingerprint") != dm.manifest["fingerprint"]:
        raise ValueError("Data/split identity differs from the checkpoint")
    learner = state["learner"]
    task = int(learner["cur_task"])
    if task < 0 or task >= dm.nb_tasks:
        raise ValueError("Invalid completed checkpoint stage")
    start = sum(dm.get_task_size(i) for i in range(task))
    seen = start + dm.get_task_size(task)
    if learner["known_classes"] != seen or learner["total_classes"] != seen:
        raise ValueError("Checkpoint is not a completed stage")
    device = torch.device(
        "cpu" if int(args["device"][0]) == -1 else "cuda:" + str(args["device"][0])
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.set_device(device)
    native_args = copy.deepcopy(args)
    native_args.update(device=[device], nb_classes=dm.nb_classes, nb_tasks=dm.nb_tasks)
    model = Learner(native_args)
    current = dm.get_dataset(np.arange(start, seen), source="train", mode="test")
    training = dm.get_dataset(np.arange(start, seen), source="train", mode="train")
    model.canonical_transform.transform = current.trsf
    model.training_transform.transform = training.trsf
    model.restore_checkpoint(learner)
    return model, dm


def fitting_pool(model, dm, previous):
    task = model._cur_task
    start = sum(dm.get_task_size(i) for i in range(task))
    seen = model._total_classes
    current = dm.get_dataset(np.arange(start, seen), source="train", mode="test")
    current_rows = list(zip(current.images, current.labels, current.ids))
    if task == 0:
        if previous is not None:
            raise ValueError("The first stage has no previous checkpoint")
        old, old_base = [], torch.empty(
            (0, model.core.feature_dim), dtype=torch.float64
        )
    else:
        if previous is None:
            raise ValueError(
                "A previous-stage checkpoint is required to recover the pre-shrink fitting pool"
            )
        learner = previous["learner"]
        if (
            learner.get("method") != "current_adapted"
            or learner["cur_task"] != task - 1
            or learner["known_classes"] != start
            or learner["total_classes"] != start
        ):
            raise ValueError(
                "Previous checkpoint must be the immediately preceding completed stage"
            )
        memory = ReplayMemory(model.args)
        memory.load_state_dict(learner["core"]["memory"])
        if memory.classes != list(range(start)):
            raise ValueError("Previous replay vocabulary is inconsistent")
        old = [
            (record.image, record.label, record.sample_id) for record in memory.records
        ]
        cached = learner["core"]["engine"]["memory_base"]
        if set(cached) != {row[2] for row in old}:
            raise ValueError("Previous replay and anchor cache IDs disagree")
        old_base = torch.stack([cached[row[2]].detach().cpu().double() for row in old])
        if old_base.shape != (len(old), model.core.feature_dim):
            raise ValueError("Invalid previous resident-anchor dimensions")
        if not torch.isfinite(old_base).all():
            raise ValueError("Nonfinite previous resident-anchor values")
    ids = [int(row[2]) for row in current_rows + old]
    if len(set(ids)) != len(ids):
        raise ValueError("The deployment fitting pool repeats image identities")
    if any(label >= start for _, label, _ in old):
        raise ValueError("The previous pool includes a future class")
    return current_rows, old, old_base


@torch.no_grad()
def encode_rows(model, rows, path):
    size = int(model.core.options["feature_batch_size"])
    if not rows:
        return torch.empty((0, model.core.feature_dim), dtype=torch.float64)
    pieces = []
    for start in range(0, len(rows), size):
        images = torch.stack(
            [model.canonical_transform(row[0]) for row in rows[start : start + size]]
        ).to(model._device)
        if path == "base":
            features = model.core.encoder.anchor_features(images)
        elif path in {"common", "full"}:
            features = model.core.encoder.main_features(images, path == "full")
        else:
            raise ValueError("Unknown feature path")
        pieces.append(features.cpu().double())
    return torch.cat(pieces)


def branch_scores(base, branch, head, prior, temperature, images):
    value = base @ prior
    if branch is not None:
        value = value + (branch - base) @ head
    # Match deployed CurrentNetwork/engine output casting before argmax.
    return (temperature * value).to(device=images.device, dtype=images.dtype)


def accuracy_report(labels, predictions, dm, task):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    correct = labels == predictions
    start, rows, counts = 0, [], []
    for j in range(task + 1):
        stop = start + dm.get_task_size(j)
        mask = (labels >= start) & (labels < stop)
        if not mask.any():
            raise ValueError("A seen class stage has no holdout examples")
        rows.append(float(correct[mask].mean()))
        counts.append(int(mask.sum()))
        start = stop
    split = sum(dm.get_task_size(i) for i in range(task))
    old = labels < split
    return {
        "accuracy": float(correct.mean()),
        "task_accuracy": rows,
        "task_counts": counts,
        "old_class_accuracy": float(correct[old].mean()) if old.any() else None,
        "new_class_accuracy": float(correct[~old].mean()),
        "evaluation_images": len(labels),
        "predictions": predictions.astype(int).tolist(),
    }


@torch.no_grad()
def evaluate_heads(model, dm, common_head):
    dataset = dm.get_dataset(
        np.arange(model._total_classes), source="test", mode="test"
    )
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_size=int(model.args.get("eval_batch_size", 64)),
        shuffle=False,
        num_workers=int(model.args.get("num_workers", 0)),
        generator=generator,
    )
    predictions = {name: [] for name in ["anchor", "common_refit", "full_deployed"]}
    labels = []
    for _, images, targets in loader:
        images = images.to(model._device)
        base = model.core.encoder.anchor_features(images).cpu().double()
        common = model.core.encoder.main_features(images, False).cpu().double()
        full = model.core.encoder.main_features(images, True).cpu().double()
        temp = model.core.options["temperature"]
        values = {
            "anchor": branch_scores(base, None, None, model.core.W0, temp, images),
            "common_refit": branch_scores(
                base, common, common_head, model.core.W0, temp, images
            ),
            "full_deployed": branch_scores(
                base, full, model.core.V, model.core.W0, temp, images
            ),
        }
        for name, logits in values.items():
            if (
                not torch.isfinite(logits).all()
                or logits.shape[1] != model._total_classes
            ):
                raise ValueError("Invalid all-seen diagnostic logits")
            predictions[name].extend(logits.argmax(1).cpu().tolist())
        labels.extend(targets.tolist())
    return {
        "labels": labels,
        "evaluation_ids_sha256": hashlib.sha256(dataset.ids.tobytes()).hexdigest(),
        "branches": {
            name: accuracy_report(labels, value, dm, model._cur_task)
            for name, value in predictions.items()
        },
    }


def diagnose_loaded(model, dm, previous=None):
    """Refit only the diagnostic common head; leave all trained state intact."""
    with preserve_rng():
        before = readonly_state(model)
        current, old, old_base = fitting_pool(model, dm, previous)
        base = torch.cat((encode_rows(model, current, "base"), old_base))
        rows = current + old
        labels = torch.tensor([int(row[1]) for row in rows])
        targets = F.one_hot(labels, num_classes=model._total_classes).double()
        weights = class_weights(labels)
        full = encode_rows(model, rows, "full")
        rebuilt = centered_residual_head(
            full - base,
            base,
            targets,
            model.core.W0,
            model.core.options["residual_ridge"],
            weights,
        )
        maximum = float((rebuilt - model.core.V).abs().max())
        scale = float(model.core.V.abs().max())
        tolerance = 1e-7 + 1e-5 * scale
        if maximum > tolerance:
            raise ValueError(
                "Reconstructed full head differs from deployed V; cannot claim a matched fitting pool"
            )
        common = encode_rows(model, rows, "common")
        common_head = centered_residual_head(
            common - base,
            base,
            targets,
            model.core.W0,
            model.core.options["residual_ridge"],
            weights,
        )
        report = evaluate_heads(model, dm, common_head)
        after = readonly_state(model)
        if before != after:
            raise RuntimeError(
                "Diagnostic changed trained model, moments, counters or RNG state"
            )
        report.update(
            {
                "stage": model._cur_task,
                "seen_classes": model._total_classes,
                "fit_pool": {
                    "current_images": len(current),
                    "pre_task_old_images": len(old),
                    "fit_ids_sha256": hashlib.sha256(
                        np.asarray([row[2] for row in rows], dtype=np.int64).tobytes()
                    ).hexdigest(),
                    "rule": "same complete current task plus previous-stage resident memory used by deployed full V",
                },
                "full_head_reconstruction": {
                    "max_abs_difference": maximum,
                    "tolerance": tolerance,
                    "passed": True,
                },
                "trained_state": before,
                "common_head_sha256": tensor_digest(common_head),
                "read_only_state_verified": True,
            }
        )
        return report


def diagnose_checkpoint(config_path, checkpoint_path, previous_path=None, device=None):
    started = time.perf_counter()
    with preserve_rng():
        raw_config = json.loads(Path(config_path).read_text())
        args = normalize_config(raw_config, device)
        state, checkpoint_hash = load_state(checkpoint_path)
        previous, previous_hash = (
            (None, None) if previous_path is None else load_state(previous_path)
        )
        if previous is not None and (
            previous.get("config_fingerprint") != state.get("config_fingerprint")
            or previous.get("data_fingerprint") != state.get("data_fingerprint")
        ):
            raise ValueError(
                "Previous checkpoint belongs to a different config/data stream"
            )
        seed_everything(int(args["seed"]))
        model, dm = load_model_and_data(args, state)
        report = diagnose_loaded(model, dm, previous)
        report.update(
            {
                "status": "complete",
                "purpose": "read-only internal-holdout readout diagnosis",
                "phase": args["phase"],
                "validation_only": True,
                "performance_claim": False,
                "official_test_read": False,
                "training_steps": 0,
                "configuration": raw_config,
                "config_fingerprint": state["config_fingerprint"],
                "config_file_sha256": file_hash(config_path),
                "checkpoint_sha256": checkpoint_hash,
                "previous_checkpoint_sha256": previous_hash,
                "data_manifest": dm.manifest,
                "script_sha256": file_hash(__file__),
                "execution_seconds": time.perf_counter() - started,
                "interpretation": "Coherent heads on the same conditional checkpoint; not retrained ablation or proof of future-class transfer",
            }
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--previous-checkpoint", type=Path)
    parser.add_argument("--device", type=int, help="-1 for CPU, otherwise CUDA index")
    parser.add_argument("--output-json", required=True, type=Path)
    options = parser.parse_args()
    if options.output_json.exists():
        parser.error("Choose a new output path")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = json.loads(options.config.read_text())
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    report = diagnose_checkpoint(
        options.config, options.checkpoint, options.previous_checkpoint, options.device
    )
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    options.output_json.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage": report["stage"],
                "branches": {
                    name: {
                        k: value[k]
                        for k in [
                            "accuracy",
                            "old_class_accuracy",
                            "new_class_accuracy",
                        ]
                    }
                    for name, value in report["branches"].items()
                },
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
