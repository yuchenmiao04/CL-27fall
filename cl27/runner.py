"""PILOT's task loop with structured records and validation-only short runs."""

from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from utils import factory
from .common import (
    UPSTREAM_COMMIT,
    NEW_METHODS,
    jsonable,
    numeric_bytes,
    seed_everything,
)
from .data import ProtocolDataManager
from .protocol import validate_protocol


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def configuration_fingerprint(args):
    ignore = {
        "output_dir",
        "resume",
        "max_tasks",
        "config",
        "device",
        "num_workers",
        "cpu_threads",
    }
    value = {k: v for k, v in args.items() if k not in ignore}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA for exact continuation")
        torch.cuda.set_rng_state_all(state["cuda"])


def _raw_metrics(raw, dm, task):
    labels = np.asarray(raw["labels"])
    predictions = np.asarray(raw["predictions"])
    if len(labels) == 0 or labels.shape != predictions.shape:
        raise ValueError("Invalid evaluation record")
    seen = sum(dm.get_task_size(i) for i in range(task + 1))
    if np.any((labels < 0) | (labels >= seen)) or np.any(
        (predictions < 0) | (predictions >= seen)
    ):
        raise ValueError("Prediction or target escaped all-seen vocabulary")
    correct = predictions == labels
    matrix = []
    counts = []
    start = 0
    for j in range(dm.nb_tasks):
        width = dm.get_task_size(j)
        mask = (labels >= start) & (labels < start + width)
        start += width
        matrix.append(float(correct[mask].mean()) if mask.any() else None)
        counts.append(int(mask.sum()))
    return {
        "accuracy": float(correct.mean()),
        "task_accuracy": matrix,
        "task_counts": counts,
        "evaluation_images": len(labels),
    }


def stage_record(model, cnn, nme, dm, task):
    raw = getattr(model, "_evaluation_record", None)
    if raw is None:
        # Upstream methods do not cache raw predictions; one explicit extra eval
        # supplies exact matrices instead of reverse-engineering rounded logs.
        p, y = model._eval_cnn(model.test_loader)
        raw = {
            "predictions": p[:, 0].astype(int).tolist(),
            "labels": y.astype(int).tolist(),
        }
        extra_eval = True
    else:
        extra_eval = False
    heads = {"cnn": _raw_metrics(raw, dm, task)}
    if nme is not None:
        p, y = model._eval_nme(model.test_loader, model._class_means)
        nme_raw = {
            "predictions": p[:, 0].astype(int).tolist(),
            "labels": y.astype(int).tolist(),
        }
        if nme_raw["labels"] != raw["labels"]:
            raise ValueError("CNN and NME evaluation rows disagree")
        heads["nme"] = _raw_metrics(nme_raw, dm, task)
    primary = getattr(model, "args", {}).get("evaluation_head", "cnn")
    if primary not in heads:
        raise ValueError("The preselected evaluation head is unavailable")
    seen = sum(dm.get_task_size(i) for i in range(task + 1))
    return {
        "task": task,
        "seen_classes": seen,
        **heads[primary],
        "prediction_space": "all_seen_classes_without_task_id",
        "cnn_report": cnn,
        "nme_report": nme,
        "evaluation_head": primary,
        "heads": heads,
        "extra_evaluation_pass_for_raw_matrix": extra_eval,
        "extra_nme_evaluation_pass_for_raw_matrix": nme is not None,
    }


def summarize(stages):
    if not stages:
        return {
            "final_accuracy": None,
            "aia": None,
            "forgetting_previous": None,
            "forgetting_including_final": None,
            "accuracy_matrix": [],
        }
    matrix = [s["task_accuracy"] for s in stages]
    n = len(stages)
    previous = []
    inclusive = []
    for j in range(n - 1):
        best = max(matrix[t][j] for t in range(j, n - 1))
        previous.append(best - matrix[-1][j])
        inclusive.append(max(best, matrix[-1][j]) - matrix[-1][j])
    return {
        "final_accuracy": stages[-1]["accuracy"],
        "aia": float(np.mean([s["accuracy"] for s in stages])),
        "forgetting_previous": float(np.mean(previous)) if previous else None,
        "forgetting_including_final": float(np.mean(inclusive)) if inclusive else None,
        "accuracy_matrix": matrix,
        "completed_tasks": n,
        "definition": "Fractions; AIA is mean stage sample-weighted all-seen accuracy; forgetting averages old tasks only.",
    }


def run(args):
    args = copy.deepcopy(args)
    args.setdefault("phase", "final")
    validate_protocol(args)
    if not args.get("smoke", False):
        import timm

        if timm.__version__ != "0.6.12":
            raise RuntimeError("All structured PILOT runs require timm==0.6.12")
    if args.get("max_tasks") is not None and not args.get("validation_only", False):
        raise ValueError("A truncated stream must be explicitly validation-only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(args.get("cpu_threads", 4)))
    seed_everything(int(args["seed"]))
    output = Path(args.get("output_dir", "runs/" + args["model_name"]))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use a new output directory; resuming also writes to a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = configuration_fingerprint(args)
    write_json(output / "config.json", args)
    devices = [
        torch.device("cpu" if int(x) == -1 else "cuda:" + str(x))
        for x in args["device"]
    ]
    if len(devices) != 1:
        raise ValueError("Independent single-GPU runs are the validated execution mode")
    device = devices[0]
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dm = ProtocolDataManager(
        args["dataset"],
        args.get("shuffle", True),
        int(args["seed"]),
        int(args["init_cls"]),
        int(args["increment"]),
        args,
    )
    write_json(output / "data_manifest.json", dm.manifest)
    args["device"] = devices
    args["nb_classes"] = dm.nb_classes
    args["nb_tasks"] = dm.nb_tasks
    model = factory.get_model(args["model_name"], args)
    stages = []
    start = 0
    if args.get("resume"):
        if args["model_name"] not in NEW_METHODS:
            raise ValueError(
                "Task-boundary resume is implemented only for the three new learners"
            )
        state = torch.load(args["resume"], map_location="cpu", weights_only=False)
        if (
            state["config_fingerprint"] != fingerprint
            or state["data_fingerprint"] != dm.manifest["fingerprint"]
        ):
            raise ValueError("Resume configuration or dataset identity changed")
        model.restore_checkpoint(state["learner"])
        stages = copy.deepcopy(state["stages"])
        start = model._cur_task + 1
        expected = sum(dm.get_task_size(i) for i in range(start))
        if model._known_classes != expected or len(stages) != start:
            raise ValueError("Resume task/class counters disagree")
        restore_rng(state["rng"])
    stop = min(dm.nb_tasks, int(args.get("max_tasks") or dm.nb_tasks))
    began = time.perf_counter()
    for task in range(start, stop):
        tick = time.perf_counter()
        model.incremental_train(dm)
        cnn, nme = model.eval_task()
        row = stage_record(model, cnn, nme, dm, task)
        model.after_task()
        row["wall_seconds"] = time.perf_counter() - tick
        stages.append(row)
        write_json(
            output / "progress.json",
            {"status": "running", "stages": stages, "metrics": summarize(stages)},
        )
        if args.get("save_checkpoints", True) and args["model_name"] in NEW_METHODS:
            state = {
                "version": 1,
                "config_fingerprint": fingerprint,
                "data_fingerprint": dm.manifest["fingerprint"],
                "learner": model.checkpoint_state(),
                "stages": stages,
                "rng": rng_state(),
            }
            temporary = output / "checkpoint.tmp"
            torch.save(state, temporary)
            temporary.replace(output / "checkpoint.pt")
        print(
            json.dumps(
                {
                    "task_complete": task,
                    "phase": args["phase"],
                    "accuracy": row["accuracy"],
                    "classes": row["seen_classes"],
                },
                allow_nan=False,
            ),
            flush=True,
        )
    status = "complete" if stop == dm.nb_tasks else "validation_partial"
    provenance = getattr(getattr(model, "backbone", None), "cl27_provenance", None)
    if provenance is None:
        provenance = getattr(model, "backbone_provenance", None)
    state_bytes = (
        numeric_bytes(model.checkpoint_core())
        if args["model_name"] in NEW_METHODS
        else None
    )
    head_metrics = {
        head: summarize([{**row, **row["heads"][head]} for row in stages])
        for head in ("cnn", "nme")
        if stages and all(head in row["heads"] for row in stages)
    }
    result = {
        "status": status,
        "method": args["model_name"],
        "phase": args["phase"],
        "performance_claim": (
            False
            if args.get("validation_only")
            else "formal evaluation, not a verified improvement"
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "config_fingerprint": fingerprint,
        "data_fingerprint": dm.manifest["fingerprint"],
        "metrics": summarize(stages),
        "stages": stages,
        "evaluation_head": args.get("evaluation_head", "cnn"),
        "metrics_by_head": head_metrics,
        "task_diagnostics": getattr(model, "task_diagnostics", []),
        "backbone_provenance": provenance,
        "execution": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device_type": device.type,
            "wall_seconds": time.perf_counter() - began,
            "numeric_checkpoint_payload_bytes": state_bytes,
            "gpu_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    write_json(output / "results.json", result)
    return result
