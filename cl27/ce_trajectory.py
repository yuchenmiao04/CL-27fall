"""Identity-checked reuse of complete, native CE adaptation calls.

This is an execution cache, not an algorithmic memory. ER records one stage's
post-training body/head and RNG states; independently initialized native runs
may reuse that call only when its complete precondition digest matches. No
feature vectors or historical images are supplied by the cache. Every learner
still executes its own legal feature encoding, moments, herding and evaluation.

Cache files are trusted local torch checkpoints, not untrusted downloadable
artifacts. Logical devices are deliberately bound; no cross-device RNG remap or
numerically approximate reuse is attempted.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import file_hash, state_hash
from .runner import RANDOM_STREAM_POLICY, restore_rng, rng_state


METHODS = frozenset({"er", "history_transport", "plastic_shared_basis"})
VERSION = 1


class CEReuseMismatch(ValueError):
    """A cache is absent or differs from the exact native CE call requested."""


def _digest(value):
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            h.update(b"torch" + str(item.dtype).encode())
            add(tuple(item.shape))
            h.update(item.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(b"numpy" + str(item.dtype).encode())
            add(tuple(item.shape))
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            add(item.item())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            add(len(item))
            for part in item:
                add(part)
        elif item is None or isinstance(item, (str, bool, int, float)):
            h.update(type(item).__name__.encode())
            h.update(json.dumps(item, allow_nan=False).encode())
            h.update(b"\0")
        else:
            raise TypeError(f"Unsupported trajectory identity value: {type(item)}")

    add(value)
    return h.hexdigest()


def _state_dict_hash(state):
    """The same state encoding as common.state_hash, without touching live weights."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise CEReuseMismatch("Cached CE state contains invalid tensors")
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _raw_image_hash(image):
    if isinstance(image, np.ndarray):
        return _digest(image)
    if isinstance(image, (str, np.str_)):
        return file_hash(str(image))
    raise CEReuseMismatch("CE reuse requires native raw-array or file-path images")


def _dataset_identity(dataset):
    if not all(hasattr(dataset, key) for key in ("images", "labels", "ids", "trsf")):
        raise CEReuseMismatch("CE reuse requires native stable-ID datasets")
    if not (len(dataset) == len(dataset.images) == len(dataset.labels) == len(dataset.ids)):
        raise CEReuseMismatch("CE dataset identity arrays are misaligned")
    return {
        "ids_sha256": _digest(np.asarray(dataset.ids)),
        "labels_sha256": _digest(np.asarray(dataset.labels)),
        "ordered_raw_images_sha256": _digest([_raw_image_hash(x) for x in dataset.images]),
        "count": len(dataset),
        "dataset_class": type(dataset).__module__ + "." + type(dataset).__qualname__,
        "training_transform": repr(dataset.trsf),
        "use_path": bool(getattr(dataset, "use_path", False)),
    }


def _replay_identity(replay):
    return {
        "fixed": replay.fixed,
        "memory_size": replay.memory_size,
        "memory_per_class": replay.memory_per_class,
        "selection": replay.selection,
        "classes": list(replay.classes),
        "seen": replay.seen,
        "selection_rng": copy.deepcopy(replay.rng.bit_generator.state),
        "ordered_records_sha256": _digest([
            (r.sample_id, r.label, _raw_image_hash(r.image)) for r in replay.records
        ]),
        "count": len(replay),
    }


def _hyperparameters(args):
    return {
        "epochs": int(args.get("epochs", 20)),
        "batch_size": int(args.get("batch_size", 64)),
        "replay_batch_size": int(args.get("replay_batch_size", 64)),
        "backbone_lr": float(args.get("backbone_lr", 1e-5)),
        "head_lr": float(args.get("head_lr", 1e-3)),
        "weight_decay": float(args.get("weight_decay", 1e-4)),
        "num_workers": int(args.get("num_workers", 0)),
        "feature_batch_size": int(args.get("feature_batch_size", 32)),
        "seed": int(args["seed"]),
    }


def configuration_reuse_identity(args):
    """Conservative CLI preflight; live call guards remain authoritative."""
    defaults = {
        "dataset": None, "phase": "final", "data_root": "data",
        "smoke": False, "validation_only": False, "validation_fraction": 0.1,
        "split_seed": 0, "validation_max_per_class": None, "max_tasks": None,
        "shuffle": True, "init_cls": None, "increment": None,
        "backbone_type": None, "pretrained": False, "cpu_threads": 4,
        "synthetic_classes": 6, "synthetic_per_class": 12,
        "image_size": 32, "smoke_feature_dim": 24, "smoke_depth": 3,
        "fixed_memory": False, "memory_size": 0, "memory_per_class": 0,
        "replay_selection": "herding",
    }
    result = {k: args.get(k, value) for k, value in defaults.items()}
    result.update(training=_hyperparameters(args),
                  device=[str(value) for value in args["device"]],
                  backbone_seed=args.get("backbone_seed", args["seed"]))
    return _digest(result)


def _numeric_policy(device):
    result = {
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "logical_device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast": torch.is_autocast_enabled(),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update(cuda=torch.version.cuda, gpu_name=p.name, capability=[p.major, p.minor])
    return result


class CEStageCache:
    """Record or replay immutable, per-stage native CE state transitions."""

    def __init__(self, directory, mode):
        if mode not in {"record", "replay"}:
            raise ValueError("CE cache mode must be record or replay")
        self.directory = Path(directory).expanduser().resolve()
        self.mode = mode
        if mode == "record":
            self.directory.mkdir(parents=True, exist_ok=True)
        elif not self.directory.is_dir():
            raise CEReuseMismatch("CE trajectory directory does not exist")

    def for_run(self, args, data_manifest):
        if args["model_name"] not in METHODS:
            raise CEReuseMismatch("Unsupported CE reuse method")
        if self.mode == "record" and args["model_name"] != "er":
            raise CEReuseMismatch("Only the plain ER control records native CE trajectories")
        if not isinstance(data_manifest.get("fingerprint"), str):
            raise CEReuseMismatch("A verified data manifest fingerprint is required")
        return _BoundCache(self, args, data_manifest)


class _BoundCache:
    def __init__(self, cache, args, manifest):
        self.cache = cache
        self.method = args["model_name"]
        self.data_fingerprint = manifest["fingerprint"]
        self.phase = args.get("phase", "final")
        self.records = []

    def report(self):
        return {
            "version": VERSION,
            "mode": self.cache.mode,
            "execution_role": (
                "internal_shared_ce_driver" if self.method == "er" else "candidate_method"
            ),
            "trajectory_directory": str(self.cache.directory),
            "stages": copy.deepcopy(self.records),
            "timing_policy": "Run wall time is actual; reused CE seconds are shared driver work, not separately executed training",
            "feature_cache": False,
            "independent_native_initialization_and_evaluation": True,
        }

    def adapt(self, owner, native, body, head, current, replay, args, device, rng):
        began = time.perf_counter()
        if owner.args["model_name"] != self.method or args.get("phase", "final") != self.phase:
            raise CEReuseMismatch("CE executor was attached to a different run")
        parameters = _hyperparameters(args)
        n, batch = len(current), parameters["batch_size"]
        if n < 1 or batch < 1:
            raise CEReuseMismatch("Invalid native CE batch plan")
        extra = min(parameters["replay_batch_size"], len(replay)) if len(replay) else 0
        sizes = [min(batch, n - i) + extra for i in range(0, n, batch)]
        request = {
            "version": VERSION,
            "random_stream_policy": RANDOM_STREAM_POLICY,
            "data_fingerprint": self.data_fingerprint,
            "phase": self.phase,
            "task": owner._cur_task,
            "known_classes": owner._known_classes,
            "total_classes": owner._total_classes,
            "native_function_sha256": hashlib.sha256(inspect.getsource(native).encode()).hexdigest(),
            "body_before_sha256": state_hash(body),
            "head_before_sha256": state_hash(head),
            "body_architecture": repr(body),
            "requires_grad": [p.requires_grad for p in body.parameters()],
            "current": _dataset_identity(current),
            "replay": _replay_identity(replay),
            "hyperparameters": parameters,
            "effective_combined_batch_sizes_per_epoch": sizes,
            "replay_sampling_rng": copy.deepcopy(rng.bit_generator.state),
            "global_rng_sha256": _digest(rng_state()),
            "numeric_policy": _numeric_policy(device),
        }
        fingerprint = _digest(request)
        filename = self.cache.directory / f"stage_{owner._cur_task:03d}.pt"
        if self.cache.mode == "record":
            if filename.exists():
                raise CEReuseMismatch("Refusing to overwrite an existing CE trajectory stage")
            tick = time.perf_counter()
            training = native(body, head, current, replay, args, device, rng)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            native_seconds = time.perf_counter() - tick
            record = {
                "version": VERSION,
                "request": request,
                "request_fingerprint": fingerprint,
                "body": {k: v.detach().cpu().clone() for k, v in body.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "body_after_sha256": state_hash(body),
                "head_after_sha256": state_hash(head),
                "global_rng_after": rng_state(),
                "replay_sampling_rng_after": copy.deepcopy(rng.bit_generator.state),
                "training": copy.deepcopy(training),
                "native_ce_seconds": native_seconds,
            }
            temporary = filename.with_suffix(".tmp")
            torch.save(record, temporary)
            temporary.replace(filename)
        else:
            if not filename.is_file():
                raise CEReuseMismatch(f"Missing complete CE trajectory stage: {filename}")
            record = torch.load(filename, map_location="cpu", weights_only=False)
            if (
                record.get("version") != VERSION
                or record.get("request_fingerprint") != fingerprint
                or _digest(record.get("request")) != fingerprint
            ):
                previous = record.get("request", {})
                changed = sorted(k for k in request if _digest(request[k]) != _digest(previous.get(k)))
                raise CEReuseMismatch("CE preconditions differ: " + ", ".join(changed))
            if (
                _state_dict_hash(record["body"]) != record["body_after_sha256"]
                or _state_dict_hash(record["head"]) != record["head_after_sha256"]
            ):
                raise CEReuseMismatch("Cached CE tensors do not match their recorded identity")
            body.load_state_dict(record["body"], strict=True)
            head.load_state_dict(record["head"], strict=True)
            body.eval()
            head.eval()
            rng.bit_generator.state = copy.deepcopy(record["replay_sampling_rng_after"])
            restore_rng(record["global_rng_after"])
            training = copy.deepcopy(record["training"])
            native_seconds = record["native_ce_seconds"]
        self.records.append({
            "task": owner._cur_task,
            "path": str(filename),
            "request_fingerprint": fingerprint,
            "body_after_sha256": record["body_after_sha256"],
            "head_after_sha256": record["head_after_sha256"],
            "reused": self.cache.mode == "replay",
            "shared_native_ce_seconds": native_seconds,
            "executor_wall_seconds": time.perf_counter() - began,
        })
        if self.cache.mode == "replay":
            print(json.dumps({"ce_training_reused": True, "method": self.method,
                              "task": owner._cur_task, "training_fingerprint": fingerprint}), flush=True)
        return training
