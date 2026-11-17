#!/usr/bin/env python3
"""Bounded numerical/speed gate for opt-in current-method prefix reuse.

Uses synthetic RGB images, the real cached AugReg ViT by default, and identical
PILOT augmentations for both engines. This is not an accuracy experiment or
hyperparameter search. Existing training processes and files are not touched.
The default timing context is shared GPU: speed is then descriptive, not a
passed isolated speed gate. A tiny CPU mode checks this script's control flow.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision import transforms

from cl27.common import build_backbone, seed_everything
from cl27.current.engine import FixedLibraryEngine
from cl27.current.episodes import sample_episode
from models.current_adapted import RawTransform

SEED = 1993
CORRECTNESS_STEPS = 2
WEIGHT_FILE = (
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--"
    "imagenet2012-steps_20k-lr_0.01-res_224.npz"
)
TOLERANCES = {
    "loss": {"atol": 1e-7, "rtol": 1e-6},
    "gradient": {"atol": 2e-7, "rtol": 2e-5},
    "parameter": {"atol": 2e-7, "rtol": 2e-6},
}


class RecordingTransform:
    def __init__(self, transform):
        self.transform = RawTransform(transform)
        self.recording = False
        self.digests = []

    def __call__(self, image):
        value = self.transform(image)
        if self.recording:
            self.digests.append(hashlib.sha256(value.numpy().tobytes()).hexdigest())
        return value

    def reset(self, recording):
        self.recording = recording
        self.digests = []


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": copy_numpy_state(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        ),
    }


def copy_numpy_state(value):
    return (value[0], value[1].copy(), *value[2:])


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"], device)


def same_rng(a, b):
    na, nb = a["numpy"], b["numpy"]
    return (
        a["python"] == b["python"]
        and na[0] == nb[0]
        and np.array_equal(na[1], nb[1])
        and na[2:] == nb[2:]
        and torch.equal(a["torch"], b["torch"])
        and (
            (a["cuda"] is None and b["cuda"] is None)
            or (
                a["cuda"] is not None
                and b["cuda"] is not None
                and torch.equal(a["cuda"], b["cuda"])
            )
        )
    )


def compare_tensors(named_a, named_b, tolerance):
    """Check every tensor with one packed scalar transfer, including None grads."""
    if list(named_a) != list(named_b):
        raise RuntimeError("Tensor names/order differ")
    scalars, names, none_count = [], [], 0
    for name, a in named_a.items():
        b = named_b[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise RuntimeError("Gradient presence differs for " + name)
            none_count += 1
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("Tensor shape/dtype differs for " + name)
        error = (a.detach() - b.detach()).abs()
        scale = tolerance["atol"] + tolerance["rtol"] * b.detach().abs()
        scalars.append(torch.stack((error.max(), (error / scale).max())).double())
        names.append(name)
    if not scalars:
        raise RuntimeError("No numerical tensors were checked")
    values = torch.stack(scalars).cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite tensor difference")
    worst = int(np.argmax(values[:, 1]))
    return {
        "passed": bool(values[:, 1].max() <= 1.0),
        "max_absolute_difference": float(values[:, 0].max()),
        "max_tolerance_ratio": float(values[:, 1].max()),
        "worst_tensor": names[worst],
        "checked_tensors": len(names),
        "matching_none_gradients": none_count,
    }


def compare_engines(control, fast, a, b):
    loss_tolerance = TOLERANCES["loss"]
    loss_ok = math.isclose(
        a["query_ce"],
        b["query_ce"],
        abs_tol=loss_tolerance["atol"],
        rel_tol=loss_tolerance["rtol"],
    )
    gradients = compare_tensors(
        {name: p.grad for name, p in control.encoder.named_parameters()},
        {name: p.grad for name, p in fast.encoder.named_parameters()},
        TOLERANCES["gradient"],
    )
    parameters = compare_tensors(
        dict(control.encoder.named_parameters()),
        dict(fast.encoder.named_parameters()),
        TOLERANCES["parameter"],
    )
    counters_ok = (
        control.counters == fast.counters
        and control.updates == fast.updates
        and control.module_updates == fast.module_updates
        and control.backbone_updates == fast.backbone_updates
    )
    return {
        "passed": loss_ok
        and gradients["passed"]
        and parameters["passed"]
        and counters_ok,
        "loss_passed": loss_ok,
        "control_loss": a["query_ce"],
        "prefix_loss": b["query_ce"],
        "gradient": gradients,
        "parameter": parameters,
        "logical_counters_equal": counters_ok,
    }


def validate_options(options):
    if options.device < 0 and not options.tiny:
        raise ValueError("The bounded CPU check requires --tiny")
    if not 2 <= options.classes <= 32:
        raise ValueError("Use 2-32 episode classes")
    if (
        options.samples_per_class < 3
        or options.classes * options.samples_per_class > 256
    ):
        raise ValueError("Use at least three samples/class and at most 256 pool rows")
    if (
        min(
            options.support_shots
            + [
                options.background_shots,
                options.target_classes,
                options.query_per_class,
            ]
        )
        < 1
    ):
        raise ValueError("Shots/classes/query counts must be positive")
    if not 1 <= options.microbatch <= 16:
        raise ValueError(
            "Use microbatch 1-16; the measured 8-GiB gate remains mandatory"
        )
    if not 0 <= options.warmup <= 2 or not 1 <= options.steps <= 5:
        raise ValueError("Use 0-2 warmup pairs and 1-5 timed pairs")
    if not 0 < options.max_seconds <= 110:
        raise ValueError("The runtime budget must be at most 110 seconds")
    if not 0 <= options.library_u_std <= 0.1:
        raise ValueError("Use a bounded nonzero-library stress initialization")
    if (
        not options.layers
        or len(set(options.layers)) != len(options.layers)
        or min(options.layers) < 0
        or max(options.layers) > 10
    ):
        raise ValueError("Library positions must be unique and before the final block")


def run_probe(options):
    validate_options(options)
    started = time.perf_counter()
    device = torch.device(
        "cpu" if options.device < 0 else "cuda:" + str(options.device)
    )
    report = {
        "status": "running",
        "seed": SEED,
        "purpose": "synthetic-input real-encoder numerical/resource gate; not accuracy or tuning",
        "genuine_pretrained_encoder": not options.tiny,
        "precision": "float32",
        "tf32": False,
        "autocast": False,
        "torch": str(torch.__version__),
        "timm": None,
        "configuration": vars(options).copy(),
        "tolerances": TOLERANCES,
        "correctness_steps": [],
        "timed_pairs": [],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "memory_scope": "this process with both comparison engines resident; excludes other GPU processes",
        "correctness_passed": False,
        "resource_limits_passed": False,
        "speed_passed": None,
        "enable_recommended": False,
    }
    report["configuration"].pop("output_json", None)
    if options.torch_home:
        report["configuration"]["torch_home"] = "explicit cache directory"
        os.environ["TORCH_HOME"] = str(options.torch_home)

    def deadline():
        if time.perf_counter() - started > options.max_seconds:
            raise RuntimeError(
                "Runtime budget exceeded; stop without relaxing the gate"
            )

    def record_memory():
        if device.type == "cuda":
            report["peak_allocated_bytes"] = max(
                report["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device)
            )
            report["peak_reserved_bytes"] = max(
                report["peak_reserved_bytes"], torch.cuda.max_memory_reserved(device)
            )
            if report["peak_reserved_bytes"] >= 8 * 1024**3:
                raise RuntimeError(
                    "The validation process exceeded its 8-GiB memory budget"
                )

    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            report["device_name"] = torch.cuda.get_device_name(device)
            free, total = torch.cuda.mem_get_info(device)
            report.update(initial_free_bytes=free, device_total_memory_bytes=total)
            if free < 6 * 1024**3:
                raise RuntimeError(
                    "Less than 6 GiB is free; do not contend with the existing job"
                )
        else:
            report["device_name"] = "CPU tiny control-flow check"
        torch.set_num_threads(options.cpu_threads)
        seed_everything(SEED)
        if not options.tiny:
            weight = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_FILE
            if not weight.is_file():
                raise FileNotFoundError(
                    "Prepare the fixed pretrained weight cache separately; this bounded gate never downloads it"
                )
            import timm

            report["timm"] = timm.__version__
        args = {
            "model_name": "current_adapted",
            "seed": SEED,
            "backbone_type": "vit_base_patch16_224",
            "pretrained": True,
        }
        if options.tiny:
            args.update(
                smoke=True,
                dataset="synthetic",
                pretrained=False,
                smoke_feature_dim=12,
                smoke_depth=12,
                image_size=16,
            )
        backbone = build_backbone(args)
        report["backbone_provenance"] = backbone.cl27_provenance
        side = 16 if options.tiny else 224
        canonical = transforms.Compose(
            [
                transforms.Resize(int(side * 256 / 224), interpolation=3),
                transforms.CenterCrop(side),
                transforms.ToTensor(),
            ]
        )
        augmented = transforms.Compose(
            [
                transforms.RandomResizedCrop(side, scale=(0.05, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        common_options = {
            "feature_batch_size": options.microbatch,
            "memory_efficient": True,
            "seed": SEED,
            "support_shots": tuple(options.support_shots),
            "background_shots": options.background_shots,
            "target_classes": options.target_classes,
            "query_per_class": options.query_per_class,
        }
        engines = []
        old_flag = os.environ.get("CL27_CURRENT_FROZEN_PREFIX_REUSE")
        try:
            for flag in ["0", "1"]:
                os.environ["CL27_CURRENT_FROZEN_PREFIX_REUSE"] = flag
                transform = RecordingTransform(augmented)
                engines.append(
                    FixedLibraryEngine(
                        backbone,
                        positions=options.layers,
                        count=4,
                        rank=8,
                        options=common_options,
                        device=device,
                        canonical_transform=RawTransform(canonical),
                        training_transform=transform,
                    )
                )
        finally:
            if old_flag is None:
                os.environ.pop("CL27_CURRENT_FROZEN_PREFIX_REUSE", None)
            else:
                os.environ["CL27_CURRENT_FROZEN_PREFIX_REUSE"] = old_flag
        del backbone
        gc.collect()
        control, fast = engines
        with torch.no_grad():
            for library in control.encoder.libraries:
                library.U.normal_(0, options.library_u_std)
        fast.encoder.load_state_dict(control.encoder.state_dict())
        report["stress_library_u_std"] = options.library_u_std
        report["supported_prefix_recipe"] = fast.encoder.frozen_prefix_kind()
        if (
            report["supported_prefix_recipe"] is None
            or control.frozen_prefix_reuse
            or not fast.frozen_prefix_reuse
        ):
            raise RuntimeError("The intended prefix path is not eligible/enabled")
        record_memory()
        n = options.classes * options.samples_per_class
        pool_rng = np.random.default_rng(SEED)
        raw = pool_rng.integers(0, 256, (n, side + 32, side + 32, 3), dtype=np.uint8)
        items = [
            (image, i // options.samples_per_class, i) for i, image in enumerate(raw)
        ]
        labels = torch.tensor([row[1] for row in items])
        canonical_features = []
        for core in engines:
            core.statistics.register(labels)
            canonical_features.append(core._canonical(items, True))
        torch.testing.assert_close(*canonical_features, rtol=0, atol=0)
        record_memory()
        episodes = {
            kind: sample_episode(
                labels,
                torch.empty(0, dtype=torch.long),
                kind,
                torch.Generator().manual_seed(SEED),
                options.support_shots,
                options.background_shots,
                options.target_classes,
                options.query_per_class,
            )
            for kind in ["transfer", "anchored"]
        }
        report["episode_rows"] = {
            kind: len(e.support_indices)
            + len(e.query_target_indices)
            + len(e.query_background_indices)
            for kind, e in episodes.items()
        }
        if max(report["episode_rows"].values()) > 64:
            raise ValueError("Episode exceeds 64 encoded rows; choose a smaller gate")
        versions = [
            tuple(p._version for p in core.encoder.main.parameters())
            for core in engines
        ]
        prefix_calls = [0]
        original = fast.encoder.main_features_from_prefix

        def count_prefix(prefix, kind):
            prefix_calls[0] += 1
            return original(prefix, kind)

        fast.encoder.main_features_from_prefix = count_prefix

        def run_one(core, index, episode):
            deadline()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            result = core.episode_step(
                episode, items, canonical_features[index], labels, len(items), "modules"
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - tick
            record_memory()
            deadline()
            return result, seconds

        def run_pair(index, record_views=False, check=False):
            kind = (
                options.episode_kind
                if options.episode_kind != "alternating"
                else ("transfer" if index % 2 == 0 else "anchored")
            )
            episode = episodes[kind]
            for core in engines:
                core.training_transform.reset(record_views)
            before = rng_state(device)
            a, ta = run_one(control, 0, episode)
            after_a = rng_state(device)
            restore_rng(before, device)
            b, tb = run_one(fast, 1, episode)
            if not same_rng(after_a, rng_state(device)):
                raise RuntimeError("Prefix reuse changed RNG consumption")
            result = {
                "kind": kind,
                "control_seconds": ta,
                "prefix_seconds": tb,
                "control_loss": a["query_ce"],
                "prefix_loss": b["query_ce"],
            }
            if record_views:
                old, new = (core.training_transform.digests for core in engines)
                if old != new or not old:
                    raise RuntimeError(
                        "The engines did not receive identical augmented views"
                    )
                result.update(
                    identical_augmented_views=True,
                    view_count=len(old),
                    augmented_sequence_sha256=hashlib.sha256(
                        "".join(old).encode()
                    ).hexdigest(),
                )
            if check:
                result["comparison"] = compare_engines(control, fast, a, b)
                if not result["comparison"]["passed"]:
                    report["correctness_steps"].append(result)
                    raise RuntimeError(
                        "Numerical/counter comparison failed; keep prefix reuse disabled"
                    )
            for core, expected in zip(engines, versions):
                if (
                    tuple(p._version for p in core.encoder.main.parameters())
                    != expected
                ):
                    raise RuntimeError("A supposedly frozen main parameter changed")
            record_memory()
            return result

        index = 0
        for _ in range(CORRECTNESS_STEPS):
            report["correctness_steps"].append(run_pair(index, True, True))
            index += 1
        for _ in range(options.warmup):
            run_pair(index)
            index += 1
        for _ in range(options.steps):
            report["timed_pairs"].append(run_pair(index))
            index += 1
        # Include the accumulated timed updates in a final numerical check.
        report["final_comparison"] = compare_engines(
            control,
            fast,
            {"query_ce": report["timed_pairs"][-1]["control_loss"]},
            {"query_ce": report["timed_pairs"][-1]["prefix_loss"]},
        )
        if not report["final_comparison"]["passed"] or prefix_calls[0] == 0:
            raise RuntimeError(
                "Accumulated updates differ or prefix reuse did not execute"
            )
        record_memory()
        deadline()
        old = [pair["control_seconds"] for pair in report["timed_pairs"]]
        new = [pair["prefix_seconds"] for pair in report["timed_pairs"]]
        ratio = statistics.median(old) / statistics.median(new)
        report.update(
            status="complete",
            correctness_passed=True,
            resource_limits_passed=True,
            prefix_replay_calls=prefix_calls[0],
            frozen_main_versions_unchanged=True,
            logical_counters=control.counters,
            optimizer_updates=control.updates,
            median_control_seconds=statistics.median(old),
            median_prefix_seconds=statistics.median(new),
            observed_median_speedup=ratio,
        )
        isolated_cuda = device.type == "cuda" and options.timing_context == "isolated"
        report["speed_passed"] = (ratio >= 1.03) if isolated_cuda else None
        report["enable_recommended"] = bool(isolated_cuda and report["speed_passed"])
        report["speed_interpretation"] = (
            "Isolated speed gate requires at least 3% median gain; repeat if timings are noisy."
            if isolated_cuda
            else "Shared-GPU/CPU timing is descriptive; repeat in an isolated CUDA slot before claiming a speed gain."
        )
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            enable_recommended=False,
        )
    report["wall_seconds"] = time.perf_counter() - started
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--tiny",
        action="store_true",
        help="Tiny synthetic CPU control-flow check; not a genuine encoder gate",
    )
    p.add_argument("--torch-home", type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--classes", type=int, default=4)
    p.add_argument("--samples-per-class", type=int, default=8)
    p.add_argument("--support-shots", type=int, nargs="+", default=[1, 5])
    p.add_argument("--background-shots", type=int, default=1)
    p.add_argument("--target-classes", type=int, default=5)
    p.add_argument("--query-per-class", type=int, default=2)
    p.add_argument("--layers", type=int, nargs="+", default=[7, 9])
    p.add_argument(
        "--microbatch",
        type=int,
        default=2,
        help="1-16; default 2, with an unchanged measured 8-GiB memory limit",
    )
    p.add_argument(
        "--library-u-std",
        type=float,
        default=0.03,
        help="Matched nonzero-library stress state; use 0 for original zero initialization",
    )
    p.add_argument(
        "--episode-kind",
        choices=["alternating", "transfer", "anchored"],
        default="alternating",
    )
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--timing-context", choices=["shared", "isolated"], default="shared")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=105)
    return p


def main():
    p = parser()
    options = p.parse_args()
    if options.output_json.exists():
        p.error("Choose a new output path")
    if not 1 <= options.cpu_threads <= 8:
        p.error("Use 1-8 CPU threads")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        report = run_probe(options)
    except Exception as error:
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "enable_recommended": False,
        }
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.output_json.with_suffix(options.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(options.output_json)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in [
                    "status",
                    "correctness_passed",
                    "resource_limits_passed",
                    "speed_passed",
                    "observed_median_speedup",
                    "peak_allocated_bytes",
                    "wall_seconds",
                    "enable_recommended",
                    "error",
                ]
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
