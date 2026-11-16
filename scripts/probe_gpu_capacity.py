#!/usr/bin/env python3
"""Synthetic FP32 ViT capacity probe; no dataset, tuning, or accuracy results.

This measures one plastic ViT and CE head, not the complete dual-branch or
module pipeline. GPU-only step timings are not end-to-end experiment ETAs.
The pinned pretrained weight may be downloaded by timm if not already cached.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATCHES = (64, 128)
SEED = 1993
WARMUP_STEPS = 1
TIMED_STEPS = 3
CLASSES = 200


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def probe_batch(torch, device, batch_size):
    from cl27.common import build_backbone, seed_everything
    from cl27.transport_training import expand_head

    seed_everything(SEED)
    backbone = build_backbone(
        {
            "model_name": "er",
            "backbone_type": "vit_base_patch16_224",
            "pretrained": True,
            "seed": SEED,
        }
    ).to(device=device, dtype=torch.float32)
    head = expand_head(None, backbone.out_dim, CLASSES, device)
    backbone.train()
    head.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    # Reuse fixed synthetic GPU tensors; no image decoding or data loading is
    # represented by these timings. The 200-output head covers ImageNet-R.
    images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32)
    labels = torch.randint(CLASSES, (batch_size,), device=device)

    def step():
        loss = torch.nn.functional.cross_entropy(head(backbone(images)), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite backbone adaptation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in backbone.parameters() if p.grad is not None]
        norm = torch.sqrt(sum(g.detach().square().sum() for g in gradients))
        gradient = float(norm)
        if not math.isfinite(gradient):
            raise FloatingPointError("Nonfinite backbone gradient")
        optimizer.step()
        # Reuse adapt_backbone's existing loss/norm host reads rather than
        # introducing a different fused or mixed-precision training loop.
        return float(loss.detach()), gradient

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(WARMUP_STEPS):
        step()
    torch.cuda.synchronize(device)
    durations = []
    losses = []
    gradient_norms = []
    for _ in range(TIMED_STEPS):
        began = time.perf_counter()
        loss, gradient = step()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - began)
        losses.append(loss)
        gradient_norms.append(gradient)
    return {
        "status": "complete",
        "logical_batch_size": batch_size,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "seconds_per_step": sum(durations) / len(durations),
        "step_seconds": durations,
        "synthetic_images_per_second": batch_size * len(durations) / sum(durations),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "trainable_parameter_count": sum(
            p.numel() for module in (backbone, head) for p in module.parameters()
            if p.requires_grad
        ),
        "parameter_count": sum(
            p.numel() for module in (backbone, head) for p in module.parameters()
        ),
        "head_classes": CLASSES,
        "synthetic_loss": losses,
        "backbone_gradient_l2": gradient_norms,
        "backbone_provenance": backbone.cl27_provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-json", type=Path, required=True)
    options = parser.parse_args()
    if options.device < 0:
        parser.error("--device must be a nonnegative CUDA index")
    output = options.output_json.expanduser().resolve()
    if output.exists():
        parser.error("--output-json already exists; choose a new report path")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "purpose": "synthetic GPU capacity and throughput only",
        "dataset_access": False,
        "accuracy_evaluation": False,
        "hyperparameter_tuning": False,
        "timing_scope": "GPU-resident synthetic supervised steps; not end-to-end ETA",
        "capacity_scope": "one plastic ViT plus CE head; no anchor or module library",
        "seed": SEED,
        "planned_logical_batches": list(BATCHES),
        "precision": "float32",
        "autocast": False,
        "tf32": False,
        "backbone_lr": 1e-5,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": options.device,
        "batches": [],
    }
    batch_size = None
    device = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this GPU-only probe")
        if options.device >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device index is unavailable")
        device = torch.device("cuda", options.device)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        report["device_name"] = properties.name
        report["device_total_memory_bytes"] = properties.total_memory
        report["compute_capability"] = [properties.major, properties.minor]
        write_report(output, report)
        for batch_size in BATCHES:
            gc.collect()
            torch.cuda.empty_cache()
            result = probe_batch(torch, device, batch_size)
            report["batches"].append(result)
            write_report(output, report)
            print(json.dumps(result, allow_nan=False), flush=True)
        report["status"] = "complete"
        write_report(output, report)
        return 0
    except Exception as error:
        report["status"] = (
            "out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError) else "error"
        )
        report["failed_logical_batch_size"] = batch_size
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if device is not None:
            report["gpu_peak_allocated_bytes_at_failure"] = (
                torch.cuda.max_memory_allocated(device)
            )
            report["gpu_peak_reserved_bytes_at_failure"] = (
                torch.cuda.max_memory_reserved(device)
            )
        write_report(output, report)
        print(json.dumps(report, allow_nan=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
