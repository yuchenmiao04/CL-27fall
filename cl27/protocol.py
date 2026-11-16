"""Fail visibly when a reported run is not the chosen historical protocol."""

import operator
from .common import NEW_METHODS

REPLAY_METHODS = {
    "history_transport",
    "current_adapted",
    "icarl",
    "coil",
    "der",
    "memo",
    "foster",
}


def validate_protocol(args):
    for key in ("init_cls", "increment"):
        value = args.get(key)
        if isinstance(value, bool):
            raise ValueError("Class increments must be positive integers")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError("Class increments must be positive integers") from error
        if value < 1:
            raise ValueError("Class increments must be positive integers")
    if args.get("max_tasks") is not None:
        value = args["max_tasks"]
        if isinstance(value, bool):
            raise ValueError("max_tasks must be a positive integer")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError("max_tasks must be a positive integer") from error
        if value < 1:
            raise ValueError("max_tasks must be a positive integer")
    if args.get("phase", "final") != "final" and not args.get("validation_only", False):
        raise ValueError(
            "Development and smoke runs must be explicitly validation-only"
        )
    if args.get("smoke", False):
        if args.get("dataset") != "synthetic" or args.get("phase") != "smoke":
            raise ValueError("Smoke dataset/phase mismatch")
        if args.get("pretrained", False):
            raise ValueError("Synthetic smoke must not be described as pretrained")
        return
    if args.get("phase", "final") == "smoke":
        raise ValueError("Non-synthetic smoke")
    if args["dataset"] not in {"cifar224", "imagenetr"}:
        raise ValueError("Unknown main protocol dataset")
    width = 10 if args["dataset"] == "cifar224" else 20
    if (args["init_cls"], args["increment"]) != (width, width):
        raise ValueError("Main comparison has ten equal stages")
    if args.get("model_name") in NEW_METHODS and args.get("backbone_type") not in {
        "vit_base_patch16_224",
        "pretrained_vit_b16_224",
    }:
        raise ValueError("New routes use the historical IN21K->IN1K checkpoint")
    if args["model_name"] in REPLAY_METHODS:
        if args["dataset"] == "cifar224" and (
            args.get("fixed_memory", False) or args.get("memory_size") != 2000
        ):
            raise ValueError("CIFAR-100 uses fixed TOTAL memory 2000")
        if args["dataset"] == "imagenetr" and (
            not args.get("fixed_memory", False) or args.get("memory_per_class") != 20
        ):
            raise ValueError("ImageNet-R uses fixed PER-CLASS memory 20")
    elif args["model_name"] == "shared_basis" and (
        args.get("memory_size", 0) or args.get("memory_per_class", 0)
    ):
        raise ValueError("Shared predictor basis is reported with zero raw replay")
    if args.get("phase", "final") == "final":
        if not args.get("shuffle", True):
            raise ValueError(
                "The direct PILOT comparison uses a seeded class permutation"
            )
        if args.get("max_tasks") is not None or args.get("validation_only", False):
            raise ValueError("Partial runs cannot be called formal final evaluation")
        if int(args["seed"]) != 1993 and not args.get("additional_order", False):
            raise ValueError(
                "The direct PILOT comparison uses seed 1993; label extra orders explicitly"
            )


def smoke_config(method, output_dir):
    return {
        "model_name": method,
        "cl27_protocol": "pilot_figure2",
        "dataset": "synthetic",
        "phase": "smoke",
        "smoke": True,
        "pretrained": False,
        "seed": [0],
        "device": [-1],
        "init_cls": 2,
        "increment": 2,
        "shuffle": True,
        "synthetic_classes": 6,
        "synthetic_per_class": 12,
        "image_size": 32,
        "smoke_feature_dim": 24,
        "smoke_depth": 3,
        "validation_only": True,
        "validation_fraction": 0.25,
        "split_seed": 0,
        "memory_size": 0 if method == "shared_basis" else 12,
        "memory_per_class": 0,
        "fixed_memory": False,
        "replay_selection": "herding",
        "epochs": 2,
        "batch_size": 8,
        "replay_batch_size": 8,
        "eval_batch_size": 12,
        "feature_batch_size": 8,
        "num_workers": 0,
        "cpu_threads": 2,
        "backbone_type": "tiny",
        "backbone_lr": 1e-3,
        "head_lr": 1e-2,
        "module_lr": 1e-3,
        "weight_decay": 1e-4,
        "basis_rank": 2,
        "basis_max_iter": 5,
        "module_layers": [0, 1],
        "modules_per_layer": 2,
        "module_rank": 2,
        "outer_updates": 2,
        "support_shots": [1],
        "background_shots": 1,
        "target_classes": 1,
        "query_per_class": 1,
        "crossfit_folds": 2,
        "output_dir": str(output_dir),
        "save_checkpoints": True,
    }
