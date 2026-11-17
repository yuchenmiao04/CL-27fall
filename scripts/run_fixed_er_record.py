#!/usr/bin/env python3
"""Run the fixed formal ER baseline and record its exact CE trajectory.

This performs one ordinary native run, with the supplied fixed training values.
It does not search parameters or invoke the internal candidate cohort driver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cl27.ce_trajectory import CEStageCache
from cl27.protocol import validate_protocol
from cl27.runner import run


def load_fixed_config(path):
    args = json.loads(Path(path).read_text())
    if not isinstance(args, dict):
        raise ValueError("Fixed ER configuration must be a JSON object")
    seed = args.get("seed")
    if isinstance(seed, list) and len(seed) == 1:
        seed = seed[0]
    if type(seed) is not int or seed != 1993:
        raise ValueError("Fixed ER requires exactly seed 1993")
    if args.get("model_name") != "er" or args.get("cl27_protocol") != "pilot_figure2":
        raise ValueError("This entry point only runs the fixed native PILOT ER baseline")
    if args.get("dataset") not in {"cifar224", "imagenetr"}:
        raise ValueError("Fixed ER uses only the two selected PILOT datasets")
    if args.get("phase") != "final" or args.get("validation_only", False) is not False:
        raise ValueError("Fixed ER requires final phase, not development or validation-only data")
    if args.get("smoke", False) is not False or args.get("pretrained") is not True:
        raise ValueError("Fixed ER requires the real pretrained backbone, not smoke data")
    if any(args.get(key) is not None for key in (
        "max_tasks", "validation_max_per_class", "joint_max_classes"
    )):
        raise ValueError("Fixed ER must run the complete formal stream without subset limits")
    if args.get("execution_role") not in {None, "fixed_baseline"}:
        raise ValueError("An internal/development driver cannot become a fixed ER baseline")
    args["seed"] = seed  # Same scalar normalization as trainer.train.
    args["validation_only"] = False
    args["execution_role"] = "fixed_baseline"
    args["config"] = str(Path(path).resolve())
    validate_protocol(args)
    return args


class _FixedBaselineBinding:
    """Change reporting metadata only; keep native record guards and adaptation."""

    def __init__(self, binding):
        self.binding = binding

    def adapt(self, *args, **kwargs):
        return self.binding.adapt(*args, **kwargs)

    def report(self):
        result = self.binding.report()
        result["execution_role"] = "fixed_baseline"
        return result


class FixedBaselineRecorder(CEStageCache):
    def __init__(self, directory):
        super().__init__(directory, "record")

    def for_run(self, args, data_manifest):
        return _FixedBaselineBinding(super().for_run(args, data_manifest))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    options = parser.parse_args(argv)
    args = load_fixed_config(options.config)
    run(args, ce_training_executor=FixedBaselineRecorder(options.trajectory_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
