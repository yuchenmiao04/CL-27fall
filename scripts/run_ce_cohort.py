#!/usr/bin/env python3
"""Run independent ER/transport/shared-prior heads using one exact CE trajectory.

Each native runner starts independently and evaluates once per stage. This
avoids changing initialization or evaluation-loader RNG consumption. The driver
records full stage body/head transitions; readers fail closed on any mismatch.
Use --replay-only to reuse an already complete trajectory for analytic-head HPO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cl27.ce_trajectory import CEStageCache, METHODS, configuration_reuse_identity
from cl27.runner import run, write_json


def load_config(path):
    args = json.loads(Path(path).read_text())
    seed = args.get("seed")
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError("A cohort contains exactly one seed per configuration")
        args["seed"] = seed[0]
    if not isinstance(args.get("seed"), int):
        raise ValueError("A cohort needs one integer seed")
    if args.get("model_name") not in METHODS:
        raise ValueError("Only ER, history_transport and plastic_shared_basis share CE trajectories")
    if args["model_name"] == "er":
        # The optimized candidate body's execution driver is not the separately
        # fixed ER benchmark. Persist this distinction in config.json, not only
        # the outer cohort report, so final result collectors can reject it.
        args["execution_role"] = "internal_shared_ce_driver"
    args["config"] = str(Path(path).resolve())
    return args


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, type=Path)
    p.add_argument("--trajectory-dir", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--replay-only", action="store_true")
    options = p.parse_args(argv)
    if options.report.exists():
        p.error("--report already exists; choose a new file")
    configs = [load_config(path) for path in options.configs]
    names = [c["model_name"] for c in configs]
    if len(set(names)) != len(names):
        p.error("A cohort contains each method at most once")
    if not options.replay_only:
        if "er" not in names:
            p.error("A recording cohort requires the ER driver")
        if len({configuration_reuse_identity(c) for c in configs}) != 1:
            p.error("Recording cohort data, training, device and stage scope must match before execution")
        configs.sort(key=lambda c: c["model_name"] != "er")
    began = time.perf_counter()
    records = []
    for i, config in enumerate(configs):
        mode = "replay" if options.replay_only or i else "record"
        result = run(config, ce_training_executor=CEStageCache(options.trajectory_dir, mode))
        records.append({"method": config["model_name"], "output_dir": config["output_dir"],
                        "execution_role": (
                            "internal_shared_ce_driver" if config["model_name"] == "er" else "candidate_method"
                        ),
                        "status": result["status"], "ce_mode": mode,
                        "metrics": result["metrics"], "execution": result["execution"]})
        write_json(options.report, {"status": "running", "runs": records,
                                    "actual_cohort_wall_seconds": time.perf_counter() - began})
    write_json(options.report, {"status": "complete", "runs": records,
                                "actual_cohort_wall_seconds": time.perf_counter() - began,
                                "timing_policy": "Shared CE is executed once; per-reader wall time excludes that driver work"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
