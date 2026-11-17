#!/usr/bin/env python3
"""Prepare private CIFAR module-selection evidence; never launch or publish.

First run without --selected-trial to make a comparison dossier after BOTH full
streams complete. Root then audits the completed checkpoint and may rerun with
an explicit reviewed selection. All generated files remain private drafts.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

TRIALS = ("cifar_current_bounded64_full_v4", "cifar_current_mb2_full_v5")
COMMIT = "bb028d926565e6db86d4b4e716406c57392a1617"
JOB = "formal_cifar_current_selected_v5"
MMBM = ["modules", "modules", "backbone", "modules"]


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def finite(x):
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError("Nonfinite numeric record")
    if isinstance(x, dict):
        for v in x.values(): finite(v)
    if isinstance(x, list):
        for v in x: finite(v)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect(run, expected, validator):
    config, result, data = [read(run / n) for n in ("config.json", "results.json", "data_manifest.json")]
    for value in (config, result, data): finite(value)
    assert run.name == expected
    assert config["model_name"] == "current_adapted" and config["dataset"] == "cifar224"
    assert result["status"] == "complete" and result["phase"] == config["phase"] == "development"
    assert config["validation_only"] is True and result["metrics"]["completed_tasks"] == 10
    assert config["seed"] in (1993, [1993]) and config["epochs"] == 5
    assert config["implementation_commit"] == COMMIT
    assert config["episode_replay_batch_size"] == 64 and config["batch_size"] == 64
    assert config["module_layers"] == [7, 9] and config["modules_per_layer"] == 4 and config["module_rank"] == 8
    assert config.get("module_scale", .1) == .1 and config["feature_normalization"] == "l2"
    schedule = config.get("schedule", MMBM)
    assert (config["outer_updates"], schedule) == ((4, MMBM) if expected == TRIALS[0] else (2, ["modules", "backbone"]))
    assert config["memory_size"] == 2000 and config["fixed_memory"] is False
    assert data["train_count"] == 45000 and data["evaluation_count"] == 5000
    assert data["official_test_read"] is False and data["evaluation_source"] == "training_holdout"
    assert result["data_fingerprint"] == data["fingerprint"]
    metrics = validator(config, result)
    assert result["task_diagnostics"][-1]["history_count"] == 45000
    assert result["task_diagnostics"][-1]["replay_images"] == 2000
    last = result["task_diagnostics"][-1]
    assert all(d["backbone_changed"] and d["modules_changed"] for d in result["task_diagnostics"])
    assert len({d["anchor_sha256"] for d in result["task_diagnostics"]}) == 1
    n = result["stages"][-1]["evaluation_images"]
    correct = round(metrics["final_accuracy"] * n)
    assert n == 5000 and math.isclose(correct / n, metrics["final_accuracy"], abs_tol=1e-14)
    return config, {"trial": expected, "metrics": metrics, "final_correct": correct,
        "final_evaluation_images": n, "data_fingerprint": data["fingerprint"],
        "source_sha256": {n: sha(run / n) for n in ("results.json", "config.json", "data_manifest.json")},
        "cost_evidence": {"sum_recorded_stage_seconds": sum(s["wall_seconds"] for s in result["stages"]),
            "current_process_wall_seconds": result.get("execution", {}).get("wall_seconds"),
            "module_updates_total": last["module_updates_total"], "backbone_updates_total": last["backbone_updates_total"],
            "cumulative_counters": last["counters"],
            "limits": "Resumed-process wall excludes its prefix; stage sums exclude checkpoint-save overhead. Different concurrency prevents a controlled speedup claim."}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--legacy-run", type=Path, required=True)
    p.add_argument("--mb-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New private/tmp draft directory")
    p.add_argument("--selected-trial", choices=TRIALS)
    p.add_argument("--root-reviewed", action="store_true")
    p.add_argument("--tie-cost-reason", default="")
    a = p.parse_args()
    paths = (a.legacy_run, a.mb_run)
    missing = [str(run / "results.json") for run in paths if not (run / "results.json").is_file()]
    if missing:
        print(json.dumps({"status": "waiting_for_both_complete_results", "missing": missing})); return 3
    output = a.output.resolve()
    assert any(str(output).startswith(x) for x in ("/tmp/", "/private/tmp/", "local_private_archive/"))
    assert not output.exists(), "Use a new private output directory"
    spec = importlib.util.spec_from_file_location("campaign_validator", a.code_root / "scripts/summarize_campaign.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    inspected = [inspect(run, name, module.validate_complete) for run, name in zip(paths, TRIALS)]
    configs, evidence = zip(*inspected)
    assert evidence[0]["data_fingerprint"] == evidence[1]["data_fingerprint"]
    output.mkdir(parents=True)
    save(output / "comparison.json", {"status": "requires_root_review_no_selection", "candidates": evidence,
        "rule": "Unrounded full10-stage Final first; exact tie only then documented cost. No formal score read.",
        "checkpoint_audit_required": "Completed selected checkpoint: N45000, G769x769/C769x100,2000 unique legal residents, immutable initial anchor, changed body/library, full two AdamW states and owned/global RNG, finite values and native restore. This helper never opens checkpoint.pt."})
    if a.selected_trial is None:
        print(json.dumps({"status": "comparison_only", "output": str(output)})); return 0
    assert a.root_reviewed, "Root must review both full results before an explicit selection"
    index = TRIALS.index(a.selected_trial); selected = evidence[index]
    assert selected["metrics"]["final_accuracy"] >= evidence[1-index]["metrics"]["final_accuracy"]
    if selected["metrics"]["final_accuracy"] == evidence[1-index]["metrics"]["final_accuracy"]:
        assert a.tie_cost_reason, "Document exact-tie cost evidence and concurrency limitations"
    config = copy.deepcopy(configs[index])
    for key in ("config", "resume", "max_tasks", "validation_max_per_class", "joint_max_classes", "validation_fraction", "split_seed", "path_export_note"):
        config.pop(key, None)
    config.update(phase="final", validation_only=False, seed=[1993], device=[0],
        output_dir="remote_runtime/campaign_20261117/formal/runs/" + JOB,
        schedule=configs[index].get("schedule", MMBM))
    save(output / (JOB + ".json"), config)
    save(output / "D0055.draft.json", {"id": "D0055", "status": "private_draft_requires_checkpoint_audit_before_publication",
        "method": "current_adapted", "dataset": "cifar224", "selected_development_trial": a.selected_trial,
        "selection_candidates": evidence, "selected_parameters": config,
        "criterion": "Root-reviewed full10-stage unrounded Final; documented cost only for an exact tie.",
        "tie_cost_reason": a.tie_cost_reason, "formal_job": JOB,
        "formal_expected_training_rows": 50000, "formal_final_replay_images": 2000,
        "fresh_refit": "Original pretrained weights,new optimizers,no development checkpoint,moments or cache; five epochs,selected MB or MMBM,bounded64,fixed library unchanged.",
        "placement": "Prefer physical/logical GPU0; selected fresh module method is not CE-cache replay. Old MMBM development ran on1,MB on0,R formal on1. Placement is an execution record,not HPO.",
        "launch_gate": "Do not launch from this draft. Root must audit the completed selected checkpoint,finalize evidence and recheck live GPU0 capacity. Avoid sharing GPU1 with the R formal-module long tail.",
        "official_test_used_for_parameter_decision": False, "formal_execution_status": "not_started"})
    save(output / "launch_spec.draft.json", {"jobs": [{"name": JOB, "area": "formal",
        "cwd": "remote_runtime/runtime_bounded_v4",
        "command": ["remote_python", "main.py", "--config", "remote_runtime/campaign_20261117/formal/configs/" + JOB + ".json"],
        "environment": {"CL27_CURRENT_FROZEN_PREFIX_REUSE": "1"},
        "metadata": {"selection_decision": "D0055", "gpu": 0, "prepared_only": True, "fresh_full_training_refit": True,
            "expected_full_training_rows": 50000, "method_source_commit": COMMIT, "selected_development_trial": a.selected_trial,
            "not_authorized_to_launch_from_draft": True}}]})
    print(json.dumps({"status": "private_drafts_only", "selected_by_explicit_root_argument": a.selected_trial, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
