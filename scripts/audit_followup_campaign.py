#!/usr/bin/env python3
"""Audit the four predeclared follow-ups without selecting runs by test score.

Only the standard library is used. Run this on the server containing the
immutable executed source and checkpoint files. ``--allow-incomplete`` permits
progress tables, but never converts missing checkpoint/restore evidence into a
completed audit. The original twelve results are not modified or reselected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__:
    from .prepare_followup_campaign import (JOBS, NPZ_SHA, check_template,
                                           source_inventory)
else:
    from prepare_followup_campaign import (JOBS, NPZ_SHA, check_template,
                                          source_inventory)


UPSTREAM = "7a6e904c5bc5cb7a4e1823b3434020be27469b63"
BODY_SHA = "1ece119a218036ffacbd3b576e1eb00a483915c88b21d81980c5e4e4a52e66e9"
RNG_POLICY = "canonical_loader_isolated_v1"
TUNA_RNG_POLICY = "native_tuna_loaders_and_one_raw_eval_v1"
METRICS = ("final_accuracy", "aia", "forgetting_previous",
           "forgetting_including_final")
# Counts are from the already verified full selected PILOT data, seed1993.
# Checking these, in addition to the complete manifest fingerprint, prevents a
# self-consistent but truncated or differently weighted accuracy matrix passing.
DATA = {
    "cifar224": {
        "title": "CIFAR-100", "classes": 100, "width": 10,
        "fingerprint": "98b4aa3a9e6bfd167c52eb391f6dc1b028f2fb920f8a20c8c48751282b8f7ed4",
        "train_tasks": [5000] * 10, "test_tasks": [1000] * 10,
    },
    "imagenetr": {
        "title": "ImageNet-R", "classes": 200, "width": 20,
        "fingerprint": "014b448f085423b32b6ca0468e7165f896460ff3e4341d1f7bcc305d96a2de76",
        "train_tasks": [2804, 2562, 2343, 2293, 2175, 2665, 2124, 2816, 1948, 2270],
        "test_tasks": [689, 632, 606, 571, 487, 660, 569, 729, 478, 579],
    },
}


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    try:
        value = json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (OSError, ValueError) as exc:
        raise AuditError(f"Cannot read JSON {path}: {exc}") from exc
    need(isinstance(value, dict), "Expected a JSON object: " + str(path))
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scalar(value, name, *, fraction=False, positive=False):
    need(not isinstance(value, bool) and isinstance(value, (int, float))
         and math.isfinite(value), name + " must be finite and numeric")
    need(not fraction or 0 <= value <= 1, name + " must be a fraction in [0,1]")
    need(not positive or value > 0, name + " must be positive")
    return float(value)


def integer(value, name, *, minimum=0):
    need(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def same(actual, expected, name):
    need(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
         name + " differs from the complete stored stage records")


def config_fingerprint(config):
    ignored = {"output_dir", "resume", "max_tasks", "config", "device",
               "num_workers", "cpu_threads"}
    return hashlib.sha256(json.dumps({k: v for k, v in config.items() if k not in ignored},
                                    sort_keys=True).encode()).hexdigest()


def normalized_prepared(config):
    config = dict(config)
    need(config.get("seed") == [1993], "Prepared seed must be the single seed [1993]")
    config["seed"] = 1993
    return config


def validate_prepared_manifest(path):
    manifest = read(path)
    need(manifest.get("version") == 1, "Unknown prepared campaign manifest version")
    rows = manifest.get("jobs")
    need(isinstance(rows, list) and len(rows) == 4, "Manifest must bind exactly four follow-ups")
    expected = {job: (gpu, dataset, method) for job, gpu, dataset, method in JOBS}
    need({row.get("job") for row in rows} == set(expected), "Unexpected or duplicate prepared jobs")
    repo = Path(manifest["repo"])
    try:
        actual_sources = source_inventory(repo)
    except (OSError, ValueError, KeyError) as exc:
        raise AuditError("Scientific source inventory failed: " + str(exc)) from exc
    need(actual_sources == manifest.get("source_files_sha256"), "Executed source snapshot differs from the prepared manifest")
    weight = manifest.get("pretrained_npz", {})
    need(weight.get("sha256") == NPZ_SHA and Path(weight.get("path", "")).is_file(), "Missing or wrong bound pretrained NPZ")
    need(sha(weight["path"]) == NPZ_SHA, "Prepared pretrained file bytes changed")
    configs = {}
    for row in rows:
        job = row["job"]
        gpu, dataset, method = expected[job]
        need((row.get("gpu"), row.get("dataset"), row.get("method")) == (gpu, dataset, method),
             "Prepared job identity changed: " + job)
        cfg_info = row.get("config", {})
        cfg_path = Path(cfg_info.get("path", ""))
        need(cfg_path.is_file() and sha(cfg_path) == cfg_info.get("sha256"), "Prepared config SHA changed: " + job)
        config = read(cfg_path)
        try:
            check_template(repo, job, config, dataset, method)
        except (OSError, ValueError, KeyError) as exc:
            raise AuditError("Prepared scientific recipe failed for " + job + ": " + str(exc)) from exc
        need(config.get("comparison_variant") == row.get("comparison_variant"), "Prepared comparison variant differs")
        need(config.get("output_dir") == row.get("output_dir"), "Prepared output directory differs")
        need(config.get("device") == [gpu], "Prepared GPU identity differs")
        configs[job] = config
    return manifest, configs


def validate_data_manifest(manifest, dataset):
    spec = DATA[dataset]
    need(manifest.get("fingerprint") == spec["fingerprint"], "Full selected data fingerprint differs")
    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    need(hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() == spec["fingerprint"],
         "Data manifest content does not match its fingerprint")
    need(manifest.get("dataset") == dataset and manifest.get("phase") == "final"
         and manifest.get("evaluation_source") == "designated_test"
         and manifest.get("official_test_read") is True and manifest.get("validation_only") is False,
         "Data manifest is not a full final designated-test split")
    need(manifest.get("train_count") == sum(spec["train_tasks"])
         and manifest.get("evaluation_count") == sum(spec["test_tasks"]), "Full data row counts differ")


def validate_metrics(result, dataset):
    spec = DATA[dataset]
    stages, metrics = result.get("stages"), result.get("metrics", {})
    need(isinstance(stages, list) and len(stages) == 10 and metrics.get("completed_tasks") == 10,
         "All ten complete stages are required")
    matrix, accuracies, stage_seconds = [], [], []
    for task, stage in enumerate(stages):
        need(stage.get("task") == task and stage.get("seen_classes") == (task + 1) * spec["width"]
             and stage.get("prediction_space") == "all_seen_classes_without_task_id"
             and stage.get("evaluation_head") == "cnn", "Wrong all-seen Class-IL stage identity")
        row, counts = stage.get("task_accuracy"), stage.get("task_counts")
        need(isinstance(row, list) and isinstance(counts, list) and len(row) == len(counts) == 10,
             "Each stage needs a full ten-column accuracy/count record")
        expected_counts = spec["test_tasks"][:task + 1] + [0] * (9 - task)
        need(all(type(count) is int for count in counts) and counts == expected_counts,
             "Stage evaluation counts differ from the full selected test groups")
        for j in range(10):
            if j > task:
                need(row[j] is None, "Future classes entered evaluation")
                continue
            value = scalar(row[j], "Task accuracy", fraction=True)
            same(value * counts[j], round(value * counts[j]), "Task integer correct count")
        weighted = sum(row[j] * counts[j] for j in range(task + 1)) / sum(counts)
        accuracy = scalar(stage.get("accuracy"), "Stage accuracy", fraction=True)
        same(accuracy, weighted, "Sample-weighted stage accuracy")
        need(stage.get("evaluation_images") == sum(counts), "Stage total evaluation count differs")
        primary = stage.get("heads", {}).get("cnn", {})
        need(primary.get("task_accuracy") == row and primary.get("task_counts") == counts
             and primary.get("evaluation_images") == sum(counts), "Primary head and stage raw records differ")
        same(scalar(primary.get("accuracy"), "CNN accuracy", fraction=True), accuracy, "CNN primary accuracy")
        stage_seconds.append(scalar(stage.get("wall_seconds"), "Measured stage wall time", positive=True))
        matrix.append(row)
        accuracies.append(accuracy)
    need(metrics.get("accuracy_matrix") == matrix, "Summary accuracy matrix differs from stored stages")
    derived = {
        "final_accuracy": accuracies[-1], "aia": sum(accuracies) / 10,
        "forgetting_previous": sum(max(matrix[t][j] for t in range(j, 9)) - matrix[9][j] for j in range(9)) / 9,
        "forgetting_including_final": sum(max(matrix[t][j] for t in range(j, 10)) - matrix[9][j] for j in range(9)) / 9,
    }
    cnn_metrics = result.get("metrics_by_head", {}).get("cnn", {})
    for name, expected in derived.items():
        same(scalar(metrics.get(name), name, fraction=name in METRICS[:2]), expected, name)
        same(scalar(cnn_metrics.get(name), "CNN " + name, fraction=name in METRICS[:2]), expected, "CNN " + name)
    return derived, stage_seconds


def validate_records(prepared, saved, result, data, sources):
    """Validate metadata/metrics without asserting checkpoint or process state."""
    expected = normalized_prepared(prepared)
    actual = {k: v for k, v in saved.items() if k != "config"}
    need(actual == expected, "Executed configuration differs from the prebound recipe")
    dataset, method = saved.get("dataset"), saved.get("model_name")
    need(dataset in DATA and method in {"current_adapted", "tuna"}, "Wrong follow-up method/dataset")
    need(saved.get("seed") == 1993 and saved.get("phase") == "final" and saved.get("pretrained") is True,
         "Wrong final phase, pretrained flag or seed")
    need(saved.get("cl27_protocol") == "pilot_figure2" and saved.get("shuffle") is True
         and saved.get("evaluation_head") == "cnn" and saved.get("validation_only") is False,
         "Wrong selected protocol")
    need(not saved.get("smoke") and all(saved.get(k) is None for k in
         ("resume", "max_tasks", "validation_max_per_class", "joint_max_classes")), "Truncated/resumed/development run")
    need((saved.get("init_cls"), saved.get("increment")) == (DATA[dataset]["width"],) * 2,
         "Wrong class split")
    need(result.get("status") == "complete" and result.get("phase") == "final"
         and result.get("method") == method and result.get("evaluation_head") == "cnn"
         and result.get("performance_claim") is not False, "Run is not a complete formal result")
    need(result.get("upstream_commit") == UPSTREAM, "Upstream snapshot changed")
    need(result.get("config_fingerprint") == config_fingerprint(saved), "Executed config fingerprint changed")
    validate_data_manifest(data, dataset)
    need(result.get("data_fingerprint") == data["fingerprint"], "Result data identity differs")
    provenance = result.get("backbone_provenance", {})
    need(provenance.get("kind") == "pretrained" and provenance.get("timm_version") == "0.6.12"
         and provenance.get("checkpoint_family") == "augreg_in21k_ft_in1k"
         and provenance.get("weight_file_sha256") == NPZ_SHA
         and provenance.get("initial_state_sha256") == BODY_SHA
         and provenance.get("upstream_commit") == UPSTREAM, "Initial backbone provenance differs")
    if method == "current_adapted":
        need(saved.get("comparison_variant") == "epochs20" and saved.get("epochs") == 20
             and saved.get("backbone_type") == "vit_base_patch16_224", "Wrong fixed-library budget/variant")
        if dataset == "cifar224":
            need(saved.get("fixed_memory") is False and saved.get("memory_size") == 2000, "Wrong fixed-total memory")
        else:
            need(saved.get("fixed_memory") is True and saved.get("memory_per_class") == 20, "Wrong fixed-per-class memory")
    else:
        need(saved.get("comparison_variant") == "pilot_native_same_protocol"
             and saved.get("backbone_type") == "vit_base_patch16_224_tuna"
             and saved.get("memory_size") == saved.get("memory_per_class") == 0, "Wrong native TUNA branch or image memory")
        need(provenance.get("algorithm_rng_policy") == TUNA_RNG_POLICY, "TUNA algorithm RNG policy changed")
        need(saved.get("tuned_epoch") == (15 if dataset == "cifar224" else 10)
             and saved.get("crct_epochs") == 30, "Native TUNA training lengths changed")
    metrics, stage_seconds = validate_metrics(result, dataset)
    diagnostics = result.get("task_diagnostics")
    need(isinstance(diagnostics, list) and len(diagnostics) == 10, "Ten method diagnostics are required")
    arrivals, cumulative = DATA[dataset]["train_tasks"], 0
    for task, diagnostic in enumerate(diagnostics):
        need(diagnostic.get("task") == task, "Method diagnostic order differs")
        cumulative += arrivals[task]
        if method == "current_adapted":
            epochs = diagnostic.get("epoch_summaries", [])
            need(diagnostic.get("epochs") == 20 and len(epochs) == 20
                 and [e.get("epoch") for e in epochs] == list(range(20)), "Fixed-library run did not complete twenty epochs in every stage")
            need(diagnostic.get("new_images") == arrivals[task]
                 and diagnostic.get("history_count") == cumulative
                 and diagnostic.get("previous_history_count") == cumulative - arrivals[task], "Fixed-library history arrivals differ")
            seen = (task + 1) * DATA[dataset]["width"]
            quota = seen * 20 if dataset == "imagenetr" else seen * (2000 // seen)
            need(diagnostic.get("replay_images") == quota and diagnostic.get("anchor_sha256") == BODY_SHA,
                 "Fixed-library memory or immutable anchor changed")
        else:
            need(diagnostic.get("arrivals") == arrivals[task]
                 and diagnostic.get("arrived_training_rows") == cumulative
                 and diagnostic.get("raw_replay_images") == 0, "TUNA arrivals or zero-image memory differs")
            need(diagnostic.get("original_vit_frozen") is True and diagnostic.get("original_vit_sha256") == BODY_SHA
                 and diagnostic.get("algorithm_rng_policy") == TUNA_RNG_POLICY, "TUNA native body/RNG policy differs")
            for name, digest in diagnostic.get("native_source_sha256", {}).items():
                need(sources.get(name) == digest, "TUNA executed native source differs: " + name)
            need(set(diagnostic.get("native_source_sha256", {})) ==
                 {"models/tuna.py", "backbone/vit_tuna.py", "utils/inc_net.py", "backbone/linears.py", "models/tuna_protocol.py"},
                 "Missing native TUNA source identities")
            training = diagnostic.get("training", {})
            need(training.get("image_epochs") == saved["tuned_epoch"]
                 and training.get("main_optimizer_steps_observed") == math.ceil(arrivals[task] / saved["batch_size"]) * saved["tuned_epoch"]
                 and training.get("class_alignment_epochs") == (30 if task else 0), "Native TUNA observed image steps or CA epochs differ")
            integer(training.get("ca_first_head_backward_calls_observed"), "Observed CA backward calls")
            need(result["stages"][task].get("extra_evaluation_pass_for_raw_matrix") is False,
                 "TUNA raw metrics must reuse the single native evaluation")
    execution = result.get("execution", {})
    need(execution.get("device_type") == "cuda" and execution.get("random_stream_policy") == RNG_POLICY,
         "Wrong measured device or runner RNG policy")
    wall = scalar(execution.get("wall_seconds"), "Measured run wall time", positive=True)
    need(wall + 1e-6 >= sum(stage_seconds), "Run wall time is shorter than its measured stage regions")
    integer(execution.get("numeric_checkpoint_payload_bytes"), "Measured numeric checkpoint bytes", minimum=1)
    integer(execution.get("gpu_peak_allocated_bytes"), "Measured peak GPU allocation", minimum=1)
    final = diagnostics[-1]
    resources = {
        "execution": {key: execution[key] for key in ("wall_seconds", "numeric_checkpoint_payload_bytes",
                       "gpu_peak_allocated_bytes", "device_type", "torch", "numpy") if key in execution},
        "stage_wall_seconds": stage_seconds,
        "final_method_record": {key: value for key, value in final.items()
                                if key in {"resources", "numeric_payload_bytes", "replay_images", "raw_replay_images", "replay_payload_bytes"}},
        "measurement_scope": "Recorded runner and method regions; no estimated training cost or matched-compute claim.",
    }
    return metrics, resources


def validate_checkpoint_evidence(path, entry, run):
    evidence = read(path)
    checkpoint = run / "checkpoint.pt"
    need(checkpoint.is_file(), "Final server-local checkpoint is missing")
    need(evidence.get("job") == entry["job"], "Checkpoint evidence job differs")
    size, digest = checkpoint.stat().st_size, sha(checkpoint)
    need(type(evidence.get("checkpoint_bytes")) is int and evidence["checkpoint_bytes"] == size
         and evidence.get("checkpoint_sha256") == digest and evidence.get("backup_sha256") == digest,
         "Original/backup checkpoint SHA or byte counts differ")
    need(evidence.get("native_restore_verified") is True, "Native restore/payload equality is not verified")
    need(evidence.get("source_process_terminal_verified") is True, "Source training process terminal state is not verified")
    required = {"config": sha(run / "config.json"), "results": sha(run / "results.json"),
                "data_manifest": sha(run / "data_manifest.json")}
    need(evidence.get("source_metadata_sha256") == required, "Checkpoint audit metadata binding differs")
    return {"checkpoint_sha256": digest, "checkpoint_bytes": size,
            "backup_sha256": evidence["backup_sha256"], "native_restore_verified": True,
            "source_process_terminal_verified": True, "evidence_sha256": sha(path),
            "evidence_path": str(path)}


def audit_campaign(manifest_path, evidence_root=None, *, allow_incomplete=False):
    manifest_path = Path(manifest_path).resolve()
    manifest, configs = validate_prepared_manifest(manifest_path)
    evidence_root = Path(evidence_root or (manifest_path.parent / "checkpoint_evidence"))
    rows = []
    for entry in manifest["jobs"]:
        job, run = entry["job"], Path(entry["output_dir"])
        row = {"job": job, "dataset": entry["dataset"], "method": entry["method"],
               "comparison_variant": entry["comparison_variant"], "seed": 1993,
               "status": "pending", "metrics": dict.fromkeys(METRICS)}
        result_path = run / "results.json"
        if result_path.exists():
            saved, result, data = (read(run / name) for name in ("config.json", "results.json", "data_manifest.json"))
            metrics, resources = validate_records(configs[job], saved, result, data,
                                                  manifest["source_files_sha256"])
            row.update(status="metrics_verified_checkpoint_evidence_pending", metrics=metrics,
                       resources=resources,
                       source={"config_sha256": sha(run / "config.json"), "results_sha256": sha(result_path),
                               "data_manifest_sha256": sha(run / "data_manifest.json"),
                               "data_fingerprint": data["fingerprint"],
                               "prepared_config_sha256": entry["config"]["sha256"],
                               "config_fingerprint": result["config_fingerprint"]})
            evidence_path = evidence_root / (job + ".json")
            if evidence_path.exists():
                row["checkpoint_audit"] = validate_checkpoint_evidence(evidence_path, entry, run)
                row["status"] = "verified_complete"
        elif (run / "progress.json").exists():
            progress = read(run / "progress.json")
            row.update(status="incomplete", completed_stage_records=len(progress.get("stages", [])))
        rows.append(row)
    complete = sum(row["status"] == "verified_complete" for row in rows)
    need(allow_incomplete or complete == 4, "Follow-up audit incomplete: " + str(complete) + "/4 verified endpoints; use --allow-incomplete for progress only")
    return {
        "schema_version": 1, "status": "verified_complete" if complete == 4 else "incomplete",
        "expected_results": 4, "verified_results": complete,
        "metrics_verified_results": sum(row["metrics"]["final_accuracy"] is not None for row in rows),
        "seed": 1993, "selection_policy": "The four prepared job identities only; never highest test score.",
        "manifest_sha256": sha(manifest_path), "source_commit": manifest.get("source_commit"),
        "source_files_sha256": manifest["source_files_sha256"], "upstream_commit": UPSTREAM,
        "metric_units": "Fractions; Markdown uses percent accuracy and percentage-point forgetting.",
        "audit_scope": "Prepared scientific recipe/source/weights, full data identity, unrounded ten-stage records, measured resources and SHA-bound server-local backup/native-restore evidence.",
        "auditor_sha256": sha(Path(__file__)), "rows": rows,
    }


def add_base_summary(summary, base_path):
    """Preserve the prior twelve rows verbatim; identify current5 separately."""
    base = read(base_path)
    original = base.get("rows", [])
    methods = {"shared_basis", "history_transport", "current_adapted", "plastic_shared_basis", "er", "joint"}
    expected = {(d, m) for d in DATA for m in methods}
    need(len(original) == 12 and {(r.get("dataset"), r.get("method")) for r in original} == expected,
         "Base summary must contain exactly the original twelve unique rows")
    need(base.get("available_results") == 12 and all(r.get("status") != "missing" for r in original),
         "Base summary is not the completed original campaign")
    rows = []
    for row in original:
        rows.append({**row, "comparison_variant": "epochs5" if row["method"] == "current_adapted" else "original",
                     "imported_previous_campaign": True})
    summary = dict(summary)
    summary["original_campaign_reference"] = {
        "sha256": sha(base_path), "source": str(base_path),
        "scope": "Original twelve rows imported unchanged; their previous independent audits are referenced, not rerun by this helper.",
    }
    summary["combined_rows"] = rows + summary["rows"]
    return summary


def markdown(summary):
    lines = ["# Additional formal experiments", "",
             f"Seed 1993; {summary['verified_results']}/4 fully verified follow-up results.",
             "Accuracy: %. Forgetting: percentage points. Pending checkpoint evidence does not constitute a completed audit.",
             "The fixed-library 20-epoch rows are additional budget variants; the original 5-epoch results remain separate.", ""]
    for dataset, spec in DATA.items():
        lines += ["## " + spec["title"], "", "| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for row in summary.get("combined_rows", summary["rows"]):
            if row["dataset"] != dataset:
                continue
            values = ["—" if row["metrics"].get(key) is None else f"{100 * row['metrics'][key]:.2f}" for key in METRICS]
            status = "prior campaign reference" if row.get("imported_previous_campaign") else row["status"]
            lines.append("| " + " | ".join([row["method"], row["comparison_variant"], *values, status]) + " |")
        lines.append("")
    lines += ["F_previous excludes the final evaluation from each historical maximum; F_including_final includes it.",
              "Resource values are the measured runner/method records in the JSON; training costs are not inferred from epoch counts.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-evidence-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--base-summary", type=Path, help="Optional completed original twelve-row summary to preserve alongside the four additions")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        summary = audit_campaign(args.manifest, args.checkpoint_evidence_root, allow_incomplete=args.allow_incomplete)
        if args.base_summary:
            summary = add_base_summary(summary, args.base_summary)
        output = args.output_dir or args.manifest.parent / "summary"
        output.mkdir(parents=True, exist_ok=True)
        for name, text in (("summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n"),
                           ("summary.md", markdown(summary))):
            path = output / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text)
            temporary.replace(path)
    except (AuditError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, "Audit failed: " + str(exc) + "\n")
    print(json.dumps({"status": summary["status"], "verified_results": summary["verified_results"],
                      "metrics_verified_results": summary["metrics_verified_results"], "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
