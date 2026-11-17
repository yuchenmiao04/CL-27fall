"""Reject plausible-looking but incomplete or mismatched follow-up evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_followup_campaign as audit


REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(dataset="cifar224", method="current_adapted"):
    short = "cifar" if dataset == "cifar224" else "imagenetr"
    suffix = "current_epochs20" if method == "current_adapted" else "tuna_native"
    job = f"formal_{short}_{suffix}_v1"
    prepared = audit.read(REPO / "experiments/extension_20261117/configs" / (job + ".json"))
    saved = {**audit.normalized_prepared(prepared), "config": "prepared/config.json"}
    data_path = next((REPO / "experiments/campaign_20261117/formal/observations" /
                      f"formal_{short}_current_selected_v5").rglob("data_manifest.json"))
    data = audit.read(data_path)
    spec = audit.DATA[dataset]
    stages = []
    for task in range(10):
        counts = spec["test_tasks"][:task + 1] + [0] * (9 - task)
        row = [(n - 10 - (task - j) * 5) / n if j <= task else None
               for j, n in enumerate(spec["test_tasks"])]
        primary = {"accuracy": sum(row[j] * counts[j] for j in range(task + 1)) / sum(counts),
                   "task_accuracy": row, "task_counts": counts, "evaluation_images": sum(counts)}
        stages.append({"task": task, "seen_classes": (task + 1) * spec["width"],
                       "prediction_space": "all_seen_classes_without_task_id", "evaluation_head": "cnn",
                       **copy.deepcopy(primary), "heads": {"cnn": copy.deepcopy(primary)},
                       "extra_evaluation_pass_for_raw_matrix": False, "wall_seconds": 100.0})
    matrix = [row["task_accuracy"] for row in stages]
    metrics = {"completed_tasks": 10, "accuracy_matrix": matrix,
               "final_accuracy": stages[-1]["accuracy"],
               "aia": sum(row["accuracy"] for row in stages) / 10,
               "forgetting_previous": sum(max(matrix[t][j] for t in range(j, 9)) - matrix[-1][j] for j in range(9)) / 9,
               "forgetting_including_final": sum(max(matrix[t][j] for t in range(j, 10)) - matrix[-1][j] for j in range(9)) / 9}
    source_names = ("models/tuna.py", "backbone/vit_tuna.py", "utils/inc_net.py",
                    "backbone/linears.py", "models/tuna_protocol.py")
    sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in source_names}
    diagnostics, cumulative = [], 0
    for task, arrivals in enumerate(spec["train_tasks"]):
        cumulative += arrivals
        seen = (task + 1) * spec["width"]
        if method == "current_adapted":
            diagnostic = {"task": task, "epochs": 20, "epoch_summaries": [{"epoch": e} for e in range(20)],
                          "new_images": arrivals, "history_count": cumulative,
                          "previous_history_count": cumulative - arrivals,
                          "replay_images": seen * 20 if dataset == "imagenetr" else seen * (2000 // seen),
                          "anchor_sha256": audit.BODY_SHA, "numeric_payload_bytes": {"statistics": 100}}
        else:
            diagnostic = {"task": task, "arrivals": arrivals, "arrived_training_rows": cumulative,
                          "raw_replay_images": 0, "original_vit_frozen": True,
                          "original_vit_sha256": audit.BODY_SHA,
                          "algorithm_rng_policy": audit.TUNA_RNG_POLICY,
                          "native_source_sha256": copy.deepcopy(sources),
                          "training": {"image_epochs": saved["tuned_epoch"],
                                       "main_optimizer_steps_observed": ((arrivals + saved["batch_size"] - 1) // saved["batch_size"]) * saved["tuned_epoch"],
                                       "class_alignment_epochs": 30 if task else 0,
                                       "ca_first_head_backward_calls_observed": 30 if task else 0},
                          "resources": {"historical_statistics_bytes": 100}}
        diagnostics.append(diagnostic)
    result = {"status": "complete", "phase": "final", "method": method,
              "evaluation_head": "cnn", "performance_claim": "formal evaluation",
              "upstream_commit": audit.UPSTREAM,
              "config_fingerprint": audit.config_fingerprint(saved), "data_fingerprint": data["fingerprint"],
              "backbone_provenance": {"kind": "pretrained", "timm_version": "0.6.12",
                                      "checkpoint_family": "augreg_in21k_ft_in1k", "weight_file_sha256": audit.NPZ_SHA,
                                      "initial_state_sha256": audit.BODY_SHA, "upstream_commit": audit.UPSTREAM,
                                      "algorithm_rng_policy": audit.TUNA_RNG_POLICY},
              "stages": stages, "metrics": copy.deepcopy(metrics), "metrics_by_head": {"cnn": copy.deepcopy(metrics)},
              "task_diagnostics": diagnostics,
              "execution": {"device_type": "cuda", "random_stream_policy": audit.RNG_POLICY,
                            "wall_seconds": 1050.0, "numeric_checkpoint_payload_bytes": 100,
                            "gpu_peak_allocated_bytes": 1000}}
    return job, prepared, saved, result, data, sources


@pytest.mark.parametrize("dataset", ["cifar224", "imagenetr"])
@pytest.mark.parametrize("method", ["current_adapted", "tuna"])
def test_all_four_exact_protocol_records_pass(dataset, method):
    _, prepared, saved, result, data, sources = fixture(dataset, method)
    metrics, resources = audit.validate_records(prepared, saved, result, data, sources)
    assert metrics["final_accuracy"] == result["metrics"]["final_accuracy"]
    assert resources["execution"]["wall_seconds"] == 1050
    assert "ca_first_head_backward_calls_observed" not in resources["execution"]


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["stages"].pop(), "ten complete"),
    (lambda r: r["stages"][3]["task_counts"].__setitem__(0, 999), "evaluation counts"),
    (lambda r: r["stages"][3]["task_accuracy"].__setitem__(8, 0.0), "Future classes"),
    (lambda r: r["metrics"].__setitem__("aia", 0.99), "aia differs"),
    (lambda r: r["stages"][4].__setitem__("accuracy", 0.99), "Sample-weighted"),
    (lambda r: r["stages"][5]["heads"]["cnn"]["task_counts"].__setitem__(0, 999), "Primary head"),
    (lambda r: r["task_diagnostics"][2]["epoch_summaries"].pop(), "twenty epochs"),
    (lambda r: r["execution"].__setitem__("wall_seconds", 100), "shorter than"),
    (lambda r: r["backbone_provenance"].__setitem__("weight_file_sha256", "0" * 64), "backbone provenance"),
])
def test_reject_incomplete_or_self_inconsistent_records(mutation, match):
    _, prepared, saved, result, data, sources = fixture()
    mutation(result)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_records(prepared, saved, result, data, sources)


def test_wrong_budget_cannot_replace_original_five_epoch_variant():
    _, prepared, saved, result, data, sources = fixture()
    prepared["epochs"] = saved["epochs"] = 5
    prepared["comparison_variant"] = saved["comparison_variant"] = "epochs5"
    result["config_fingerprint"] = audit.config_fingerprint(saved)
    with pytest.raises(audit.AuditError, match="budget/variant"):
        audit.validate_records(prepared, saved, result, data, sources)


def test_tampered_executed_recipe_is_rejected_even_if_fingerprint_updated():
    _, prepared, saved, result, data, sources = fixture()
    saved["module_lr"] *= 2
    result["config_fingerprint"] = audit.config_fingerprint(saved)
    with pytest.raises(audit.AuditError, match="prebound recipe"):
        audit.validate_records(prepared, saved, result, data, sources)


def test_data_manifest_content_is_bound_not_just_its_claimed_fingerprint():
    _, prepared, saved, result, data, sources = fixture()
    data["class_order"][0], data["class_order"][1] = data["class_order"][1], data["class_order"][0]
    with pytest.raises(audit.AuditError, match="manifest content"):
        audit.validate_records(prepared, saved, result, data, sources)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["task_diagnostics"][4]["training"].__setitem__("main_optimizer_steps_observed", 1), "observed image steps"),
    (lambda r: r["task_diagnostics"][4]["native_source_sha256"].__setitem__("models/tuna.py", "0" * 64), "native source"),
    (lambda r: r["stages"][4].__setitem__("extra_evaluation_pass_for_raw_matrix", True), "single native evaluation"),
    (lambda r: r["task_diagnostics"][4]["training"].__setitem__("ca_first_head_backward_calls_observed", True), "integer"),
])
def test_tuna_algorithm_execution_contract(mutation, match):
    _, prepared, saved, result, data, sources = fixture(method="tuna")
    mutation(result)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_records(prepared, saved, result, data, sources)


def checkpoint_fixture(tmp_path):
    job, prepared, saved, result, data, sources = fixture()
    run = tmp_path / job
    for name, value in (("config.json", saved), ("results.json", result), ("data_manifest.json", data)):
        write(run / name, value)
    (run / "checkpoint.pt").write_bytes(b"server-local-test-payload")
    digest = audit.sha(run / "checkpoint.pt")
    evidence = {"job": job, "checkpoint_sha256": digest,
                "checkpoint_bytes": (run / "checkpoint.pt").stat().st_size,
                "backup_sha256": digest, "native_restore_verified": True,
                "source_process_terminal_verified": True,
                "source_metadata_sha256": {"config": audit.sha(run / "config.json"),
                                           "results": audit.sha(run / "results.json"),
                                           "data_manifest": audit.sha(run / "data_manifest.json")}}
    evidence_path = tmp_path / "evidence" / (job + ".json")
    write(evidence_path, evidence)
    return job, run, evidence_path, evidence


@pytest.mark.parametrize("field,value,match", [
    ("backup_sha256", "0" * 64, "checkpoint SHA"),
    ("native_restore_verified", False, "Native restore"),
    ("source_process_terminal_verified", False, "terminal state"),
    ("checkpoint_bytes", True, "checkpoint SHA"),
])
def test_checkpoint_claim_requires_all_independent_evidence(tmp_path, field, value, match):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    audit.validate_checkpoint_evidence(path, {"job": job}, run)
    evidence[field] = value
    write(path, evidence)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_checkpoint_evidence(path, {"job": job}, run)


def test_checkpoint_evidence_binds_current_result_bytes(tmp_path):
    job, run, path, _ = checkpoint_fixture(tmp_path)
    with (run / "results.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(audit.AuditError, match="metadata binding"):
        audit.validate_checkpoint_evidence(path, {"job": job}, run)


def test_metrics_only_progress_never_counts_as_completed_checkpoint_audit(tmp_path, monkeypatch):
    entries, configs = [], {}
    for dataset in audit.DATA:
        for method in ("current_adapted", "tuna"):
            job, prepared, saved, result, data, sources = fixture(dataset, method)
            run = tmp_path / job
            for name, value in (("config.json", saved), ("results.json", result), ("data_manifest.json", data)):
                write(run / name, value)
            entries.append({"job": job, "dataset": dataset, "method": method,
                            "comparison_variant": prepared["comparison_variant"], "output_dir": str(run),
                            "config": {"sha256": "a" * 64}})
            configs[job] = prepared
    manifest = {"jobs": entries, "source_files_sha256": sources, "source_commit": "b" * 40}
    path = tmp_path / "manifest.json"
    write(path, manifest)
    monkeypatch.setattr(audit, "validate_prepared_manifest", lambda p: (manifest, configs))
    summary = audit.audit_campaign(path, allow_incomplete=True)
    assert summary["metrics_verified_results"] == 4
    assert summary["verified_results"] == 0 and summary["status"] == "incomplete"
    assert all(r["status"] == "metrics_verified_checkpoint_evidence_pending" for r in summary["rows"])
    with pytest.raises(audit.AuditError, match="0/4"):
        audit.audit_campaign(path)


def test_optional_original_summary_keeps_all_twelve_rows_and_current5(tmp_path):
    rows = [{"dataset": d, "method": m, "metrics": {"final_accuracy": 0.5}}
            for d in audit.DATA for m in ("shared_basis", "history_transport", "current_adapted", "plastic_shared_basis", "er", "joint")]
    base = tmp_path / "base.json"
    write(base, {"available_results": 12, "rows": rows})
    extra = {"rows": [{"dataset": "cifar224", "method": "current_adapted", "comparison_variant": "epochs20"}]}
    combined = audit.add_base_summary(extra, base)
    assert len(combined["combined_rows"]) == 13
    assert [r["comparison_variant"] for r in combined["combined_rows"] if r["method"] == "current_adapted"] == ["epochs5", "epochs5", "epochs20"]
    assert [{k: v for k, v in r.items() if k not in {"comparison_variant", "imported_previous_campaign"}}
            for r in combined["combined_rows"][:12]] == rows


def prepared_manifest_fixture(tmp_path, monkeypatch):
    sources = {"scientific.py": "c" * 64}
    monkeypatch.setattr(audit, "source_inventory", lambda repo: dict(sources))
    weight = tmp_path / "weights.npz"
    weight.write_bytes(b"fixture weight file")
    actual_sha = audit.sha
    monkeypatch.setattr(audit, "sha", lambda path: audit.NPZ_SHA if Path(path) == weight else actual_sha(path))
    entries = []
    for job, gpu, dataset, method in audit.JOBS:
        cfg = audit.read(REPO / "experiments/extension_20261117/configs" / (job + ".json"))
        cfg["output_dir"] = str(tmp_path / "runs" / job)
        path = tmp_path / "configs" / (job + ".json")
        write(path, cfg)
        entries.append({"job": job, "gpu": gpu, "dataset": dataset, "method": method,
                        "comparison_variant": cfg["comparison_variant"], "output_dir": cfg["output_dir"],
                        "config": {"path": str(path), "sha256": actual_sha(path)}})
    manifest = {"version": 1, "repo": str(REPO), "jobs": entries,
                "source_files_sha256": dict(sources),
                "pretrained_npz": {"path": str(weight), "sha256": audit.NPZ_SHA}}
    path = tmp_path / "manifest.json"
    write(path, manifest)
    return path, manifest, sources


def test_prepared_manifest_exactly_binds_four_recipes(tmp_path, monkeypatch):
    path, _, _ = prepared_manifest_fixture(tmp_path, monkeypatch)
    _, configs = audit.validate_prepared_manifest(path)
    assert len(configs) == 4


def test_prepared_source_inventory_change_rejects(tmp_path, monkeypatch):
    path, _, sources = prepared_manifest_fixture(tmp_path, monkeypatch)
    sources["scientific.py"] = "d" * 64
    with pytest.raises(audit.AuditError, match="source snapshot"):
        audit.validate_prepared_manifest(path)


@pytest.mark.parametrize("rebind_sha", [False, True])
def test_prepared_config_tamper_or_rebound_scientific_recipe_rejects(tmp_path, monkeypatch, rebind_sha):
    path, manifest, _ = prepared_manifest_fixture(tmp_path, monkeypatch)
    entry = manifest["jobs"][0]
    cfg_path = Path(entry["config"]["path"])
    cfg = audit.read(cfg_path)
    cfg["module_lr"] *= 2
    write(cfg_path, cfg)
    if rebind_sha:
        entry["config"]["sha256"] = audit.sha(cfg_path)
        write(path, manifest)
    with pytest.raises(audit.AuditError, match="scientific recipe" if rebind_sha else "config SHA"):
        audit.validate_prepared_manifest(path)


def test_prepared_duplicate_job_is_not_a_four_run_campaign(tmp_path, monkeypatch):
    path, manifest, _ = prepared_manifest_fixture(tmp_path, monkeypatch)
    manifest["jobs"][-1] = copy.deepcopy(manifest["jobs"][0])
    write(path, manifest)
    with pytest.raises(audit.AuditError, match="duplicate prepared jobs"):
        audit.validate_prepared_manifest(path)
