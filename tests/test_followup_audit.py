"""Reject plausible-looking but incomplete or mismatched follow-up evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_followup_campaign as audit


REPO = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "a" * 64
CHECKPOINT_SOURCES = {
    name: hashlib.sha256(name.encode()).hexdigest()
    for name in ("models/current_adapted.py", "models/tuna_protocol.py")
}
HOST_KEY = "SHA256:" + "A" * 43


def fake_runtime(marker):
    return {"fingerprint_sha256": marker * 64,
            "gpu_models": ["NVIDIA GeForce RTX 5090"] * 2, "rule": audit.RUNTIME_ID_RULE}


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


def checkpoint_fixture(tmp_path, dataset="cifar224", method="current_adapted"):
    job, prepared, saved, result, data, sources = fixture(dataset, method)
    run = tmp_path / job
    for name, value in (("config.json", saved), ("results.json", result), ("data_manifest.json", data)):
        write(run / name, value)
    (run / "checkpoint.pt").write_bytes(b"server-local-test-payload")
    digest = audit.sha(run / "checkpoint.pt")
    backup = tmp_path / "backups" / job / "checkpoint.pt"
    backup.parent.mkdir(parents=True)
    backup.write_bytes((run / "checkpoint.pt").read_bytes())
    wrapper = "models/" + ("tuna_protocol" if method == "tuna" else "current_adapted") + ".py"
    report = {"native_cpu_restore_executed": True, "exact_typed_bitwise_comparison": True,
              "compared_nodes": 10, "ignored_paths": [],
              "reference_semantic_sha256": "b" * 64, "restored_semantic_sha256": "b" * 64,
              "actual_training_rows": sum(audit.DATA[dataset]["train_tasks"]),
              "actual_seen_classes": audit.DATA[dataset]["classes"],
              "source_data_manifest_rebuilt": True, "source_training_arrivals_verified": True,
              "global_rng_payloads_validated_on_cpu": True, "cuda_rng_applied": False,
              "extra_training_steps": 0, "accuracy_evaluation_performed": False,
              "pretrained_npz_sha256": audit.NPZ_SHA,
              "runtime_source_files_sha256": {wrapper: CHECKPOINT_SOURCES[wrapper]},
              "persisted_mode_map_compared": method == "tuna"}
    report_path = backup.parent / "native_restore.json"
    write(report_path, report)
    evidence = {"job": job, "checkpoint_sha256": digest,
                "checkpoint_bytes": (run / "checkpoint.pt").stat().st_size,
                "backup_sha256": digest, "native_restore_verified": True,
                "validation_only": False, "formal_checkpoint_evidence": True,
                "source_manifest_sha256": MANIFEST_SHA,
                "source_process_terminal_verified": True,
                "backup_checkpoint_path": str(backup),
                "restore_report_path": str(report_path),
                "restore_report_sha256": audit.sha(report_path), "restore_report": report,
                "source_metadata_sha256": {"config": audit.sha(run / "config.json"),
                                           "results": audit.sha(run / "results.json"),
                                           "data_manifest": audit.sha(run / "data_manifest.json")}}
    evidence_path = tmp_path / "evidence" / (job + ".json")
    write(evidence_path, evidence)
    return job, run, evidence_path, evidence


def check_checkpoint(job, run, path):
    return audit.validate_checkpoint_evidence(
        path, {"job": job}, run, sources=CHECKPOINT_SOURCES,
        manifest_sha256=MANIFEST_SHA)


def rebind_restore_report(path, evidence, report):
    report_path = Path(evidence["restore_report_path"])
    write(report_path, report)
    evidence["restore_report"] = report
    evidence["restore_report_sha256"] = audit.sha(report_path)
    write(path, evidence)


@pytest.mark.parametrize("field,value,match", [
    ("backup_sha256", "0" * 64, "checkpoint SHA"),
    ("native_restore_verified", False, "Native restore"),
    ("source_process_terminal_verified", False, "terminal state"),
    ("checkpoint_bytes", True, "checkpoint SHA"),
])
def test_checkpoint_claim_requires_all_independent_evidence(tmp_path, field, value, match):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    check_checkpoint(job, run, path)
    evidence[field] = value
    write(path, evidence)
    with pytest.raises(audit.AuditError, match=match):
        check_checkpoint(job, run, path)


def test_checkpoint_evidence_binds_current_result_bytes(tmp_path):
    job, run, path, _ = checkpoint_fixture(tmp_path)
    with (run / "results.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(audit.AuditError, match="metadata binding"):
        check_checkpoint(job, run, path)


@pytest.mark.parametrize("dataset", ["cifar224", "imagenetr"])
@pytest.mark.parametrize("method", ["current_adapted", "tuna"])
def test_checkpoint_closure_reopens_real_backup_and_report(tmp_path, dataset, method):
    job, run, path, evidence = checkpoint_fixture(tmp_path, dataset, method)
    result = check_checkpoint(job, run, path)
    assert result["actual_backup_rehashed"] is True
    assert result["formal_checkpoint_evidence"] is True
    assert result["restore_report_sha256"] == audit.sha(Path(evidence["restore_report_path"]))
    assert result["reference_semantic_sha256"] == result["restored_semantic_sha256"]


@pytest.mark.parametrize("field,value,match", [
    ("validation_only", True, "Validation-only"),
    ("formal_checkpoint_evidence", False, "Validation-only"),
    ("source_manifest_sha256", "0" * 64, "prepared-manifest binding"),
])
def test_checkpoint_rejects_nonformal_or_unbound_evidence(tmp_path, field, value, match):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    evidence[field] = value
    write(path, evidence)
    with pytest.raises(audit.AuditError, match=match):
        check_checkpoint(job, run, path)


@pytest.mark.parametrize("target,operation,match", [
    ("backup_checkpoint_path", "remove", "backup is missing"),
    ("backup_checkpoint_path", "tamper", "backup bytes differ"),
    ("restore_report_path", "remove", "report file is missing"),
    ("restore_report_path", "tamper", "report file SHA differs"),
])
def test_prior_green_evidence_cannot_hide_missing_or_tampered_server_files(
    tmp_path, target, operation, match,
):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    recorded = Path(evidence[target])
    if operation == "remove":
        recorded.unlink()
    else:
        with recorded.open("ab") as stream:
            stream.write(b"\n")
    with pytest.raises(audit.AuditError, match=match):
        check_checkpoint(job, run, path)


def test_source_checkpoint_cannot_be_its_own_backup(tmp_path):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    evidence["backup_checkpoint_path"] = str(run / "checkpoint.pt")
    write(path, evidence)
    with pytest.raises(audit.AuditError, match="Separate server-local backup"):
        check_checkpoint(job, run, path)


def test_embedded_restore_report_cannot_diverge_from_its_file(tmp_path):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    evidence["restore_report"]["compared_nodes"] += 1
    write(path, evidence)
    with pytest.raises(audit.AuditError, match="Embedded and server-local"):
        check_checkpoint(job, run, path)


@pytest.mark.parametrize("field,value,match", [
    ("native_cpu_restore_executed", False, "CPU exact comparison"),
    ("exact_typed_bitwise_comparison", False, "CPU exact comparison"),
    ("ignored_paths", ["learner.core.optimizer"], "ignored learner fields"),
    ("compared_nodes", 0, "Compared native learner nodes"),
    ("restored_semantic_sha256", "0" * 64, "semantic digest"),
    ("reference_semantic_sha256", None, "semantic digest"),
    ("actual_training_rows", 100, "full-data endpoint"),
    ("actual_seen_classes", 10, "full-data endpoint"),
    ("source_data_manifest_rebuilt", False, "full-data endpoint"),
    ("source_training_arrivals_verified", False, "full-data endpoint"),
    ("global_rng_payloads_validated_on_cpu", False, "restore execution"),
    ("cuda_rng_applied", True, "restore execution"),
    ("extra_training_steps", 1, "restore execution"),
    ("extra_training_steps", False, "restore execution"),
    ("accuracy_evaluation_performed", True, "restore execution"),
    ("pretrained_npz_sha256", "0" * 64, "restore execution"),
    ("runtime_source_files_sha256", {"models/current_adapted.py": "0" * 64}, "imported source"),
    ("runtime_source_files_sha256", {}, "imported source"),
])
def test_rebound_report_still_requires_executed_exact_full_native_restore(
    tmp_path, field, value, match,
):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    report = copy.deepcopy(evidence["restore_report"])
    report[field] = value
    rebind_restore_report(path, evidence, report)
    with pytest.raises(audit.AuditError, match=match):
        check_checkpoint(job, run, path)


def test_unknown_imported_source_cannot_match_absent_mapping_value(tmp_path):
    job, run, path, evidence = checkpoint_fixture(tmp_path)
    report = copy.deepcopy(evidence["restore_report"])
    report["runtime_source_files_sha256"]["outside/unknown.py"] = None
    rebind_restore_report(path, evidence, report)
    with pytest.raises(audit.AuditError, match="imported source"):
        check_checkpoint(job, run, path)


def test_tuna_restore_must_compare_its_persisted_modes(tmp_path):
    job, run, path, evidence = checkpoint_fixture(tmp_path, method="tuna")
    report = copy.deepcopy(evidence["restore_report"])
    report["persisted_mode_map_compared"] = False
    rebind_restore_report(path, evidence, report)
    with pytest.raises(audit.AuditError, match="persisted module modes"):
        check_checkpoint(job, run, path)


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


def completed_campaign_fixture(tmp_path, monkeypatch):
    entries, configs, evidence_rows = [], {}, []
    sources = dict(CHECKPOINT_SOURCES)
    for dataset in audit.DATA:
        for method in ("current_adapted", "tuna"):
            job, run, _, evidence = checkpoint_fixture(tmp_path / (dataset + method), dataset, method)
            _, prepared, _, _, _, native_sources = fixture(dataset, method)
            prepared["output_dir"] = str(run)
            prepared_path = tmp_path / "prepared_configs" / (job + ".json")
            write(prepared_path, prepared)
            saved = audit.read(run / "config.json")
            saved["output_dir"] = str(run)
            write(run / "config.json", saved)
            evidence["source_metadata_sha256"]["config"] = audit.sha(run / "config.json")
            gpu = 0 if dataset == "cifar224" else 1
            entries.append({"job": job, "dataset": dataset, "method": method,
                            "gpu": gpu,
                            "environment": {},
                            "comparison_variant": prepared["comparison_variant"], "output_dir": str(run),
                            "config": {"path": str(prepared_path), "sha256": audit.sha(prepared_path)}})
            configs[job] = prepared
            sources.update(native_sources)
            evidence_rows.append((job, evidence))
    manifest = {"version": 1, "jobs": entries, "source_files_sha256": sources,
                "campaign_root": str(tmp_path), "python": "/env/python", "repo": str(REPO),
                "environment": {},
                "source_commit": "b" * 40, "pretrained_npz": {"sha256": audit.NPZ_SHA}}
    path = tmp_path / "manifest.json"
    write(path, manifest)
    evidence_root = tmp_path / "final_evidence"
    for job, evidence in evidence_rows:
        evidence["source_manifest_sha256"] = audit.sha(path)
        write(evidence_root / (job + ".json"), evidence)
    monkeypatch.setattr(audit, "_validate_base_manifest", lambda p: (manifest, configs))
    monkeypatch.setattr(audit, "BASE_MANIFEST_SHA256", audit.sha(path))
    return path, evidence_root, evidence_rows


def tuna20_campaign_fixture(base_path, base_evidence_root):
    base = audit.read(base_path)
    root = base_path.parent / "campaign_tuna20"
    manifest = copy.deepcopy(base)
    manifest.update(campaign_root=str(root), user_requested_image_epochs=20,
                    base_manifest_sha256=audit.sha(base_path), base_manifest_path=str(base_path),
                    base_manifest_text=base_path.read_bytes().decode("utf-8"))
    evidence_root = root / "checkpoint_evidence"
    for old, row in zip(base["jobs"], manifest["jobs"]):
        job = old["job"]
        cfg = audit.read(old["config"]["path"])
        run = root / "formal/runs" / job
        cfg["output_dir"] = row["output_dir"] = str(run)
        if row["method"] == "tuna":
            cfg.update(tuned_epoch=20, prefix="followup_tuna_epochs20", training_budget_variant="epochs20")
        cfg_path = root / "formal/configs" / (job + ".json")
        write(cfg_path, cfg)
        row["config"] = {"path": str(cfg_path), "sha256": audit.sha(cfg_path)}
        row["command"] = [base["python"], str(Path(base["repo"]) / "main.py"), "--config", str(cfg_path)]
        if row["method"] != "tuna":
            continue
        original = Path(old["output_dir"])
        saved = {**audit.normalized_prepared(cfg), "config": str(cfg_path)}
        result, data = audit.read(original / "results.json"), audit.read(original / "data_manifest.json")
        result["config_fingerprint"] = audit.config_fingerprint(saved)
        for diagnostic in result["task_diagnostics"]:
            diagnostic["training"]["image_epochs"] = 20
            diagnostic["training"]["main_optimizer_steps_observed"] = (
                (diagnostic["arrivals"] + saved["batch_size"] - 1) // saved["batch_size"]) * 20
        for name, value in (("config", saved), ("results", result), ("data_manifest", data)):
            write(run / (name + ".json"), value)
        (run / "checkpoint.pt").write_bytes((original / "checkpoint.pt").read_bytes())
        evidence = audit.read(base_evidence_root / (job + ".json"))
        backup = root / "backups" / job / "checkpoint.pt"
        backup.parent.mkdir(parents=True)
        backup.write_bytes((run / "checkpoint.pt").read_bytes())
        report_path = backup.parent / "native_restore.json"
        write(report_path, evidence["restore_report"])
        evidence.update(backup_checkpoint_path=str(backup), restore_report_path=str(report_path),
                        restore_report_sha256=audit.sha(report_path),
                        source_metadata_sha256={name: audit.sha(run / (name + ".json"))
                                                for name in ("config", "results", "data_manifest")})
        write(evidence_root / (job + ".json"), evidence)
    path = root / "manifest.json"
    write(path, manifest)
    for evidence_path in evidence_root.glob("*.json"):
        evidence = audit.read(evidence_path)
        evidence["source_manifest_sha256"] = audit.sha(path)
        write(evidence_path, evidence)
    return path, evidence_root


def recovery_fixture(source_path, source_evidence_root, partition, monkeypatch):
    original = audit.read(source_path)
    root = source_path.parent / ("recovery-" + partition)
    rows, configs, final_data = [], {}, {}
    for old in original["jobs"]:
        if old["job"] not in audit.PARTITIONS[partition]:
            continue
        job, dataset = old["job"], old["dataset"]
        count = 2 if partition == "current20" else (4 if dataset == "cifar224" else 6)
        old_run = Path(old["output_dir"])
        result, data = audit.read(old_run / "results.json"), audit.read(old_run / "data_manifest.json")
        prefix = copy.deepcopy(result["stages"][:count])
        progress_path = old_run / "progress.json"
        write(progress_path, {"status": "running", "stages": prefix})
        old_log = Path(original["campaign_root"]) / "formal/logs" / (job + ".log")
        old_log.parent.mkdir(parents=True, exist_ok=True)
        old_log.write_text(json.dumps({"epoch_progress": True, "task": count, "epoch": 1, "wall_seconds": 7.5}) + "\n")
        snapshot = root / "resume_sources" / job / "checkpoint.pt"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes((old_run / "checkpoint.pt").read_bytes())
        cp_sha = audit.sha(snapshot)
        bound = lambda path: {"path": str(path), "sha256": audit.sha(path)}
        prepared = audit.read(old["config"]["path"])
        run, cfg_path = root / "runs" / job, root / "configs" / (job + ".json")
        cfg = {**prepared, "resume": str(snapshot), "output_dir": str(run)}
        write(cfg_path, cfg)
        lineage = {"original_manifest": bound(source_path), "original_prepared_config": old["config"],
                   "original_saved_config": bound(old_run / "config.json"), "original_output_dir": str(old_run),
                   "original_progress": bound(progress_path), "original_data_manifest": bound(old_run / "data_manifest.json"),
                   "source_checkpoint": {**bound(old_run / "checkpoint.pt"), "bytes": snapshot.stat().st_size},
                   "resume_checkpoint": {**bound(snapshot), "bytes": snapshot.stat().st_size},
                   "config_fingerprint": audit.config_fingerprint(audit.normalized_prepared(prepared)),
                   "data_fingerprint": data["fingerprint"], "random_stream_policy": audit.RNG_POLICY,
                   "saved_stage_count": count, "prefix_stages": prefix, "prefix_stages_sha256": audit.canonical_sha(prefix),
                   "original_receipt": {"observation": "original_pid_absent"}, "original_log": bound(old_log)}
        row = {key: old[key] for key in ("job", "gpu", "dataset", "method", "comparison_variant", "environment")}
        row.update(config=bound(cfg_path), output_dir=str(run), lineage=lineage,
                   command=[original["python"], str(Path(original["repo"]) / "main.py"), "--config", str(cfg_path)])
        rows.append(row)
        configs[job] = cfg
        saved = {**audit.normalized_prepared(cfg), "config": str(cfg_path)}
        result["execution"]["wall_seconds"] = 100.0 * (10 - count) + 50.0
        for key, value in (("config", saved), ("results", result), ("data_manifest", data)):
            write(run / (key + ".json"), value)
        (run / "checkpoint.pt").write_bytes(snapshot.read_bytes() + b"-resumed")
        final_data[job] = (old, row, run, result, cp_sha)
        (old_run / "results.json").unlink()  # Interrupted originals remain incomplete.
    manifest = {key: copy.deepcopy(original[key]) for key in
                ("repo", "python", "environment", "pretrained_npz", "source_files_sha256", "source_commit")}
    manifest.update(schema_version=1, kind="followup_task_boundary_recovery", partition=partition,
                    source_manifest={"path": str(source_path), "sha256": audit.sha(source_path)}, recovery_root=str(root), jobs=rows)
    path = root / "manifest.json"
    write(path, manifest)
    recovery_sha = audit.sha(path)
    evidence_root = root / "checkpoint_evidence"
    for job, (_, row, run, result, cp_sha) in final_data.items():
        lineage = row["lineage"]
        receipt = {"job": job, "pid": 100, "proc_start_ticks": 1000, "manifest_sha256": recovery_sha,
                   "config_sha256": row["config"]["sha256"], "command": row["command"], "gpu": row["gpu"],
                   "environment": {**manifest["environment"], **row["environment"]},
                   "cwd": manifest["repo"], "source_commit": manifest["source_commit"],
                   "source_inventory_sha256": hashlib.sha256(json.dumps(manifest["source_files_sha256"], sort_keys=True).encode()).hexdigest()}
        receipt_path = root / "receipts" / (job + ".json")
        write(receipt_path, receipt)
        metadata_shas = {key: audit.sha(run / (key + ".json")) for key in ("config", "results", "data_manifest")}
        seal = {"job": job, "recovery_manifest_sha256": recovery_sha, "lineage_sha256": audit.canonical_sha(lineage),
                "prefix_stages_sha256": lineage["prefix_stages_sha256"], "saved_stage_count": lineage["saved_stage_count"],
                "config_sha256": metadata_shas["config"], "result_sha256": metadata_shas["results"],
                "data_manifest_sha256": metadata_shas["data_manifest"], "original_process_state": "absent", "observed_exit_code": None}
        write(root / "queue" / (job + ".complete.json"), seal)
        evidence = audit.read(source_evidence_root / (job + ".json"))
        backup = root / "backups" / job / "checkpoint.pt"
        backup.parent.mkdir(parents=True)
        backup.write_bytes((run / "checkpoint.pt").read_bytes())
        report = copy.deepcopy(evidence["restore_report"])
        report.update(recovery_prefix_verified=True, saved_stage_count=lineage["saved_stage_count"],
                      source_checkpoint_stages_sha256=lineage["prefix_stages_sha256"], restored_final_prefix_sha256=lineage["prefix_stages_sha256"],
                      original_checkpoint_sha256=cp_sha, resume_checkpoint_sha256=cp_sha,
                      recovery_prefix_compared_nodes=10, recovery_prefix_ignored_paths=[],
                      source_checkpoint_prefix_semantic_sha256="c" * 64, source_checkpoint_rng_semantic_sha256="d" * 64)
        report_path = backup.parent / "native_restore.json"
        write(report_path, report)
        expected_lineage = {key: lineage[key] for key in ("saved_stage_count", "prefix_stages_sha256", "config_fingerprint", "data_fingerprint", "random_stream_policy")}
        expected_lineage.update(original_checkpoint_sha256=cp_sha, resume_checkpoint_sha256=cp_sha, prefix_verified=True)
        evidence.update(checkpoint_sha256=audit.sha(run / "checkpoint.pt"), checkpoint_bytes=(run / "checkpoint.pt").stat().st_size,
                        backup_sha256=audit.sha(backup), backup_checkpoint_path=str(backup), restore_report=report,
                        restore_report_path=str(report_path), restore_report_sha256=audit.sha(report_path),
                        source_metadata_sha256=metadata_shas, source_receipt_sha256=audit.sha(receipt_path),
                        recovery_manifest_sha256=recovery_sha, recovery_lineage=expected_lineage)
        write(evidence_root / (job + ".json"), evidence)
    checked = {"manifest": manifest, "configs": configs, "source_manifest": original}
    monkeypatch.setattr(audit, "read_guarded_recovery", lambda p: checked)
    return path, evidence_root


def test_all_four_final_rows_require_and_accept_actual_backup_closure(tmp_path, monkeypatch):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    summary = audit.audit_campaign(path, evidence_root)
    assert summary["verified_results"] == summary["metrics_verified_results"] == 4
    assert summary["status"] == "verified_complete"
    assert all(row["checkpoint_audit"]["actual_backup_rehashed"] is True for row in summary["rows"])


@pytest.mark.parametrize("malformation", ["missing_backup", "not_executed"])
def test_allow_incomplete_never_turns_malformed_present_closure_green(
    tmp_path, monkeypatch, malformation,
):
    path, evidence_root, rows = completed_campaign_fixture(tmp_path, monkeypatch)
    job, evidence = rows[0]
    evidence_path = evidence_root / (job + ".json")
    if malformation == "missing_backup":
        Path(evidence["backup_checkpoint_path"]).unlink()
    else:
        report = copy.deepcopy(evidence["restore_report"])
        report["native_cpu_restore_executed"] = False
        rebind_restore_report(evidence_path, evidence, report)
    with pytest.raises(audit.AuditError, match="backup is missing|CPU exact comparison"):
        audit.audit_campaign(path, evidence_root, allow_incomplete=True)


@pytest.mark.parametrize("partition", ["current20", "tuna"])
def test_partition_audits_exactly_its_two_local_jobs(tmp_path, monkeypatch, partition):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    if partition == "tuna":
        path, evidence_root = tuna20_campaign_fixture(path, evidence_root)
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("1"))
    summary = audit.audit_campaign(path, evidence_root, partition=partition,
                                   host_alias=partition + "-host", host_key_fingerprint=HOST_KEY)
    assert summary["expected_results"] == summary["verified_results"] == 2
    assert summary["global_expected_results"] == 4
    assert summary["partition_jobs"] == list(audit.PARTITIONS[partition])
    assert {row["job"] for row in summary["rows"]} == set(audit.PARTITIONS[partition])
    assert set(summary["external_jobs"]) == set(audit.PARTITIONS["all"]) - set(audit.PARTITIONS[partition])
    assert summary["external_job_status"] == "assigned_to_other_partition"
    assert all(set(row["metadata_exports"]) == {"prepared_config", "config", "results", "data_manifest"}
               for row in summary["rows"])
    assert summary["host"]["runtime_identity"]["fingerprint_sha256"] == "1" * 64


@pytest.mark.parametrize("alias,fingerprint", [(None, None), ("host", None), (None, HOST_KEY), ("host", "invalid")])
def test_partition_requires_inspected_host_identity(tmp_path, monkeypatch, alias, fingerprint):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    with pytest.raises(audit.AuditError, match="host alias|host-key fingerprint"):
        audit.audit_campaign(path, evidence_root, partition="current20",
                             host_alias=alias, host_key_fingerprint=fingerprint)


def test_excluded_host_jobs_are_external_even_when_their_outputs_are_absent(tmp_path, monkeypatch):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    manifest = audit.read(path)
    for entry in manifest["jobs"]:
        if entry["method"] == "tuna":
            for file in Path(entry["output_dir"]).iterdir():
                file.unlink()
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("1"))
    summary = audit.audit_campaign(path, evidence_root, partition="current20",
                                   host_alias="current-host", host_key_fingerprint=HOST_KEY)
    assert summary["status"] == "verified_complete" and len(summary["external_jobs"]) == 2


def test_partition_progress_counts_only_its_pending_local_proof(tmp_path, monkeypatch):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    (evidence_root / (audit.PARTITIONS["current20"][0] + ".json")).unlink()
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("1"))
    summary = audit.audit_campaign(path, evidence_root, partition="current20", allow_incomplete=True,
                                   host_alias="current-host", host_key_fingerprint=HOST_KEY)
    assert summary["status"] == "incomplete" and summary["verified_results"] == 1
    assert summary["metrics_verified_results"] == summary["expected_results"] == 2
    assert len(summary["external_jobs"]) == 2


def test_tuna20_manifest_is_an_exact_overlay_and_preserves_original(tmp_path, monkeypatch):
    base_path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    original_bytes = base_path.read_bytes()
    path, _ = tuna20_campaign_fixture(base_path, evidence_root)
    manifest, configs = audit.validate_prepared_manifest(path)
    assert base_path.read_bytes() == original_bytes
    assert manifest["base_manifest_sha256"] == audit.sha(base_path)
    for row in manifest["jobs"]:
        cfg = configs[row["job"]]
        if row["method"] == "tuna":
            assert cfg["tuned_epoch"] == 20 and cfg["crct_epochs"] == 30
            assert cfg["training_budget_variant"] == "epochs20"
        else:
            assert cfg["epochs"] == 20 and "training_budget_variant" not in cfg
            assert not Path(cfg["output_dir"]).exists()  # Unused copies are never trained here.


@pytest.mark.parametrize("field,value", [
    ("tuned_epoch", 15), ("tuned_epoch", 20.0), ("crct_epochs", 31),
    ("init_lr", 0.02), ("batch_size", 32), ("training_budget_variant", "native"),
    ("comparison_variant", "tuna20"), ("prefix", "followup_tuna_native"),
])
def test_overlay_rejects_rebound_tuna_config_with_any_other_delta(tmp_path, monkeypatch, field, value):
    base_path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    path, _ = tuna20_campaign_fixture(base_path, evidence_root)
    manifest = audit.read(path)
    row = next(row for row in manifest["jobs"] if row["method"] == "tuna" and row["dataset"] == "cifar224")
    cfg_path = Path(row["config"]["path"])
    cfg = audit.read(cfg_path)
    cfg[field] = value
    write(cfg_path, cfg)
    row["config"]["sha256"] = audit.sha(cfg_path)
    write(path, manifest)
    with pytest.raises(audit.AuditError, match="authorized config delta"):
        audit.validate_prepared_manifest(path)


def test_overlay_also_guards_unused_current_config_copies(tmp_path, monkeypatch):
    base_path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    path, _ = tuna20_campaign_fixture(base_path, evidence_root)
    manifest = audit.read(path)
    row = next(row for row in manifest["jobs"] if row["method"] == "current_adapted")
    cfg_path = Path(row["config"]["path"])
    cfg = audit.read(cfg_path)
    cfg["module_rank"] *= 2
    write(cfg_path, cfg)
    row["config"]["sha256"] = audit.sha(cfg_path)
    write(path, manifest)
    with pytest.raises(audit.AuditError, match="authorized config delta"):
        audit.validate_prepared_manifest(path)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.__setitem__("base_manifest_sha256", "0" * 64), "authoritative user budget"),
    (lambda m: m.__setitem__("base_manifest_text", "{}"), "Metadata export SHA"),
    (lambda m: m.__setitem__("user_requested_image_epochs", True), "authoritative user budget"),
    (lambda m: m.__setitem__("source_commit", "0" * 40), "common source/data"),
    (lambda m: m["pretrained_npz"].__setitem__("sha256", "0" * 64), "common source/data"),
    (lambda m: m.__setitem__("extra_scientific_override", True), "common source/data"),
    (lambda m: m["jobs"][0].__setitem__("gpu", 1), "nonadministrative job field"),
])
def test_overlay_rejects_unbound_or_scientifically_changed_manifest(tmp_path, monkeypatch, mutation, match):
    base_path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    path, _ = tuna20_campaign_fixture(base_path, evidence_root)
    manifest = audit.read(path)
    mutation(manifest)
    write(path, manifest)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_prepared_manifest(path)


def test_original_native_tuna_reports_cannot_enter_final_partition(tmp_path, monkeypatch):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("2"))
    with pytest.raises(audit.AuditError, match="user-authorized epochs20 overlay"):
        audit.audit_campaign(path, evidence_root, partition="tuna",
                             host_alias="tuna-host", host_key_fingerprint=HOST_KEY)


@pytest.mark.parametrize("partition", ["all", "current20"])
def test_unused_current_overlay_jobs_are_not_an_authorized_local_partition(tmp_path, monkeypatch, partition):
    path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    overlay, _ = tuna20_campaign_fixture(path, evidence_root)
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("2"))
    with pytest.raises(audit.AuditError, match="authorized only for the tuna partition"):
        audit.audit_campaign(overlay, partition=partition,
                             host_alias="tuna-host", host_key_fingerprint=HOST_KEY)


@pytest.mark.parametrize("partition", ["current20", "tuna"])
def test_guarded_recovery_preserves_prefix_and_scopes_wall_and_peak(tmp_path, monkeypatch, partition):
    original, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    if partition == "tuna":
        original, evidence_root = tuna20_campaign_fixture(original, evidence_root)
    recovery, recovered_evidence = recovery_fixture(original, evidence_root, partition, monkeypatch)
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("1"))
    summary = audit.audit_campaign(original, recovered_evidence, partition=partition,
                                   host_alias="restarted-host", host_key_fingerprint=HOST_KEY,
                                   recovery_manifest_path=recovery)
    assert summary["verified_results"] == 2
    assert summary["manifest_sha256"] == audit.sha(original)
    assert summary["recovery_manifest_sha256"] == audit.sha(recovery)
    for row in summary["rows"]:
        resources = row["resources"]
        assert resources["full_run_wall_seconds"] is None
        assert resources["completed_stage_wall_seconds_total"] == 1000
        assert resources["execution"]["wall_seconds"] < 1000
        assert "Resumed segment only" in resources["execution_scope"]
        assert "Resumed process only" in resources["gpu_peak_scope"]
        assert resources["interrupted_work"]["known_seconds_lower_bound"] == 7.5
        assert resources["interrupted_work"]["total_interrupted_work_seconds"] is None
        assert row["checkpoint_audit"]["recovery_lineage"]["prefix_verified"] is True


def recovery_audit_fixture(tmp_path, monkeypatch):
    original, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    recovery, recovered_evidence = recovery_fixture(original, evidence_root, "current20", monkeypatch)
    monkeypatch.setattr(audit, "runtime_identity", lambda: fake_runtime("1"))
    def check():
        return audit.audit_campaign(original, recovered_evidence, partition="current20",
                                     host_alias="restarted-host", host_key_fingerprint=HOST_KEY,
                                     recovery_manifest_path=recovery)
    return original, recovery, recovered_evidence, check


def test_recovery_does_not_allow_unbound_resume_config(tmp_path, monkeypatch):
    original, recovery, _, _ = recovery_audit_fixture(tmp_path, monkeypatch)
    old = audit.read(original)["jobs"][0]
    recovered = audit.read(recovery)["jobs"][0]
    run = Path(recovered["output_dir"])
    with pytest.raises(audit.AuditError, match="prebound recipe"):
        audit.validate_records(audit.read(old["config"]["path"]), audit.read(run / "config.json"),
                               audit.read(run / "results.json"), audit.read(run / "data_manifest.json"),
                               audit.read(original)["source_files_sha256"])


def test_recovery_never_changes_original_prefix_stage_timers(tmp_path, monkeypatch):
    _, recovery, _, check = recovery_audit_fixture(tmp_path, monkeypatch)
    entry = audit.read(recovery)["jobs"][0]
    result_path = Path(entry["output_dir"]) / "results.json"
    result = audit.read(result_path)
    result["stages"][0]["wall_seconds"] += 0.001
    write(result_path, result)
    with pytest.raises(audit.AuditError, match="exact source-checkpoint prefix"):
        check()


@pytest.mark.parametrize("field,value", [("backbone_lr", 0.1), ("epochs", 5), ("seed", [1994])])
def test_recovery_scientific_changes_are_rejected_even_with_updated_config_sha(tmp_path, monkeypatch, field, value):
    _, recovery, _, check = recovery_audit_fixture(tmp_path, monkeypatch)
    checked = audit.read_guarded_recovery(recovery)
    entry = checked["manifest"]["jobs"][0]
    checked["configs"][entry["job"]][field] = value
    with pytest.raises(audit.AuditError, match="only resume and output_dir"):
        check()


@pytest.mark.parametrize("field,value,match", [
    ("recovery_prefix_verified", False, "prefix proof differs"),
    ("source_checkpoint_rng_semantic_sha256", None, "prefix/RNG verification"),
    ("restored_final_prefix_sha256", "0" * 64, "prefix proof differs"),
])
def test_recovery_requires_independent_native_source_prefix_proof(tmp_path, monkeypatch, field, value, match):
    _, _, evidence_root, check = recovery_audit_fixture(tmp_path, monkeypatch)
    path = next(evidence_root.glob("*.json"))
    evidence = audit.read(path)
    report = copy.deepcopy(evidence["restore_report"])
    report[field] = value
    rebind_restore_report(path, evidence, report)
    with pytest.raises(audit.AuditError, match=match):
        check()


def test_recovery_rejects_a_new_completion_seal_without_lineage_binding(tmp_path, monkeypatch):
    _, recovery, _, check = recovery_audit_fixture(tmp_path, monkeypatch)
    entry = audit.read(recovery)["jobs"][0]
    path = recovery.parent / "queue" / (entry["job"] + ".complete.json")
    seal = audit.read(path)
    seal["lineage_sha256"] = "0" * 64
    write(path, seal)
    with pytest.raises(audit.AuditError, match="terminal lineage"):
        check()


def test_recovery_still_requires_wall_time_covering_the_resumed_stage_regions(tmp_path, monkeypatch):
    _, recovery, _, check = recovery_audit_fixture(tmp_path, monkeypatch)
    entry = audit.read(recovery)["jobs"][0]
    path = Path(entry["output_dir"]) / "results.json"
    result = audit.read(path)
    result["execution"]["wall_seconds"] = 100
    write(path, result)
    with pytest.raises(audit.AuditError, match="shorter than"):
        check()


def partition_reports_fixture(tmp_path, monkeypatch, *, recovered=False):
    manifest_path, evidence_root, _ = completed_campaign_fixture(tmp_path, monkeypatch)
    overlay_path, overlay_evidence = tuna20_campaign_fixture(manifest_path, evidence_root)
    paths = []
    for group, marker in (("current20", "1"), ("tuna", "2")):
        monkeypatch.setattr(audit, "runtime_identity", lambda m=marker: fake_runtime(m))
        active_path, active_evidence = ((overlay_path, overlay_evidence) if group == "tuna"
                                        else (manifest_path, evidence_root))
        recovery_path = None
        if recovered:
            recovery_path, active_evidence = recovery_fixture(active_path, active_evidence, group, monkeypatch)
        summary = audit.audit_campaign(active_path, active_evidence, partition=group,
                                       host_alias=group + "-host", host_key_fingerprint=HOST_KEY,
                                       recovery_manifest_path=recovery_path)
        path = tmp_path / (group + "_partition.json")
        write(path, summary)
        paths.append(path)
    return manifest_path, paths


def merge_reports(manifest_path, paths):
    return audit.merge_partition_summaries(
        manifest_path, paths, expected_summary_sha256=[audit.sha(path) for path in paths])


def test_global_merge_recomputes_four_rows_without_opening_remote_models(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    actual_sha = audit.sha
    def metadata_only_sha(path):
        assert Path(path).suffix != ".pt", "Global merge must not reopen model files"
        return actual_sha(path)
    monkeypatch.setattr(audit, "sha", metadata_only_sha)
    merged = merge_reports(manifest, paths)
    assert merged["verified_results"] == merged["metrics_verified_results"] == 4
    assert merged["partition"] == "merged" and merged["external_jobs"] == []
    assert len(merged["partition_reports"]) == 2
    assert merged["remote_checkpoints_reopened_by_merge"] is False
    assert {row["job"] for row in merged["rows"]} == set(audit.PARTITIONS["all"])
    # AutoDL images can share one SSH host key. The live runtime hashes distinguish them.
    assert len({proof["host"]["host_key_fingerprint"] for proof in merged["partition_reports"]}) == 1
    assert len({proof["host"]["runtime_identity"]["fingerprint_sha256"]
                for proof in merged["partition_reports"]}) == 2
    assert all(proof["summary_sha256"] == proof["server_captured_summary_sha256"]
               for proof in merged["partition_reports"])
    assert len({proof["manifest_sha256"] for proof in merged["partition_reports"]}) == 2
    overlay = merged["training_budget_overlays"][0]
    assert overlay["base_manifest_sha256"] == audit.sha(manifest)
    assert overlay["manifest_sha256"] != overlay["base_manifest_sha256"]
    assert overlay["manifest_export"]["sha256"] == overlay["manifest_sha256"]
    assert all(row["training_budget_variant"] == "epochs20" for row in merged["rows"] if row["method"] == "tuna")
    assert merged["common_identity"]["pretrained_npz_sha256"] == audit.NPZ_SHA


def test_global_merge_verifies_four_recovered_endpoints_without_parent_or_final_models(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch, recovered=True)
    actual_sha = audit.sha
    def metadata_only(path):
        assert Path(path).suffix != ".pt", "Global merge cannot reopen source/resume/final checkpoints"
        return actual_sha(path)
    monkeypatch.setattr(audit, "sha", metadata_only)
    merged = merge_reports(manifest, paths)
    assert merged["verified_results"] == 4 and len(merged["recovery_manifests"]) == 2
    assert {row["checkpoint_audit"]["recovery_lineage"]["saved_stage_count"] for row in merged["rows"]} == {2, 4, 6}
    for row in merged["rows"]:
        assert row["resources"]["full_run_wall_seconds"] is None
        assert row["resources"]["completed_stage_wall_seconds_total"] == 1000
        assert row["checkpoint_audit"]["recovery_prefix_verification"]["recovery_prefix_verified"] is True
    assert "only the resumed process" in audit.markdown(merged)


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["rows"][0]["resources"].__setitem__("full_run_wall_seconds", 850.0), "resource export differs"),
    (lambda s: s["rows"][0]["checkpoint_audit"]["recovery_lineage"].__setitem__("prefix_verified", False), "parent checkpoint lineage"),
    (lambda s: s["rows"][0]["checkpoint_audit"]["recovery_prefix_verification"].__setitem__("source_checkpoint_rng_semantic_sha256", None), "prefix/RNG verification"),
    (lambda s: s["rows"][0]["recovery_exports"]["interrupted_work"].__setitem__("known_seconds_lower_bound", 9999.0), "lower bound differs"),
])
def test_global_merge_rejects_unbound_recovery_or_full_cost_claims(tmp_path, monkeypatch, mutation, match):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch, recovered=True)
    summary = audit.read(paths[0])
    mutation(summary)
    write(paths[0], summary)
    with pytest.raises(audit.AuditError, match=match):
        merge_reports(manifest, paths)


def test_global_recovered_rows_need_the_explicit_recovery_manifest(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch, recovered=True)
    summary = audit.read(paths[0])
    del summary["recovery_manifest_export"]
    del summary["recovery_manifest_sha256"]
    write(paths[0], summary)
    with pytest.raises(audit.AuditError, match="explicit recovery manifest proof"):
        merge_reports(manifest, paths)


def test_global_recovery_config_has_no_scientific_override_even_if_hashes_rebound(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch, recovered=True)
    summary = audit.read(paths[0])
    recovery = json.loads(summary["recovery_manifest_export"]["text"])
    row = summary["rows"][0]
    recovered = next(entry for entry in recovery["jobs"] if entry["job"] == row["job"])
    export = row["recovery_exports"]["prepared_config"]
    cfg = json.loads(export["text"])
    cfg["backbone_lr"] *= 2
    export["text"] = json.dumps(cfg)
    export["sha256"] = recovered["config"]["sha256"] = hashlib.sha256(export["text"].encode()).hexdigest()
    summary["recovery_manifest_export"]["text"] = json.dumps(recovery)
    summary["recovery_manifest_export"]["sha256"] = summary["recovery_manifest_sha256"] = hashlib.sha256(
        summary["recovery_manifest_export"]["text"].encode()).hexdigest()
    write(paths[0], summary)
    with pytest.raises(audit.AuditError, match="only resume and output_dir"):
        merge_reports(manifest, paths)


def rebind_overlay_manifest(summary, manifest):
    record = summary["training_budget_overlay"]["manifest_export"]
    record["text"] = json.dumps(manifest, indent=2) + "\n"
    record["sha256"] = summary["manifest_sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
    for row in summary["rows"]:
        row["checkpoint_audit"]["source_manifest_sha256"] = summary["manifest_sha256"]


def test_global_merge_rejects_an_original_native_budget_report_even_if_claimed_complete(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[1])
    del summary["training_budget_overlay"]
    summary["manifest_sha256"] = audit.sha(manifest)
    write(paths[1], summary)
    with pytest.raises(audit.AuditError, match="authoritative user-requested epochs20 overlay"):
        merge_reports(manifest, paths)


def test_global_merge_rechecks_common_overlay_source_identity(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[1])
    overlay_manifest = json.loads(summary["training_budget_overlay"]["manifest_export"]["text"])
    overlay_manifest["source_files_sha256"]["models/tuna.py"] = "0" * 64
    rebind_overlay_manifest(summary, overlay_manifest)
    write(paths[1], summary)
    with pytest.raises(audit.AuditError, match="common source/data"):
        merge_reports(manifest, paths)


@pytest.mark.parametrize("method,field,value", [
    ("current_adapted", "module_lr", 0.123), ("tuna", "crct_epochs", 31),
])
def test_global_merge_rechecks_all_overlay_config_deltas_after_hash_rebinding(
    tmp_path, monkeypatch, method, field, value,
):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[1])
    proof = summary["training_budget_overlay"]
    overlay_manifest = json.loads(proof["manifest_export"]["text"])
    entry = next(row for row in overlay_manifest["jobs"] if row["method"] == method)
    record = proof["config_exports"][entry["job"]]
    cfg = json.loads(record["text"])
    cfg[field] = value
    record["text"] = json.dumps(cfg)
    record["sha256"] = entry["config"]["sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
    rebind_overlay_manifest(summary, overlay_manifest)
    write(paths[1], summary)
    with pytest.raises(audit.AuditError, match="authorized config delta"):
        merge_reports(manifest, paths)


def test_global_merge_cannot_count_native_fifteen_epoch_execution_as_twenty(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[1])
    row = next(row for row in summary["rows"] if row["dataset"] == "cifar224")
    export = row["metadata_exports"]["results"]
    result = json.loads(export["text"])
    for diagnostic in result["task_diagnostics"]:
        diagnostic["training"]["image_epochs"] = 15
    export["text"] = json.dumps(result)
    export["sha256"] = row["source"]["results_sha256"] = hashlib.sha256(export["text"].encode()).hexdigest()
    write(paths[1], summary)
    with pytest.raises(audit.AuditError, match="observed image steps"):
        merge_reports(manifest, paths)


def test_global_merge_requires_server_captured_report_digest(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    original_digests = [audit.sha(path) for path in paths]
    with paths[0].open("a") as stream:
        stream.write("\n")
    with pytest.raises(audit.AuditError, match="server-captured SHA"):
        audit.merge_partition_summaries(manifest, paths, expected_summary_sha256=original_digests)


@pytest.mark.parametrize("digests", [None, [], ["0" * 64], ["G" * 64, "0" * 64]])
def test_global_merge_rejects_missing_or_invalid_expected_report_digests(tmp_path, monkeypatch, digests):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    with pytest.raises(audit.AuditError, match="server-captured partition report SHAs"):
        audit.merge_partition_summaries(manifest, paths, expected_summary_sha256=digests)


def test_global_merge_scientific_recipe_is_bound_to_full_manifest_bytes(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[0])
    row = summary["rows"][0]
    export = row["metadata_exports"]["prepared_config"]
    prepared = json.loads(export["text"])
    prepared["module_lr"] *= 2
    export["text"] = json.dumps(prepared)
    export["sha256"] = hashlib.sha256(export["text"].encode()).hexdigest()
    row["source"]["prepared_config_sha256"] = export["sha256"]
    write(paths[0], summary)
    with pytest.raises(audit.AuditError, match="Metadata export SHA differs: prepared_config"):
        merge_reports(manifest, paths)


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s.__setitem__("status", "incomplete"), "locally verified complete"),
    (lambda s: s.__setitem__("manifest_sha256", "0" * 64), "scientific identity"),
    (lambda s: s.__setitem__("source_commit", "0" * 40), "scientific identity"),
    (lambda s: s.__setitem__("auditor_sha256", "0" * 64), "scientific identity"),
    (lambda s: s["source_files_sha256"].__setitem__("models/tuna.py", "0" * 64), "scientific identity"),
    (lambda s: s["rows"][0].__setitem__("job", s["rows"][1]["job"]), "overlap or omit"),
    (lambda s: s.__setitem__("external_jobs", []), "external-job declaration"),
    (lambda s: s["rows"][0]["metrics"].__setitem__("aia", 0.99), "aia differs"),
    (lambda s: s["rows"][0]["checkpoint_audit"].__setitem__("actual_backup_rehashed", False), "strict local checkpoint proof"),
    (lambda s: s["rows"][0]["metadata_exports"]["results"].__setitem__("text", "{}"), "Metadata export SHA"),
])
def test_global_merge_rejects_incomplete_or_unbound_partition_exports(tmp_path, monkeypatch, mutation, match):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[0])
    mutation(summary)
    write(paths[0], summary)
    with pytest.raises(audit.AuditError, match=match):
        merge_reports(manifest, paths)


def test_global_merge_requires_complementary_groups(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    with pytest.raises(audit.AuditError, match="one current20 and one tuna"):
        merge_reports(manifest, [paths[0], paths[0]])


@pytest.mark.parametrize("field,value,match", [
    ("alias", "current20-host", "host aliases must be distinct"),
    ("runtime_identity", fake_runtime("1"), "distinct physical runtime"),
])
def test_global_merge_cannot_attribute_two_partitions_to_one_runtime(tmp_path, monkeypatch, field, value, match):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    summary = audit.read(paths[1])
    summary["host"][field] = value
    write(paths[1], summary)
    with pytest.raises(audit.AuditError, match=match):
        merge_reports(manifest, paths)


def test_runtime_fingerprint_hides_identifiers_and_is_order_invariant(monkeypatch):
    boot_id = "12345678-1234-1234-1234-123456789abc"
    original_read = Path.read_text
    def read_boot(path, *args, **kwargs):
        return boot_id if path == Path("/proc/sys/kernel/random/boot_id") else original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_boot)
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **kw: "GPU-private-A, RTX 5090\nGPU-private-B, RTX 5080\n")
    first = audit.runtime_identity()
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **kw: "GPU-private-B, RTX 5080\nGPU-private-A, RTX 5090\n")
    assert audit.runtime_identity() == first
    assert audit.validate_runtime_identity(first) == first
    assert "GPU-private" not in json.dumps(first) and boot_id not in json.dumps(first)
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **kw: "GPU-private-A, RTX 5090\nGPU-private-C, RTX 5080\n")
    assert audit.runtime_identity()["fingerprint_sha256"] != first["fingerprint_sha256"]
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **kw: "GPU-private-A, RTX 5090\nGPU-private-B, RTX 5080\n")
    boot_id = "87654321-1234-1234-1234-123456789abc"
    assert audit.runtime_identity()["fingerprint_sha256"] != first["fingerprint_sha256"]


def test_partition_runtime_record_cannot_publish_raw_identifiers():
    with pytest.raises(audit.AuditError, match="exposes raw identifiers"):
        audit.validate_runtime_identity({**fake_runtime("1"), "boot_id": "raw-private-id"})


def test_merge_cli_writes_two_eight_row_tables_and_preserves_original_twelve(tmp_path, monkeypatch):
    manifest, paths = partition_reports_fixture(tmp_path, monkeypatch)
    base = REPO / "experiments/campaign_20261117/summary/summary.json"
    output = tmp_path / "merged_delivery"
    monkeypatch.setattr(audit.sys, "argv", [
        "audit_followup_campaign.py", "--manifest", str(manifest),
        "--merge-partition-summaries", *(str(path) for path in paths),
        "--partition-summary-sha256", *(audit.sha(path) for path in paths),
        "--base-summary", str(base), "--output-dir", str(output)])
    assert audit.main() == 0
    result = audit.read(output / "summary.json")
    assert result["verified_results"] == 4 and len(result["combined_rows"]) == 16
    assert all(sum(row["dataset"] == dataset for row in result["combined_rows"]) == 8 for dataset in audit.DATA)
    original = audit.read(base)["rows"]
    assert [{k: v for k, v in row.items() if k not in {"comparison_variant", "imported_previous_campaign"}}
            for row in result["combined_rows"][:12]] == original
    for dataset in audit.DATA:
        assert {row["comparison_variant"] for row in result["combined_rows"]
                if row["dataset"] == dataset and row["method"] == "current_adapted"} == {"epochs5", "epochs20"}
    text = (output / "summary.md").read_text()
    assert "does not reopen remote model files" in text
    assert "epochs20; native mechanism" in text and "not reproduction of the original native image-training budget" in text


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
