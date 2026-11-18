#!/usr/bin/env python3
"""Audit the four predeclared follow-ups without selecting runs by test score.

Local audits run on the server containing the immutable source and checkpoints;
legitimate recovery additionally uses the shared pinned-runtime lineage guard.
The standard-library partition merge instead rechecks exact
metadata exports and the two SHA-bound host-local proofs, without reopening
remote model files. ``--allow-incomplete`` never turns missing checkpoint proof
into completion. The original twelve results are not modified or reselected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
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
PARTITIONS = {
    "all": tuple(job for job, *_ in JOBS),
    "current20": tuple(job for job, _, _, method in JOBS if method == "current_adapted"),
    "tuna": tuple(job for job, _, _, method in JOBS if method == "tuna"),
}
RUNTIME_ID_RULE = "sha256(canonical JSON of boot_id and sorted GPU UUIDs)"
BASE_MANIFEST_SHA256 = "8a70c7e5a6011bb293c833387d71046173503d4ba94efca3c3ee2587b82e4c43"
FIRST_CURRENT_RECOVERY_SHA256 = "194f569ee2adbe981383134ceb7d1b66576ee58959d61fce6d41eac555d8613f"
SEALED_TUNA_SUMMARY_SHA256 = "aee0a3ff3629b40e2ae98b6d2446722d72257757c30591231a11c8174eac0373"
SEALED_TUNA_AUDITOR_SHA256 = "ae30b86235b9417f2fbacf65066bc2949bb6d5e4915dd1f6bd173246e5006c03"
TUNA20_OVERLAY_KEYS = {"user_requested_image_epochs", "base_manifest_sha256",
                       "base_manifest_path", "base_manifest_text"}
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


def host_identity(alias, fingerprint):
    need(isinstance(alias, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", alias),
         "Partition audits require a nonsensitive host alias")
    need(isinstance(fingerprint, str)
         and re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}=?", fingerprint),
         "Partition audits require the inspected SHA256 host-key fingerprint")
    return {"alias": alias, "host_key_fingerprint": fingerprint}


def runtime_identity():
    """Identify the physical runtime without publishing boot IDs or GPU UUIDs."""
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    need(bool(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", boot_id)),
         "Cannot identify the live host boot")
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True)
    rows = [[part.strip() for part in row] for row in csv.reader(text.splitlines()) if row]
    need(len(rows) == 2 and all(len(row) == 2 and row[0].startswith("GPU-") and row[1] for row in rows)
         and len({row[0] for row in rows}) == 2, "Cannot identify the two physical GPUs")
    encoded = json.dumps({"boot_id": boot_id, "gpu_uuids": sorted(row[0] for row in rows)},
                         sort_keys=True, separators=(",", ":")).encode()
    return {"fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
            "gpu_models": sorted(row[1] for row in rows), "rule": RUNTIME_ID_RULE}


def validate_runtime_identity(value):
    need(isinstance(value, dict) and set(value) == {"fingerprint_sha256", "gpu_models", "rule"}
         and value.get("rule") == RUNTIME_ID_RULE
         and isinstance(value.get("fingerprint_sha256"), str)
         and re.fullmatch(r"[0-9a-f]{64}", value["fingerprint_sha256"])
         and isinstance(value.get("gpu_models"), list) and len(value["gpu_models"]) == 2
         and all(isinstance(name, str) and name and not any(c in name for c in "\r\n\0")
                 for name in value["gpu_models"]), "Partition runtime identity is incomplete or exposes raw identifiers")
    return value


def metadata_export(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "text": data.decode("utf-8")}


def same_json_value(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_guarded_recovery(path):
    # The shared local validator checks retained files/source identities but
    # never deserializes weights. Its pinned-runtime imports remain server-local.
    if __package__:
        from .resume_followup_after_reboot import validate_recovery_manifest
    else:
        from resume_followup_after_reboot import validate_recovery_manifest
    try:
        return validate_recovery_manifest(path)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise AuditError("Guarded recovery lineage failed: " + str(error)) from error


def validate_recovery_header(recovery, original, original_sha, partition):
    version = recovery.get("schema_version")
    need(type(version) is int and version in (1, 2) and recovery.get("kind") == "followup_task_boundary_recovery"
         and recovery.get("partition") == partition and partition in {"current20", "tuna"},
         "Wrong explicit task-boundary recovery schema/partition")
    need(recovery.get("source_manifest") == {"path": str(Path(original["campaign_root"]) / "manifest.json"),
                                             "sha256": original_sha}, "Recovery original manifest binding differs")
    need(all(same_json_value(recovery.get(key), original.get(key)) for key in
             ("repo", "python", "environment", "pretrained_npz", "source_files_sha256", "source_commit")),
         "Recovery scientific source/runtime/weights changed")
    rows = recovery.get("jobs", [])
    need(isinstance(rows, list) and len(rows) == 2 and all(isinstance(row, dict) for row in rows)
         and {row.get("job") for row in rows} == set(PARTITIONS[partition]), "Recovery must bind exactly its two assigned jobs")
    root = Path(recovery["recovery_root"])
    need(root.is_absolute() and str(root) != original["campaign_root"], "Recovery needs a fresh absolute output root")
    if version == 2:
        binding = recovery.get("parent_recovery_manifest", {})
        need(partition == "current20" and original_sha == BASE_MANIFEST_SHA256
             and binding.get("sha256") == FIRST_CURRENT_RECOVERY_SHA256
             and Path(binding.get("path", "")) != root / "manifest.json",
             "Chained recovery must refer to the exact first current20 recovery")
    else:
        need("parent_recovery_manifest" not in recovery, "First recovery cannot declare an earlier recovery")
    return {row["job"]: row for row in rows}


def recovery_context(recovery, recovery_sha, entry, original_entry, prepared, actual, metadata, *, parent=None):
    root, lineage = Path(recovery["recovery_root"]), entry["lineage"]
    job = entry["job"]
    need(all(entry.get(key) == original_entry.get(key) for key in
             ("job", "gpu", "dataset", "method", "comparison_variant", "environment")), "Recovery job identity changed")
    resume = str(root / "resume_sources" / job / "checkpoint.pt")
    output, cfg_path = str(root / "runs" / job), str(root / "configs" / (job + ".json"))
    need(entry["output_dir"] == output and entry["config"]["path"] == cfg_path
         and entry["command"] == [recovery["python"], str(Path(recovery["repo"]) / "main.py"), "--config", cfg_path],
         "Recovery command/output/config paths differ")
    need(same_json_value(actual, {**prepared, "resume": resume, "output_dir": output}),
         "Recovery may change only resume and output_dir")
    expected_source = recovery.get("parent_recovery_manifest", recovery["source_manifest"])
    need(lineage["original_manifest"] == expected_source
         and lineage["original_prepared_config"] == original_entry["config"]
         and lineage["original_output_dir"] == original_entry["output_dir"], "Recovery original job/config binding differs")
    for key, filename in (("config", "config.json"), ("progress", "progress.json"), ("data_manifest", "data_manifest.json")):
        need(lineage["original_" + ("saved_config" if key == "config" else key)]["path"]
             == str(Path(original_entry["output_dir"]) / filename), "Recovery original metadata path differs")
    need(same_json_value({k: v for k, v in metadata["config"].items() if k != "config"}, normalized_prepared(prepared)),
         "Recovery original executed scientific configuration differs")
    source, snapshot = lineage["source_checkpoint"], lineage["resume_checkpoint"]
    need(source["path"] == str(Path(original_entry["output_dir"]) / "checkpoint.pt") and snapshot["path"] == resume
         and isinstance(source.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
         and source["sha256"] == snapshot["sha256"] and type(source.get("bytes")) is int
         and source["bytes"] == snapshot["bytes"] > 0, "Recovery retained checkpoint snapshot identity differs")
    count, prefix = lineage["saved_stage_count"], lineage["prefix_stages"]
    need(type(count) is int and 0 < count < 10 and isinstance(prefix, list) and len(prefix) == count
         and [row.get("task") for row in prefix] == list(range(count))
         and canonical_sha(prefix) == lineage["prefix_stages_sha256"]
         and same_json_value(metadata["progress"].get("stages", [])[:count], prefix),
         "Recovery checkpoint prefix metadata differs")
    fingerprint = config_fingerprint(normalized_prepared(prepared))
    need(lineage["config_fingerprint"] == fingerprint == config_fingerprint(normalized_prepared(actual))
         and lineage["random_stream_policy"] == RNG_POLICY
         and lineage["data_fingerprint"] == metadata["data_manifest"].get("fingerprint") == DATA[entry["dataset"]]["fingerprint"],
         "Recovery configuration/data/RNG fingerprint differs")
    if recovery["schema_version"] == 2:
        need(parent is not None and parent["manifest_sha256"] == expected_source["sha256"]
             and expected_source["path"] == str(Path(parent["manifest"]["recovery_root"]) / "manifest.json")
             and parent["manifest"]["schema_version"] == 1
             and parent["entry"] == original_entry
             and parent["saved_stage_count"] < count
             and same_json_value(prefix[:parent["saved_stage_count"]], parent["prefix_stages"])
             and same_json_value(metadata["data_manifest"], parent["original_metadata"]["data_manifest"]),
             "Latest source does not extend the exact first recovery prefix/data")
    else:
        need(parent is None, "First recovery cannot have an extra parent context")
    return {"manifest_sha256": recovery_sha, "manifest": recovery, "entry": entry, "lineage": lineage, "parent": parent,
            "actual_config": actual, "original_metadata": metadata, "saved_stage_count": count,
            "prefix_stages": prefix, "prefix_stages_sha256": lineage["prefix_stages_sha256"]}


def interrupted_work_summary(context, events=None):
    """Only disjoint completed-epoch regions are a known discarded-work bound."""
    lineage, count = context["lineage"], context["saved_stage_count"]
    rows = []
    for event in events or []:
        need(isinstance(event, dict) and type(event.get("task")) is int and count <= event["task"] < 10
             and type(event.get("epoch")) is int and 1 <= event["epoch"] <= 20,
             "Invalid interrupted-work epoch observation")
        scalar(event.get("wall_seconds"), "Interrupted completed-epoch seconds", positive=True)
        rows.append({key: event[key] for key in ("task", "epoch", "wall_seconds")})
    need(len({(row["task"], row["epoch"]) for row in rows}) == len(rows), "Interrupted-work epoch observations repeat")
    progress_tail = context["original_metadata"]["progress"].get("stages", [])[count:]
    progress_seconds = sum(scalar(row["wall_seconds"], "Interrupted completed-stage seconds", positive=True) for row in progress_tail)
    seconds = max(sum(row["wall_seconds"] for row in rows), progress_seconds)
    return {"source_log_sha256": lineage.get("original_log", {}).get("sha256"), "completed_epoch_regions": rows,
            "completed_extra_stage_seconds": progress_seconds, "known_seconds_lower_bound": seconds,
            "total_interrupted_work_seconds": None,
            "scope": "Lower bound from retained completed epoch/stage regions after the saved checkpoint; overlapping bounds use max, not sum. Unrecorded/incomplete work is unknown."}


def read_interrupted_work(context):
    log = context["lineage"].get("original_log")
    events = []
    if log:
        need(sha(log["path"]) == log["sha256"], "Recovery retained interrupted-work log changed")
        for line in Path(log["path"]).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("epoch_progress") and row.get("task", -1) >= context["saved_stage_count"]:
                events.append(row)
    return interrupted_work_summary(context, events)


def bind_interrupted_work(context, latest):
    context["interrupted_attempt"] = latest
    parent = context.get("parent")
    if parent is None:
        context["interrupted_work"] = latest
        return
    attempts = [parent["interrupted_attempt"], latest]
    context["interrupted_work"] = {
        "attempts": attempts,
        "known_seconds_lower_bound": sum(row["known_seconds_lower_bound"] for row in attempts),
        "total_interrupted_work_seconds": None,
        "scope": "Distinct interrupted processes in chronological order. Each includes only measured work discarded after its last retained checkpoint."}


def recovery_metadata_exports(entry):
    lineage = entry["lineage"]
    return {"prepared_config": metadata_export(entry["config"]["path"]),
            **{key: metadata_export(lineage[name]["path"]) for key, name in (
                ("original_config", "original_saved_config"), ("original_progress", "original_progress"),
                ("original_data_manifest", "original_data_manifest"))}}


def context_from_exports(manifest, digest, entry, original_entry, prepared, exports, *, parent=None):
    lineage = entry["lineage"]
    actual = load_metadata_export(exports, "prepared_config", entry["config"]["sha256"])
    metadata = {key: load_metadata_export(exports, export, lineage[binding]["sha256"]) for key, export, binding in (
        ("config", "original_config", "original_saved_config"), ("progress", "original_progress", "original_progress"),
        ("data_manifest", "original_data_manifest", "original_data_manifest"))}
    return recovery_context(manifest, digest, entry, original_entry, prepared, actual, metadata, parent=parent)


def verify_exported_cost(context, exports):
    cost = exports.get("interrupted_work")
    need(isinstance(cost, dict), "Recovery interrupted-work observations are missing")
    derived = interrupted_work_summary(context, cost.get("completed_epoch_regions"))
    need(same_json_value(cost, derived), "Recovery interrupted-work lower bound differs from its retained observations")
    bind_interrupted_work(context, derived)


def validate_recovery_seal(context, receipt, seal, metadata_shas):
    manifest, entry, lineage = context["manifest"], context["entry"], context["lineage"]
    need(receipt.get("job") == entry["job"] and receipt.get("manifest_sha256") == context["manifest_sha256"]
         and receipt.get("config_sha256") == entry["config"]["sha256"] and receipt.get("command") == entry["command"]
         and receipt.get("gpu") == entry["gpu"] and receipt.get("cwd") == manifest["repo"]
         and same_json_value(receipt.get("environment"), {**manifest["environment"], **entry["environment"]})
         and receipt.get("source_commit") == manifest["source_commit"]
         and receipt.get("source_inventory_sha256") == hashlib.sha256(json.dumps(manifest["source_files_sha256"], sort_keys=True).encode()).hexdigest(),
         "Recovery process receipt is not bound to its source/config/lineage")
    integer(receipt.get("pid"), "Recovery process PID", minimum=1)
    integer(receipt.get("proc_start_ticks"), "Recovery process start ticks", minimum=1)
    need(seal.get("job") == entry["job"] and seal.get("recovery_manifest_sha256") == context["manifest_sha256"]
         and seal.get("lineage_sha256") == canonical_sha(lineage)
         and seal.get("prefix_stages_sha256") == context["prefix_stages_sha256"]
         and seal.get("saved_stage_count") == context["saved_stage_count"]
         and seal.get("config_sha256") == metadata_shas["config"] and seal.get("result_sha256") == metadata_shas["results"]
         and seal.get("data_manifest_sha256") == metadata_shas["data_manifest"]
         and seal.get("original_process_state") in {"absent", "original_pid_reused", "Z", "X", "x"}
         and seal.get("observed_exit_code") in (None, 0), "Recovery completion seal does not prove its exact terminal lineage")


def recovery_expected_proof(context):
    lineage = context["lineage"]
    return {"original_checkpoint_sha256": lineage["source_checkpoint"]["sha256"],
            "resume_checkpoint_sha256": lineage["resume_checkpoint"]["sha256"],
            "saved_stage_count": lineage["saved_stage_count"], "prefix_stages_sha256": lineage["prefix_stages_sha256"],
            "config_fingerprint": lineage["config_fingerprint"], "data_fingerprint": lineage["data_fingerprint"],
            "random_stream_policy": lineage["random_stream_policy"], "prefix_verified": True}


def validate_recovery_prefix_proof(proof, context):
    need(isinstance(proof, dict) and proof.get("recovery_manifest_sha256") == context["manifest_sha256"]
         and same_json_value(proof.get("recovery_lineage"), recovery_expected_proof(context)),
         "Native recovery evidence is not bound to the exact parent checkpoint lineage")
    validate_native_source_prefix(proof.get("recovery_prefix_verification", {}), context)
    parent = context.get("parent")
    if parent is not None:
        need(proof.get("parent_recovery_manifest_sha256") == parent["manifest_sha256"],
             "Original native proof lacks the first recovery manifest binding")
        validate_native_source_prefix(proof.get("parent_recovery_prefix_verification", {}), parent)


def validate_native_source_prefix(report, context):
    lineage = context["lineage"]
    need(report.get("recovery_prefix_verified") is True
         and report.get("saved_stage_count") == context["saved_stage_count"]
         and report.get("source_checkpoint_stages_sha256") == report.get("restored_final_prefix_sha256") == context["prefix_stages_sha256"]
         and report.get("original_checkpoint_sha256") == lineage["source_checkpoint"]["sha256"]
         and report.get("resume_checkpoint_sha256") == lineage["resume_checkpoint"]["sha256"]
         and report.get("recovery_prefix_ignored_paths") == [], "Native recovery source/final checkpoint prefix proof differs")
    integer(report.get("recovery_prefix_compared_nodes"), "Recovery exact prefix compared nodes", minimum=1)
    for key in ("source_checkpoint_prefix_semantic_sha256", "source_checkpoint_rng_semantic_sha256"):
        need(isinstance(report.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", report[key]),
             "Native recovery source prefix/RNG verification is missing")


def load_metadata_export(exports, name, expected_sha):
    need(isinstance(exports, dict), "Missing exact metadata exports")
    record = exports.get(name, {})
    need(isinstance(record, dict), "Invalid metadata export: " + name)
    text = record.get("text")
    need(isinstance(text, str), "Missing exact metadata export: " + name)
    try:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        value = json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, UnicodeError) as error:
        raise AuditError("Invalid metadata export: " + name) from error
    need(digest == record.get("sha256") == expected_sha,
         "Metadata export SHA differs: " + name)
    need(isinstance(value, dict), "Metadata export must contain a JSON object: " + name)
    return value


def validate_overlay_structure(manifest, base):
    """Allow only the user-authorized image-epoch overlay on the pinned base."""
    need(manifest.get("base_manifest_sha256") == BASE_MANIFEST_SHA256
         and type(manifest.get("user_requested_image_epochs")) is int
         and manifest["user_requested_image_epochs"] == 20,
         "TUNA20 overlay is not bound to the authoritative user budget and base")
    embedded = load_metadata_export(
        {"base": {"sha256": manifest["base_manifest_sha256"], "text": manifest.get("base_manifest_text")}},
        "base", BASE_MANIFEST_SHA256)
    need(same_json_value(embedded, base) and not any(key in base for key in TUNA20_OVERLAY_KEYS),
         "TUNA20 overlay changed or recursively replaced its original base")
    need(manifest.get("base_manifest_path") == str(Path(base["campaign_root"]) / "manifest.json"),
         "TUNA20 overlay original manifest path differs")
    mutable = TUNA20_OVERLAY_KEYS | {"campaign_root", "jobs"}
    need(same_json_value({k: v for k, v in manifest.items() if k not in mutable},
                         {k: v for k, v in base.items() if k not in {"campaign_root", "jobs"}}),
         "TUNA20 overlay changed common source/data/environment/weight identity")
    root = Path(manifest["campaign_root"])
    need(root.is_absolute() and str(root) != base["campaign_root"],
         "TUNA20 overlay must use a fresh absolute campaign root")
    original, rows = base["jobs"], manifest.get("jobs")
    need(isinstance(rows, list) and len(rows) == len(original) == 4
         and all(isinstance(row, dict) for row in rows)
         and [row.get("job") for row in rows] == [row["job"] for row in original],
         "TUNA20 overlay job identities differ from the four original jobs")
    for old, row in zip(original, rows):
        job = old["job"]
        config_path = root / "formal/configs" / (job + ".json")
        need(isinstance(row.get("config"), dict), "TUNA20 overlay config binding is missing")
        expected = {**old, "output_dir": str(root / "formal/runs" / job),
                    "config": {"path": str(config_path), "sha256": row.get("config", {}).get("sha256")},
                    "command": [base["python"], str(Path(base["repo"]) / "main.py"), "--config", str(config_path)]}
        need(same_json_value(row, expected) and isinstance(row["config"]["sha256"], str)
             and re.fullmatch(r"[0-9a-f]{64}", row["config"]["sha256"]),
             "TUNA20 overlay changed a nonadministrative job field: " + job)


def validate_overlay_configs(manifest, base, configs, base_configs):
    expected_jobs = set(PARTITIONS["all"])
    need(set(configs) == set(base_configs) == expected_jobs, "TUNA20 overlay needs all four exact config exports")
    for row in manifest["jobs"]:
        job = row["job"]
        expected = {**base_configs[job], "output_dir": row["output_dir"]}
        if row["method"] == "tuna":
            need(base_configs[job].get("tuned_epoch") == (15 if row["dataset"] == "cifar224" else 10)
                 and "training_budget_variant" not in base_configs[job], "TUNA20 original native budget differs")
            expected.update(tuned_epoch=20, prefix="followup_tuna_epochs20", training_budget_variant="epochs20")
        need(same_json_value(configs[job], expected),
             "TUNA20 overlay changed more than the authorized config delta: " + job)


def overlay_from_summary(summary, base, base_digest):
    proof = summary.get("training_budget_overlay", {})
    need(isinstance(proof, dict) and proof.get("user_requested_image_epochs") == 20
         and proof.get("base_manifest_sha256") == base_digest == BASE_MANIFEST_SHA256
         and proof.get("exact_config_delta_verified") is True,
         "Final TUNA report requires the authoritative user-requested epochs20 overlay")
    manifest = load_metadata_export(
        {"manifest": proof.get("manifest_export")}, "manifest", summary.get("manifest_sha256"))
    validate_overlay_structure(manifest, base)
    base_configs = {row["job"]: load_metadata_export(
        proof.get("base_config_exports"), row["job"], row["config"]["sha256"]) for row in base["jobs"]}
    configs = {row["job"]: load_metadata_export(
        proof.get("config_exports"), row["job"], row["config"]["sha256"]) for row in manifest["jobs"]}
    validate_overlay_configs(manifest, base, configs, base_configs)
    return manifest, proof


def _validate_base_manifest(path):
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


def validate_prepared_manifest(path):
    manifest = read(path)
    if not any(key in manifest for key in TUNA20_OVERLAY_KEYS):
        return _validate_base_manifest(path)
    base_path = Path(manifest.get("base_manifest_path", ""))
    need(base_path.is_absolute() and base_path.is_file() and sha(base_path) == BASE_MANIFEST_SHA256,
         "TUNA20 authoritative original manifest is missing or changed")
    base, base_configs = _validate_base_manifest(base_path)
    validate_overlay_structure(manifest, base)
    need(Path(path).resolve() == (Path(manifest["campaign_root"]) / "manifest.json").resolve(),
         "TUNA20 manifest is outside its declared new campaign")
    configs = {}
    for row in manifest["jobs"]:
        config_path = Path(row["config"]["path"])
        need(config_path.is_file() and sha(config_path) == row["config"]["sha256"],
             "TUNA20 prepared config SHA changed: " + row["job"])
        configs[row["job"]] = read(config_path)
    validate_overlay_configs(manifest, base, configs, base_configs)
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


def validate_records(prepared, saved, result, data, sources, *, recovery=None):
    """Validate metadata/metrics without asserting checkpoint or process state."""
    expected = normalized_prepared(prepared)
    if recovery is not None:
        need(same_json_value(recovery["actual_config"], {**prepared,
             "resume": recovery["lineage"]["resume_checkpoint"]["path"],
             "output_dir": recovery["entry"]["output_dir"]}), "Recovery scientific configuration changed")
        expected = normalized_prepared(recovery["actual_config"])
    actual = {k: v for k, v in saved.items() if k != "config"}
    need(actual == expected, "Executed configuration differs from the prebound recipe")
    dataset, method = saved.get("dataset"), saved.get("model_name")
    need(dataset in DATA and method in {"current_adapted", "tuna"}, "Wrong follow-up method/dataset")
    need(saved.get("seed") == 1993 and saved.get("phase") == "final" and saved.get("pretrained") is True,
         "Wrong final phase, pretrained flag or seed")
    need(saved.get("cl27_protocol") == "pilot_figure2" and saved.get("shuffle") is True
         and saved.get("evaluation_head") == "cnn" and saved.get("validation_only") is False,
         "Wrong selected protocol")
    forbidden = ("max_tasks", "validation_max_per_class", "joint_max_classes")
    need(not saved.get("smoke") and all(saved.get(k) is None for k in forbidden)
         and (recovery is not None or saved.get("resume") is None), "Truncated/resumed/development run")
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
        budget = saved.get("training_budget_variant")
        need(budget in (None, "epochs20"), "Unknown TUNA image-training budget variant")
        image_epochs = 20 if budget == "epochs20" else (15 if dataset == "cifar224" else 10)
        need(saved.get("tuned_epoch") == image_epochs
             and saved.get("crct_epochs") == 30, "Native TUNA training lengths changed")
    metrics, stage_seconds = validate_metrics(result, dataset)
    prefix_count = 0
    if recovery is not None:
        prefix_count = recovery["saved_stage_count"]
        need(same_json_value(result["stages"][:prefix_count], recovery["prefix_stages"])
             and canonical_sha(result["stages"][:prefix_count]) == recovery["prefix_stages_sha256"],
             "Final resumed results changed the exact source-checkpoint prefix")
        need(same_json_value(data, recovery["original_metadata"]["data_manifest"]),
             "Resumed data manifest differs from its source trajectory")
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
    need(wall + 1e-6 >= sum(stage_seconds[prefix_count:]), "Run wall time is shorter than its measured stage regions")
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
    if recovery is not None:
        resources.update(
            execution_scope="Resumed segment only, measured by the unchanged runner after checkpoint restoration.",
            gpu_peak_scope="Resumed process only; earlier-process peaks are not reconstructed.",
            full_run_wall_seconds=None,
            completed_stage_wall_seconds_total=sum(stage_seconds),
            original_checkpoint_stage_wall_seconds_total=sum(stage_seconds[:prefix_count]),
            resumed_stage_wall_seconds_total=sum(stage_seconds[prefix_count:]),
            interrupted_work=recovery.get("interrupted_work") or interrupted_work_summary(recovery),
            measurement_scope="Stage totals retain original prefix timers. Raw wall/peak cover only the resumed segment; no complete-lineage runtime or matched-compute claim.")
        if recovery.get("parent") is not None:
            boundaries = (0, recovery["parent"]["saved_stage_count"], prefix_count, len(stage_seconds))
            resources["completed_stage_segments"] = [
                {"first_stage": start + 1, "last_stage": end, "wall_seconds": sum(stage_seconds[start:end])}
                for start, end in zip(boundaries, boundaries[1:])]
    return metrics, resources


def validate_checkpoint_evidence(path, entry, run, *, sources, manifest_sha256, recovery=None):
    evidence = read(path)
    checkpoint = run / "checkpoint.pt"
    need(checkpoint.is_file(), "Final server-local checkpoint is missing")
    need(evidence.get("job") == entry["job"], "Checkpoint evidence job differs")
    need(evidence.get("validation_only") is False
         and evidence.get("formal_checkpoint_evidence") is True,
         "Validation-only evidence cannot finalize a formal checkpoint")
    need(evidence.get("source_manifest_sha256") == manifest_sha256,
         "Checkpoint evidence prepared-manifest binding differs")
    size, digest = checkpoint.stat().st_size, sha(checkpoint)
    need(type(evidence.get("checkpoint_bytes")) is int and evidence["checkpoint_bytes"] == size
         and evidence.get("checkpoint_sha256") == digest and evidence.get("backup_sha256") == digest,
         "Original/backup checkpoint SHA or byte counts differ")
    need(evidence.get("native_restore_verified") is True, "Native restore/payload equality is not verified")
    need(evidence.get("source_process_terminal_verified") is True, "Source training process terminal state is not verified")
    required = {"config": sha(run / "config.json"), "results": sha(run / "results.json"),
                "data_manifest": sha(run / "data_manifest.json")}
    need(evidence.get("source_metadata_sha256") == required, "Checkpoint audit metadata binding differs")

    # A previously successful copy is not proof that its backup still exists.
    # Reopen the actual server-local model and report at final audit time.
    backup_name, report_name = (evidence.get(name) for name in
                                ("backup_checkpoint_path", "restore_report_path"))
    need(isinstance(backup_name, str) and isinstance(report_name, str),
         "Missing server-local backup or restore-report path")
    backup, report_path = Path(backup_name), Path(report_name)
    need(backup.is_absolute() and backup.is_file()
         and not backup.samefile(checkpoint), "Separate server-local backup is missing")
    need(backup.stat().st_size == size and sha(backup) == digest,
         "Actual server-local backup bytes differ")
    need(report_path.is_absolute() and report_path.is_file(),
         "Native restore report file is missing")
    report_digest = sha(report_path)
    need(evidence.get("restore_report_sha256") == report_digest,
         "Native restore report file SHA differs")
    report = read(report_path)
    need(report == evidence.get("restore_report") and sha(report_path) == report_digest,
         "Embedded and server-local native restore reports differ")
    need(report.get("native_cpu_restore_executed") is True
         and report.get("exact_typed_bitwise_comparison") is True
         and report.get("ignored_paths") == [],
         "Native CPU exact comparison is incomplete or ignored learner fields")
    integer(report.get("compared_nodes"), "Compared native learner nodes", minimum=1)
    semantic = report.get("reference_semantic_sha256")
    need(isinstance(semantic, str) and len(semantic) == 64
         and all(character in "0123456789abcdef" for character in semantic)
         and report.get("restored_semantic_sha256") == semantic,
         "Native restored learner semantic digest differs")
    config = read(run / "config.json")
    spec = DATA[config["dataset"]]
    need(report.get("actual_training_rows") == sum(spec["train_tasks"])
         and report.get("actual_seen_classes") == spec["classes"]
         and report.get("source_data_manifest_rebuilt") is True
         and report.get("source_training_arrivals_verified") is True,
         "Native restored full-data endpoint is not verified")
    need(report.get("global_rng_payloads_validated_on_cpu") is True
         and report.get("cuda_rng_applied") is False
         and type(report.get("extra_training_steps")) is int
         and report["extra_training_steps"] == 0
         and report.get("accuracy_evaluation_performed") is False
         and report.get("pretrained_npz_sha256") == NPZ_SHA,
         "Native restore execution or pretrained identity differs")
    imported = report.get("runtime_source_files_sha256")
    wrapper = "models/" + ("tuna_protocol" if config["model_name"] == "tuna" else "current_adapted") + ".py"
    need(isinstance(imported, dict) and wrapper in imported
         and all(name in sources and sources[name] == value
                 for name, value in imported.items()),
         "Native restore imported source differs from the prepared snapshot")
    if config["model_name"] == "tuna":
        need(report.get("persisted_mode_map_compared") is True,
             "Native TUNA persisted module modes are not verified")
    proof = {"checkpoint_sha256": digest, "checkpoint_bytes": size,
            "backup_sha256": evidence["backup_sha256"], "native_restore_verified": True,
            "source_process_terminal_verified": True, "evidence_sha256": sha(path),
            "evidence_path": str(path), "actual_backup_rehashed": True,
            "backup_checkpoint_path": str(backup), "restore_report_path": str(report_path),
            "restore_report_sha256": report_digest, "compared_nodes": report["compared_nodes"],
            "reference_semantic_sha256": semantic, "restored_semantic_sha256": semantic,
            "formal_checkpoint_evidence": True, "source_manifest_sha256": manifest_sha256}
    if recovery is not None:
        proof.update(recovery_manifest_sha256=evidence.get("recovery_manifest_sha256"),
                     source_receipt_sha256=evidence.get("source_receipt_sha256"),
                     recovery_lineage=evidence.get("recovery_lineage"),
                     recovery_prefix_verification={key: report.get(key) for key in (
                         "recovery_prefix_verified", "saved_stage_count", "source_checkpoint_stages_sha256",
                         "restored_final_prefix_sha256", "original_checkpoint_sha256", "resume_checkpoint_sha256",
                         "recovery_prefix_compared_nodes", "recovery_prefix_ignored_paths",
                         "source_checkpoint_prefix_semantic_sha256", "source_checkpoint_rng_semantic_sha256")})
        if recovery.get("parent") is not None:
            proof["parent_recovery_prefix_verification"] = report.get("parent_recovery_prefix_verification", {})
            proof["parent_recovery_manifest_sha256"] = report.get("parent_recovery_manifest_sha256")
        validate_recovery_prefix_proof(proof, recovery)
        need(isinstance(proof["source_receipt_sha256"], str)
             and re.fullmatch(r"[0-9a-f]{64}", proof["source_receipt_sha256"]),
             "Native recovery evidence is missing its actual source receipt SHA")
    else:
        need(evidence.get("recovery_manifest_sha256") is None,
             "Recovered checkpoint needs its explicitly validated recovery manifest")
    return proof


def audit_campaign(manifest_path, evidence_root=None, *, allow_incomplete=False,
                   partition="all", host_alias=None, host_key_fingerprint=None, recovery_manifest_path=None):
    need(partition in PARTITIONS, "Unknown local audit partition")
    host = (host_identity(host_alias, host_key_fingerprint)
            if partition != "all" or host_alias is not None or host_key_fingerprint is not None else None)
    if host is not None:
        host["runtime_identity"] = validate_runtime_identity(runtime_identity())
    selected = PARTITIONS[partition]
    manifest_path = Path(manifest_path).resolve()
    manifest, configs = validate_prepared_manifest(manifest_path)
    manifest_digest = sha(manifest_path)
    overlay = any(key in manifest for key in TUNA20_OVERLAY_KEYS)
    need(not overlay or partition == "tuna", "TUNA20 overlay is authorized only for the tuna partition")
    if partition == "tuna":
        need(overlay, "Final tuna partition requires the user-authorized epochs20 overlay")
    if partition == "current20":
        need(not overlay and manifest_digest == BASE_MANIFEST_SHA256,
             "Current20 partition must retain the authoritative original manifest")
    recovery_manifest, recovery_rows, recovery_configs = None, {}, {}
    parent_manifest, parent_rows, parent_export = None, {}, None
    if recovery_manifest_path is not None:
        recovery_manifest_path = Path(recovery_manifest_path).resolve()
        checked = read_guarded_recovery(recovery_manifest_path)
        need(same_json_value(checked["source_manifest"], manifest), "Recovery selected original manifest changed")
        recovery_manifest, recovery_configs = checked["manifest"], checked["configs"]
        recovery_rows = validate_recovery_header(recovery_manifest, manifest, manifest_digest, partition)
        if recovery_manifest["schema_version"] == 2:
            binding = recovery_manifest["parent_recovery_manifest"]
            parent_export = metadata_export(binding["path"])
            parent_manifest = load_metadata_export({"parent": parent_export}, "parent", binding["sha256"])
            parent_rows = validate_recovery_header(parent_manifest, manifest, manifest_digest, partition)
    evidence_root = Path(evidence_root or ((recovery_manifest_path.parent if recovery_manifest is not None
                                          else manifest_path.parent) / "checkpoint_evidence"))
    rows = []
    for entry in manifest["jobs"]:
        job, run = entry["job"], Path(entry["output_dir"])
        if job not in selected:
            continue
        recovery = None
        if recovery_manifest is not None:
            recovered = recovery_rows[job]
            row_exports = recovery_metadata_exports(recovered)
            source_entry, source_config, parent = entry, configs[job], None
            if parent_manifest is not None:
                source_entry = parent_rows[job]
                parent_exports = recovery_metadata_exports(source_entry)
                parent = context_from_exports(parent_manifest, parent_export["sha256"], source_entry,
                                              entry, configs[job], parent_exports)
                bind_interrupted_work(parent, read_interrupted_work(parent))
                parent_exports["interrupted_work"] = parent["interrupted_attempt"]
                row_exports["parent"] = parent_exports
                source_config = parent["actual_config"]
            recovery = context_from_exports(recovery_manifest, sha(recovery_manifest_path), recovered,
                                            source_entry, source_config, row_exports, parent=parent)
            need(same_json_value(recovery["actual_config"], recovery_configs[job]),
                 "Recovery may change only resume and output_dir")
            bind_interrupted_work(recovery, read_interrupted_work(recovery))
            row_exports["interrupted_work"] = recovery["interrupted_attempt"]
            run = Path(recovered["output_dir"])
        row = {"job": job, "dataset": entry["dataset"], "method": entry["method"],
               "comparison_variant": entry["comparison_variant"], "seed": 1993,
               "status": "pending", "metrics": dict.fromkeys(METRICS)}
        if configs[job].get("training_budget_variant") == "epochs20":
            row["training_budget_variant"] = "epochs20"
            row["method_setting"] = "Native TUNA mechanism with user-requested 20 image epochs; not native image-budget reproduction."
        result_path = run / "results.json"
        if result_path.exists():
            saved, result, data = (read(run / name) for name in ("config.json", "results.json", "data_manifest.json"))
            metrics, resources = validate_records(configs[job], saved, result, data,
                                                  manifest["source_files_sha256"], recovery=recovery)
            row.update(status="metrics_verified_checkpoint_evidence_pending", metrics=metrics,
                       resources=resources,
                       source={"config_sha256": sha(run / "config.json"), "results_sha256": sha(result_path),
                               "data_manifest_sha256": sha(run / "data_manifest.json"),
                               "data_fingerprint": data["fingerprint"],
                               "prepared_config_sha256": entry["config"]["sha256"],
                               "config_fingerprint": result["config_fingerprint"]})
            if partition != "all":
                row["metadata_exports"] = {
                    name: metadata_export(file) for name, file in (
                        ("prepared_config", entry["config"]["path"]),
                        ("config", run / "config.json"), ("results", result_path),
                        ("data_manifest", run / "data_manifest.json"))}
            evidence_path = evidence_root / (job + ".json")
            if evidence_path.exists():
                row["checkpoint_audit"] = validate_checkpoint_evidence(
                    evidence_path, entry, run, sources=manifest["source_files_sha256"],
                    manifest_sha256=manifest_digest, recovery=recovery)
                row["status"] = "verified_complete"
            if recovery is not None:
                recovery_root = Path(recovery_manifest["recovery_root"])
                row["recovery_exports"] = row_exports
                if row["status"] == "verified_complete":
                    receipt_path = recovery_root / "receipts" / (job + ".json")
                    seal_path = recovery_root / "queue" / (job + ".complete.json")
                    receipt, seal = read(receipt_path), read(seal_path)
                    need(sha(receipt_path) == row["checkpoint_audit"]["source_receipt_sha256"],
                         "Recovery process receipt differs from the native backup evidence")
                    validate_recovery_seal(recovery, receipt, seal, {key: row["source"][key + "_sha256"]
                                                                  for key in ("config", "results", "data_manifest")})
                    row["recovery_exports"].update(receipt=metadata_export(receipt_path), completion_seal=metadata_export(seal_path))
        elif (run / "progress.json").exists():
            progress = read(run / "progress.json")
            row.update(status="incomplete", completed_stage_records=len(progress.get("stages", [])))
        rows.append(row)
    complete = sum(row["status"] == "verified_complete" for row in rows)
    expected = len(selected)
    need(allow_incomplete or complete == expected,
         "Follow-up audit incomplete: " + str(complete) + "/" + str(expected)
         + " verified endpoints; use --allow-incomplete for progress only")
    summary = {
        "schema_version": 1, "status": "verified_complete" if complete == expected else "incomplete",
        "partition": partition, "partition_jobs": list(selected), "host": host,
        "external_jobs": [job for job in PARTITIONS["all"] if job not in selected],
        "external_job_status": "assigned_to_other_partition",
        "global_expected_results": 4, "expected_results": expected, "verified_results": complete,
        "metrics_verified_results": sum(row["metrics"]["final_accuracy"] is not None for row in rows),
        "seed": 1993, "selection_policy": "The four prepared job identities only; never highest test score.",
        "manifest_sha256": manifest_digest, "source_commit": manifest.get("source_commit"),
        "source_files_sha256": manifest["source_files_sha256"], "upstream_commit": UPSTREAM,
        "metric_units": "Fractions; Markdown uses percent accuracy and percentage-point forgetting.",
        "audit_scope": "Prepared scientific recipe/source/weights, full data identity, unrounded ten-stage records, measured resources and SHA-bound server-local backup/native-restore evidence.",
        "auditor_sha256": sha(Path(__file__)), "rows": rows,
    }
    if overlay:
        base = read(manifest["base_manifest_path"])
        summary["training_budget_overlay"] = {
            "user_requested_image_epochs": 20, "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "manifest_export": metadata_export(manifest_path),
            "base_config_exports": {row["job"]: metadata_export(row["config"]["path"]) for row in base["jobs"]},
            "config_exports": {row["job"]: metadata_export(row["config"]["path"]) for row in manifest["jobs"]},
            "exact_config_delta_verified": True}
    if recovery_manifest is not None:
        summary["recovery_manifest_export"] = metadata_export(recovery_manifest_path)
        summary["recovery_manifest_sha256"] = sha(recovery_manifest_path)
    if parent_export is not None:
        summary["parent_recovery_manifest_export"] = parent_export
    return summary


def merge_partition_summaries(manifest_path, summary_paths, *, expected_summary_sha256):
    """Combine two host-local proofs using exact metadata, never remote models."""
    need(len(summary_paths) == 2, "Global merge requires two partition reports")
    need(isinstance(expected_summary_sha256, (list, tuple)) and len(expected_summary_sha256) == 2
         and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                 for value in expected_summary_sha256),
         "Global merge requires the two server-captured partition report SHAs")
    manifest_path = Path(manifest_path).resolve()
    manifest, manifest_digest = read(manifest_path), sha(manifest_path)
    need(manifest_digest == BASE_MANIFEST_SHA256
         and not any(key in manifest for key in TUNA20_OVERLAY_KEYS),
         "Global merge requires the authoritative original base manifest")
    entries = manifest.get("jobs", [])
    expected = {job: (gpu, dataset, method) for job, gpu, dataset, method in JOBS}
    need(manifest.get("version") == 1 and isinstance(entries, list) and len(entries) == 4
         and all(isinstance(entry, dict) for entry in entries)
         and {entry.get("job") for entry in entries} == set(expected),
         "Global manifest must bind the four exact job identities")
    source_commit, sources = manifest.get("source_commit"), manifest.get("source_files_sha256")
    need(isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit)
         and isinstance(sources, dict) and bool(sources)
         and manifest.get("pretrained_npz", {}).get("sha256") == NPZ_SHA,
         "Global manifest source or pretrained identity is incomplete")
    by_job = {entry["job"]: entry for entry in entries}
    for job, entry in by_job.items():
        need((entry.get("gpu"), entry.get("dataset"), entry.get("method")) == expected[job],
             "Global manifest job assignment differs: " + job)
    auditor_digest = sha(Path(__file__))
    combined, proofs, groups, aliases, runtimes, overlays, recoveries = {}, [], set(), set(), set(), [], []
    for path, expected_report_digest in zip(summary_paths, expected_summary_sha256):
        path = Path(path).resolve()
        report_digest = sha(path)
        need(report_digest == expected_report_digest,
             "Partition report differs from its server-captured SHA")
        summary = read(path)
        need(sha(path) == report_digest, "Partition report changed during merge")
        group = summary.get("partition")
        need(group in {"current20", "tuna"} and group not in groups,
             "Global merge requires one current20 and one tuna partition")
        groups.add(group)
        jobs = PARTITIONS[group]
        active_manifest, active_digest, budget_proof = manifest, manifest_digest, None
        if group == "tuna":
            active_manifest, budget_proof = overlay_from_summary(summary, manifest, manifest_digest)
            active_digest = summary["manifest_sha256"]
            overlays.append({"partition": "tuna", "manifest_sha256": active_digest,
                             "base_manifest_sha256": manifest_digest, "user_requested_image_epochs": 20,
                             "manifest_export": budget_proof["manifest_export"], "exact_config_delta_verified": True})
        else:
            need(not summary.get("training_budget_overlay"), "Current20 report must retain its original budget manifest")
        active_jobs = {entry["job"]: entry for entry in active_manifest["jobs"]}
        recovered_manifest, recovered_rows, recovered_digest = None, {}, None
        parent_manifest, parent_rows = None, {}
        if "recovery_manifest_export" in summary or "recovery_manifest_sha256" in summary:
            recovered_digest = summary.get("recovery_manifest_sha256")
            recovered_manifest = load_metadata_export(
                {"recovery": summary.get("recovery_manifest_export")}, "recovery", recovered_digest)
            recovered_rows = validate_recovery_header(recovered_manifest, active_manifest, active_digest, group)
            recoveries.append({"partition": group, "manifest_sha256": recovered_digest,
                               "source_manifest_sha256": active_digest,
                               "manifest_export": summary["recovery_manifest_export"]})
            if recovered_manifest["schema_version"] == 2:
                binding = recovered_manifest["parent_recovery_manifest"]
                parent_export = summary.get("parent_recovery_manifest_export")
                parent_manifest = load_metadata_export({"parent": parent_export}, "parent", binding["sha256"])
                parent_rows = validate_recovery_header(parent_manifest, active_manifest, active_digest, group)
                recoveries[-1]["parent_recovery_manifest_export"] = parent_export
        need(parent_manifest is not None or "parent_recovery_manifest_export" not in summary,
             "Extra parent export without a chained recovery")
        need(summary.get("status") == "verified_complete"
             and summary.get("expected_results") == summary.get("verified_results")
             == summary.get("metrics_verified_results") == 2
             and summary.get("global_expected_results") == 4,
             "Partition is not locally verified complete")
        need(summary.get("partition_jobs") == list(jobs)
             and summary.get("external_jobs") == [job for job in PARTITIONS["all"] if job not in jobs]
             and summary.get("external_job_status") == "assigned_to_other_partition",
             "Partition coverage or external-job declaration differs")
        host = summary.get("host", {})
        need(isinstance(host, dict), "Partition host identity is missing")
        runtime = validate_runtime_identity(host.get("runtime_identity"))
        host = {**host_identity(host.get("alias"), host.get("host_key_fingerprint")),
                "runtime_identity": runtime}
        need(host["alias"] not in aliases, "Partition host aliases must be distinct")
        aliases.add(host["alias"])
        need(runtime["fingerprint_sha256"] not in runtimes,
             "Partitions must have distinct physical runtime fingerprints")
        runtimes.add(runtime["fingerprint_sha256"])
        producer = summary.get("auditor_sha256")
        exact_historical_report = (group == "tuna" and report_digest == SEALED_TUNA_SUMMARY_SHA256
                                   and producer == SEALED_TUNA_AUDITOR_SHA256)
        need(summary.get("manifest_sha256") == active_digest
             and summary.get("source_commit") == source_commit
             and summary.get("source_files_sha256") == sources
             and summary.get("upstream_commit") == UPSTREAM
             and (producer == auditor_digest or exact_historical_report)
             and summary.get("seed") == 1993,
             "Partition manifest/source/auditor/scientific identity differs")
        rows = summary.get("rows", [])
        need(isinstance(rows, list) and len(rows) == 2
             and all(isinstance(row, dict) for row in rows)
             and {row.get("job") for row in rows} == set(jobs),
             "Partition rows overlap or omit assigned jobs")
        proof = {"partition": group, "host": host, "jobs": list(jobs),
                 "summary_path": str(path), "summary_sha256": report_digest,
                 "server_captured_summary_sha256": expected_report_digest,
                 "auditor_sha256": producer, "manifest_sha256": active_digest,
                 "base_manifest_sha256": manifest_digest}
        if budget_proof is not None:
            proof["training_budget_overlay"] = overlays[-1]
        if recovered_manifest is not None:
            proof["recovery_manifest_sha256"] = recovered_digest
        proofs.append(proof)
        for row in rows:
            job = row["job"]
            entry = active_jobs[job]
            need(job not in combined and row.get("status") == "verified_complete"
                 and (row.get("dataset"), row.get("method"), row.get("comparison_variant"), row.get("seed"))
                 == (entry["dataset"], entry["method"], entry["comparison_variant"], 1993),
                 "Partition row is not its unique verified assignment")
            source, exports = row.get("source", {}), row.get("metadata_exports")
            need(isinstance(source, dict), "Partition source metadata is missing")
            prepared = load_metadata_export(exports, "prepared_config", entry["config"]["sha256"])
            saved = load_metadata_export(exports, "config", source.get("config_sha256"))
            result = load_metadata_export(exports, "results", source.get("results_sha256"))
            data = load_metadata_export(exports, "data_manifest", source.get("data_manifest_sha256"))
            need(prepared.get("output_dir") == entry["output_dir"]
                 and prepared.get("device") == [entry["gpu"]]
                 and source.get("prepared_config_sha256") == entry["config"]["sha256"]
                 and source.get("data_fingerprint") == data.get("fingerprint")
                 and source.get("config_fingerprint") == result.get("config_fingerprint"),
                 "Partition metadata binding differs from the global manifest")
            recovery = None
            if recovered_manifest is not None:
                recovery_exports = row.get("recovery_exports", {})
                recovered = recovered_rows[job]
                source_entry, source_config, parent = entry, prepared, None
                if parent_manifest is not None:
                    source_entry = parent_rows[job]
                    parent_exports = recovery_exports.get("parent")
                    parent = context_from_exports(parent_manifest, recovered_manifest["parent_recovery_manifest"]["sha256"],
                                                  source_entry, entry, prepared, parent_exports)
                    verify_exported_cost(parent, parent_exports)
                    source_config = parent["actual_config"]
                recovery = context_from_exports(recovered_manifest, recovered_digest, recovered, source_entry,
                                                source_config, recovery_exports, parent=parent)
                verify_exported_cost(recovery, recovery_exports)
                receipt = load_metadata_export(recovery_exports, "receipt", row.get("checkpoint_audit", {}).get("source_receipt_sha256"))
                seal = load_metadata_export(recovery_exports, "completion_seal", recovery_exports.get("completion_seal", {}).get("sha256"))
                validate_recovery_seal(recovery, receipt, seal, {key: source[key + "_sha256"]
                                                               for key in ("config", "results", "data_manifest")})
            else:
                need(not row.get("recovery_exports"), "Recovered row needs its explicit recovery manifest proof")
            metrics, resources = validate_records(prepared, saved, result, data, sources, recovery=recovery)
            if group == "tuna":
                need(prepared.get("training_budget_variant") == row.get("training_budget_variant") == "epochs20"
                     and prepared.get("tuned_epoch") == 20, "Final TUNA row is not the requested epochs20 setting")
            for metric in METRICS:
                same(scalar(row.get("metrics", {}).get(metric), "Partition " + metric), metrics[metric], metric)
            need(row.get("resources") == resources, "Partition resource export differs from its raw records")
            checkpoint = row.get("checkpoint_audit", {})
            need(isinstance(checkpoint, dict), "Partition local checkpoint proof is missing")
            need(all(checkpoint.get(name) is True for name in
                     ("native_restore_verified", "source_process_terminal_verified",
                      "actual_backup_rehashed", "formal_checkpoint_evidence"))
                 and checkpoint.get("source_manifest_sha256") == active_digest,
                 "Partition lacks its strict local checkpoint proof")
            if recovery is not None:
                validate_recovery_prefix_proof(checkpoint, recovery)
            else:
                need(checkpoint.get("recovery_manifest_sha256") is None, "Recovered checkpoint lacks its explicit recovery manifest proof")
            for name in ("checkpoint_sha256", "backup_sha256", "evidence_sha256",
                         "restore_report_sha256", "reference_semantic_sha256", "restored_semantic_sha256"):
                need(isinstance(checkpoint.get(name), str)
                     and re.fullmatch(r"[0-9a-f]{64}", checkpoint[name]),
                     "Missing partition checkpoint proof digest: " + name)
            need(checkpoint["checkpoint_sha256"] == checkpoint["backup_sha256"]
                 and checkpoint["reference_semantic_sha256"] == checkpoint["restored_semantic_sha256"],
                 "Partition checkpoint or native semantic equality differs")
            integer(checkpoint.get("checkpoint_bytes"), "Partition checkpoint bytes", minimum=1)
            integer(checkpoint.get("compared_nodes"), "Partition compared native nodes", minimum=1)
            need(all(isinstance(checkpoint.get(name), str) and Path(checkpoint[name]).is_absolute()
                     for name in ("evidence_path", "backup_checkpoint_path", "restore_report_path")),
                 "Partition checkpoint proof paths are incomplete")
            combined[job] = {**row, "partition_proof": proof,
                             "checkpoint_verification_scope": "Strictly verified on the recorded partition host; not reopened by this metadata merge."}
    need(groups == {"current20", "tuna"} and set(combined) == set(expected),
         "Global merge must cover all four nonoverlapping jobs")
    return {
        "schema_version": 1, "status": "verified_complete", "partition": "merged",
        "partition_jobs": list(PARTITIONS["all"]), "external_jobs": [],
        "global_expected_results": 4, "expected_results": 4, "verified_results": 4,
        "metrics_verified_results": 4, "seed": 1993, "manifest_sha256": manifest_digest,
        "source_commit": source_commit, "source_files_sha256": sources, "upstream_commit": UPSTREAM,
        "selection_policy": "The four prepared job identities only; never highest test score.",
        "metric_units": "Fractions; Markdown uses percent accuracy and percentage-point forgetting.",
        "audit_scope": "Original-manifest current20 and user-authorized TUNA20 overlay, both strictly audited on their hosts; metrics/resources recomputed from exact metadata. Remote model files are not reopened or copied by this merge.",
        "training_budget_overlays": overlays,
        "recovery_manifests": recoveries,
        "common_identity": {"base_manifest_sha256": manifest_digest, "source_commit": source_commit,
                            "source_files_sha256": sources, "pretrained_npz_sha256": NPZ_SHA,
                            "data_fingerprints": {dataset: spec["fingerprint"] for dataset, spec in DATA.items()}},
        "remote_checkpoints_reopened_by_merge": False, "partition_reports": proofs,
        "auditor_sha256": auditor_digest, "rows": [combined[job] for job in PARTITIONS["all"]],
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
             f"Seed 1993; {summary['verified_results']}/{summary['expected_results']} fully verified follow-up results.",
             "Accuracy: %. Forgetting: percentage points. Pending checkpoint evidence does not constitute a completed audit.",
             "The fixed-library 20-epoch rows are additional budget variants; the original 5-epoch results remain separate.", ""]
    if summary.get("partition", "all") != "all":
        lines += ["Audit partition: `" + summary["partition"] + "`.", ""]
    if summary.get("external_jobs"):
        lines += ["Jobs assigned to the other host partition: " + ", ".join(summary["external_jobs"]) + ".", ""]
    if summary.get("partition") == "merged":
        lines += ["Checkpoint files were verified on their recorded hosts. This merge rechecks exported metadata and does not reopen remote model files.", ""]
    if any(row.get("training_budget_variant") == "epochs20" and row["method"] == "tuna"
           for row in summary["rows"]):
        lines += ["TUNA retains the native mechanism and 30-epoch classifier alignment, with the user-requested 20 image epochs per stage. This is not reproduction of the original native image-training budget.", ""]
    if summary.get("recovery_manifest_export") or summary.get("recovery_manifests"):
        lines += ["Checkpoint recovery preserves the exact earlier stage records. Raw execution wall time and GPU peak cover only the resumed process; full-lineage runtime/peak are unknown. Stage-time totals and any discarded-work lower bounds are reported separately.", ""]
    for dataset, spec in DATA.items():
        lines += ["## " + spec["title"], "", "| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for row in summary.get("combined_rows", summary["rows"]):
            if row["dataset"] != dataset:
                continue
            values = ["—" if row["metrics"].get(key) is None else f"{100 * row['metrics'][key]:.2f}" for key in METRICS]
            status = "prior campaign reference" if row.get("imported_previous_campaign") else row["status"]
            variant = ("epochs20; native mechanism" if row.get("training_budget_variant") == "epochs20"
                       and row["method"] == "tuna" else row["comparison_variant"])
            lines.append("| " + " | ".join([row["method"], variant, *values, status]) + " |")
        lines.append("")
    lines += ["F_previous excludes the final evaluation from each historical maximum; F_including_final includes it.",
              "Resource values are the measured runner/method records in the JSON; training costs are not inferred from epoch counts.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-evidence-root", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--partition", choices=tuple(PARTITIONS), default="all")
    parser.add_argument("--host-alias")
    parser.add_argument("--host-key-fingerprint")
    parser.add_argument("--merge-partition-summaries", type=Path, nargs=2, metavar=("CURRENT20", "TUNA"))
    parser.add_argument("--partition-summary-sha256", nargs=2, metavar=("CURRENT20_SHA", "TUNA_SHA"))
    parser.add_argument("--base-summary", type=Path, help="Optional completed original twelve-row summary to preserve alongside the four additions")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.merge_partition_summaries:
            need(args.partition == "all" and not args.allow_incomplete
                 and args.checkpoint_evidence_root is None and args.host_alias is None
                 and args.host_key_fingerprint is None and args.recovery_manifest is None,
                 "Merge requires complete partition reports; local-host options belong to their audits")
            need(args.partition_summary_sha256 is not None,
                 "Merge requires --partition-summary-sha256 captured on the two servers")
            summary = merge_partition_summaries(
                args.manifest, args.merge_partition_summaries,
                expected_summary_sha256=args.partition_summary_sha256)
        else:
            need(args.partition_summary_sha256 is None,
                 "Partition report SHAs are supplied only for a global merge")
            summary = audit_campaign(args.manifest, args.checkpoint_evidence_root,
                                     allow_incomplete=args.allow_incomplete, partition=args.partition,
                                     host_alias=args.host_alias, host_key_fingerprint=args.host_key_fingerprint,
                                     recovery_manifest_path=args.recovery_manifest)
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
