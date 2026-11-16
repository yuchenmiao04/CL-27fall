"""Synthetic CLI/continuation checks and protocol invariants, never full runs."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cl27 import data as data_module
from cl27.data import ProtocolDataManager, split_training
from cl27.protocol import smoke_config, validate_protocol
from cl27.replay import ReplayMemory
from cl27.runner import run, stage_record, summarize


METHODS = ("shared_basis", "history_transport", "current_adapted")
REPO = Path(__file__).resolve().parents[1]


def small_config(method, output):
    config = smoke_config(method, output)
    config.update(
        {
            "smoke_feature_dim": 6,
            "smoke_depth": 2,
            "image_size": 16,
            "module_layers": [0],
            "synthetic_per_class": 6,
            "epochs": 1,
            "outer_updates": 1,
            "batch_size": 4,
            "feature_batch_size": 4,
            "cpu_threads": 1,
        }
    )
    return config


def assert_same_state(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same_state(a, b)
    elif hasattr(first, "__dataclass_fields__"):
        assert_same_state(vars(first), vars(second))
    else:
        assert first == second


@pytest.mark.parametrize("method", METHODS)
def test_original_main_cli_dispatches_three_native_methods(method, tmp_path):
    output = tmp_path / method
    config = small_config(method, output)
    config["max_tasks"] = 1
    config_path = tmp_path / (method + ".json")
    config_path.write_text(json.dumps(config))
    process = subprocess.run(
        [sys.executable, "main.py", "--config", str(config_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, PYTHONHASHSEED="0"),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((output / "results.json").read_text())
    assert result["method"] == method
    assert result["status"] == "validation_partial"
    assert result["phase"] == "smoke" and result["performance_claim"] is False
    assert result["backbone_provenance"]["kind"] == "synthetic_random"
    manifest = json.loads((output / "data_manifest.json").read_text())
    assert manifest["official_test_read"] is False
    assert manifest["evaluation_source"] == "training_holdout"
    assert result["stages"][0]["seen_classes"] == 2
    assert (output / "checkpoint.pt").is_file()


@pytest.mark.parametrize("method", METHODS)
def test_runner_task_boundary_resume_matches_full_synthetic_stream(method, tmp_path):
    config = small_config(method, tmp_path / "continuous")
    config["seed"] = 0
    full = run(config)
    first_args = dict(config, output_dir=str(tmp_path / "first"), max_tasks=1)
    first = run(first_args)
    assert first["status"] == "validation_partial"
    resumed_args = dict(
        config,
        output_dir=str(tmp_path / "resumed"),
        resume=str(tmp_path / "first" / "checkpoint.pt"),
    )
    resumed = run(resumed_args)
    assert resumed["status"] == "complete"
    assert_same_state(full["metrics"], resumed["metrics"])
    assert_same_state(full["task_diagnostics"], resumed["task_diagnostics"])
    a = torch.load(tmp_path / "continuous" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    assert_same_state(a["learner"], b["learner"])
    assert_same_state(a["rng"], b["rng"])


def test_development_uses_stratified_training_holdout_without_reading_official_test(
    monkeypatch,
):
    labels = np.repeat(np.arange(100), 4)
    images = np.zeros((400, 32, 32, 3), dtype=np.uint8)
    calls = []

    def loader(root, split, download):
        calls.append(split)
        assert split == "train", "Official test must not be opened during development"
        return images, labels, "training-file-hash", False, None

    monkeypatch.setattr(data_module, "cifar_arrays", loader)
    args = {
        "phase": "development",
        "validation_only": True,
        "validation_fraction": 0.25,
        "split_seed": 3,
        "backbone_type": "vit_base_patch16_224",
        "model_name": "shared_basis",
    }
    dm = ProtocolDataManager("cifar224", True, 1993, 10, 10, args)
    assert calls == ["train"] and dm.nb_tasks == 10
    assert dm._class_order == np.random.RandomState(1993).permutation(100).tolist()
    assert set(dm._train_ids).isdisjoint(dm._test_ids)
    assert len(dm._train_ids) == 300 and len(dm._test_ids) == 100
    assert np.bincount(dm._train_targets).tolist() == [3] * 100
    assert np.bincount(dm._test_targets).tolist() == [1] * 100
    rows = dm.get_dataset([17, 2], source="train", mode="test")
    assert rows.labels.tolist() == [17] * 3 + [2] * 3
    assert (
        labels[rows.ids].tolist()
        == [dm._class_order[17]] * 3 + [dm._class_order[2]] * 3
    )


def test_training_holdout_split_is_reproducible_disjoint_and_not_class_order_rng():
    labels = np.repeat([5, 2, 9], [4, 6, 8])
    a, b = split_training(labels, 0.25, 11)
    again_a, again_b = split_training(labels, 0.25, 11)
    np.testing.assert_array_equal(a, again_a)
    np.testing.assert_array_equal(b, again_b)
    assert set(a).isdisjoint(b) and sorted(np.r_[a, b].tolist()) == list(
        range(len(labels))
    )
    assert all(np.any(labels[a] == c) and np.any(labels[b] == c) for c in [5, 2, 9])


def formal_config(dataset, method="history_transport"):
    width = 10 if dataset == "cifar224" else 20
    return {
        "dataset": dataset,
        "phase": "final",
        "seed": 1993,
        "model_name": method,
        "init_cls": width,
        "increment": width,
        "backbone_type": "vit_base_patch16_224",
        "fixed_memory": dataset == "imagenetr",
        "memory_size": 2000,
        "memory_per_class": 20,
        "shuffle": True,
        "pretrained": True,
    }


@pytest.mark.parametrize("dataset", ["cifar224", "imagenetr"])
def test_protocol_accepts_historical_dataset_specific_memory_rules(dataset):
    validate_protocol(formal_config(dataset))
    shared = formal_config(dataset, "shared_basis")
    shared.update(memory_size=0, memory_per_class=0)
    validate_protocol(shared)


@pytest.mark.parametrize(
    "change",
    [
        {"memory_size": 1000},
        {"fixed_memory": True},
        {"increment": 5},
        {"seed": 0},
        {"max_tasks": 2},
        {"validation_only": True},
        {"shuffle": False},
    ],
)
def test_formal_cifar_rejects_protocol_mismatches(change):
    with pytest.raises(ValueError):
        validate_protocol(dict(formal_config("cifar224"), **change))


@pytest.mark.parametrize("change", [{"memory_per_class": 10}, {"fixed_memory": False}])
def test_formal_imagenetr_rejects_wrong_memory_rule(change):
    with pytest.raises(ValueError):
        validate_protocol(dict(formal_config("imagenetr"), **change))


def records_dataset(classes, per_class=5, start_id=0):
    labels = np.repeat(classes, per_class)
    return SimpleNamespace(
        labels=labels,
        ids=np.arange(start_id, start_id + len(labels)),
        images=np.zeros((len(labels), 4, 4, 3), dtype=np.uint8),
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_replay_herding_retains_prefix_and_obeys_total_or_per_class_quota(fixed):
    memory = ReplayMemory(
        {"memory_size": 8, "memory_per_class": 2, "fixed_memory": fixed, "seed": 0}
    )
    first = records_dataset([0, 1])
    features = np.random.default_rng(7).normal(size=(10, 3))
    memory.update(first, features, [0, 1])
    saved_prefix = {
        c: [r.sample_id for r in memory.records if r.label == c][:2] for c in [0, 1]
    }
    assert len(memory) == (4 if fixed else 8)
    second = records_dataset([2, 3], start_id=10)
    memory.update(second, features, [0, 1, 2, 3])
    assert len(memory) == 8
    assert [sum(r.label == c for r in memory.records) for c in range(4)] == [2] * 4
    assert {
        c: [r.sample_id for r in memory.records if r.label == c] for c in [0, 1]
    } == saved_prefix


def test_replay_rejects_lost_vocabulary_and_corrupted_checkpoint_atomically():
    memory = ReplayMemory({"memory_size": 6, "seed": 0})
    first = records_dataset([0, 1])
    features = np.random.default_rng(7).normal(size=(10, 3))
    memory.update(first, features, [0, 1])
    previous = memory.state_dict()
    second = records_dataset([2], start_id=10)
    with pytest.raises(ValueError, match="preserve"):
        memory.update(second, features[:5], [1, 2])
    assert_same_state(memory.state_dict(), previous)
    bad = copy.deepcopy(previous)
    next(r for r in bad["records"] if r.label == 1).label = (
        0  # Same total size, wrong per-class quota.
    )
    with pytest.raises(ValueError, match="per-class"):
        memory.load_state_dict(bad)
    assert_same_state(memory.state_dict(), previous)
    bad = copy.deepcopy(previous)
    bad["rng"]["bit_generator"] = "MT19937"
    with pytest.raises(ValueError):
        memory.load_state_dict(bad)
    assert_same_state(memory.state_dict(), previous)


def test_metrics_keep_signed_and_final_inclusive_forgetting_explicit():
    stages = [
        {"accuracy": 0.8, "task_accuracy": [0.8, None, None]},
        {"accuracy": 0.6, "task_accuracy": [0.5, 0.7, None]},
        {"accuracy": 0.7, "task_accuracy": [0.9, 0.4, 0.8]},
    ]
    result = summarize(stages)
    assert result["final_accuracy"] == 0.7
    assert result["aia"] == pytest.approx(0.7)
    assert result["forgetting_previous"] == pytest.approx((-0.1 + 0.3) / 2)
    assert result["forgetting_including_final"] == pytest.approx(0.3 / 2)


def test_raw_head_metrics_use_preselected_nme_even_when_cnn_is_better():
    labels = np.array([0, 0, 1, 2, 2, 2, 3, 3])
    cnn = np.array([0, 1, 1, 2, 0, 0, 3, 0])
    nme = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    model = SimpleNamespace(
        args={"evaluation_head": "nme"},
        test_loader=None,
        _class_means=np.zeros((4, 2)),
        _evaluation_record=None,
    )
    model._eval_cnn = lambda loader: (cnn[:, None], labels)
    model._eval_nme = lambda loader, means: (nme[:, None], labels)
    dm = SimpleNamespace(nb_tasks=2, get_task_size=lambda task: 2)
    result = stage_record(model, {"top1": 99.99}, {"top1": 88.88}, dm, 1)
    assert result["accuracy"] == 0 and result["evaluation_head"] == "nme"
    assert result["heads"]["cnn"]["accuracy"] == 0.5
    assert result["heads"]["cnn"]["task_accuracy"] == [2 / 3, 2 / 5]
    assert result["task_counts"] == [3, 5]
    assert result["extra_evaluation_pass_for_raw_matrix"]
    assert result["extra_nme_evaluation_pass_for_raw_matrix"]
    model._eval_nme = lambda loader, means: (nme[:, None], labels[::-1])
    with pytest.raises(ValueError, match="rows disagree"):
        stage_record(model, {}, {}, dm, 1)


@pytest.mark.parametrize(
    "change",
    [
        {"validation_only": False},
        {"init_cls": 0},
        {"increment": 0},
        {"increment": 1.5},
        {"max_tasks": 0},
    ],
)
def test_smoke_rejects_unsafe_or_mislabeled_stream_shapes(change):
    with pytest.raises(ValueError):
        validate_protocol(dict(small_config("shared_basis", "unused"), **change))


def test_multiseed_driver_isolates_outputs_only_for_structured_protocol(
    monkeypatch, tmp_path
):
    import trainer

    captured = []
    monkeypatch.setattr(
        trainer, "_train", lambda args: captured.append(copy.deepcopy(args))
    )
    config = small_config("shared_basis", tmp_path / "runs")
    config["seed"] = [0, 1]
    trainer.train(config)
    assert [Path(x["output_dir"]).name for x in captured] == ["seed_0", "seed_1"]
    assert [x["seed"] for x in captured] == [0, 1]
    captured.clear()
    legacy = dict(config, cl27_protocol="legacy", seed=[3, 4], output_dir="unchanged")
    trainer.train(legacy)
    assert [x["output_dir"] for x in captured] == ["unchanged", "unchanged"]
