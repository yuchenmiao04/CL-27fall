"""Independent ER / final-only Joint protocol and native-entry checks.

All runs use tiny synthetic train/holdout data. They verify algorithmic/data
contracts and continuation, not benchmark accuracy or an offline upper bound.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from cl27 import runner
from cl27.common import build_backbone, extract_features, state_hash
from cl27.data import ProtocolDataManager
from cl27.protocol import smoke_config, validate_protocol
from utils import factory


ROOT = Path(__file__).resolve().parents[1]


def configuration(method, output):
    config = smoke_config(method, output)
    config.update(
        {
            "smoke_feature_dim": 6,
            "smoke_depth": 2,
            "image_size": 16,
            "synthetic_per_class": 6,
            "epochs": 1,
            "batch_size": 4,
            "feature_batch_size": 4,
            "cpu_threads": 1,
            "memory_size": 0 if method == "joint" else 12,
            "memory_per_class": 0,
            "evaluation_head": "cnn",
        }
    )
    return config


def same_state(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            same_state(first, second)
    elif hasattr(a, "__dataclass_fields__"):
        same_state(vars(a), vars(b))
    else:
        assert a == b


@pytest.mark.parametrize("method", ["er", "joint"])
def test_original_cli_runs_er_stream_and_joint_final_only(method, tmp_path):
    config = configuration(method, tmp_path / method)
    filename = tmp_path / (method + ".json")
    filename.write_text(json.dumps(config))
    process = subprocess.run(
        [sys.executable, "main.py", "--config", str(filename)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, PYTHONHASHSEED="0"),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((tmp_path / method / "results.json").read_text())
    assert result["method"] == method and result["status"] == "complete"
    assert result["performance_claim"] is False
    assert result["backbone_provenance"]["kind"] == "synthetic_random"
    manifest = json.loads((tmp_path / method / "data_manifest.json").read_text())
    assert not manifest["official_test_read"]
    assert manifest["evaluation_source"] == "training_holdout"
    if method == "er":
        assert [stage["seen_classes"] for stage in result["stages"]] == [2, 4, 6]
        assert result["metrics"]["aia"] is not None
        assert len(result["task_diagnostics"]) == 3
    else:
        assert result["evaluation_mode"] == "offline_joint_final"
        assert len(result["stages"]) == len(result["task_diagnostics"]) == 1
        assert result["completed_fits"] == 1
        final = result["stages"][0]
        assert final["seen_classes"] == 6 and final["task"] == 2
        assert final["evaluation_images"] == 6  # One holdout per class.
        assert result["metrics"]["final_accuracy"] == final["accuracy"]
        assert result["metrics"]["accuracy_matrix"] == []
        for metrics in [result["metrics"], result["metrics_by_head"]["cnn"]]:
            for key in ("aia", "forgetting_previous", "forgetting_including_final"):
                assert metrics[key] is None
        assert result["task_diagnostics"][0]["unique_training_images"] == 30
        assert (
            result["task_diagnostics"][0]["training"]["current_training_visits"] == 30
        )


def test_er_task_boundary_resume_matches_body_head_memory_and_rng(tmp_path):
    config = configuration("er", tmp_path / "whole")
    config["seed"] = 0
    whole = runner.run(config)
    runner.run(dict(config, output_dir=str(tmp_path / "prefix"), max_tasks=1))
    resumed = runner.run(
        dict(
            config,
            output_dir=str(tmp_path / "resumed"),
            resume=str(tmp_path / "prefix" / "checkpoint.pt"),
        )
    )
    same_state(whole["metrics"], resumed["metrics"])
    same_state(whole["task_diagnostics"], resumed["task_diagnostics"])
    first = torch.load(tmp_path / "whole" / "checkpoint.pt", weights_only=False)
    second = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    same_state(first["learner"], second["learner"])
    same_state(first["rng"], second["rng"])


def test_completed_joint_resume_does_not_fit_or_evaluate_again(monkeypatch, tmp_path):
    config = configuration("joint", tmp_path / "fit")
    config["seed"] = 0
    first = runner.run(config)
    real_factory = factory.get_model

    def fresh_model(name, args):
        model = real_factory(name, args)

        def forbidden(*unused_args, **unused_kwargs):
            raise AssertionError(
                "Completed Joint checkpoint must not refit or re-evaluate"
            )

        model.incremental_train = forbidden
        model.eval_task = forbidden
        return model

    monkeypatch.setattr(factory, "get_model", fresh_model)
    restored = runner.run(
        dict(
            config,
            output_dir=str(tmp_path / "read_completed"),
            resume=str(tmp_path / "fit" / "checkpoint.pt"),
        )
    )
    same_state(first["metrics"], restored["metrics"])
    same_state(first["stages"], restored["stages"])
    assert restored["completed_fits"] == 1


def test_explicit_null_joint_class_limit_still_fits_the_full_union(tmp_path):
    config = configuration("joint", tmp_path / "joint_null")
    config.update(seed=0, joint_max_classes=None)
    result = runner.run(config)
    assert result["status"] == "complete"
    assert result["selected_classes"] == result["dataset_classes"] == 6
    assert result["completed_fits"] == 1
    assert result["metrics"]["aia"] is None


@pytest.mark.parametrize("method", ["er", "joint"])
def test_resume_rejects_checkpoint_from_an_undeclared_old_random_stream(
    method, tmp_path
):
    config = configuration(method, tmp_path / "initial")
    config["seed"] = 0
    initial = dict(config, max_tasks=1) if method == "er" else config
    runner.run(initial)
    path = tmp_path / "initial" / "checkpoint.pt"
    state = torch.load(path, weights_only=False)
    state.pop("random_stream_policy")
    torch.save(state, path)
    with pytest.raises(ValueError, match="random-stream policy"):
        runner.run(dict(config, output_dir=str(tmp_path / "resume"), resume=str(path)))


def test_joint_selected_prefix_is_one_union_fit_without_holdout_training(
    monkeypatch, tmp_path
):
    phase = {"fitting": False}
    requests, read_during_fit, initial_hashes, created = [], [], [], []

    class View:
        def __init__(self, dataset, source):
            self.dataset, self.source = dataset, source

        def __len__(self):
            return len(self.dataset)

        def __getattr__(self, name):
            return getattr(self.dataset, name)

        def __getitem__(self, index):
            if phase["fitting"]:
                read_during_fit.append((self.source, int(self.dataset.ids[index])))
            return self.dataset[index]

    class TrackingData(ProtocolDataManager):
        def get_dataset(self, indices, source, mode, **kwargs):
            result = super().get_dataset(indices, source, mode, **kwargs)
            requests.append((tuple(map(int, indices)), source, mode))
            return View(result, source)

    real_factory = factory.get_model

    def wrapped_factory(name, args):
        model = real_factory(name, args)
        initial_hashes.append(state_hash(model.backbone))
        created.append(model)
        original = model.incremental_train

        def fitting(dm):
            assert not phase["fitting"]
            phase["fitting"] = True
            try:
                return original(dm)
            finally:
                phase["fitting"] = False

        model.incremental_train = fitting
        return model

    monkeypatch.setattr(runner, "ProtocolDataManager", TrackingData)
    monkeypatch.setattr(factory, "get_model", wrapped_factory)
    config = configuration("joint", tmp_path / "joint_prefix")
    config.update(seed=0, joint_max_classes=4)
    result = runner.run(config)
    assert result["status"] == "validation_partial"
    assert len(result["stages"]) == 1 and result["stages"][0]["seen_classes"] == 4
    assert result["stages"][0]["task"] == 1
    assert all(source == "train" for source, _ in read_during_fit)
    assert len(read_during_fit) == 20  # Four classes × five fitting rows, one epoch.
    assert len(set(uid for _, uid in read_during_fit)) == 20
    assert all(classes == (0, 1, 2, 3) for classes, _, _ in requests)
    assert all(mode == "train" for _, source, mode in requests if source == "train")
    assert created[0]._known_classes == created[0]._total_classes == 4
    with pytest.raises(RuntimeError):
        created[0].incremental_train(TrackingData("synthetic", True, 0, 2, 2, config))
    # Same fixed synthetic initialization is used for ER and Joint (not a
    # previously trained incremental checkpoint passed into Joint).
    runner.run(dict(configuration("er", tmp_path / "er"), seed=0, max_tasks=1))
    assert initial_hashes[0] == initial_hashes[1]


@pytest.mark.parametrize(
    "change", [{"max_tasks": 1}, {"max_tasks": 3}, {"memory_size": 12}]
)
def test_joint_rejects_ambiguous_truncation_or_replay_protocol(change):
    config = configuration("joint", "unused")
    config["seed"] = 0
    with pytest.raises(ValueError):
        validate_protocol(dict(config, **change))


def test_joint_validation_prefix_must_end_at_a_protocol_boundary(tmp_path):
    config = configuration("joint", tmp_path / "wrong_prefix")
    config.update(seed=0, joint_max_classes=3)
    with pytest.raises(ValueError, match="task boundary"):
        runner.run(config)


@pytest.mark.parametrize("dataset", ["cifar224", "imagenetr"])
def test_formal_er_requires_the_same_dataset_specific_replay_capacity(dataset):
    width = 10 if dataset == "cifar224" else 20
    args = {
        "model_name": "er",
        "dataset": dataset,
        "phase": "final",
        "seed": 1993,
        "shuffle": True,
        "init_cls": width,
        "increment": width,
        "backbone_type": "vit_base_patch16_224",
        "fixed_memory": dataset == "imagenetr",
        "memory_size": 2000 if dataset == "cifar224" else 4000,
        "memory_per_class": 20,
    }
    validate_protocol(args)
    wrong = (
        dict(args, memory_size=1000)
        if dataset == "cifar224"
        else dict(args, memory_per_class=10)
    )
    with pytest.raises(ValueError):
        validate_protocol(wrong)


def test_canonical_feature_encoding_preserves_training_rng_and_numerical_outputs():
    config = configuration("er", "unused")
    config["seed"] = 0
    dm = ProtocolDataManager("synthetic", True, 0, 2, 2, config)
    dataset = dm.get_dataset([0, 1], source="train", mode="test")
    body = build_backbone(config).eval()
    torch.manual_seed(93)
    before = torch.get_rng_state().clone()
    body_hash = state_hash(body)
    features, labels, ids = extract_features(body, dataset, "cpu", batch_size=4)
    assert torch.equal(torch.get_rng_state(), before)
    assert state_hash(body) == body_hash
    # Compare with the same batch-shaped deterministic forward computation;
    # removing the loader's incidental RNG draw must not change its features.
    expected = []
    with torch.no_grad():
        for start in range(0, len(dataset), 4):
            images = torch.stack(
                [
                    dataset[index][1]
                    for index in range(start, min(start + 4, len(dataset)))
                ]
            )
            expected.append(
                torch.nn.functional.normalize(body(images), dim=-1).double().numpy()
            )
    np.testing.assert_array_equal(features, np.concatenate(expected))
    np.testing.assert_array_equal(labels, dataset.labels)
    np.testing.assert_array_equal(ids, dataset.ids)
