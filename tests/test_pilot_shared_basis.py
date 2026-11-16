"""PILOT lifecycle, data-access, immutability and resume checks (synthetic only)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from cl27.common import state_hash
from models import shared_basis


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 4
        self.linear = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.linear(x))


class Rows(Dataset):
    def __init__(self, manager, indices, source):
        self.manager = manager
        self.indices = indices
        self.source = source
        self.ids = indices + (10000 if source == "test" else 0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        self.manager.reads[self.source].append(row)
        return index, self.manager.x[row], self.manager.y[row]


class Manager:
    nb_tasks = 2

    def __init__(self):
        rng = np.random.default_rng(7)
        self.y = np.repeat(np.arange(4), 6)
        self.x = torch.tensor(
            rng.normal(size=(24, 4)) + np.eye(4)[self.y] * 2, dtype=torch.float32
        )
        self.requests = []
        self.reads = {"train": [], "test": []}

    def get_task_size(self, task):
        return 2

    def get_dataset(self, classes, source, mode):
        self.requests.append((list(classes), source, mode))
        return Rows(self, np.flatnonzero(np.isin(self.y, classes)), source)


@pytest.fixture
def setup(monkeypatch):
    def build(_args):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            return SmallBackbone()

    monkeypatch.setattr(shared_basis, "build_backbone", build)
    args = {
        "model_name": "shared_basis",
        "device": [torch.device("cpu")],
        "memory_size": 2000,
        "init_cls": 2,
        "increment": 2,
        "seed": 1,
        "basis_rank": 2,
        "basis_epsilon": 0.1,
        "basis_max_iter": 8,
        "feature_normalization": "none",
        "batch_size": 5,
    }
    return args, Manager()


def test_native_task_lifecycle_commits_current_images_once_and_freezes_backbone(setup):
    args, dm = setup
    model = shared_basis.Learner(args)
    initial = state_hash(model.backbone)
    model.incremental_train(dm)
    assert dm.reads["test"] == []  # Creating the evaluator does not fit on it.
    assert dm.reads["train"] == list(range(12))
    assert model.core.classes == [0, 1]
    assert model.core.n_samples == 12
    model.after_task()
    model.incremental_train(dm)
    assert dm.reads["train"] == list(range(24))
    assert dm.requests == [
        ([0, 1], "train", "test"),
        ([0, 1], "test", "test"),
        ([2, 3], "train", "test"),
        ([0, 1, 2, 3], "test", "test"),
    ]
    assert model.core.n_samples == 24 and model.exemplar_size == 0
    assert model.core.classes == [0, 1, 2, 3]
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert state_hash(model.backbone) == initial
    assert all(row["raw_replay_images"] == 0 for row in model.task_diagnostics)
    assert all(
        row["backbone_state_sha256"] == initial for row in model.task_diagnostics
    )
    evaluation, nme = model.eval_task()
    assert nme is None and np.isfinite(evaluation["top1"])
    with torch.no_grad():
        raw = model.backbone(dm.x).numpy().astype(np.float64)
        expected = model.core.predict_scores(raw)
        logits = model._network(dm.x)["logits"].numpy()
    np.testing.assert_allclose(logits, expected, atol=1e-6)
    assert logits.shape == (24, 4)
    assert state_hash(model.backbone) == initial


def test_task_boundary_checkpoint_resume_matches_continuous_fit(setup):
    args, dm = setup
    original = shared_basis.Learner(args)
    original.incremental_train(dm)
    original.after_task()
    state = original.checkpoint_state()
    restored = shared_basis.Learner(args)
    restored.restore_checkpoint(state)
    assert restored._known_classes == 2 and restored._cur_task == 0
    restored.incremental_train(Manager())
    original.incremental_train(dm)
    np.testing.assert_array_equal(restored.core.G, original.core.G)
    np.testing.assert_array_equal(restored.core.D, original.core.D)
    np.testing.assert_array_equal(restored.core.weights, original.core.weights)
    assert restored.task_diagnostics == original.task_diagnostics
    # Checkpoint tensor payloads own their storage.
    for tensor in state["core"]["backbone"].values():
        tensor.zero_()
    assert state_hash(restored.backbone) == original._frozen_hash


def test_modified_backbone_is_rejected_before_new_data_access(setup):
    args, dm = setup
    model = shared_basis.Learner(args)
    with torch.no_grad():
        model.backbone.linear.weight[0, 0] += 0.01
    with pytest.raises(RuntimeError, match="immutable backbone"):
        model.incremental_train(dm)
    assert dm.requests == [] and model.core.n_samples == 0


def test_checkpoint_corruption_does_not_modify_live_history_or_backbone(setup):
    args, dm = setup
    model = shared_basis.Learner(args)
    model.incremental_train(dm)
    model.after_task()
    state = model.checkpoint_state()
    old_hash, old_weights = state_hash(model.backbone), model.core.weights
    corrupt = copy.deepcopy(state)
    corrupt["core"]["predictor"]["counts"][0] += 1
    with pytest.raises(ValueError):
        model.restore_checkpoint(corrupt)
    np.testing.assert_array_equal(model.core.weights, old_weights)
    assert state_hash(model.backbone) == old_hash
    corrupt = copy.deepcopy(state)
    corrupt["core"]["backbone"]["linear.weight"][0, 0] += 0.1
    with pytest.raises(ValueError, match="hash"):
        model.restore_checkpoint(corrupt)
    assert state_hash(model.backbone) == old_hash


def test_normalization_is_part_of_checkpoint_feature_map(setup):
    args, dm = setup
    model = shared_basis.Learner(args)
    model.incremental_train(dm)
    model.after_task()
    other = shared_basis.Learner(dict(args, feature_normalization="l2"))
    with pytest.raises(ValueError, match="normalization differs"):
        other.restore_checkpoint(model.checkpoint_state())
