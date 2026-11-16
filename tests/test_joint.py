"""Offline Joint data-access, final-only evaluation and completed-state checks."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest
import torch

from cl27.common import seed_everything, state_hash
from models import joint
from utils.data_manager import DummyDataset


def image_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


class ReadDataset(DummyDataset):
    def __init__(self, manager, classes, source):
        labels = manager.train_y if source == "train" else manager.test_y
        images = manager.train_x if source == "train" else manager.test_x
        rows = np.flatnonzero(np.isin(labels, classes))
        super().__init__(images[rows], labels[rows], image_tensor)
        self.ids = rows + (1000 if source == "test" else 0)
        self.manager = manager
        self.source = source

    def __getitem__(self, i):
        self.manager.reads[self.source].append(int(self.ids[i]))
        return super().__getitem__(i)


class Tasks:
    nb_tasks = 3
    nb_classes = 6

    def __init__(self):
        rng = np.random.default_rng(716)
        self.train_y = np.repeat(np.arange(6), 6)
        self.test_y = np.repeat(np.arange(6), 3)
        self.train_x = rng.integers(0, 255, (36, 16, 16, 3), dtype=np.uint8)
        self.test_x = rng.integers(0, 255, (18, 16, 16, 3), dtype=np.uint8)
        self.requests = []
        self.reads = {"train": [], "test": []}
        self.events = []

    def get_task_size(self, t):
        return 2

    def get_dataset(self, classes, source, mode):
        self.requests.append((tuple(map(int, classes)), source, mode))
        self.events.append("request_" + source)
        return ReadDataset(self, classes, source)


def args(**overrides):
    return {
        "model_name": "joint",
        "dataset": "synthetic",
        "smoke": True,
        "pretrained": False,
        "phase": "smoke",
        "validation_only": True,
        "device": [torch.device("cpu")],
        "seed": 7,
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": False,
        "smoke_feature_dim": 6,
        "smoke_depth": 2,
        "image_size": 16,
        "epochs": 2,
        "batch_size": 4,
        "replay_batch_size": 0,
        "eval_batch_size": 6,
        "num_workers": 0,
        "backbone_lr": 0.001,
        "head_lr": 0.01,
        "weight_decay": 1e-4,
        "init_cls": 2,
        "increment": 2,
        **overrides,
    }


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_everything(19)
    yield
    torch.set_num_threads(previous)


def test_joint_fits_the_whole_union_once_and_only_requests_evaluation_after_updates(
    monkeypatch,
):
    dm = Tasks()
    model = joint.Learner(args())
    initial = state_hash(model.backbone)
    real_fit = joint.adapt_backbone

    def fit(*values, **keywords):
        assert dm.requests == [(tuple(range(6)), "train", "train")]
        assert values[1].out_features == 6 and len(values[3]) == 0
        dm.events.append("fit_start")
        result = real_fit(*values, **keywords)
        dm.events.append("fit_end")
        assert dm.reads["test"] == []
        return result

    monkeypatch.setattr(joint, "adapt_backbone", fit)
    assert model.head is None
    model.incremental_train(dm)
    assert dm.events == ["request_train", "fit_start", "fit_end", "request_test"]
    assert dm.reads["test"] == []
    assert Counter(dm.reads["train"]) == Counter({i: 2 for i in range(36)})
    assert (
        model._cur_task == 2 and model._total_classes == 6 and model._known_classes == 0
    )
    assert model.exemplar_size == 0 and state_hash(model.backbone) != initial
    diag = model.task_diagnostics[0]
    assert len(model.task_diagnostics) == diag["offline_fit_count"] == 1
    assert diag["selected_classes"] == 6 and diag["unique_training_images"] == 36
    assert diag["training"]["current_training_visits"] == 72
    assert diag["training"]["optimizer_updates"] == 18
    assert diag["training"]["replay_training_visits"] == 0
    assert diag["training"]["backbone_gradient_l2_max"] > 0
    assert (
        not diag["evaluation_during_training"]
        and not diag["aia_applicable"]
        and not diag["forgetting_applicable"]
    )
    evaluated, nme = model.eval_task()
    assert nme is None and np.isfinite(evaluated["top1"])
    assert len(dm.reads["test"]) == 18
    images = torch.stack([image_tensor(image) for image in dm.test_x])
    with torch.no_grad():
        result = model._network(images)["logits"]
        reference = model.head(model.backbone(images))
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert result.shape == (18, 6) and torch.isfinite(result).all()
    assert max(model._evaluation_record["predictions"]) < 6
    model.after_task()
    assert model._known_classes == 6
    with pytest.raises(RuntimeError, match="exactly one"):
        model.incremental_train(dm)


def test_bounded_development_joint_uses_a_complete_class_prefix():
    dm = Tasks()
    model = joint.Learner(args(joint_max_classes=4))
    model.incremental_train(dm)
    model.after_task()
    assert model._cur_task == 1 and model._known_classes == model.head.out_features == 4
    assert dm.requests == [
        (tuple(range(4)), "train", "train"),
        (tuple(range(4)), "test", "test"),
    ]
    assert set(dm.reads["train"]) == set(range(24)) and dm.reads["test"] == []
    assert model.selection["protocol_total_classes"] == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"joint_max_classes": 3},
        {"joint_max_classes": 8},
        {"joint_max_classes": 4, "phase": "final", "validation_only": False},
    ],
)
def test_invalid_class_prefix_is_rejected_before_reading_training_images(overrides):
    dm = Tasks()
    model = joint.Learner(args(**overrides))
    with pytest.raises(ValueError):
        model.incremental_train(dm)
    assert dm.requests == [] and dm.reads["train"] == []


@pytest.mark.parametrize(
    "overrides", [{"memory_size": 2000}, {"memory_per_class": 20}, {"max_tasks": 2}]
)
def test_joint_rejects_replay_or_ambiguous_continual_limits(overrides):
    with pytest.raises(ValueError):
        joint.Learner(args(**overrides))


def test_no_untrained_or_partial_checkpoint_and_completed_restore_is_readonly():
    model = joint.Learner(args())
    with pytest.raises(RuntimeError, match="after its full fit"):
        model.eval_task()
    with pytest.raises(RuntimeError, match="completed-fit only"):
        model.checkpoint_state()
    model.incremental_train(Tasks())
    with pytest.raises(RuntimeError, match="completed task"):
        model.checkpoint_state()
    model.after_task()
    state = model.checkpoint_state()
    restored = joint.Learner(args())
    rng = torch.get_rng_state().clone()
    restored.restore_checkpoint(state)
    assert torch.equal(rng, torch.get_rng_state())
    assert state_hash(restored._network) == state_hash(model._network)
    assert (
        restored.selection == model.selection
        and restored._cur_task == 2
        and restored._known_classes == 6
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        restored.incremental_train(Tasks())
    with pytest.raises(RuntimeError, match="no new evaluation loader"):
        restored.eval_task()
    for tensor in state["core"]["head"].values():
        tensor.zero_()
    assert torch.count_nonzero(restored.head.weight) > 0


@pytest.mark.parametrize("corruption", ["source", "nonfinite_head", "phase"])
def test_invalid_completed_state_is_rejected_without_changing_live_model(corruption):
    trained = joint.Learner(args())
    trained.incremental_train(Tasks())
    trained.after_task()
    state = copy.deepcopy(trained.checkpoint_state())
    fresh = joint.Learner(args())
    initial = state_hash(fresh.backbone)
    if corruption == "source":
        state["core"]["initial_backbone_hash"] = "0" * 64
    elif corruption == "nonfinite_head":
        state["core"]["head"]["weight"][0, 0] = float("nan")
    else:
        state["cur_task"] = 0
    with pytest.raises(ValueError):
        fresh.restore_checkpoint(state)
    assert (
        state_hash(fresh.backbone) == initial
        and fresh.head is None
        and not fresh._fit_complete
    )
