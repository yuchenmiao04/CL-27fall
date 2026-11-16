"""Small correctness tests for the native supervised experience-replay control."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from cl27.common import extract_features, seed_everything, state_hash
from cl27.supervised import CEClassifier
from models import er
from models.history_transport import Learner as HistoryLearner
from utils.data_manager import DummyDataset


def image_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def training_tensor(image):
    value = image_tensor(image)
    return value.flip(-1) if torch.rand(()) < 0.5 else value


class TracedDataset(DummyDataset):
    def __getitem__(self, index):
        self.reads.append(int(self.ids[index]))
        return super().__getitem__(index)


class SyntheticTasks:
    nb_tasks = 3

    def __init__(self):
        rng = np.random.default_rng(701)
        self.labels = np.repeat(np.arange(6), 6)
        self.images = rng.integers(0, 255, (36, 16, 16, 3), dtype=np.uint8)
        self.requests = []
        self.test_reads = []
        self.train_reads = []

    def get_task_size(self, task):
        return 2

    def get_dataset(self, classes, source, mode):
        self.requests.append((tuple(map(int, classes)), source, mode))
        indices = np.flatnonzero(np.isin(self.labels, classes))
        result = TracedDataset(
            self.images[indices],
            self.labels[indices],
            training_tensor if mode == "train" else image_tensor,
        )
        result.ids = indices + (1000 if source == "test" else 0)
        result.reads = self.test_reads if source == "test" else self.train_reads
        return result


def args(fixed=False, **overrides):
    return {
        "model_name": "er",
        "dataset": "synthetic",
        "smoke": True,
        "pretrained": False,
        "device": [torch.device("cpu")],
        "seed": 3,
        "smoke_feature_dim": 6,
        "smoke_depth": 2,
        "image_size": 16,
        "memory_size": 8,
        "fixed_memory": fixed,
        "memory_per_class": 2,
        "replay_selection": "herding",
        "epochs": 2,
        "batch_size": 4,
        "replay_batch_size": 3,
        "backbone_lr": 0.001,
        "head_lr": 0.01,
        "weight_decay": 1e-4,
        "feature_batch_size": 4,
        "eval_batch_size": 8,
        "num_workers": 0,
        "init_cls": 2,
        "increment": 2,
        "feature_normalization": "l2",
        "mean_shrinkage": 2.0,
        "crossfit_folds": 2,
        **overrides,
    }


@pytest.fixture(autouse=True)
def deterministic_cpu():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_everything(13)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("fixed", [False, True])
def test_er_trains_single_body_uses_bounded_replay_and_seen_ce_logits(fixed):
    manager = SyntheticTasks()
    model = er.Learner(args(fixed))
    assert not hasattr(model, "anchor") and not hasattr(model, "core")
    for task in range(3):
        body = state_hash(model.backbone)
        test_reads_before = list(manager.test_reads)
        model.incremental_train(manager)
        assert manager.test_reads == test_reads_before
        assert state_hash(model.backbone) != body
        assert model.completed_tasks == task + 1
        assert model.replay.seen == (task + 1) * 12
        quota = 2 if fixed else 8 // ((task + 1) * 2)
        assert len(model.replay) == quota * ((task + 1) * 2)
        assert model.replay.classes == list(range((task + 1) * 2))
        assert model.ce_head.out_features == (task + 1) * 2
        diagnostic = model.task_diagnostics[-1]
        assert diagnostic["method"] == "er"
        assert diagnostic["training"]["current_training_visits"] == 24
        assert diagnostic["training"]["replay_training_visits"] == (
            0 if not task else 18
        )
        assert diagnostic["training"]["optimizer_updates"] == 6
        assert diagnostic["training"]["backbone_gradient_l2_max"] > 0
        assert diagnostic["unique_fitting_images_seen"] == model.replay.seen
        assert diagnostic["resources"]["training_frozen_anchor_parameter_bytes"] == 0
        assert diagnostic["resources"]["historical_statistics_bytes"] == 0
        assert diagnostic["resources"]["inference_backbones"] == 1
        # Probe includes old classes after later tasks. No task-ID/mask is given.
        images = torch.stack([image_tensor(manager.images[i]) for i in [0, 7, 13, 19]])
        with torch.no_grad():
            scores = model._network(images)["logits"]
            expected = model.ce_head(model.backbone(images))
        torch.testing.assert_close(scores, expected, rtol=0, atol=0)
        assert scores.shape == (4, (task + 1) * 2) and torch.isfinite(scores).all()
        model.eval_task()
        model.after_task()
    assert [r[0] for r in manager.requests if r[1:] == ("train", "train")] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert all(
        r[0] in [(0, 1), (2, 3), (4, 5)] for r in manager.requests if r[1] == "train"
    )


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_er_exactly_matches_history_body_ce_head_and_herding_trajectory(mode):
    settings = args(transport_mode=mode)
    model = er.Learner(settings)
    history = HistoryLearner(dict(settings, model_name="history_transport"))
    left, right = SyntheticTasks(), SyntheticTasks()
    for task in range(3):
        rng_before = torch.get_rng_state()
        model.incremental_train(left)
        rng_after = torch.get_rng_state()
        torch.set_rng_state(rng_before)
        history.incremental_train(right)
        # Read-only canonical feature extraction must not alter the stochastic
        # training trajectory just because one method has extra analytic paths.
        torch.testing.assert_close(torch.get_rng_state(), rng_after, rtol=0, atol=0)
        assert state_hash(model.backbone) == state_hash(history.backbone)
        assert state_hash(model.ce_head) == state_hash(history.ce_head)
        assert (
            model.task_diagnostics[-1]["training"]
            == history.task_diagnostics[-1]["training"]
        )
        assert [r.sample_id for r in model.replay.records] == [
            r.sample_id for r in history.replay.records
        ]
        np.testing.assert_equal(
            model.rng.bit_generator.state, history.rng.bit_generator.state
        )
        model.after_task()
        history.after_task()


def test_expansion_preserves_old_ce_rows_before_new_task_updates(monkeypatch):
    model = er.Learner(args())
    manager = SyntheticTasks()
    model.incremental_train(manager)
    model.after_task()
    weight, bias = (
        model.ce_head.weight.detach().clone(),
        model.ce_head.bias.detach().clone(),
    )
    original = er.adapt_backbone
    observed = []

    def check_expansion(body, head, *remaining):
        assert head.out_features == 4
        torch.testing.assert_close(head.weight[:2], weight, rtol=0, atol=0)
        torch.testing.assert_close(head.bias[:2], bias, rtol=0, atol=0)
        observed.append(True)
        return original(body, head, *remaining)

    monkeypatch.setattr(er, "adapt_backbone", check_expansion)
    model.incremental_train(manager)
    assert observed == [True]


def test_checkpoint_resume_matches_stochastic_training_and_raw_ce_predictions():
    manager = SyntheticTasks()
    model = er.Learner(args())
    model.incremental_train(manager)
    model.after_task()
    saved = copy.deepcopy(model.checkpoint_state())
    resumed = er.Learner(args())
    resumed.restore_checkpoint(saved)
    image = image_tensor(manager.images[0]).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(
            model._network(image)["logits"],
            resumed._network(image)["logits"],
            rtol=0,
            atol=0,
        )
    rng = torch.get_rng_state()
    model.incremental_train(manager)
    torch.set_rng_state(rng)
    resumed.incremental_train(SyntheticTasks())
    assert state_hash(model.backbone) == state_hash(resumed.backbone)
    assert state_hash(model.ce_head) == state_hash(resumed.ce_head)
    assert model.task_diagnostics == resumed.task_diagnostics
    assert [r.sample_id for r in model.replay.records] == [
        r.sample_id for r in resumed.replay.records
    ]
    saved["core"]["ce_head"]["weight"].zero_()
    assert torch.count_nonzero(resumed.ce_head.weight) > 0


@pytest.mark.parametrize(
    "corruption", ["head", "nonfinite", "provenance", "count", "outer"]
)
def test_corrupt_checkpoint_rejection_is_atomic(corruption):
    manager = SyntheticTasks()
    model = er.Learner(args())
    model.incremental_train(manager)
    model.after_task()
    first = copy.deepcopy(model.checkpoint_state())
    model.incremental_train(manager)
    model.after_task()
    before_body, before_head = state_hash(model.backbone), state_hash(model.ce_head)
    before_ids = [r.sample_id for r in model.replay.records]
    if corruption == "head":
        first["core"]["ce_head"]["weight"] = first["core"]["ce_head"]["weight"][:1]
    elif corruption == "nonfinite":
        first["core"]["ce_head"]["weight"][0, 0] = float("nan")
    elif corruption == "provenance":
        first["core"]["initial_backbone_hash"] = "untrusted"
    elif corruption == "count":
        first["core"]["task_arrivals"][0] += 1
    else:
        first["cur_task"] += 1
    with pytest.raises(ValueError):
        model.restore_checkpoint(first)
    assert (
        state_hash(model.backbone) == before_body
        and state_hash(model.ce_head) == before_head
    )
    assert [r.sample_id for r in model.replay.records] == before_ids
    assert model.completed_tasks == 2 and model._known_classes == 4


def test_ce_classifier_does_not_create_a_throwaway_head_or_normalize_features():
    model = er.Learner(args())
    before = torch.get_rng_state()
    classifier = CEClassifier(model.backbone)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    image = image_tensor(SyntheticTasks().images[0]).unsqueeze(0)
    with pytest.raises(RuntimeError, match="no registered classes"):
        classifier(image)
    assert classifier.feature_dim == model.backbone.out_dim
    with torch.no_grad():
        torch.testing.assert_close(
            classifier.extract_vector(image), model.backbone(image), rtol=0, atol=0
        )


def test_canonical_feature_encoding_does_not_consume_training_rng():
    model = er.Learner(args())
    rows = SyntheticTasks().get_dataset([0, 1], "train", "test")
    before = torch.get_rng_state()
    features, labels, ids = extract_features(
        model.backbone, rows, "cpu", 4, normalization="none"
    )
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert features.shape == (12, 6) and len(labels) == len(ids) == 12
