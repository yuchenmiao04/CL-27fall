"""Fourth-route native lifecycle checks, on small synthetic RGB images only."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from cl27.common import extract_features, seed_everything, state_hash
from cl27.cores.transport import TransportHistory
from models.history_transport import Learner as HistoryLearner
from models.plastic_shared_basis import Learner
from utils.data_manager import DummyDataset


def image_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


class SyntheticTasks:
    nb_tasks = 3

    def __init__(self):
        rng = np.random.default_rng(810)
        self.labels = np.repeat(np.arange(6), 6)
        self.images = rng.integers(0, 255, (36, 16, 16, 3), dtype=np.uint8)
        self.requests = []

    def get_task_size(self, task):
        return 2

    def get_dataset(self, classes, source, mode):
        self.requests.append((tuple(map(int, classes)), source, mode))
        rows = np.flatnonzero(np.isin(self.labels, classes))
        result = DummyDataset(self.images[rows], self.labels[rows], image_tensor)
        result.ids = rows + (1000 if source == "test" else 0)
        return result


def args(mode="direct", fixed=False, **overrides):
    return {
        "model_name": "plastic_shared_basis",
        "dataset": "synthetic",
        "smoke": True,
        "pretrained": False,
        "device": [torch.device("cpu")],
        "seed": 0,
        "smoke_feature_dim": 6,
        "smoke_depth": 2,
        "image_size": 16,
        "memory_size": 8,
        "fixed_memory": fixed,
        "memory_per_class": 2,
        "replay_selection": "herding",
        "transport_mode": mode,
        "residual_model": "class_conditional_psd",
        "head_ridge": 0.03,
        "epochs": 2,
        "batch_size": 4,
        "replay_batch_size": 3,
        "backbone_lr": 0.001,
        "head_lr": 0.01,
        "feature_batch_size": 4,
        "eval_batch_size": 8,
        "num_workers": 0,
        "init_cls": 2,
        "increment": 2,
        "feature_normalization": "l2",
        "mean_shrinkage": 2.0,
        "crossfit_folds": 2,
        "basis_rank": 2,
        "basis_epsilon": 0.03,
        "basis_noise_variance": 0.4,
        "basis_max_iter": 6,
        "basis_tolerance": 1e-8,
        **overrides,
    }


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_everything(15)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("mode", ["direct", "chained"])
@pytest.mark.parametrize("fixed", [False, True])
def test_fourth_route_updates_body_and_refits_full_moments_once(mode, fixed):
    manager = SyntheticTasks()
    model = Learner(args(mode, fixed))
    frozen = state_hash(model.anchor)
    for task in range(3):
        body = state_hash(model.backbone)
        model.incremental_train(manager)
        assert state_hash(model.backbone) != body
        assert state_hash(model.anchor) == frozen
        assert all(not p.requires_grad for p in model.anchor.parameters())
        assert model.core.tasks == task + 1
        expected_count = (task + 1) * 12
        assert model.core.basis.n_samples == model.replay.seen == expected_count
        np.testing.assert_array_equal(model.core.counts, np.repeat(6, (task + 1) * 2))
        np.testing.assert_allclose(
            model.core.basis.G, model.core.current_G[:-1, :-1], atol=1e-12
        )
        np.testing.assert_allclose(
            model.core.basis.C, model.core.current_C[:-1], atol=1e-12
        )
        np.testing.assert_allclose(
            model.core.basis.feature_sum, model.core.current_G[:-1, -1], atol=1e-12
        )
        np.testing.assert_array_equal(model.core.basis.counts, model.core.counts)
        assert (
            model.core.basis.classes
            == model.core.classes
            == list(range((task + 1) * 2))
        )
        quota = 2 if fixed else 8 // ((task + 1) * 2)
        assert len(model.replay) == quota * ((task + 1) * 2)
        diagnostic = model.task_diagnostics[-1]
        assert diagnostic["method"] == "plastic_shared_basis"
        assert diagnostic["training"]["current_training_visits"] == 24
        assert diagnostic["training"]["backbone_gradient_l2_max"] > 0
        assert diagnostic["resources"]["inference_backbones"] == 1
        assert np.linalg.eigvalsh(model.core.current_G).min() >= -1e-8
        probe = manager.get_dataset(range((task + 1) * 2), "test", "test")
        x, _, _ = extract_features(model.backbone, probe, "cpu", 4)
        expected = model.core.basis.predict_scores(x)
        np.testing.assert_allclose(model.core.predict_scores(x), expected, atol=1e-12)
        np.testing.assert_allclose(
            np.column_stack((x, np.ones(len(x)))) @ model.core.head,
            expected,
            atol=1e-12,
        )
        images = torch.stack([probe[i][1] for i in range(len(probe))])
        with torch.no_grad():
            predicted = model._network(images)["logits"].numpy()
        np.testing.assert_allclose(predicted, expected, rtol=1e-4, atol=3e-6)
        model.after_task()
    assert [r[0] for r in manager.requests if r[1:] == ("train", "train")] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_stagewise_coupling_preserves_the_history_route_body_and_transport(mode):
    """EB changes deployment readout, not the CE gradient trajectory in this version."""
    settings = args(mode)
    combined = Learner(settings)
    history = HistoryLearner(dict(settings, model_name="history_transport"))
    assert type(history.core) is TransportHistory
    a, b = SyntheticTasks(), SyntheticTasks()
    for task in range(3):
        torch.manual_seed(51 + task)
        combined.incremental_train(a)
        torch.manual_seed(51 + task)
        history.incremental_train(b)
        assert state_hash(combined.backbone) == state_hash(history.backbone)
        np.testing.assert_array_equal(combined.core.current_G, history.core.current_G)
        np.testing.assert_array_equal(combined.core.current_C, history.core.current_C)
        assert [r.sample_id for r in combined.replay.records] == [
            r.sample_id for r in history.replay.records
        ]
        assert (
            combined.task_diagnostics[-1]["training"]
            == history.task_diagnostics[-1]["training"]
        )
        combined.after_task()
        history.after_task()


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_checkpoint_restores_fourth_route_and_next_task_exactly(mode):
    model = Learner(args(mode))
    manager = SyntheticTasks()
    model.incremental_train(manager)
    model.after_task()
    saved = copy.deepcopy(model.checkpoint_state())
    restored = Learner(args(mode))
    restored.restore_checkpoint(saved)
    np.testing.assert_array_equal(restored.core.head, model.core.head)
    np.testing.assert_array_equal(restored.core.basis.D, model.core.basis.D)
    torch.manual_seed(97)
    model.incremental_train(manager)
    torch.manual_seed(97)
    restored.incremental_train(SyntheticTasks())
    assert state_hash(model.backbone) == state_hash(restored.backbone)
    np.testing.assert_array_equal(model.core.head, restored.core.head)
    np.testing.assert_array_equal(model.core.basis.D, restored.core.basis.D)
    assert model.task_diagnostics == restored.task_diagnostics


def test_failed_restore_rejects_basis_configuration_atomically():
    model = Learner(args())
    manager = SyntheticTasks()
    model.incremental_train(manager)
    model.after_task()
    saved = copy.deepcopy(model.checkpoint_state())
    other = Learner(args(basis_epsilon=0.2))
    old_body, old_anchor = state_hash(other.backbone), state_hash(other.anchor)
    with pytest.raises(ValueError):
        other.restore_checkpoint(saved)
    assert (
        state_hash(other.backbone) == old_body
        and state_hash(other.anchor) == old_anchor
    )
    assert other.core.tasks == 0 and other.replay.seen == 0


def test_full_resident_refresh_matches_explicit_new_body_joint_moments():
    model = Learner(args(memory_size=100))
    manager = SyntheticTasks()
    for task in range(3):
        model.incremental_train(manager)
        seen = manager.get_dataset(range((task + 1) * 2), "train", "test")
        x, labels, _ = extract_features(model.backbone, seen, "cpu", 4)
        z = np.column_stack((x, np.ones(len(x))))
        y = np.eye((task + 1) * 2)[labels]
        # FP32 batch regrouping may perturb coordinates within the declared
        # encoder budget. No nonresident approximation is needed in this case.
        np.testing.assert_allclose(model.core.current_G, z.T @ z, rtol=3e-6, atol=1e-5)
        np.testing.assert_allclose(model.core.current_C, z.T @ y, rtol=3e-6, atol=1e-5)
        assert model.core.current_G[-1, -1] == len(seen)
        model.after_task()


def test_unsupported_stale_basis_warm_start_is_rejected():
    with pytest.raises(ValueError, match="basis_reset_each_task=True"):
        Learner(args(basis_reset_each_task=False))


def test_deployment_uses_a_nonzero_learned_basis_not_the_auxiliary_ridge_head():
    # Explicit synthetic-only noise setting exercises a nontrivial basis. A
    # large fixed noise can correctly collapse the constrained optimum to D=0.
    model = Learner(args(basis_noise_variance=0.01))
    manager = SyntheticTasks()
    model.incremental_train(manager)
    assert model.core.basis.effective_rank > 0
    assert not np.allclose(model.core.head, model.core.transport.head)
    probe = manager.get_dataset([0, 1], "test", "test")
    images = torch.stack([probe[i][1] for i in range(4)])
    with torch.no_grad():
        raw = model.backbone(images)
        features = torch.nn.functional.normalize(raw, dim=-1).double().numpy()
        deployed = model._network(images)["logits"].double().numpy()
    basis_scores = model.core.basis.predict_scores(features)
    shadow_scores = (
        np.column_stack((features, np.ones(len(features)))) @ model.core.transport.head
    )
    np.testing.assert_allclose(deployed, basis_scores, rtol=1e-6, atol=1e-6)
    assert not np.allclose(deployed, shadow_scores)
