"""Small native PILOT lifecycle tests; synthetic data and random tiny ViT only."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from cl27.common import extract_features, seed_everything, state_hash
from models.history_transport import Learner
from utils.data_manager import DummyDataset


def image_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


class SyntheticTasks:
    nb_tasks = 3

    def __init__(self):
        rng = np.random.default_rng(88)
        self.labels = np.repeat(np.arange(6), 8)
        self.images = rng.integers(0, 255, size=(48, 16, 16, 3), dtype=np.uint8)
        # Distinctive spatial/color signals keep the training test interpretable.
        for i, label in enumerate(self.labels):
            self.images[i, :8, :, label % 3] = 30 + 35 * label
        self.requests = []

    def get_task_size(self, task):
        return 2

    def get_dataset(self, classes, source, mode):
        classes = np.asarray(classes)
        self.requests.append((tuple(map(int, classes)), source, mode))
        mask = np.isin(self.labels, classes)
        result = DummyDataset(self.images[mask], self.labels[mask], image_tensor)
        result.ids = np.flatnonzero(mask) + (1000 if source == "test" else 0)
        return result


def args(mode="direct", fixed=False):
    return {
        "model_name": "history_transport",
        "dataset": "synthetic",
        "smoke": True,
        "pretrained": False,
        "device": [torch.device("cpu")],
        "seed": 3,
        "smoke_feature_dim": 6,
        "smoke_depth": 1,
        "image_size": 16,
        "memory_size": 8,
        "fixed_memory": fixed,
        "memory_per_class": 2,
        "replay_selection": "herding",
        "transport_mode": mode,
        "residual_model": "class_conditional_psd",
        "head_ridge": 0.03,
        "epochs": 2,
        "batch_size": 8,
        "replay_batch_size": 3,
        "backbone_lr": 0.001,
        "head_lr": 0.01,
        "feature_batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 0,
        "init_cls": 2,
        "increment": 2,
        "normalization": "l2",
        "mean_shrinkage": 2.0,
        "crossfit_folds": 2,
    }


@pytest.fixture(autouse=True)
def deterministic_small_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_everything(5)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("mode", ["direct", "chained"])
@pytest.mark.parametrize("fixed", [False, True])
def test_native_plastic_training_once_only_statistics_and_replay_budget(mode, fixed):
    dm = SyntheticTasks()
    model = Learner(args(mode, fixed))
    anchor = state_hash(model.anchor)
    for task in range(3):
        before = state_hash(model.backbone)
        model.incremental_train(dm)
        assert state_hash(model.backbone) != before
        assert state_hash(model.anchor) == anchor
        assert all(not p.requires_grad for p in model.anchor.parameters())
        assert model.core.tasks == task + 1
        np.testing.assert_array_equal(model.core.counts, np.repeat(8, (task + 1) * 2))
        assert model.replay.seen == (task + 1) * 16
        quota = 2 if fixed else 8 // ((task + 1) * 2)
        assert len(model.replay) == quota * ((task + 1) * 2)
        assert max(r.label for r in model.replay.records) < (task + 1) * 2
        diag = model.task_diagnostics[-1]
        assert diag["training"]["current_training_visits"] == 32
        assert diag["training"]["optimizer_updates"] == 4
        assert diag["training"]["backbone_gradient_l2_max"] > 0
        assert diag["training"]["replay_training_visits"] == (0 if task == 0 else 12)
        assert not diag["transport_unbiasedness_claimed"]
        assert model.core.current_G[-1, -1] == pytest.approx((task + 1) * 16)
        assert np.linalg.eigvalsh(model.core.current_G).min() > -1e-8
        evaluated = dm.get_dataset(np.arange((task + 1) * 2), "test", "test")
        vectors, labels, _ = extract_features(model.backbone, evaluated, "cpu", 8)
        logits = model.core.predict_scores(vectors)
        assert logits.shape == (len(evaluated), (task + 1) * 2)
        assert np.isfinite(logits).all()
        if task == 0:
            # The once-only moments use the exact same dtype/device feature
            # normalization as the deployed scorer, without a second encoder run.
            augmented = np.column_stack((vectors, np.ones(len(vectors))))
            np.testing.assert_allclose(
                model.core.current_G, augmented.T @ augmented, atol=1e-12
            )
        assert diag["resources"]["inference_backbones"] == 1
        model.after_task()
    # Training data requests contain only the current class block; old images
    # are supplied by the bounded replay object, never by the data manager.
    assert [r[0] for r in dm.requests if r[1:] == ("train", "train")] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_task_boundary_checkpoint_resume_matches_next_task(mode):
    dm = SyntheticTasks()
    original = Learner(args(mode))
    original.incremental_train(dm)
    original.after_task()
    saved = copy.deepcopy(original.checkpoint_state())
    restored = Learner(args(mode))
    restored.restore_checkpoint(saved)
    probe = dm.get_dataset([0, 1], "test", "test")
    x = torch.stack([probe[i][1] for i in range(5)])
    torch.testing.assert_close(
        original._network(x)["logits"], restored._network(x)["logits"]
    )
    torch.manual_seed(101)
    original.incremental_train(dm)
    torch.manual_seed(101)
    restored.incremental_train(dm)
    assert state_hash(original.backbone) == state_hash(restored.backbone)
    assert state_hash(original.anchor) == state_hash(restored.anchor)
    np.testing.assert_allclose(
        original.core.current_G, restored.core.current_G, atol=1e-12
    )
    np.testing.assert_allclose(original.core.head, restored.core.head, atol=1e-12)
    assert [r.sample_id for r in original.replay.records] == [
        r.sample_id for r in restored.replay.records
    ]


def test_checkpoint_rejects_changed_anchor_or_replay_counts():
    dm = SyntheticTasks()
    model = Learner(args())
    model.incremental_train(dm)
    model.after_task()
    saved = model.checkpoint_state()
    wrong = copy.deepcopy(saved)
    key = next(iter(wrong["core"]["anchor"]))
    wrong["core"]["anchor"][key].view(-1)[0] += 0.1
    with pytest.raises(ValueError, match="anchor checkpoint hash"):
        Learner(args()).restore_checkpoint(wrong)
    wrong = copy.deepcopy(saved)
    wrong["core"]["replay"]["seen"] += 1
    with pytest.raises(ValueError, match="historical statistics disagree"):
        Learner(args()).restore_checkpoint(wrong)


def test_failed_restore_leaves_existing_body_statistics_and_replay_unchanged():
    dm = SyntheticTasks()
    model = Learner(args())
    model.incremental_train(dm)
    model.after_task()
    checkpoint = copy.deepcopy(model.checkpoint_state())
    model.incremental_train(dm)
    model.after_task()
    before = copy.deepcopy(model.checkpoint_state())
    wrong = copy.deepcopy(checkpoint)
    wrong["core"]["replay"]["seen"] += 1
    with pytest.raises(ValueError, match="historical statistics disagree"):
        model.restore_checkpoint(wrong)
    assert (
        state_hash(model.backbone)
        == model.task_diagnostics[-1]["backbone_sha256_after"]
    )
    assert model.core.tasks == before["core"]["analytic"]["tasks"] == 2
    np.testing.assert_array_equal(
        model.core.current_G, before["core"]["analytic"]["current_G"]
    )
    assert [r.sample_id for r in model.replay.records] == [
        r.sample_id for r in before["core"]["replay"]["records"]
    ]
    assert model._known_classes == 4 and model._cur_task == 1


def test_augmented_data_ids_must_equal_once_only_commit_ids():
    dm = SyntheticTasks()
    original_get = dm.get_dataset

    def wrong_get(classes, source, mode):
        data = original_get(classes, source, mode)
        if source == "train" and mode == "train":
            data.ids = data.ids[::-1].copy()
        return data

    dm.get_dataset = wrong_get
    model = Learner(args())
    with pytest.raises(ValueError, match="canonical task samples differ"):
        model.incremental_train(dm)
    assert model.core.tasks == 0 and model.replay.seen == 0


def test_conflicting_normalization_keys_are_rejected():
    settings = args()
    settings["feature_normalization"] = "none"
    with pytest.raises(ValueError):
        Learner(settings)


def test_anchor_change_is_rejected_before_task_or_image_access():
    dm = SyntheticTasks()
    model = Learner(args())
    with torch.no_grad():
        next(model.anchor.parameters()).view(-1)[0] += 0.1
    with pytest.raises(RuntimeError, match="before task access"):
        model.incremental_train(dm)
    assert dm.requests == [] and model.core.tasks == 0 and model._cur_task == -1


@pytest.mark.parametrize("field", ["initial_anchor_hash", "backbone_provenance"])
def test_checkpoint_must_match_trusted_initial_encoder_provenance(field):
    dm = SyntheticTasks()
    model = Learner(args())
    model.incremental_train(dm)
    model.after_task()
    saved = copy.deepcopy(model.checkpoint_state())
    if field == "initial_anchor_hash":
        saved["core"][field] = "untrusted"
    else:
        saved["core"][field]["initial_state_sha256"] = "untrusted"
    before = state_hash(model.backbone)
    with pytest.raises(ValueError, match="provenance differs"):
        model.restore_checkpoint(saved)
    assert state_hash(model.backbone) == before
