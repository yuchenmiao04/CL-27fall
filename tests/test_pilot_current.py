"""Exact objective/lifecycle checks, using synthetic images and a native tiny ViT."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cl27.common import TinyBackbone, seed_everything, state_hash
from cl27.current.engine import FixedLibraryEngine, NativeDualEncoder
from cl27.current.episodes import Episode, sample_episode
from cl27.current.readout import ridge
from models.current_adapted import Learner, RawTransform
from utils.data_manager import DummyDataset


def image_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


class SyntheticTasks:
    nb_tasks = 3

    def __init__(self):
        rng = np.random.default_rng(71)
        self.labels = np.repeat(np.arange(6), 6)
        self.images = rng.integers(0, 255, (36, 16, 16, 3), dtype=np.uint8)
        self.requests = []

    def get_task_size(self, task):
        return 2

    def get_dataset(self, classes, source, mode):
        self.requests.append((tuple(map(int, classes)), source, mode))
        mask = np.isin(self.labels, classes)
        result = DummyDataset(self.images[mask], self.labels[mask], image_tensor)
        result.ids = np.flatnonzero(mask) + (1000 if source == "test" else 0)
        return result


def args(**overrides):
    return {
        "model_name": "current_adapted",
        "dataset": "synthetic",
        "smoke": True,
        "pretrained": False,
        "device": [torch.device("cpu")],
        "seed": 3,
        "smoke_feature_dim": 6,
        "smoke_depth": 3,
        "image_size": 16,
        "memory_size": 12,
        "memory_per_class": 2,
        "fixed_memory": False,
        "replay_selection": "herding",
        "epochs": 2,
        "batch_size": 12,
        "feature_batch_size": 3,
        "eval_batch_size": 12,
        "num_workers": 0,
        "init_cls": 2,
        "increment": 2,
        "module_layers": [0, 1],
        "modules_per_layer": 2,
        "module_rank": 2,
        "module_lr": 0.003,
        "backbone_lr": 0.001,
        "outer_updates": 4,
        "target_classes": 1,
        "support_shots": (1, 2),
        "background_shots": 2,
        "query_per_class": 1,
        **overrides,
    }


@pytest.fixture(autouse=True)
def small_deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_everything(29)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("fixed", [False, True])
def test_taskwise_exact_history_native_blocks_and_learning(fixed):
    dm = SyntheticTasks()
    model = Learner(args(fixed_memory=fixed))
    anchor = state_hash(model.core.encoder.anchor)
    modules = state_hash(model.core.encoder.libraries)
    assert isinstance(model.core.encoder.main, TinyBackbone)
    assert len(model.core.encoder.main.blocks[0]._forward_hooks) == 1
    assert len(model.core.encoder.main.blocks[1]._forward_hooks) == 1
    for task in range(3):
        before = state_hash(model.core.encoder.main)
        model.incremental_train(dm)
        count = (task + 1) * 12
        assert model.core.statistics.count == count
        assert model.core.statistics.G[-1, -1] == count
        assert model.core.classes == list(range((task + 1) * 2))
        assert state_hash(model.core.encoder.anchor) == anchor
        assert all(not p.requires_grad for p in model.core.encoder.anchor.parameters())
        assert state_hash(model.core.encoder.main) != before
        quota = 2 if fixed else 12 // ((task + 1) * 2)
        assert len(model.memory) == min(quota, 6) * (task + 1) * 2
        assert set(model.core.memory_base) == {
            r.sample_id for r in model.memory.records
        }
        diag = model.task_diagnostics[-1]
        assert diag["previous_history_count"] == task * 12
        assert diag["counters"]["logical_current_presentations"] == count * 2
        assert diag["module_updates_total"] > 0 and diag["backbone_updates_total"] > 0
        # The full ridge head is exactly joint ridge on *frozen* coordinates.
        all_rows = list(zip(dm.images[:count], dm.labels[:count], range(count)))
        z = model.core._canonical(all_rows, True)
        ys = torch.nn.functional.one_hot(
            torch.tensor(dm.labels[:count]), (task + 1) * 2
        ).double()
        torch.testing.assert_close(
            model.core.W0, ridge(z, ys, 0.01), atol=1e-11, rtol=1e-11
        )
        batch = next(iter(model.test_loader))[1]
        with torch.no_grad():
            logits = model._network(batch)["logits"]
        assert (
            logits.shape == (len(batch), (task + 1) * 2)
            and torch.isfinite(logits).all()
        )
        model.after_task()
    assert state_hash(model.core.encoder.libraries) != modules
    assert [r[0] for r in dm.requests if r[1:] == ("train", "train")] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert all(r[0] in [(0, 1), (2, 3), (4, 5)] for r in dm.requests if r[1] == "train")


def test_native_library_zero_initialization_and_token_effect():
    encoder = NativeDualEncoder(
        TinyBackbone(width=6, depth=3, image_size=16), (0, 1), 2, 2
    )
    x = torch.rand(3, 3, 16, 16)
    with torch.no_grad():
        # Adding a zero residual can change tensor strides / native kernel
        # dispatch, so identity is checked to FP32 arithmetic tolerance.
        torch.testing.assert_close(
            encoder.raw_main(x, True), encoder.anchor(x), atol=1e-6, rtol=1e-6
        )
        encoder.libraries[0].U.normal_(0, 0.4)
        assert not torch.allclose(encoder.raw_main(x, True), encoder.raw_main(x, False))
    assert all(not p.requires_grad for p in encoder.anchor.parameters())


def engine(cached=True, dtype=torch.float32):
    fn = RawTransform(lambda image: image_tensor(image).to(dtype=dtype))
    result = FixedLibraryEngine(
        TinyBackbone(width=6, depth=3, image_size=16).to(dtype=dtype),
        positions=(0, 1),
        count=2,
        rank=2,
        options={
            "memory_efficient": cached,
            "feature_batch_size": 3,
            "seed": 3,
            "support_shots": (1, 2),
            "target_classes": 1,
            "query_per_class": 1,
        },
        canonical_transform=fn,
        training_transform=fn,
    )
    result.encoder.to(dtype=dtype)
    return result


def prepare_episode(core):
    dm = SyntheticTasks()
    items = list(zip(dm.images[:12], dm.labels[:12], range(12)))
    labels = torch.tensor(dm.labels[:12])
    core.statistics.register(labels)
    canonical = core._canonical(items, True)
    episode = sample_episode(
        labels,
        torch.empty(0, dtype=torch.long),
        "transfer",
        torch.Generator().manual_seed(5),
        (1, 2),
        2,
        1,
        1,
    )
    return episode, items, canonical, labels


@pytest.mark.parametrize("phase", ["modules", "backbone"])
def test_feature_chain_vjp_equals_direct_objective_gradient(phase):
    # Compare the chain rule in FP64. Native CPU kernels may dispatch differently
    # under no_grad versus direct autograd; FP32 feature rounding is amplified
    # by a small ridge and does not define a different mathematical objective.
    cached = engine(True, torch.float64)
    direct = engine(False, torch.float64)
    # Nonzero modules exercise gate/V gradients as well as the zero-init U gradient.
    with torch.no_grad():
        for library in cached.encoder.libraries:
            library.U.normal_(0, 0.08)
    direct.encoder.load_state_dict(cached.encoder.state_dict())
    ec, items, z, labels = prepare_episode(cached)
    prepare_episode(direct)
    a = cached.episode_step(ec, items, z, labels, len(items), phase, step=False)
    b = direct.episode_step(ec, items, z, labels, len(items), phase, step=False)
    assert a["query_ce"] == pytest.approx(b["query_ce"], abs=1e-10, rel=1e-12)
    count = 0
    for (name, p), (other, q) in zip(
        cached.encoder.named_parameters(), direct.encoder.named_parameters()
    ):
        assert name == other
        if p.grad is None:
            assert q.grad is None
            continue
        count += 1
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-10, msg=name)
    assert count > 0 and a["gradient_l2"] > 0


def test_anchored_old_query_is_exactly_downdated_without_history_mutation():
    core = engine()
    dm = SyntheticTasks()
    old_items = list(zip(dm.images[:12], dm.labels[:12], range(12)))
    zold = core._canonical(old_items, True)
    oldy = torch.tensor(dm.labels[:12])
    core.statistics.add(zold, oldy)
    core.statistics.register([2, 3])
    new_items = list(zip(dm.images[12:24], dm.labels[12:24], range(12, 24)))
    znew = core._canonical(new_items, True)
    newy = torch.tensor(dm.labels[12:24])
    z = torch.cat((znew, zold))
    labels = torch.cat((newy, oldy))
    episode = Episode(
        "anchored",
        torch.tensor([0, 6, 12, 18]),
        torch.tensor([1, 13]),
        torch.tensor([7, 19]),
        (0, 2),
    )
    before = copy.deepcopy(core.statistics.state_dict())
    actual = core._temporary_base(episode, z, labels, 12, z[episode.support_indices])
    keep = torch.tensor([i for i in range(12) if i not in {1, 7}])
    expected_z = torch.cat((zold[keep], znew[[0, 6]]))
    expected_y = torch.cat((oldy[keep], newy[[0, 6]]))
    targets = torch.nn.functional.one_hot(expected_y, num_classes=4).double()
    torch.testing.assert_close(
        actual, ridge(expected_z, targets, 0.01), rtol=1e-10, atol=1e-10
    )
    after = core.statistics.state_dict()
    for name in ("G", "C"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert before["count"] == after["count"] == 12


def test_task_checkpoint_resume_matches_next_task():
    dm = SyntheticTasks()
    original = Learner(args())
    original.incremental_train(dm)
    original.after_task()
    checkpoint = original.checkpoint_state()
    resumed = Learner(args())
    resumed.restore_checkpoint(checkpoint)
    original.incremental_train(dm)
    resumed.incremental_train(SyntheticTasks())
    assert state_hash(original.core.encoder) == state_hash(resumed.core.encoder)
    for name in ("W0", "V"):
        torch.testing.assert_close(
            getattr(original.core, name), getattr(resumed.core, name), rtol=0, atol=0
        )
    assert original.task_diagnostics == resumed.task_diagnostics
    assert original.core.memory_base.keys() == resumed.core.memory_base.keys()
    # A saved state owns its CPU tensor storage.
    checkpoint["core"]["engine"]["W0"].zero_()
    assert torch.count_nonzero(resumed.core.W0) > 0


@pytest.mark.parametrize(
    "overrides", [{"module_layers": [1, 0]}, {"module_scale": 0.2}]
)
def test_checkpoint_binds_hook_positions_and_scale(overrides):
    original = Learner(args())
    original.incremental_train(SyntheticTasks())
    original.after_task()
    other = Learner(args(**overrides))
    with pytest.raises(ValueError, match="library architecture"):
        other.restore_checkpoint(original.checkpoint_state())


def test_current_normalization_is_explicitly_bound():
    with pytest.raises(ValueError, match="l2 feature normalization"):
        Learner(args(feature_normalization="none"))


def test_rejected_restore_keeps_live_encoder_history_and_replay():
    model = Learner(args())
    model.incremental_train(SyntheticTasks())
    model.after_task()
    before = model.checkpoint_state()
    corrupt = copy.deepcopy(before)
    corrupt["core"]["memory"]["records"].pop()
    corrupt["core"]["engine"]["W0"].zero_()
    weights = state_hash(model.core.encoder)
    with pytest.raises(ValueError, match="historical moments"):
        model.restore_checkpoint(corrupt)
    assert state_hash(model.core.encoder) == weights
    assert len(model.memory) == len(before["core"]["memory"]["records"])
    torch.testing.assert_close(
        model.core.W0, before["core"]["engine"]["W0"], rtol=0, atol=0
    )
