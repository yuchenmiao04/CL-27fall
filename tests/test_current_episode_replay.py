"""The opt-in bounded old pool preserves alignment and history semantics."""

import copy

import numpy as np
import pytest
import torch

from cl27.common import TinyBackbone
from cl27.current.engine import FixedLibraryEngine
from models.current_adapted import Learner


def make_engine(budget=None, **options):
    torch.manual_seed(1993)

    def transform(image):
        return torch.from_numpy(image.copy()).permute(2, 0, 1).double() / 255

    core = FixedLibraryEngine(
        TinyBackbone(width=6, depth=3, image_size=16).double(),
        positions=(0, 1),
        count=2,
        rank=2,
        options={
            "seed": 1993,
            "support_shots": (1, 2),
            "background_shots": 2,
            "query_per_class": 1,
            "target_classes": 1,
            "feature_batch_size": 4,
            "episode_replay_batch_size": budget,
            **options,
        },
        canonical_transform=transform,
        training_transform=transform,
    )
    core.encoder.double()
    return core


def rows(task):
    rng = np.random.default_rng(1993 + task)
    images = rng.integers(0, 256, (12, 16, 16, 3), dtype=np.uint8)
    return [(image, 2 * task + i // 6, 12 * task + i) for i, image in enumerate(images)]


def fit(core, task, old=(), **kwargs):
    current = rows(task)
    candidates = list(old) + current
    # Keep four original rows per class, independent of episode selection.
    resident = [
        row
        for label in sorted({r[1] for r in candidates})
        for row in [r for r in candidates if r[1] == label][:4]
    ]
    event = core.fit_task(
        current,
        old,
        epochs=1,
        batch_size=6,
        seed=1993 + task,
        on_commit=lambda _features: [r[2] for r in resident],
        **kwargs,
    )
    return event, resident


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def small_cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.delenv("CL27_CURRENT_FROZEN_PREFIX_REUSE", raising=False)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_rejected(budget):
    with pytest.raises(ValueError, match="positive"):
        make_engine(budget)


@pytest.mark.parametrize("budget", [True, False, 1.5, "64"])
def test_noninteger_budget_rejected(budget):
    with pytest.raises(TypeError, match="integer or None"):
        make_engine(budget)


def test_budget_and_support_query_sized_class_groups_use_only_engine_rng():
    core = make_engine(64, support_shots=(1, 5), background_shots=5, query_per_class=2)
    labels = torch.arange(20).repeat_interleave(100)
    global_rng = torch.get_rng_state().clone()
    before = core.rng.get_state().clone()
    selected = core._sample_episode_replay_indices(labels)
    assert len(selected) == len(selected.unique()) == 64
    assert torch.equal(selected, selected.sort().values)
    _, counts = labels[selected].unique(return_counts=True)
    assert len(counts) == 9 and sorted(counts.tolist()) == [7] * 8 + [8]
    assert not torch.equal(before, core.rng.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())


@pytest.mark.parametrize("budget,n", [(None, 17), (64, 17), (17, 17), (64, 0)])
def test_full_or_empty_buffer_preserves_order_and_consumes_no_rng(budget, n):
    core = make_engine(budget)
    rng = core.rng.get_state().clone()
    selected = core._sample_episode_replay_indices(torch.arange(n))
    assert torch.equal(selected, torch.arange(n))
    assert torch.equal(rng, core.rng.get_state())


def test_sparse_capacity_expands_classes_and_redistributes_budget():
    core = make_engine(10)
    # Initial floor(10/3)=3 classes may have insufficient capacity.
    labels = torch.arange(12)
    selected = core._sample_episode_replay_indices(labels)
    assert len(selected) == len(selected.unique()) == 10
    assert len(labels[selected].unique()) == 10
    # Exhausted classes are skipped while unused capacity receives their quota.
    labels = torch.tensor([0] * 40 + [1] + [2] + [3])
    selected = core._sample_episode_replay_indices(labels)
    assert len(selected) == len(selected.unique()) == 10
    counts = torch.bincount(labels[selected], minlength=4)
    assert 7 <= counts[0] <= 8 and (counts[1:] <= 1).all()
    assert len(labels[selected].unique()) >= 3


def test_budget_below_full_group_remains_valid_and_nonempty():
    core = make_engine(1)
    labels = torch.arange(10).repeat_interleave(4)
    selected = core._sample_episode_replay_indices(labels)
    assert selected.shape == (1,) and 0 <= selected.item() < len(labels)


def test_sampler_is_reproducible_and_uses_different_old_groups_over_time():
    left, right = make_engine(8), make_engine(8)
    labels = torch.arange(12).repeat_interleave(20)
    groups = set()
    for _ in range(10):
        a = left._sample_episode_replay_indices(labels)
        b = right._sample_episode_replay_indices(labels)
        assert torch.equal(a, b)
        groups.add(tuple(labels[a].unique().tolist()))
    assert len(groups) > 1


def test_bounded_pool_alignment_and_full_history_and_final_readout(monkeypatch, capsys):
    core = make_engine(6)
    _, old = fit(core, 0)
    original_step = core.episode_step
    original_canonical = core._canonical
    episode_pools = []
    full_fit_ids = []

    def inspect(episode, items, canonical, labels, n_current, phase, **kwargs):
        assert labels.tolist() == [row[1] for row in items]
        assert len(items) == len(canonical) == n_current + 6
        for index, row in enumerate(items[n_current:], n_current):
            torch.testing.assert_close(
                canonical[index], core.memory_base[row[2]], rtol=0, atol=0
            )
        assert core.classes == [0, 1, 2, 3]  # No sampled-class output masking.
        episode_pools.append(tuple(row[2] for row in items[n_current:]))
        return original_step(
            episode, items, canonical, labels, n_current, phase, **kwargs
        )

    def inspect_final(items, anchor=False):
        if not anchor:
            full_fit_ids.extend(row[2] for row in items)
        return original_canonical(items, anchor)

    monkeypatch.setattr(core, "episode_step", inspect)
    monkeypatch.setattr(core, "_canonical", inspect_final)
    event, resident = fit(core, 1, old)
    assert len(episode_pools) == 8
    assert len(set(episode_pools[:4])) == len(set(episode_pools[4:])) == 1
    assert set(full_fit_ids) == {r[2] for r in rows(1) + old}
    assert len(full_fit_ids) == 12 + len(old)  # Full legal buffer for V, not six.
    assert core.statistics.count == core.statistics.G[-1, -1] == 24
    assert set(core.memory_base) == {r[2] for r in resident}
    all_rows = rows(0) + rows(1)
    expected = original_canonical(all_rows, True)
    torch.testing.assert_close(
        core.statistics.G, expected.T @ expected, rtol=1e-12, atol=1e-12
    )
    counters = event["epoch_summaries"][0]["episode_replay_sampling"]
    assert counters["old_rows_min"] == counters["old_rows_max"] == 6
    assert counters["old_rows_total"] == 12 and counters["minibatches"] == 2
    assert counters["combined_pool_rows_max"] == 12
    import json

    epochs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert epochs[-1]["episode_replay_sampling"] == counters


def test_none_path_never_samples_and_matches_full_budget_numerics_and_rng(monkeypatch):
    legacy, full = make_engine(), make_engine(64)

    def forbidden(_labels):
        raise AssertionError("Legacy path must not call the new sampler")

    monkeypatch.setattr(legacy, "_sample_episode_replay_indices", forbidden)
    old = []
    for task in range(3):
        left, resident = fit(legacy, task, old)
        right, other = fit(full, task, old)
        a, b = legacy.training_state(), full.training_state()
        a.pop("options")
        b.pop("options")
        assert_tree_equal(a, b)
        assert left["counters"] == right["counters"]
        assert (
            left["epoch_summaries"][0]["mean_query_ce"]
            == right["epoch_summaries"][0]["mean_query_ce"]
        )
        assert [r[2] for r in resident] == [r[2] for r in other]
        old = resident


def test_bounded_checkpoint_resume_exactly_reproduces_next_task():
    continuous = make_engine(6)
    _, old = fit(continuous, 0)
    # Checkpoint while bounded sampling really has consumed RNG, not just task0.
    _, old = fit(continuous, 1, old)
    checkpoint = continuous.training_state()
    restored = make_engine(6)
    restored.load_training_state(checkpoint)
    event, _ = fit(continuous, 2, old)
    resumed, _ = fit(restored, 2, old)
    assert_tree_equal(continuous.training_state(), restored.training_state())
    assert event == resumed


@pytest.mark.parametrize("stored_budget,target_budget", [(None, 6), (6, None), (6, 7)])
def test_changed_replay_budget_rejected_without_mutating_live_state(
    stored_budget, target_budget
):
    source = make_engine(stored_budget)
    fit(source, 0)
    target = make_engine(target_budget)
    before = target.training_state()
    with pytest.raises(ValueError, match="Episode replay budget changed"):
        target.load_training_state(source.training_state())
    assert_tree_equal(before, target.training_state())


def test_legacy_checkpoint_missing_option_only_loads_in_none_mode():
    source = make_engine()
    fit(source, 0)
    old_checkpoint = copy.deepcopy(source.training_state())
    old_checkpoint["options"].pop("episode_replay_batch_size")
    restored = make_engine()
    restored.load_training_state(old_checkpoint)
    assert_tree_equal(source.training_state(), restored.training_state())
    assert "episode_replay_batch_size" not in old_checkpoint["options"]
    with pytest.raises(ValueError, match="start a fresh bounded"):
        make_engine(6).load_training_state(old_checkpoint)


@pytest.mark.parametrize("budget", [None, 64])
def test_wrapper_exposes_distinct_episode_quota_and_keeps_ce_quota_separate(budget):
    learner = Learner(
        {
            "model_name": "current_adapted",
            "dataset": "synthetic",
            "smoke": True,
            "pretrained": False,
            "device": [torch.device("cpu")],
            "seed": 1993,
            "smoke_feature_dim": 6,
            "smoke_depth": 3,
            "image_size": 16,
            "module_layers": [0, 1],
            "module_rank": 2,
            "modules_per_layer": 2,
            "memory_size": 2000,
            "fixed_memory": False,
            "replay_selection": "herding",
            "replay_batch_size": 1,
            "episode_replay_batch_size": budget,
        }
    )
    assert learner.core.options["episode_replay_batch_size"] == budget
    assert "replay_batch_size" not in learner.core.options
    assert learner.memory.quota(20) == 100
