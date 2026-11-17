"""The opt-in frozen-prefix path must preserve the existing episode update."""

import copy

import numpy as np
import pytest
import torch

from cl27.common import TinyBackbone
from cl27.current.engine import FixedLibraryEngine
from cl27.current.episodes import sample_episode
from models.current_adapted import RawTransform


def image_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def make_engine(kind="tiny", dtype=torch.float64, cached=True, positions=(0, 1)):
    torch.manual_seed(81)
    if kind == "timm":
        from timm.models.vision_transformer import VisionTransformer

        backbone = VisionTransformer(
            img_size=16,
            patch_size=8,
            embed_dim=12,
            depth=3,
            num_heads=3,
            mlp_ratio=2,
            num_classes=0,
        )
        backbone.out_dim = 12
    else:

        class OtherTiny(TinyBackbone):
            pass

        cls = TinyBackbone if kind == "tiny" else OtherTiny
        backbone = cls(width=12, depth=3, image_size=16)
    fn = RawTransform(lambda image: image_tensor(image).to(dtype=dtype))
    result = FixedLibraryEngine(
        backbone.to(dtype=dtype),
        positions=positions,
        count=2,
        rank=2,
        options={
            "memory_efficient": cached,
            "feature_batch_size": 3,
            "seed": 1993,
            "outer_updates": 4,
            "target_classes": 1,
            "support_shots": (1, 2),
            "background_shots": 2,
            "query_per_class": 1,
        },
        canonical_transform=fn,
        training_transform=fn,
    )
    result.encoder.to(dtype=dtype)
    with torch.no_grad():
        for library in result.encoder.libraries:
            library.U.normal_(0, 0.06)
    return result


def rows(start=0):
    rng = np.random.default_rng(71 + start)
    images = rng.integers(0, 255, (12, 16, 16, 3), dtype=np.uint8)
    return [(image, start // 6 + i // 6, start + i) for i, image in enumerate(images)]


def prepare(core, items):
    labels = torch.tensor([row[1] for row in items])
    core.statistics.register(labels)
    canonical = core._canonical(items, True)
    episode = sample_episode(
        labels,
        torch.empty(0, dtype=torch.long),
        "transfer",
        torch.Generator().manual_seed(1993),
        (1, 2),
        2,
        1,
        1,
    )
    return episode, canonical, labels


@pytest.fixture(autouse=True)
def small_cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.delenv("CL27_CURRENT_FROZEN_PREFIX_REUSE", raising=False)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ["tiny", "timm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("phase", ["modules", "backbone"])
@pytest.mark.parametrize("cached", [True, False])
def test_prefix_reuse_preserves_features_loss_gradients_and_updates(
    kind, dtype, phase, cached
):
    control = make_engine(kind, dtype, cached)
    fast = make_engine(kind, dtype, cached)
    fast.encoder.load_state_dict(control.encoder.state_dict())
    fast.frozen_prefix_reuse = True
    items = rows()
    ec, zc, y = prepare(control, items)
    ef, zf, yf = prepare(fast, items)
    torch.testing.assert_close(zc, zf, rtol=0, atol=0)
    replayed = []
    original = fast.encoder.main_features_from_prefix

    def counted(prefix, recipe):
        replayed.append(len(prefix))
        return original(prefix, recipe)

    fast.encoder.main_features_from_prefix = counted
    tolerance = 2e-6 if dtype == torch.float32 else 1e-10
    for _ in range(2):
        rng = torch.get_rng_state().clone()
        old = control.episode_step(ec, items, zc, y, len(items), phase)
        assert torch.equal(torch.get_rng_state(), rng)
        new = fast.episode_step(ef, items, zf, yf, len(items), phase)
        assert torch.equal(torch.get_rng_state(), rng)
        assert old.keys() == new.keys()
        assert old["query_ce"] == pytest.approx(
            new["query_ce"], rel=tolerance, abs=tolerance
        )
        assert old["gradient_l2"] == pytest.approx(
            new["gradient_l2"], rel=tolerance, abs=tolerance
        )
        for (name, p), (other, q) in zip(
            control.encoder.named_parameters(), fast.encoder.named_parameters()
        ):
            assert name == other
            torch.testing.assert_close(p, q, rtol=tolerance, atol=tolerance, msg=name)
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(
                    p.grad, q.grad, rtol=tolerance, atol=tolerance, msg=name
                )
        assert fast.encoder._prefix_capture is None
        assert fast.encoder._modules_active is False
    assert bool(replayed) == (phase == "modules" and cached)
    assert control.options == fast.options
    assert control.counters == fast.counters
    assert set(control.encoder.state_dict()) == set(fast.encoder.state_dict())


def test_prefix_split_uses_earliest_position_not_library_list_order():
    control = make_engine(positions=(1, 0))
    fast = make_engine(positions=(1, 0))
    fast.frozen_prefix_reuse = True
    items = rows()
    e, z, y = prepare(control, items)
    prepare(fast, items)
    old = control.episode_step(e, items, z, y, len(items), "modules")
    new = fast.episode_step(e, items, z, y, len(items), "modules")
    assert old["query_ce"] == pytest.approx(new["query_ce"], rel=1e-11, abs=1e-11)
    for a, b in zip(control.encoder.parameters(), fast.encoder.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind", ["tiny", "timm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_suffix_features_match_the_original_module_forward(kind, dtype):
    core = make_engine(kind, dtype)
    images = torch.stack([image_tensor(row[0]).to(dtype) for row in rows()[:3]])
    with core.phase("modules"):
        with torch.no_grad():
            reference_prepass = core.encoder.main_features(images, True)
            features, prefix = core.encoder.main_features_and_prefix(images)
        torch.testing.assert_close(features, reference_prepass, rtol=0, atol=0)
        reference = core.encoder.main_features(images, True)
        replay = core.encoder.main_features_from_prefix(
            prefix, core.encoder.frozen_prefix_kind()
        )
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(replay, reference, rtol=tolerance, atol=tolerance)
        assert replay.requires_grad


@pytest.mark.parametrize("kind", ["other", "hook"])
def test_unknown_native_behavior_falls_back(kind):
    core = make_engine("other" if kind == "other" else "tiny")
    core.frozen_prefix_reuse = True
    if kind == "hook":
        handle = core.encoder.main.patch_embed.register_forward_hook(lambda m, a, y: y)
    assert core.encoder.frozen_prefix_kind() is None

    def forbidden(*args):
        raise AssertionError("Unsupported models must use the original full path")

    core.encoder.main_features_from_prefix = forbidden
    items = rows()
    e, z, y = prepare(core, items)
    assert np.isfinite(
        core.episode_step(e, items, z, y, len(items), "modules")["query_ce"]
    )
    if kind == "hook":
        handle.remove()


def test_execution_flag_is_off_by_default_and_old_checkpoint_stays_loadable(
    monkeypatch,
):
    control = make_engine()
    assert control.frozen_prefix_reuse is False
    control.fit_task(rows(), [], epochs=1, batch_size=6, seed=1993)
    old = copy.deepcopy(control.training_state())
    assert (
        "frozen_prefix_reuse" not in old and "frozen_prefix_reuse" not in old["options"]
    )
    monkeypatch.setenv("CL27_CURRENT_FROZEN_PREFIX_REUSE", "1")
    restored = make_engine()
    restored.load_training_state(old)
    assert restored.frozen_prefix_reuse is True
    assert restored.options == control.options
    assert set(restored.training_state()) == set(old)
    control.fit_task(rows(12), [], epochs=1, batch_size=6, seed=1994)
    restored.fit_task(rows(12), [], epochs=1, batch_size=6, seed=1994)
    for a, b in zip(control.encoder.parameters(), restored.encoder.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    for field in ["W0", "V"]:
        torch.testing.assert_close(
            getattr(control, field), getattr(restored, field), rtol=1e-10, atol=1e-10
        )


def test_prefix_capture_cleans_up_after_an_exception():
    core = make_engine()
    images = torch.stack([image_tensor(row[0]).double() for row in rows()[:3]])
    original = core.encoder.raw_main

    def broken(*args):
        raise RuntimeError("intentional capture failure")

    core.encoder.raw_main = broken
    with core.phase("modules"), torch.no_grad(), pytest.raises(
        RuntimeError, match="intentional"
    ):
        core.encoder.main_features_and_prefix(images)
    assert core.encoder._prefix_capture is None
    core.encoder.raw_main = original


def test_a_main_update_between_capture_and_replay_is_rejected():
    core = make_engine()
    core.frozen_prefix_reuse = True
    items = rows()
    e, z, y = prepare(core, items)
    objective = core._episode_objective

    def mutate(*args):
        loss = objective(*args)
        with torch.no_grad():
            next(core.encoder.main.parameters()).add_(0.01)
        return loss

    core._episode_objective = mutate
    with pytest.raises(RuntimeError, match="main changed"):
        core.episode_step(e, items, z, y, len(items), "modules")
    assert core.encoder._prefix_capture is None


def test_unsupported_forward_or_checkpoint_recipe_is_not_selected():
    core = make_engine("timm")
    assert core.encoder.frozen_prefix_kind() == "timm_token"
    core.encoder.main.grad_checkpointing = True
    assert core.encoder.frozen_prefix_kind() is None
    core = make_engine()
    original = core.encoder.main.forward
    core.encoder.main.forward = lambda images: original(images)
    assert core.encoder.frozen_prefix_kind() is None


def test_global_forward_instrumentation_uses_the_original_path():
    core = make_engine()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda m, a, y: y)
    try:
        assert core.encoder.frozen_prefix_kind() is None
    finally:
        handle.remove()
