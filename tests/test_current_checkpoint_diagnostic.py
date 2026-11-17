"""Readout diagnostics must reconstruct the pre-shrink pool and stay read-only."""

import copy
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from cl27.protocol import smoke_config
from cl27.runner import run
from scripts.diagnose_current_checkpoint import (
    diagnose_checkpoint,
    fitting_pool,
    load_model_and_data,
    load_state,
    normalize_config,
)


@pytest.fixture
def current_pair(tmp_path):
    args = smoke_config("current_adapted", str(tmp_path / "stage0"))
    args.update(
        seed=1993,
        device=[-1],
        synthetic_per_class=6,
        image_size=16,
        smoke_feature_dim=6,
        smoke_depth=3,
        module_layers=[0, 1],
        modules_per_layer=2,
        module_rank=2,
        feature_batch_size=3,
        memory_size=4,
        epochs=1,
        batch_size=8,
        max_tasks=1,
        save_checkpoints=True,
        outer_updates=4,
        support_shots=[1, 2],
        background_shots=2,
        target_classes=1,
        query_per_class=1,
    )
    run(args)
    previous = tmp_path / "previous.pt"
    shutil.copyfile(tmp_path / "stage0/checkpoint.pt", previous)
    args.update(output_dir=str(tmp_path / "stage1"), resume=str(previous), max_tasks=2)
    result = run(args)
    return (
        tmp_path / "stage1/config.json",
        tmp_path / "stage1/checkpoint.pt",
        previous,
        result,
    )


def test_same_pool_common_refit_matches_deployed_full_and_preserves_rng(current_pair):
    config, checkpoint, previous, result = current_pair
    random.seed(61)
    np.random.seed(62)
    torch.manual_seed(63)
    py = random.getstate()
    numpy = np.random.get_state()
    tensor = torch.get_rng_state().clone()
    report = diagnose_checkpoint(config, checkpoint, previous, device=-1)
    assert random.getstate() == py
    now = np.random.get_state()
    assert (
        now[0] == numpy[0] and np.array_equal(now[1], numpy[1]) and now[2:] == numpy[2:]
    )
    assert torch.equal(torch.get_rng_state(), tensor)
    assert report["read_only_state_verified"]
    assert report["training_steps"] == 0 and report["official_test_read"] is False
    assert report["performance_claim"] is False
    assert report["fit_pool"]["pre_task_old_images"] == 4
    assert report["fit_pool"]["current_images"] == 10
    assert report["full_head_reconstruction"]["passed"]
    assert (
        report["branches"]["full_deployed"]["accuracy"]
        == result["stages"][-1]["accuracy"]
    )
    assert (
        report["branches"]["full_deployed"]["task_accuracy"]
        == result["stages"][-1]["task_accuracy"][:2]
    )
    assert (
        report["branches"]["full_deployed"]["task_counts"]
        == result["stages"][-1]["task_counts"][:2]
    )
    assert all(len(row["predictions"]) == 4 for row in report["branches"].values())


def test_second_stage_requires_actual_previous_pool(current_pair):
    config, checkpoint, previous, _ = current_pair
    with pytest.raises(ValueError, match="previous-stage checkpoint"):
        diagnose_checkpoint(config, checkpoint, device=-1)
    state, _ = load_state(checkpoint)
    args = normalize_config(json.loads(config.read_text()), -1)
    model, dm = load_model_and_data(args, state)
    # The post-task resident pool has only two old images after shrinkage.
    assert sum(record.label < 2 for record in model.memory.records) == 2
    before, _ = load_state(previous)
    current, old, cached = fitting_pool(model, dm, before)
    assert len(old) == 4 and len(current) == 10 and len(cached) == 4
    with pytest.raises(ValueError, match="immediately preceding"):
        fitting_pool(model, dm, state)


def test_formal_test_config_is_rejected_before_any_data_access(tmp_path):
    config = smoke_config("current_adapted", str(tmp_path / "unused"))
    config.update(phase="final", validation_only=False, seed=[1993])
    with pytest.raises(ValueError, match="internal-training-holdout"):
        normalize_config(config, -1)
