"""Training-only cache identity and incremental evaluation, synthetic fixtures."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from cl27.cores.shared_basis import SharedPredictorBasis
from cl27.data import split_training
from scripts import tune_frozen_basis as tuning


def checksum(value):
    return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()


def cache_fixture():
    """Small-dimensional artificial values with real protocol row counts."""
    original = np.repeat(np.arange(100), 500)
    fit, holdout = split_training(original)
    order = np.random.RandomState(1993).permutation(100)
    inverse = np.argsort(order)
    arrays = {}
    for prefix, ids in (("fit", fit), ("holdout", holdout)):
        labels = inverse[original[ids]]
        permutation = np.argsort(labels, kind="stable")
        labels, ids = labels[permutation], ids[permutation]
        arrays.update(
            {
                prefix + "_features": np.ones((len(ids), 1), dtype=np.float64),
                prefix + "_labels": labels.astype(np.int64),
                prefix + "_ids": ids.astype(np.int64),
            }
        )
    manifest = {
        "phase": "development",
        "official_test_read": False,
        "evaluation_source": "training_holdout",
        "train_count": len(fit),
        "evaluation_count": len(holdout),
        "fit_ids_sha256": checksum(fit),
        "evaluation_ids_sha256": checksum(holdout),
        "class_order": order.tolist(),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    metadata = {
        "cache_version": 1,
        "purpose": "training_holdout_development_only",
        "protocol": {
            "dataset": "cifar224",
            "seed": 1993,
            "split_seed": 0,
            "validation_fraction": 0.1,
            "class_count": 100,
            "increment": 10,
        },
        "data_manifest": manifest,
        "feature_map": {
            "normalization": "l2",
            "backbone_type": "vit_base_patch16_224",
            "feature_dim": 1,
            "frozen_state_sha256": "fixture-state",
            "backbone_provenance": {
                "kind": "pretrained",
                "timm_version": "0.6.12",
                "weight_file_sha256": "fixture-weight",
                "initial_state_sha256": "fixture-state",
            },
        },
        "execution": {"wall_seconds": 0.0},
    }
    return metadata, arrays


def test_cache_identity_checks_payload_coordinates_and_requested_dataset(tmp_path):
    metadata, arrays = cache_fixture()
    directory = tmp_path / "cache"
    saved = tuning.save_cache(directory, metadata, arrays)
    restored, rows = tuning.load_cache(directory, "cifar224")
    assert restored["cache_identity_sha256"] == saved["cache_identity_sha256"]
    np.testing.assert_array_equal(rows["fit_ids"], arrays["fit_ids"])
    with pytest.raises(ValueError, match="dataset differs"):
        tuning.load_cache(directory, "imagenetr")
    changed = copy.deepcopy(saved)
    changed["feature_map"]["normalization"] = "none"
    tuning.write_json(directory / "metadata.json", changed)
    with pytest.raises(ValueError, match="Cache identity differs"):
        tuning.load_cache(directory, "cifar224")
    tuning.write_json(directory / "metadata.json", saved)
    with (directory / "features.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="payload checksum differs"):
        tuning.load_cache(directory, "cifar224")


def test_rehashed_but_wrong_split_original_ids_are_rejected(tmp_path):
    metadata, arrays = cache_fixture()
    # Recompute payload checksums to test semantic ID checks, not only file hash.
    arrays["holdout_ids"][0] = arrays["fit_ids"][0]
    tuning.save_cache(tmp_path / "cache", metadata, arrays)
    with pytest.raises(ValueError, match="original IDs differ"):
        tuning.load_cache(tmp_path / "cache", "cifar224")


def test_cached_stream_equals_direct_core_and_never_fits_future_classes():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(100), 4)
    features = rng.normal(size=(400, 6))
    arrays = {
        "fit_features": features,
        "fit_labels": labels,
        "holdout_features": rng.normal(size=(200, 6)),
        "holdout_labels": np.repeat(np.arange(100), 2),
    }
    params = {
        "basis_rank": 3,
        "basis_epsilon": 0.1,
        "basis_noise_variance": 0.02,
        "basis_max_iter": 4,
    }
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        stages, diagnostics, metrics = tuning.evaluate_arrays(
            arrays, classes=100, width=10, parameters=params, max_tasks=2
        )
        direct = SharedPredictorBasis(
            6, rank=3, epsilon=0.1, noise_variance=0.02, max_iter=4, seed=1993
        )
        expected = []
        for task in range(2):
            new = (labels >= 10 * task) & (labels < 10 * (task + 1))
            direct.observe(features[new], labels[new])
            chosen = arrays["holdout_labels"] < 10 * (task + 1)
            scores = direct.predict_scores(arrays["holdout_features"][chosen])
            predictions = (
                torch.from_numpy(scores).float().topk(5, dim=1)[1][:, 0].numpy()
            )
            expected.append(
                float(np.mean(predictions == arrays["holdout_labels"][chosen]))
            )
    finally:
        torch.set_num_threads(original_threads)
    assert [row["accuracy"] for row in stages] == expected
    assert metrics["final_accuracy"] == expected[-1]
    assert metrics["aia"] == sum(expected) / 2
    assert [row["samples"] for row in diagnostics] == [40, 80]
    assert all(len(row["task_accuracy"]) == 10 for row in stages)
    assert stages[0]["task_counts"] == [20] + [0] * 9
    assert stages[1]["task_counts"] == [20, 20] + [0] * 8
    assert diagnostics[-1]["classes"] == 20
    with pytest.raises(ValueError, match="explicit basis parameter"):
        tuning.evaluate_arrays(
            arrays, classes=100, width=10, parameters={"basis_rank": [1, 2]}
        )


def test_extraction_config_forces_train_holdout_without_reducing_current_data():
    source = json.loads(
        (tuning.ROOT / "exps/pilot_figure2/shared_basis_cifar224.json").read_text()
    )
    config, classes, width = tuning.development_config(source)
    assert source["phase"] == "final"
    assert config["phase"] == "development" and config["validation_only"]
    assert config["validation_fraction"] == 0.1 and config["split_seed"] == 0
    assert config["download"] is False and config["seed"] == 1993
    assert (classes, width) == (100, 10)
    with pytest.raises(ValueError, match="complete development"):
        tuning.development_config(dict(source, validation_max_per_class=3))
    with pytest.raises(ValueError, match="only seed"):
        tuning.development_config(dict(source, seed=[0]))
