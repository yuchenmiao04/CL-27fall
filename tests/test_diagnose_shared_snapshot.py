"""Scientific read-only diagnostics, using small synthetic class streams."""

import json

import numpy as np
import pytest

from cl27.cores.shared_basis import SharedPredictorBasis
from scripts import diagnose_shared_snapshot as diagnostic


def stream():
    rng = np.random.default_rng(81)
    labels = np.repeat(np.arange(20), 8)
    means = rng.normal(size=(20, 5))
    features = means[labels] + 0.3 * rng.normal(size=(len(labels), 5))
    return {"fit_features": features, "fit_labels": labels}


def test_rank_zero_is_ridge_and_one_mapping_cannot_mutate_or_change_predictions():
    arrays = stream()
    x, y = arrays["fit_features"], arrays["fit_labels"]
    core = SharedPredictorBasis(5, rank=0, epsilon=0.1, noise_variance=0.2)
    core.observe(x, y)
    centered_x = x - x.mean(axis=0)
    targets = np.eye(20)[y]
    expected = np.linalg.solve(
        centered_x.T @ centered_x + 2.0 * np.eye(5),
        centered_x.T @ (targets - targets.mean(axis=0)),
    )
    np.testing.assert_allclose(core.weights, expected, rtol=1e-12, atol=1e-12)
    before = diagnostic.numeric_identity(core)
    report = diagnostic.diagnose_core(core, x, y, width=10)
    assert report["read_only_state_verified"]
    assert diagnostic.numeric_identity(core) == before
    assert report["isotropic_residual_ridge_lambda"] == 2.0
    assert report["one_em_map"]["covariance_relative_fro_change"] == 0
    assert report["one_em_map"]["posterior_mean_relative_fro_change"] == 0
    assert report["holdout"]["all"]["prediction_flips"] == 0


def test_reconstruction_preserves_original_warm_starts_and_em_step_is_non_decreasing():
    arrays = stream()
    params = {
        "basis_rank": 3,
        "basis_epsilon": 0.001,
        "basis_noise_variance": 0.001,
        "basis_max_iter": 2,
        "basis_tolerance": 1e-12,
    }
    reconstructed = diagnostic.reconstruct_core(arrays, params, width=10, stages=2)
    direct = SharedPredictorBasis(
        5, rank=3, epsilon=0.001, noise_variance=0.001,
        max_iter=2, tol=1e-12, seed=1993,
    )
    for task in range(2):
        chosen = (arrays["fit_labels"] // 10) == task
        direct.observe(arrays["fit_features"][chosen], arrays["fit_labels"][chosen])
    assert diagnostic.numeric_identity(direct) == diagnostic.numeric_identity(reconstructed)
    report = diagnostic.diagnose_core(
        reconstructed, arrays["fit_features"], arrays["fit_labels"], width=10
    )
    step = report["one_em_map"]
    assert step["marginal_log_likelihood_gain"] >= -1e-8
    assert step["q_first_unretained_eigenvalue"] is not None
    assert step["q_first_unretained_over_epsilon"] > 0
    assert report["holdout"]["all"]["samples"] == 160
    assert report["holdout"]["old"]["samples"] == 80
    assert report["holdout"]["current"]["samples"] == 80


def test_formal_trial_is_rejected_before_any_cache_access(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "parameters.json").write_text("{}")
    (trial / "results.json").write_text(json.dumps({"phase": "final"}))
    monkeypatch.setattr(
        diagnostic, "load_cache",
        lambda *args: pytest.fail("must reject a formal trial before cache access"),
    )
    with pytest.raises(ValueError, match="training-holdout"):
        diagnostic.run(
            cache_dir=tmp_path / "unread", dataset="cifar224", trial_dir=trial,
            output_json=tmp_path / "report.json", cpu_threads=1,
        )
    assert not (tmp_path / "report.json").exists()
