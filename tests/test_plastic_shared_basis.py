"""Replacement moments and composed plastic shared-prior mathematics."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cl27.cores.plastic_shared_basis import PlasticSharedPredictorBasis
from cl27.cores.shared_basis import SharedPredictorBasis


def snapshot(x, y, classes=None):
    classes = sorted(set(map(int, y))) if classes is None else classes
    target = np.asarray(
        [[label == c for c in classes] for label in y], dtype=np.float64
    )
    return dict(
        gram=x.T @ x,
        cross=x.T @ target,
        feature_sum=x.sum(0),
        counts=target.sum(0),
        n_samples=len(x),
        classes=classes,
    )


def fitting_data(seed=4, dim=4, classes=4, per_class=12):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(classes), per_class)
    centers = rng.normal(size=(classes, dim))
    x = centers[y] + rng.normal(scale=0.3, size=(len(y), dim))
    return x, y


def same_state(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            same_state(first, second)
    else:
        assert a == b


def test_replacement_matches_explicit_full_history_and_never_double_accumulates():
    x, y = fitting_data()
    options = dict(feature_dim=4, rank=2, epsilon=0.2, noise_variance=0.1, max_iter=15)
    model = SharedPredictorBasis(**options)
    model.observe(x[y < 2], y[y < 2])
    expected = SharedPredictorBasis(**options)
    expected.observe(x, y)
    result = model.fit_moments(
        **snapshot(x, y), tasks=2, moment_kind="exact_fixed_features"
    )
    assert result["snapshot_replaced"] and model.n_samples == len(x)
    np.testing.assert_array_equal(model.G, x.T @ x)
    np.testing.assert_array_equal(model.D, expected.D)
    # Matrix multiplication and per-class summation differ only by float64
    # reduction roundoff in C; both solve the same full-history regression.
    np.testing.assert_allclose(
        model.predict_scores(x), expected.predict_scores(x), atol=2e-14, rtol=0
    )
    previous = model.state_dict()
    model.fit_moments(**snapshot(x, y), tasks=2, moment_kind="exact_fixed_features")
    same_state(previous, model.state_dict())


def test_replacement_resets_stale_prior_and_reestimates_all_class_predictors():
    x, y = fitting_data()
    options = dict(
        feature_dim=4, rank=2, epsilon=0.05, noise_variance=0.02, max_iter=10
    )
    model = SharedPredictorBasis(**options)
    model.observe(x[y < 2], y[y < 2])
    stale = model.state_dict()
    stale["D"] = np.full((4, 1), 50.0)
    model.load_state_dict(stale)
    expected = SharedPredictorBasis(**options)
    expected.fit_moments(**snapshot(x, y), tasks=2)
    result = model.fit_moments(**snapshot(x, y), tasks=2)
    np.testing.assert_array_equal(model.D, expected.D)
    np.testing.assert_array_equal(model.weights, expected.weights)
    assert result["prior_initialization"].startswith("reset to isotropic")
    assert result["objective_interpretation"].startswith(
        "centered Gaussian moment surrogate"
    )
    assert (
        result["history"]
        == "caller-supplied surrogate full snapshot; true nonlinear history unknown"
    )


@pytest.mark.parametrize(
    "corruption", ["gram", "cross", "feature_sum", "counts", "classes", "tasks"]
)
def test_invalid_replacement_snapshot_is_rejected_without_mutation(corruption):
    x, y = fitting_data()
    model = SharedPredictorBasis(4, rank=2, epsilon=0.2, max_iter=3)
    model.observe(x[y < 2], y[y < 2])
    previous = model.state_dict()
    supplied = snapshot(x, y)
    supplied["tasks"] = 2
    if corruption == "gram":
        supplied["gram"][0, 1] += 2
    elif corruption == "cross":
        supplied["cross"][0, 0] += 1000
        supplied["cross"][0, 1] -= 1000  # Preserve sum, but destroy joint PSD.
    elif corruption == "feature_sum":
        supplied["feature_sum"][0] += 1
    elif corruption == "counts":
        supplied["counts"][0] += 0.25
    elif corruption == "classes":
        supplied["classes"][1] = supplied["classes"][0]
    else:
        supplied["tasks"] = 5
    with pytest.raises(ValueError):
        model.fit_moments(**supplied)
    same_state(previous, model.state_dict())


def test_population_surrogate_is_not_rejected_by_empirical_sample_rank_bound():
    supplied = dict(
        gram=np.eye(3),
        cross=np.zeros((3, 2)),
        feature_sum=np.zeros(3),
        counts=np.ones(2),
        n_samples=2,
        classes=[0, 1],
        tasks=1,
    )
    # Full-rank population scatter with N-C=0 cannot be a two-row labeled sample.
    model = SharedPredictorBasis(3, rank=1, max_iter=3)
    report = model.fit_moments(**supplied, moment_kind="surrogate")
    assert "surrogate" in report["objective_interpretation"]
    assert np.isfinite(model.predict_scores(np.eye(3))).all()
    with pytest.raises(ValueError, match="N-C sample-rank"):
        model.fit_moments(**supplied, moment_kind="exact_fixed_features")


def test_replacement_rejects_boolean_counts_and_complex_moments():
    supplied = dict(
        gram=np.eye(3),
        cross=np.zeros((3, 2)),
        feature_sum=np.zeros(3),
        counts=np.ones(2),
        n_samples=2,
        classes=[0, 1],
        tasks=1,
    )
    model = SharedPredictorBasis(3)
    with pytest.raises(ValueError, match="numeric integer"):
        model.fit_moments(**dict(supplied, counts=[True, True]))
    with pytest.raises(ValueError, match="real valued"):
        model.fit_moments(**dict(supplied, gram=np.eye(3) + 1j))


def test_affine_feature_pushforward_uses_inverse_coefficient_map_and_changes_floor():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(9, 3))
    a = np.diag([2.0, 1.0, 0.5])
    offset = np.array([0.3, -0.2, 0.4])
    weights, bias = rng.normal(size=(3, 2)), rng.normal(size=2)
    new_weights = np.linalg.solve(a, weights)
    new_bias = bias - offset @ new_weights
    np.testing.assert_allclose(
        (x @ a + offset) @ new_weights + new_bias, x @ weights + bias, atol=1e-14
    )
    epsilon = 0.2
    loading = np.array([[0.0], [1.0], [0.0]])
    covariance = epsilon * np.eye(3) + loading @ loading.T
    inverse = np.linalg.inv(a)
    pushed = inverse @ covariance @ inverse.T
    np.testing.assert_allclose(
        (x @ a) @ pushed @ (x @ a).T, x @ covariance @ x.T, atol=1e-14
    )
    # A fixed epsilon-I floor is not closed under nonorthogonal reparameterization.
    assert np.linalg.eigvalsh(pushed).min() < epsilon


def test_isotropic_reset_reports_sufficient_but_not_necessary_collapse_condition():
    x, y = fitting_data(dim=6, classes=10, per_class=3)
    model = SharedPredictorBasis(
        6, rank=4, epsilon=0.1, noise_variance=1.0, max_iter=10
    )
    report = model.fit_moments(**snapshot(x, y), tasks=1)
    assert report["reset_collapse_threshold_noise_variance"] == pytest.approx(0.3)
    assert report["reset_collapse_sufficient_condition"] is True
    assert model.effective_rank == 0
    # Failing the sufficient condition does not guarantee that a basis grows.
    model = SharedPredictorBasis(3, rank=1, epsilon=0.1, noise_variance=0.1, max_iter=5)
    report = model.fit_moments(
        np.eye(3), np.zeros((3, 2)), np.zeros(3), [5, 5], 10, [0, 1], 1
    )
    assert report["reset_collapse_sufficient_condition"] is False
    assert model.effective_rank == 0


def two_tasks():
    x, y = fitting_data(dim=3, per_class=10)
    first, second = y < 2, y >= 2
    # Partial residents still identify the global affine drift exactly.
    resident = np.array([0, 3, 6, 10, 13, 16])
    a = np.array([[1.1, 0.1, -0.2], [0.0, 0.8, 0.2], [0.1, -0.1, 1.2]])
    offset = np.array([0.2, -0.4, 0.1])
    task0 = (x[first], x[first], y[first])
    task1 = (x[second], x[second] @ a + offset, y[second])
    replay = dict(
        replay_anchor=x[resident],
        replay_reference=x[resident],
        replay_new=x[resident] @ a + offset,
        replay_labels=y[resident],
    )
    return x, y, a, offset, task0, task1, replay


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_composition_uses_reconstructed_full_moments_once_with_reset_prior(mode):
    x, y, a, offset, first, second, replay = two_tasks()
    model = PlasticSharedPredictorBasis(
        3,
        mode=mode,
        residual="none",
        map_ridge=0,
        basis_rank=2,
        basis_epsilon=0.2,
        basis_noise_variance=0.1,
        basis_max_iter=10,
    )
    model.update_task(*first)
    report = model.update_task(*second, **replay)
    final = x @ a + offset
    expected = snapshot(final, y)
    np.testing.assert_allclose(model.current_G[:-1, :-1], expected["gram"], atol=1e-10)
    np.testing.assert_allclose(model.current_C[:-1], expected["cross"], atol=1e-11)
    np.testing.assert_allclose(model.basis.G, expected["gram"], atol=1e-10)
    assert model.basis.n_samples == len(x) and model.tasks == 2
    assert model.head.shape == (4, 4)
    assert report["basis_reset_each_task"] is True
    assert report["anchor_history_exact"] and not report["predictive_history_exact"]
    assert report["statistics_count"] == len(x)
    assert report["shared_basis"]["samples"] == len(x)
    np.testing.assert_allclose(
        np.column_stack((final, np.ones(len(x)))) @ model.head,
        model.predict_scores(final),
        atol=1e-13,
    )
    assert np.diff(report["shared_basis"]["log_likelihood_history"]).min() >= -1e-9


def test_rank_zero_reduces_to_flat_bias_ridge_only_when_penalties_match():
    x, y, a, offset, first, second, replay = two_tasks()
    epsilon, noise = 0.2, 0.01
    model = PlasticSharedPredictorBasis(
        3,
        residual="none",
        map_ridge=0,
        ridge=noise / epsilon,
        basis_rank=0,
        basis_epsilon=epsilon,
        basis_noise_variance=noise,
    )
    model.update_task(*first)
    model.update_task(*second, **replay)
    np.testing.assert_allclose(model.head, model.transport.head, atol=1e-12)
    assert model.basis.effective_rank == 0
    unequal = PlasticSharedPredictorBasis(
        3,
        residual="none",
        map_ridge=0,
        ridge=2.0,
        basis_rank=0,
        basis_epsilon=epsilon,
        basis_noise_variance=noise,
    )
    unequal.update_task(*first)
    unequal.update_task(*second, **replay)
    assert not np.allclose(unequal.head, unequal.transport.head, atol=1e-4)


def test_composed_update_rolls_back_both_parts_when_shared_prior_fit_fails(monkeypatch):
    _, _, _, _, first, second, replay = two_tasks()
    model = PlasticSharedPredictorBasis(3, basis_rank=1, basis_max_iter=3)
    model.update_task(*first)
    previous = model.state_dict()

    def failure(*args, **kwargs):
        raise FloatingPointError("injected shared-prior failure")

    monkeypatch.setattr(SharedPredictorBasis, "fit_moments", failure)
    with pytest.raises(FloatingPointError, match="injected"):
        model.update_task(*second, **replay)
    same_state(model.state_dict(), previous)


def test_composed_checkpoint_and_resume_match_without_retaining_external_rows():
    x, _, _, _, first, second, replay = two_tasks()
    options = dict(
        feature_dim=3,
        residual="none",
        map_ridge=0,
        basis_rank=2,
        basis_max_iter=6,
        basis_epsilon=0.1,
    )
    model = PlasticSharedPredictorBasis(**options)
    model.update_task(*first)
    state = model.state_dict()
    resumed = PlasticSharedPredictorBasis(**options)
    resumed.load_state_dict(state)
    model.update_task(*second, **replay)
    resumed.update_task(*second, **replay)
    same_state(model.state_dict(), resumed.state_dict())
    assert (
        model.resource_bytes()
        == model.transport.resource_bytes() + model.basis.resource_bytes()
    )
    previous = model.predict_scores(x)
    first[0][:] = 900  # Caller data mutation cannot change stored history.
    np.testing.assert_array_equal(model.predict_scores(x), previous)


def test_composite_restore_rejects_disagreeing_valid_snapshots_atomically():
    _, _, _, _, first, _, _ = two_tasks()
    model = PlasticSharedPredictorBasis(3, basis_rank=1, basis_max_iter=3)
    model.update_task(*first)
    previous = model.state_dict()
    bad = copy.deepcopy(previous)
    bad["basis"]["G"] += np.eye(3)  # Valid PSD snapshot, but not transport's snapshot.
    with pytest.raises(ValueError, match="moment snapshots differ"):
        model.load_state_dict(bad)
    same_state(model.state_dict(), previous)
