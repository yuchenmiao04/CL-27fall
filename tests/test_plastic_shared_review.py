"""Independent reference checks for the fourth route's statistical contracts."""

from __future__ import annotations

import numpy as np
import pytest

from cl27.cores.plastic_shared_basis import PlasticSharedPredictorBasis
from cl27.cores.shared_basis import SharedPredictorBasis


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_composed_missing_history_flat_intercept_handles_a_pure_translation(mode):
    rng = np.random.default_rng(381)
    y = np.repeat(np.arange(4), 9)
    x = rng.normal(size=(36, 5)) + y[:, None] * 0.3
    first = y < 2
    second = ~first
    resident = np.array([0, 4, 9, 13])
    shift = np.array([0.4, -0.2, 0.7, -0.3, 0.1])
    settings = dict(
        feature_dim=5,
        mode=mode,
        map_ridge=0.01,
        residual="class_conditional_psd",
        basis_rank=2,
        basis_epsilon=0.1,
        basis_noise_variance=0.05,
        basis_max_iter=10,
        mean_shrinkage=2.0,
        crossfit_folds=2,
    )
    unchanged = PlasticSharedPredictorBasis(**settings)
    translated = PlasticSharedPredictorBasis(**settings)
    for model in (unchanged, translated):
        model.update_task(x[first], x[first], y[first])
    unchanged.update_task(
        x[second],
        x[second],
        y[second],
        replay_anchor=x[resident],
        replay_reference=x[resident],
        replay_new=x[resident],
        replay_labels=y[resident],
    )
    result = translated.update_task(
        x[second],
        x[second] + shift,
        y[second],
        replay_anchor=x[resident],
        replay_reference=x[resident],
        replay_new=x[resident] + shift,
        replay_labels=y[resident],
    )
    assert result["nonresident_historical_count"] == 14
    # This exercises actual transport + snapshot replacement + prior refit,
    # rather than merely multiplying arbitrary coefficient matrices.
    np.testing.assert_allclose(
        translated.basis.Sigma, unchanged.basis.Sigma, atol=2e-11, rtol=2e-10
    )
    np.testing.assert_allclose(
        translated.basis.weights, unchanged.basis.weights, atol=2e-11, rtol=2e-10
    )
    np.testing.assert_allclose(
        translated.basis.intercept,
        unchanged.basis.intercept - shift @ unchanged.basis.weights,
        atol=2e-11,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        translated.predict_scores(x + shift),
        unchanged.predict_scores(x),
        atol=2e-11,
        rtol=2e-10,
    )


def coherent_population_snapshot(seed=5):
    rng = np.random.default_rng(seed)
    d, k, n_per = 9, 3, 2
    means = rng.normal(scale=0.4, size=(k, d))
    means -= means.mean(0)
    counts = np.repeat(n_per, k)
    cross = means.T * counts
    within = 0.2 * np.eye(d)
    gram = within + (means.T * counts) @ means
    return dict(
        gram=gram,
        cross=cross,
        feature_sum=np.zeros(d),
        counts=counts,
        n_samples=int(counts.sum()),
        classes=list(range(k)),
        tasks=1,
    )


def test_surrogate_evidence_matches_factorized_joint_gaussian_reference():
    data = coherent_population_snapshot()
    d, k = 9, 3
    model = SharedPredictorBasis(d, rank=2, epsilon=0.2, noise_variance=0.1, max_iter=7)
    report = model.fit_moments(**data, moment_kind="surrogate")
    counts = data["counts"]
    n = data["n_samples"]
    target_gram = np.diag(counts) - np.outer(counts, counts) / n
    joint = np.block([[data["gram"], data["cross"]], [data["cross"].T, target_gram]])
    values, vectors = np.linalg.eigh(joint)
    keep = values > 1e-11
    # This joint moment is valid, yet its full-rank within-class scatter cannot
    # arise from N=6 actual labeled rows (N-C=3). A factorization gives an exact
    # Gaussian pseudo-observation interpretation of the surrogate objective.
    factor = np.sqrt(values[keep])[:, None] * vectors[:, keep].T
    design, target = factor[:, :d], factor[:, d:]
    rows = len(factor)
    assert rows > n - 1
    covariance = model.noise_variance * np.eye(rows) + design @ model.Sigma @ design.T
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    reference = -0.5 * (
        k * rows * np.log(2 * np.pi)
        + k * logdet
        + np.sum(target * np.linalg.solve(covariance, target))
    )
    # The implementation keeps the nominal N-1 normalization. Its difference
    # from this realized pseudo-design is constant in Sigma for fixed noise.
    correction = -0.5 * k * (n - 1 - rows) * np.log(2 * np.pi * model.noise_variance)
    assert report["marginal_log_likelihood"] == pytest.approx(
        reference + correction, abs=1e-10
    )
    assert np.diff(report["log_likelihood_history"]).min() >= -1e-10


@pytest.mark.parametrize("seed", [5, 13])
def test_sufficient_noise_collapse_bound_also_holds_for_population_proxy(seed):
    data = coherent_population_snapshot(seed)
    d, k = 9, 3
    epsilon = 0.2
    noise = float(data["counts"].max()) / k
    g, c = data["gram"], data["cross"]
    posterior = np.linalg.solve(g / noise + np.eye(d) / epsilon, np.eye(d))
    mean = posterior @ c / noise
    q = posterior + mean @ mean.T / k
    bound = (
        (epsilon / noise) * (g @ posterior) * (data["counts"].max() / (k * noise) - 1.0)
    )
    assert np.linalg.eigvalsh(bound - (q - epsilon * np.eye(d))).min() > -1e-12
    model = SharedPredictorBasis(
        d, rank=2, epsilon=epsilon, noise_variance=noise, max_iter=4
    )
    report = model.fit_moments(**data, moment_kind="surrogate")
    assert report["reset_collapse_sufficient_condition"] and model.effective_rank == 0
