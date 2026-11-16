"""Mathematical and state-consistency checks for the shared predictor prior."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cl27.cores.shared_basis import SharedPredictorBasis, _posterior


def data(seed=3, classes=(8, 3, 21), per_class=13, dim=5):
    rng = np.random.default_rng(seed)
    labels = np.repeat(classes, per_class)
    features = rng.normal(size=(len(labels), dim)) + np.repeat(
        rng.normal(size=(len(classes), dim)), per_class, axis=0
    )
    return features, labels


def onehot(labels, classes):
    return np.asarray([[float(label == c) for c in classes] for label in labels])


def test_moments_match_explicit_joint_data_and_new_columns_are_zero_on_history():
    x, y = data()
    model = SharedPredictorBasis(5, rank=0, epsilon=0.2)
    first = y != 21
    model.observe(x[first], y[first])
    old_cross, old_gram = model.C, model.G
    model.observe(x[~first], y[~first])
    assert model.classes == [3, 8, 21]
    np.testing.assert_allclose(model.G, x.T @ x, atol=1e-12)
    np.testing.assert_allclose(model.C, x.T @ onehot(y, model.classes), atol=1e-12)
    np.testing.assert_array_equal(model.C[:, :2], old_cross)
    np.testing.assert_allclose(model.G - old_gram, x[~first].T @ x[~first], atol=1e-12)
    np.testing.assert_allclose(model.feature_sum, x.sum(0), atol=1e-12)
    assert model.n_samples == len(x)
    np.testing.assert_array_equal(model.counts, [13, 13, 13])


def test_flat_intercept_matches_augmented_explicit_generalized_ridge():
    x, y = data()
    model = SharedPredictorBasis(
        5, rank=2, epsilon=0.15, noise_variance=0.7, max_iter=12
    )
    model.observe(x, y)
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.zeros((6, 6))
    penalty[:5, :5] = model.noise_variance * np.linalg.inv(model.Sigma)
    expected = np.linalg.solve(
        design.T @ design + penalty, design.T @ onehot(y, model.classes)
    )
    np.testing.assert_allclose(model.weights, expected[:-1], atol=2e-12)
    np.testing.assert_allclose(model.intercept, expected[-1], atol=2e-12)
    np.testing.assert_allclose(model.predict_scores(x), design @ expected, atol=3e-12)


def test_flat_bias_is_feature_translation_invariant_and_scores_sum_to_one():
    x, y = data(classes=(2, 6, 11, 17), dim=7)
    shift = np.array([2.0, -3.0, 1.0, 0.0, 4.0, -0.7, 1.1])
    settings = dict(feature_dim=7, rank=2, epsilon=0.07, max_iter=20, tol=1e-10)
    first = SharedPredictorBasis(**settings)
    second = SharedPredictorBasis(**settings)
    first.observe(x, y)
    second.observe(x + shift, y)
    np.testing.assert_allclose(
        first.predict_scores(x), second.predict_scores(x + shift), atol=3e-12
    )
    np.testing.assert_allclose(first.predict_scores(x).sum(1), 1.0, atol=1e-13)


def test_posterior_matches_direct_precision_inverse():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(24, 6))
    center = x - x.mean(0)
    gram = center.T @ center
    cross = rng.normal(size=(6, 4))
    d = rng.normal(size=(6, 2))
    covariance = d @ d.T + 0.3 * np.eye(6)
    mean, s, logdet = _posterior(gram, cross, covariance, 0.4)
    expected = np.linalg.inv(gram / 0.4 + np.linalg.inv(covariance))
    np.testing.assert_allclose(s, expected, atol=2e-13)
    np.testing.assert_allclose(mean, expected @ cross / 0.4, atol=2e-13)
    assert logdet == pytest.approx(
        np.linalg.slogdet(np.eye(6) + covariance @ gram / 0.4)[1]
    )


def test_marginal_likelihood_matches_centered_observation_space_gaussian():
    x, y = data(per_class=4, dim=5)
    model = SharedPredictorBasis(5, rank=2, epsilon=0.2, noise_variance=0.7, max_iter=8)
    report = model.observe(x, y)
    # A QR basis for the complement of the constant vector makes the flat-bias
    # integration explicit and avoids a singular centered observation covariance.
    q, _ = np.linalg.qr(np.column_stack([np.ones(len(x)), np.eye(len(x))[:, 1:]]))
    contrast = q[:, 1:]
    design = contrast.T @ x
    target = contrast.T @ onehot(y, model.classes)
    covariance = 0.7 * np.eye(len(x) - 1) + design @ model.Sigma @ design.T
    expected = -0.5 * (
        len(model.classes)
        * ((len(x) - 1) * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1])
        + np.sum(target * np.linalg.solve(covariance, target))
    )
    assert report["marginal_log_likelihood"] == pytest.approx(expected, abs=2e-11)


def test_fixed_floor_covariance_and_monotone_constrained_em():
    x, y = data(classes=tuple(range(7)), per_class=15, dim=9)
    model = SharedPredictorBasis(
        9, rank=3, epsilon=0.03, noise_variance=0.08, max_iter=40, tol=1e-10
    )
    report = model.observe(x, y)
    eig = np.linalg.eigvalsh(model.Sigma)
    assert eig.min() >= 0.03 - 1e-12
    np.testing.assert_allclose(eig[:6], 0.03, atol=3e-12)
    np.testing.assert_allclose(model.Sigma, model.D @ model.D.T + 0.03 * np.eye(9))
    assert model.D.shape == (9, 3)
    assert np.min(np.diff(report["log_likelihood_history"])) >= -1e-9
    assert report["class_identified_rank_cap"] == 3
    for column in model.D.T:
        assert column[np.argmax(np.abs(column))] >= 0


def test_rank_is_limited_by_observed_class_contrasts_and_floor_subtracted():
    x, y = data(classes=(10, 40), per_class=20, dim=6)
    model = SharedPredictorBasis(6, rank=64, epsilon=0.12, max_iter=1)
    gc = (x - x.mean(0)).T @ (x - x.mean(0))
    target = onehot(y, [10, 40])
    cc = (x - x.mean(0)).T @ (target - target.mean(0))
    mean, posterior_covariance, _ = _posterior(gc, cc, 0.12 * np.eye(6), 1.0)
    q = posterior_covariance + mean @ mean.T / 2
    expected_energy = max(float(np.linalg.eigvalsh(q)[-1]) - 0.12, 0.0)
    report = model.observe(x, y)
    assert model.D.shape == (6, 1)
    assert report["class_identified_rank_cap"] == 1
    assert float(np.sum(model.D**2)) == pytest.approx(expected_energy, abs=1e-13)


def test_basis_can_freeze_while_novel_classes_update_posterior():
    x, y = data(classes=(2, 4, 6, 8), dim=7)
    old = y != 8
    model = SharedPredictorBasis(7, rank=2, epsilon=0.03, max_iter=30)
    model.observe(x[old], y[old])
    basis = model.D
    report = model.observe(x[~old], y[~old], update_basis=False)
    np.testing.assert_array_equal(model.D, basis)
    assert model.classes == [2, 4, 6, 8]
    assert report["basis_updated"] is False
    assert report["stop_reason"] == "basis_frozen"
    assert model.predict_scores(x).shape == (len(x), 4)


def test_observe_cannot_access_or_mutate_future_input_arrays():
    x, y = data(classes=(3, 7))
    x_copy, y_copy = x.copy(), y.copy()
    model = SharedPredictorBasis(5, rank=1)
    model.observe(x, y)
    scores = model.predict_scores(x)
    x[:] = -900
    y[:] = -900
    np.testing.assert_array_equal(model.predict_scores(x_copy), scores)
    np.testing.assert_array_equal(model.counts, [13, 13])
    assert model.classes == [3, 7]
    assert not any(
        np.shares_memory(a, x) for a in [model.G, model.C, model.feature_sum, model.D]
    )
    assert y_copy.tolist() != y.tolist()


def test_single_class_and_empty_predictions_are_well_defined():
    model = SharedPredictorBasis(3, rank=64)
    assert model.predict_scores(np.empty((0, 3))).shape == (0, 0)
    report = model.observe(np.array([[2.0, -1.0, 5.0]]), np.array([-7]))
    assert model.classes == [-7]
    assert report["class_identified_rank_cap"] == 0
    assert model.effective_rank == 0
    np.testing.assert_array_equal(model.weights, np.zeros((3, 1)))
    np.testing.assert_allclose(model.predict_scores(np.zeros((5, 3))), np.ones((5, 1)))


def test_code_only_control_refits_intercept_in_restricted_model():
    x, y = data(classes=(2, 4, 7, 8), dim=6)
    model = SharedPredictorBasis(
        6, rank=2, epsilon=0.01, noise_variance=0.2, max_iter=40
    )
    model.observe(x, y)
    projected = x @ model.D
    design = np.column_stack([projected, np.ones(len(x))])
    penalty = np.diag([0.2] * model.D.shape[1] + [0.0])
    expected = np.linalg.solve(
        design.T @ design + penalty, design.T @ onehot(y, model.classes)
    )
    np.testing.assert_allclose(model.code_only_scores(x), design @ expected, atol=1e-12)


def test_state_roundtrip_and_resume_equal_uninterrupted_fit():
    x, y = data(classes=(5, 9, 18, 23), dim=7)
    config = dict(
        feature_dim=7, rank=2, epsilon=0.03, noise_variance=0.4, max_iter=15, seed=12
    )
    model = SharedPredictorBasis(**config)
    old = y != 23
    model.observe(x[old], y[old])
    state = model.state_dict()
    restored = SharedPredictorBasis(**config)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.predict_scores(x), model.predict_scores(x))
    restored.observe(x[~old], y[~old])
    model.observe(x[~old], y[~old])
    np.testing.assert_array_equal(restored.predict_scores(x), model.predict_scores(x))
    assert restored.diagnostics == model.diagnostics
    state["G"][:] = 0
    assert np.any(restored.G)


def test_empty_state_roundtrip_numeric_payload_and_copy_properties():
    model = SharedPredictorBasis(4)
    copy_model = SharedPredictorBasis(4)
    copy_model.load_state_dict(model.state_dict())
    assert copy_model.classes == []
    assert model.resource_bytes() == 8 * (4 * 4 + 4)
    x, y = data(dim=4)
    model.observe(x, y)
    expected = sum(
        a.nbytes
        for a in [
            model.G,
            model.C,
            model.feature_sum,
            model.counts,
            model.D,
            model.weights,
            model.intercept,
        ]
    )
    assert model.resource_bytes() == expected
    external = model.G
    external[:] = 0
    assert np.any(model.G)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, 5), np.nan),
        np.full((2, 5), np.inf),
        np.ones((2, 6)),
        np.empty((0, 5)),
    ],
)
def test_invalid_features_are_rejected_without_committing(bad):
    model = SharedPredictorBasis(5)
    with pytest.raises(ValueError):
        model.observe(bad, np.array([1, 2]))
    assert model.n_samples == 0 and model.classes == []


def test_duplicate_classes_and_invalid_labels_leave_existing_state_unchanged():
    x, y = data(classes=(2, 4))
    model = SharedPredictorBasis(5)
    model.observe(x, y)
    old_scores = model.predict_scores(x)
    for bad in [y, np.asarray(y, dtype=float), np.ones(len(y), dtype=bool)]:
        with pytest.raises((ValueError, TypeError)):
            model.observe(x, bad)
        np.testing.assert_array_equal(model.predict_scores(x), old_scores)
    with pytest.raises(ValueError):
        model.observe(x, y[:-1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("epsilon", 0),
        ("noise_variance", float("nan")),
        ("rank", True),
        ("feature_dim", 0),
        ("max_iter", -1),
        ("tol", 0),
    ],
)
def test_invalid_constructor_settings(key, value):
    config = {"feature_dim": 5, key: value}
    with pytest.raises((ValueError, TypeError)):
        SharedPredictorBasis(**config)


def test_corrupted_checkpoint_is_rejected_atomically():
    x, y = data()
    model = SharedPredictorBasis(5, rank=2)
    model.observe(x, y)
    before = model.predict_scores(x)
    broken = copy.deepcopy(model.state_dict())
    broken["counts"][0] += 1
    with pytest.raises(ValueError):
        model.load_state_dict(broken)
    np.testing.assert_array_equal(model.predict_scores(x), before)
    broken = copy.deepcopy(model.state_dict())
    broken["C"][0, 0] += 1
    with pytest.raises(ValueError):
        model.load_state_dict(broken)
    broken = copy.deepcopy(model.state_dict())
    broken["C"][0, 0] += 1000
    broken["C"][0, 1] -= 1000
    with pytest.raises(ValueError, match="joint moments"):
        model.load_state_dict(broken)
    broken = copy.deepcopy(model.state_dict())
    broken["D"] = np.ones((5, 5))
    with pytest.raises(ValueError):
        model.load_state_dict(broken)
    other = SharedPredictorBasis(5, rank=1)
    with pytest.raises(ValueError):
        other.load_state_dict(model.state_dict())


def test_rotating_loadings_preserves_prior_posterior_and_predictions():
    x, y = data(classes=(0, 1, 2, 3), dim=6)
    config = dict(feature_dim=6, rank=2, epsilon=0.02, max_iter=30)
    model = SharedPredictorBasis(**config)
    model.observe(x, y)
    state = model.state_dict()
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    state["D"] = state["D"] @ rotation
    rotated = SharedPredictorBasis(**config)
    rotated.load_state_dict(state)
    np.testing.assert_allclose(rotated.Sigma, model.Sigma, atol=1e-13)
    np.testing.assert_allclose(
        rotated.predict_scores(x), model.predict_scores(x), atol=1e-12
    )
    # Individual latent coordinates change under the same model: units are not
    # semantically identified merely by canonicalizing eigenvector signs.
    np.testing.assert_allclose(
        rotated.class_codes, rotation.T @ model.class_codes, atol=1e-12
    )
