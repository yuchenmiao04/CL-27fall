"""Mathematical and causal contracts for the historical-transport core."""

import copy

import numpy as np
import pytest

from cl27.cores.transport import (
    TransportHistory,
    _augment,
    _conditional_moments,
    _moments,
)


def oracle(x, labels, classes, ridge=0.01):
    z = _augment(x)
    y = np.eye(len(classes))[[classes.index(int(c)) for c in labels]]
    penalty = np.diag(np.r_[np.ones(x.shape[1]), 0.0])
    return z.T @ z, z.T @ y, np.linalg.solve(z.T @ z + ridge * penalty, z.T @ y)


def affine(x, t):
    return (_augment(x) @ t)[:, :-1]


def transform(d, seed):
    rng = np.random.default_rng(seed)
    t = np.eye(d + 1)
    t[:-1, :-1] = np.eye(d) + rng.normal(scale=0.15, size=(d, d))
    t[-1, :-1] = rng.normal(scale=0.2, size=d)
    assert np.linalg.cond(t) < 100
    return t


@pytest.mark.parametrize("mode", ["direct", "chained"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_affine_transport_matches_all_history_oracle_with_shrinking_replay(
    mode, seed
):
    rng = np.random.default_rng(seed)
    d, per_class = 4, 30
    labels = np.repeat(np.arange(6), per_class)
    anchor = rng.normal(size=(len(labels), d)) + labels[:, None] * 0.1
    learner = TransportHistory(d, mode=mode, residual="none", map_ridge=0, ridge=0.01)
    resident = np.array([], dtype=int)
    previous = None
    for task in range(3):
        t = transform(d, seed * 10 + task + 1)
        current = np.flatnonzero((labels // 2) == task)
        args = {}
        if task:
            args = {
                "replay_anchor": anchor[resident],
                "replay_reference": affine(anchor[resident], previous),
                "replay_new": affine(anchor[resident], t),
                "replay_labels": labels[resident],
            }
        result = learner.update_task(
            anchor[current], affine(anchor[current], t), labels[current], **args
        )
        history = np.flatnonzero((labels // 2) <= task)
        g, c, w = oracle(affine(anchor[history], t), labels[history], learner.classes)
        np.testing.assert_allclose(learner.current_G, g, rtol=2e-11, atol=2e-10)
        np.testing.assert_allclose(learner.current_C, c, rtol=2e-11, atol=2e-10)
        np.testing.assert_allclose(learner.head, w, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(
            learner.predict_scores(affine(anchor[history], t)),
            _augment(affine(anchor[history], t)) @ w,
            atol=2e-10,
        )
        np.testing.assert_array_equal(
            learner.counts, np.repeat(per_class, (task + 1) * 2)
        )
        assert result["statistics_count"] == len(history)
        assert result["map_bias_column_error"] == 0
        ga, ca, _ = oracle(anchor[history], labels[history], learner.classes)
        np.testing.assert_allclose(learner.anchor_G, ga, atol=1e-10)
        np.testing.assert_allclose(learner.anchor_C, ca, atol=1e-10)
        # Old residents only shrink; new-class residents come from this task.
        resident = np.concatenate(
            [
                (
                    resident[labels[resident] == c]
                    if c < task * 2
                    else current[labels[current] == c]
                )[: 5 - task]
                for c in learner.classes
            ]
        )
        previous = t


def test_conditional_psd_reconstruction_includes_class_mean_outer_products_and_cross_terms():
    x = np.r_[np.tile([-1.0, 1.0], 25), np.tile([-3.0, 3.0], 25)][:, None]
    y = np.repeat([0, 1], 50)
    residual = x * x - 5.0
    current = x + residual
    g, c, n = _moments(_augment(x), y, [0, 1])
    reconstructed_g, reconstructed_c, check = _conditional_moments(
        g, c, n, np.eye(2), np.array([[-4.0], [4.0]]), np.zeros((1, 1))
    )
    exact_g, exact_c, _ = _moments(_augment(current), y, [0, 1])
    np.testing.assert_allclose(reconstructed_g, exact_g, atol=1e-10)
    np.testing.assert_allclose(reconstructed_c, exact_c, atol=1e-10)
    assert np.linalg.norm(g - exact_g) / np.linalg.norm(exact_g) > 0.25
    assert check["roundoff_clip_frobenius"] == 0


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_nonlinear_residual_estimator_has_coherent_psd_moments_and_counts(mode):
    rng = np.random.default_rng(9)
    d = 3
    labels = np.repeat(np.arange(6), 24)
    x = rng.normal(size=(len(labels), d)) + 0.2 * labels[:, None]
    learner = TransportHistory(d, mode=mode, crossfit_folds=3, seed=19)
    resident = np.array([], dtype=int)
    previous_features = x
    for task in range(3):
        current = np.flatnonzero(labels // 2 == task)
        new_features = x + (task + 1) * 0.1 * np.square(x)
        args = {}
        if task:
            args = {
                "replay_anchor": x[resident],
                "replay_reference": previous_features[resident],
                "replay_new": new_features[resident],
                "replay_labels": labels[resident],
            }
        result = learner.update_task(
            x[current], new_features[current], labels[current], **args
        )
        assert np.linalg.eigvalsh(learner.current_G).min() > -1e-8
        np.testing.assert_allclose(learner.current_G[-1, -1], (task + 1) * 48)
        np.testing.assert_allclose(learner.current_C[-1], learner.counts)
        np.testing.assert_allclose(learner.current_G[:, -1], learner.current_C.sum(1))
        means = learner.current_C.T / learner.counts[:, None]
        within = learner.current_G - (means.T * learner.counts) @ means
        assert np.linalg.eigvalsh(within).min() > -1e-8
        assert np.isfinite(learner.head).all()
        if task:
            assert result["crossfit_folds_used"] == 3
            assert result["map_bias_column_error"] == 0
            assert "not assumed" in result["replay_gradient_independence"]
        resident = np.concatenate(
            [
                (
                    resident[labels[resident] == c]
                    if c < task * 2
                    else current[labels[current] == c]
                )[: 6 - task]
                for c in learner.classes
            ]
        )
        previous_features = new_features


def test_first_task_exact_head_has_flat_bias_and_no_duplicate_arrival_statistics():
    x = np.zeros((8, 2))
    labels = np.array([4] * 2 + [7] * 6)
    learner = TransportHistory(2)
    learner.update_task(x, x, labels)
    np.testing.assert_allclose(learner.head[:-1], 0)
    np.testing.assert_allclose(learner.head[-1], [0.25, 0.75])
    np.testing.assert_allclose(
        learner.predict_scores(np.ones((3, 2))), [[0.25, 0.75]] * 3
    )
    before = learner.state_dict()
    with pytest.raises(ValueError, match="committed twice"):
        learner.update_task(x, x, labels)
    np.testing.assert_array_equal(learner.anchor_G, before["anchor_G"])
    assert learner.tasks == 1


def test_missing_replay_class_is_rejected_transactionally():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    labels = np.repeat([0, 1], 10)
    learner = TransportHistory(2)
    learner.update_task(x, x, labels)
    before = learner.state_dict()
    with pytest.raises(ValueError, match="No replay"):
        learner.update_task(
            x[:4],
            x[:4],
            np.repeat(2, 4),
            replay_anchor=x[:3],
            replay_new=x[:3],
            replay_labels=labels[:3],
        )
    assert learner.classes == [0, 1]
    np.testing.assert_array_equal(learner.current_G, before["current_G"])


def test_replay_overcount_and_future_labels_are_rejected():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 2))
    labels = np.repeat([0, 1], 6)
    learner = TransportHistory(2)
    learner.update_task(x, x, labels)
    for replay_y, message in [
        (np.repeat(0, 7), "exceeds"),
        (np.repeat(9, 7), "future"),
    ]:
        with pytest.raises(ValueError, match=message):
            learner.update_task(
                x[:2],
                x[:2],
                [2, 2],
                replay_anchor=np.tile(x[0], (7, 1)),
                replay_new=np.tile(x[0], (7, 1)),
                replay_labels=replay_y,
            )


def test_materially_invalid_chained_reference_is_not_silently_psd_projected():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 2))
    labels = np.repeat([0, 1], 10)
    learner = TransportHistory(2, mode="chained", reference_feature_rtol=1e-5)
    learner.update_task(x, x, labels)
    before = learner.state_dict()
    with pytest.raises(ValueError, match="non-PSD|Inconsistent"):
        learner.update_task(
            x[:2],
            x[:2],
            [2, 2],
            replay_reference=np.ones((2, 2)) * 100,
            replay_anchor=x[[0, 10]],
            replay_new=x[[0, 10]],
            replay_labels=[0, 1],
        )
    np.testing.assert_array_equal(learner.current_G, before["current_G"])


def test_checkpoint_roundtrip_preserves_next_update_and_numerical_state_payload():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(36, 3))
    labels = np.repeat(np.arange(3), 12)
    first = TransportHistory(3, seed=7, crossfit_folds=3)
    first.update_task(x[:12], x[:12], labels[:12])
    state = first.state_dict()
    clone = TransportHistory(3, seed=7, crossfit_folds=3)
    clone.load_state_dict(state)
    assert clone.resource_bytes() == first.resource_bytes()
    replay = [0, 1, 2, 3, 4, 5]
    args = dict(
        replay_anchor=x[replay],
        replay_new=x[replay] + 0.1 * x[replay] ** 2,
        replay_labels=labels[replay],
    )
    for obj in (first, clone):
        obj.update_task(x[12:24], x[12:24] + 0.1 * x[12:24] ** 2, labels[12:24], **args)
    np.testing.assert_array_equal(clone.head, first.head)
    np.testing.assert_array_equal(clone.current_G, first.current_G)
    assert clone.last_diagnostics == first.last_diagnostics
    state["anchor_G"][:] = 0
    assert np.linalg.norm(first.anchor_G) > 0


def test_small_per_class_replay_reports_in_sample_fallback_without_independence_claim():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(12, 2))
    labels = np.repeat([0, 1], 6)
    learner = TransportHistory(2, crossfit_folds=5)
    learner.update_task(x, x, labels)
    rows = [0, 6]
    result = learner.update_task(
        x[:4],
        x[:4] + 0.2,
        [2] * 4,
        replay_anchor=x[rows],
        replay_new=x[rows] + 0.2,
        replay_labels=labels[rows],
    )
    assert result["crossfit_folds_used"] == 1
    assert "insufficient" in result["crossfit_fallback"]
    assert result["covariance_unestimable_classes"] == [0, 1]
    assert result["residual_covariance_trace"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": True},
        {"feature_dim": 0},
        {"feature_dim": 2, "ridge": 0},
        {"feature_dim": 2, "map_ridge": -1},
        {"feature_dim": 2, "covariance_shrinkage": 1.1},
        {"feature_dim": 2, "crossfit_folds": 0},
        {"feature_dim": 2, "mode": "joint"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TransportHistory(**kwargs)


def test_malformed_inputs_and_checkpoint_are_rejected():
    learner = TransportHistory(2)
    with pytest.raises(ValueError, match="Nonfinite"):
        learner.update_task([[np.nan, 0]], [[0, 0]], [0])
    with pytest.raises(ValueError, match="integer"):
        learner.update_task([[0, 0]], [[0, 0]], [0.0])
    with pytest.raises(ValueError, match="together"):
        learner.update_task([[0, 0]], [[0, 0]], [0], replay_anchor=np.empty((0, 2)))
    with pytest.raises(ValueError, match="shape"):
        learner.update_task(
            [[0, 0]], [[0, 0]], [0], replay_new=np.empty((0, 3)), replay_labels=[]
        )
    learner.update_task([[0, 0], [1, 1]], [[0, 0], [1, 1]], [0, 0])
    state = learner.state_dict()
    state["head"][0, 0] += 1
    with pytest.raises(ValueError, match="head"):
        TransportHistory(2).load_state_dict(state)
    state = learner.state_dict()
    state["last_transport"][-1, -1] = 0
    with pytest.raises(ValueError, match="bias"):
        TransportHistory(2).load_state_dict(state)
    with pytest.raises(ValueError, match="configuration"):
        TransportHistory(2, ridge=1).load_state_dict(learner.state_dict())


def test_empty_untrained_state_roundtrip_and_prediction_shape():
    learner = TransportHistory(3)
    clone = TransportHistory(3)
    clone.load_state_dict(learner.state_dict())
    assert clone.predict_scores(np.empty((0, 3))).shape == (0, 0)
    assert clone.predict_scores(np.zeros((2, 3))).shape == (2, 0)


@pytest.mark.parametrize("mode", ["direct", "chained"])
def test_full_coverage_reencoded_residents_have_exact_zero_missing_population(mode):
    rng = np.random.default_rng(19)
    old = rng.normal(size=(20, 3))
    labels = np.repeat([0, 1], 10)
    reencoded = old + rng.normal(size=old.shape) * 2e-7
    current = rng.normal(size=(4, 3))
    learner = TransportHistory(3, mode=mode, reference_feature_rtol=1e-5)
    learner.update_task(old, old, labels)
    args = {
        "replay_anchor": reencoded,
        "replay_reference": reencoded,
        "replay_new": old + 0.2,
        "replay_labels": labels,
    }
    result = learner.update_task(current, current + 0.2, np.repeat(2, 4), **args)
    expected = np.r_[old + 0.2, current + 0.2]
    exact_g, exact_c, _ = oracle(
        expected, np.r_[labels, np.repeat(2, 4)], learner.classes
    )
    np.testing.assert_allclose(learner.current_G, exact_g, atol=1e-12)
    np.testing.assert_allclose(learner.current_C, exact_c, atol=1e-12)
    assert result["nonresident_historical_count"] == 0
    assert result["psd_checks"]["missing_reference"]["zero_population"]
    strict = TransportHistory(3, mode=mode)
    strict.update_task(old, old, labels)
    with pytest.raises(ValueError, match="reencoding tolerance"):
        strict.update_task(current, current + 0.2, np.repeat(2, 4), **args)


@pytest.mark.parametrize("residual", ["none", "class_conditional_psd"])
def test_partial_reencoding_budget_repairs_only_roundoff_with_coherent_means(residual):
    # Missing observations are exactly at each class mean, so their true
    # within-class scatter is zero. Reencoding the variable residents with
    # float32-scale perturbation makes direct subtraction slightly negative.
    old = np.r_[
        [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 4.0, 2.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    ][:, None]
    labels = np.repeat([0, 1], 10)
    resident = np.r_[np.arange(4), np.arange(10, 14)]
    class_mean = np.where(labels[resident] == 0, 0.0, 3.0)[:, None]
    reencoded = class_mean + (old[resident] - class_mean) * (1 + 2e-7)
    learner = TransportHistory(1, residual=residual, reference_feature_rtol=1e-5)
    learner.update_task(old, old, labels)
    current = np.array([[5.0], [6.0]])
    result = learner.update_task(
        current,
        current,
        [2, 2],
        replay_anchor=reencoded,
        replay_new=old[resident],
        replay_labels=labels[resident],
    )
    check = result["psd_checks"]["missing_reference"]["within_class_psd_check"]
    assert check["min_eigenvalue_before_clip"] < -1e-7
    assert check["absolute_roundoff_budget"] > abs(check["min_eigenvalue_before_clip"])
    assert check["roundoff_clip_frobenius"] > 0
    assert np.linalg.eigvalsh(learner.current_G).min() >= -1e-10
    np.testing.assert_allclose(
        learner.current_G[:, -1], learner.current_C.sum(1), atol=1e-12
    )
    assert learner.current_G[-1, -1] == 22


def test_encoder_precision_budget_does_not_hide_material_reference_mismatch():
    rng = np.random.default_rng(20)
    old = rng.normal(size=(20, 3))
    labels = np.repeat([0, 1], 10)
    model = TransportHistory(3, reference_feature_rtol=1e-5)
    model.update_task(old, old, labels)
    before = model.state_dict()
    with pytest.raises(ValueError, match="reencoding tolerance"):
        model.update_task(
            old[:2],
            old[:2],
            [2, 2],
            replay_anchor=old * 1.2,
            replay_new=old,
            replay_labels=labels,
        )
    np.testing.assert_array_equal(model.current_G, before["current_G"])
