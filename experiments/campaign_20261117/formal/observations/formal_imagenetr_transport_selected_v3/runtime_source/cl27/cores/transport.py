"""Approximate historical moments for a plastic feature map.

Inputs contain examples in rows and *exclude* the constant coordinate. The
module appends one exact, unpenalized intercept internally. Exact anchor moments
are updated once per new class-incremental task. Direct transport starts from
those moments; chained transport starts from the preceding approximate moments.

The optional residual model is class conditional. It transports within-class
scatter, reconstructs class means including residual mean shifts, and adds a
pooled positive-semidefinite residual covariance. Zero conditional cross
covariance is a modelling assumption, not a consequence of cross-fitting or
ridge regression. Replay need not be independent of encoder training, so the
estimator is not claimed unbiased. No discarded input or oracle feature is read.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {name}")
    return value


def _features(value, dim: int, name: str, *, empty: bool = False) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != dim or (not empty and not len(x)):
        raise ValueError(f"{name} must have shape [N, {dim}]")
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite {name}")
    return x


def _labels(value, n: int, name: str) -> np.ndarray:
    y = np.asarray(value)
    if y.ndim != 1 or len(y) != n:
        raise ValueError(f"{name} must have one label per example")
    if n and (not np.issubdtype(y.dtype, np.integer) or y.dtype == np.bool_):
        raise ValueError(f"{name} must contain integer labels")
    return y.astype(np.int64)


def _augment(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x, np.ones(len(x))))


def _moments(z: np.ndarray, y: np.ndarray, classes: list[int]):
    lookup = {c: j for j, c in enumerate(classes)}
    try:
        columns = np.array([lookup[int(c)] for c in y], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("A moment label is not in the registered vocabulary") from exc
    c = np.zeros((z.shape[1], len(classes)))
    if len(z):
        np.add.at(c.T, columns, z)
    counts = np.bincount(columns, minlength=len(classes)).astype(np.float64)
    return z.T @ z, c, counts


def _checked_psd(
    matrix: np.ndarray,
    name: str,
    tolerance: float = 1e-9,
    absolute_tolerance: float = 0.0,
):
    """Reject material violations; clip and report only rounding-scale negatives."""
    a = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(a)
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    minimum = float(values.min()) if values.size else 0.0
    if minimum < -(tolerance * scale + absolute_tolerance):
        raise ValueError(f"{name} is materially non-PSD: min_eigenvalue={minimum}")
    negative = np.minimum(values, 0.0)
    if np.any(negative):
        a = (vectors * np.maximum(values, 0.0)) @ vectors.T
        a = (a + a.T) * 0.5
    return a, {
        "min_eigenvalue_before_clip": minimum,
        "absolute_roundoff_budget": float(absolute_tolerance),
        "roundoff_clip_frobenius": float(np.linalg.norm(negative)),
        "negative_eigenvalues_clipped": int(np.count_nonzero(negative)),
    }


def _validate_moments(g, c, counts, name: str):
    """Check the feature/one-hot/bias aggregate identities."""
    if not all(np.isfinite(a).all() for a in (g, c, counts)) or (counts < -1e-8).any():
        raise ValueError(f"Nonfinite or negative {name} moments")
    scale = max(1.0, float(counts.sum()), float(np.linalg.norm(c)))
    atol = 1e-8 * scale
    if not np.allclose(g, g.T, rtol=1e-9, atol=atol):
        raise ValueError(f"Asymmetric {name} Gram")
    if (
        not np.allclose(c[-1], counts, rtol=1e-9, atol=atol)
        or not np.isclose(g[-1, -1], counts.sum(), rtol=1e-9, atol=atol)
        or not np.allclose(g[:, -1], c.sum(axis=1), rtol=1e-9, atol=atol)
    ):
        raise ValueError(f"Inconsistent {name} bias or class-count moments")


def _flat_bias_ridge(g: np.ndarray, c: np.ndarray, counts: np.ndarray, ridge: float):
    n = float(counts.sum())
    if n <= 0:
        return np.empty((g.shape[0], 0)), {
            "min_eigenvalue_before_clip": 0.0,
            "roundoff_clip_frobenius": 0.0,
            "negative_eigenvalues_clipped": 0,
        }
    sx = g[:-1, -1]
    centered = g[:-1, :-1] - np.outer(sx, sx) / n
    centered, check = _checked_psd(centered, "centered classifier Gram")
    rhs = c[:-1] - np.outer(sx, counts) / n
    slopes = np.linalg.solve(centered + ridge * np.eye(len(sx)), rhs)
    bias = counts / n - (sx / n) @ slopes
    return np.vstack((slopes, bias)), check


def _fit_map(z, target, weights, gamma):
    """Weighted affine map, identity prior on slopes and flat offset prior."""
    d = target.shape[1]
    root = np.sqrt(weights)
    design, response = z * root[:, None], target * root[:, None]
    rank = int(np.linalg.matrix_rank(design))
    if gamma == 0:
        slopes = np.linalg.lstsq(design, response, rcond=None)[0]
    else:
        penalty = np.diag(np.r_[np.ones(d), 0.0])
        prior = np.vstack((np.eye(d), np.zeros((1, d))))
        slopes = np.linalg.solve(
            design.T @ design + gamma * penalty, design.T @ response + gamma * prior
        )
    transform = np.column_stack((slopes, np.r_[np.zeros(d), 1.0]))
    return transform, rank


def _conditional_moments(g, c, counts, transform, shifts, covariance):
    """Coherent PSD conditional model on a non-resident historical population.

    ``shifts`` has one non-bias residual mean per class. The pooled covariance
    excludes bias. The conditional centered predictor/residual cross covariance
    is assumed zero. The returned C uses the same estimated means as G.
    """
    p, classes = c.shape
    means = np.zeros((classes, p))
    valid = counts > 0
    means[valid] = c[:, valid].T / counts[valid, None]
    between = (means.T * counts) @ means
    within = g[:-1, :-1] - between[:-1, :-1]
    within, check = _checked_psd(within, "missing within-class scatter")
    full_within = np.zeros((p, p))
    full_within[:-1, :-1] = within
    new_means = means @ transform
    new_means[:, :-1] += shifts
    new_means[valid, -1] = 1.0
    new_means[~valid] = 0.0
    extra = np.zeros((p, p))
    extra[:-1, :-1] = covariance * counts.sum()
    new_g = (
        transform.T @ full_within @ transform
        + (new_means.T * counts) @ new_means
        + extra
    )
    new_c = (new_means * counts[:, None]).T
    return (new_g + new_g.T) * 0.5, new_c, check


def _missing_moments(
    reference_g,
    reference_c,
    counts,
    resident_g,
    resident_c,
    resident_counts,
    resident_features,
    feature_rtol,
):
    """Subtract residents with an explicit, bounded encoder-reencoding budget.

    A float32 encoder can emit slightly different coordinates when images are
    rebatched. Float64 moment arithmetic cannot remove that input perturbation.
    The caller declares a relative feature-norm budget, separately from the
    strict float64 solver tolerance. Reconstructing scatter plus means keeps a
    rounding repair coherent with both class sums and the exact bias coordinate.
    """
    missing_counts = np.maximum(counts - resident_counts, 0)
    raw_g, raw_c = reference_g - resident_g, reference_c - resident_c
    reencoding_error = feature_rtol * float(np.linalg.norm(resident_features[:, :-1]))
    gram_budget = (
        2 * float(np.linalg.norm(resident_features)) * reencoding_error
        + reencoding_error**2
    )
    sum_budget = np.sqrt(len(resident_features)) * reencoding_error
    float64_g = 1e-9 * max(
        1.0, float(np.linalg.norm(reference_g)), float(np.linalg.norm(resident_g))
    )
    float64_c = 1e-9 * max(
        1.0, float(np.linalg.norm(reference_c)), float(np.linalg.norm(resident_c))
    )
    empty = missing_counts == 0
    # A zero-population class has exactly zero sum. This is stronger than
    # subtracting two differently batched float32 feature sums and retaining noise.
    if np.linalg.norm(raw_c[:, empty]) > sum_budget + float64_c:
        raise ValueError(
            "Inconsistent zero-count reference class sums exceed reencoding tolerance"
        )
    corrected_c = raw_c.copy()
    corrected_c[:, empty] = 0.0
    check = {
        "reference_feature_rtol": float(feature_rtol),
        "feature_error_frobenius_budget": float(reencoding_error),
        "gram_reencoding_absolute_budget": float(gram_budget),
        "class_sum_reencoding_absolute_budget": float(sum_budget),
        "subtraction_gram_frobenius": float(np.linalg.norm(raw_g)),
        "zero_count_class_sum_correction_frobenius": float(
            np.linalg.norm(raw_c[:, empty])
        ),
        "zero_population": not bool(missing_counts.sum()),
    }
    if not missing_counts.sum():
        if np.linalg.norm(raw_g) > gram_budget + float64_g:
            raise ValueError(
                "Inconsistent zero-population reference Gram exceeds reencoding tolerance"
            )
        return np.zeros_like(raw_g), np.zeros_like(raw_c), missing_counts, check
    means = np.zeros_like(corrected_c.T)
    present = missing_counts > 0
    means[present] = corrected_c[:, present].T / missing_counts[present, None]
    means[present, -1] = 1.0
    between = (means.T * missing_counts) @ means
    scatter = raw_g[:-1, :-1] - between[:-1, :-1]
    # Propagate the class-sum perturbation through the mean outer products.
    weighted_means = means[present, :-1] * np.sqrt(missing_counts[present, None])
    weighted_mean_error = sum_budget / np.sqrt(missing_counts[present].min())
    between_budget = (
        2 * float(np.linalg.norm(weighted_means)) * weighted_mean_error
        + weighted_mean_error**2
    )
    scatter, scatter_check = _checked_psd(
        scatter,
        "missing reference within-class scatter",
        absolute_tolerance=gram_budget + between_budget + float64_g,
    )
    result = between.copy()
    result[:-1, :-1] += scatter
    # Exact count/bias identities use the corrected missing class sums.
    corrected_c[-1] = missing_counts
    result[:, -1] = corrected_c.sum(axis=1)
    result[-1] = result[:, -1]
    check.update(
        {
            "within_class_psd_check": scatter_check,
            "mean_outer_product_reencoding_budget": float(between_budget),
            "gram_total_roundoff_correction_frobenius": float(
                np.linalg.norm(result - raw_g)
            ),
        }
    )
    return result, corrected_c, missing_counts, check


class TransportHistory:
    """Bounded-statistics direct or chained historical feature transport.

    The caller must supply the resident replay in its true reference/current
    coordinates. For chained transport, resident replay must be a subset of the
    preceding accessible set (previous replay plus previous current task), as
    is true for a shrinking class-balanced buffer. Subtracting arbitrary samples
    from approximate chained history is not identified and can fail PSD checks.

    ``resource_bytes`` counts numerical state payload only, not Python overhead
    or an encoder/buffer owned by the caller. ``last_transport`` and diagnostics
    are inspection state; prediction uses only the current classifier.
    """

    def __init__(
        self,
        feature_dim,
        mode="direct",
        residual="class_conditional_psd",
        ridge=0.01,
        map_ridge=0.01,
        covariance_shrinkage=0.1,
        mean_shrinkage=20.0,
        crossfit_folds=5,
        seed=0,
        reference_feature_rtol=0.0,
    ):
        self.feature_dim = _integer(feature_dim, "feature_dim")
        if mode not in {"direct", "chained"}:
            raise ValueError("mode must be direct or chained")
        if residual not in {"none", "class_conditional_psd"}:
            raise ValueError("Unknown residual model")
        self.mode, self.residual = mode, residual
        self.ridge = _scalar(ridge, "ridge", positive=True)
        self.map_ridge = _scalar(map_ridge, "map_ridge")
        self.covariance_shrinkage = _scalar(
            covariance_shrinkage, "covariance_shrinkage"
        )
        if self.covariance_shrinkage > 1:
            raise ValueError("covariance_shrinkage must lie in [0,1]")
        self.mean_shrinkage = _scalar(mean_shrinkage, "mean_shrinkage")
        self.crossfit_folds = _integer(crossfit_folds, "crossfit_folds")
        self.seed = _integer(seed, "seed", minimum=0)
        self.reference_feature_rtol = _scalar(
            reference_feature_rtol, "reference_feature_rtol"
        )
        p = self.feature_dim + 1
        self._classes: list[int] = []
        self.anchor_G, self.current_G = np.zeros((p, p)), np.zeros((p, p))
        self.anchor_C, self.current_C = np.empty((p, 0)), np.empty((p, 0))
        self.counts = np.empty(0)
        self.head = np.empty((p, 0))
        self.last_transport = np.eye(p)
        self.tasks = 0
        self.last_diagnostics: dict[str, Any] = {}

    @property
    def classes(self):
        return self._classes.copy()

    def _config(self):
        return {
            key: getattr(self, key)
            for key in (
                "feature_dim",
                "mode",
                "residual",
                "ridge",
                "map_ridge",
                "covariance_shrinkage",
                "mean_shrinkage",
                "crossfit_folds",
                "seed",
                "reference_feature_rtol",
            )
        }

    def _residual_estimates(
        self, reference, target, labels, weights, counts_missing, transform, old_classes
    ):
        n, d = target.shape
        active = weights > 0
        local_counts = {c: int(np.sum((labels == c) & active)) for c in old_classes}
        present = [m for m in local_counts.values() if m]
        folds = min(self.crossfit_folds, min(present)) if present else 1
        residuals = target - reference @ transform[:, :-1]
        fallback = None
        if folds >= 2:
            assignment = np.full(n, -1, dtype=np.int64)
            rng = np.random.default_rng(self.seed + self.tasks * 104729)
            for cls in old_classes:
                rows = np.flatnonzero((labels == cls) & active)
                rows = rng.permutation(rows)
                assignment[rows] = np.arange(len(rows)) % folds
            for fold in range(folds):
                train = active & (assignment != fold)
                test = active & (assignment == fold)
                fold_map, _ = _fit_map(
                    reference[train], target[train], weights[train], self.map_ridge
                )
                residuals[test] = target[test] - reference[test] @ fold_map[:, :-1]
        elif self.crossfit_folds > 1:
            fallback = "insufficient per-class replay; in-sample residuals used"
        shifts = np.zeros((len(old_classes), d))
        covariance = np.zeros((d, d))
        covariance_mass = 0.0
        unestimable = []
        factors = []
        for j, cls in enumerate(old_classes):
            rows = np.flatnonzero(labels == cls)
            m, missing = len(rows), float(counts_missing[j])
            factor = m / (m + self.mean_shrinkage) if m else 0.0
            factors.append(factor)
            if not m or missing <= 0:
                continue
            mean = residuals[rows].mean(axis=0)
            shifts[j] = factor * mean
            if m >= 2:
                centered = residuals[rows] - mean
                covariance += missing * (centered.T @ centered) / (m - 1)
                covariance_mass += missing
            else:
                unestimable.append(int(cls))
        if covariance_mass:
            covariance /= covariance_mass
        covariance = (
            1 - self.covariance_shrinkage
        ) * covariance + self.covariance_shrinkage * np.diag(np.diag(covariance))
        covariance, check = _checked_psd(covariance, "estimated residual covariance")
        return (
            shifts,
            covariance,
            {
                "crossfit_folds_requested": self.crossfit_folds,
                "crossfit_folds_used": folds,
                "crossfit_fallback": fallback,
                "residual_class_mean_shrink_factors": factors,
                "covariance_unestimable_classes": unestimable,
                "covariance_estimated_missing_mass": covariance_mass,
                "residual_covariance_trace": float(np.trace(covariance)),
                "residual_covariance_psd_check": check,
                "residual_estimation": (
                    "OOF affine residuals"
                    if folds >= 2
                    else "in-sample affine residuals"
                ),
                "replay_gradient_independence": "not assumed; resident replay may have trained the encoder",
            },
        )

    def update_task(
        self,
        current_anchor,
        current_new,
        current_labels,
        *,
        replay_anchor=None,
        replay_reference=None,
        replay_new=None,
        replay_labels=None,
    ):
        """Commit one new-class task; repeat gradient epochs outside this method.

        No state is changed if validation, identifiability or PSD checks fail.
        Missing classes cannot be estimated without resident replay. The method
        does not manufacture a zero-replay transport estimate from new classes.
        """
        d, p = self.feature_dim, self.feature_dim + 1
        xa = _features(current_anchor, d, "current_anchor")
        xt = _features(current_new, d, "current_new")
        y = _labels(current_labels, len(xa), "current_labels")
        if len(xt) != len(xa):
            raise ValueError("Current feature paths must have equal row counts")
        new_classes = sorted(set(map(int, y)))
        if set(new_classes).intersection(self._classes):
            raise ValueError(
                "Current labels must be new classes; a task cannot be committed twice"
            )
        old_classes = self._classes.copy()
        classes = old_classes + new_classes
        old_c, extra_c = len(old_classes), len(new_classes)
        ca = np.column_stack((self.anchor_C, np.zeros((p, extra_c))))
        source_c = np.column_stack((self.current_C, np.zeros((p, extra_c))))
        old_counts = np.r_[self.counts, np.zeros(extra_c)]
        ga_new, ca_new, new_counts = _moments(_augment(xa), y, classes)
        gt_new, ct_new, _ = _moments(_augment(xt), y, classes)
        transform, map_rank = np.eye(p), 0
        checks: dict[str, Any] = {}
        residual_info: dict[str, Any] = {"residual_estimation": "not required"}
        replay_count, missing_count = 0, 0.0

        provided = [replay_new is not None, replay_labels is not None]
        any_replay = any(
            value is not None
            for value in (replay_anchor, replay_reference, replay_new, replay_labels)
        )
        if any_replay and not all(provided):
            raise ValueError("Replay new features and labels must be supplied together")
        if old_c:
            if not all(provided):
                raise ValueError(
                    "Historical transport requires resident old-class replay"
                )
            rt = _features(replay_new, d, "replay_new", empty=True)
            ry = _labels(replay_labels, len(rt), "replay_labels")
            if not set(map(int, ry)).issubset(old_classes):
                raise ValueError("Replay contains future or current-task classes")
            ref_input = replay_anchor if self.mode == "direct" else replay_reference
            if ref_input is None:
                raise ValueError(
                    "The selected transport mode requires replay reference features"
                )
            rr = _features(ref_input, d, "replay_reference", empty=True)
            if len(rr) != len(rt):
                raise ValueError("Replay feature paths have unequal row counts")
            if replay_anchor is not None:
                ra = _features(replay_anchor, d, "replay_anchor", empty=True)
                if len(ra) != len(rt):
                    raise ValueError("Replay anchor rows do not match")
            if replay_reference is not None:
                rp = _features(replay_reference, d, "replay_reference", empty=True)
                if len(rp) != len(rt):
                    raise ValueError("Replay previous-reference rows do not match")
            z_ref, z_new = _augment(rr), _augment(rt)
            replay_g, replay_c, replay_counts = _moments(z_ref, ry, classes)
            new_replay_g, new_replay_c, _ = _moments(z_new, ry, classes)
            reference_g = self.anchor_G if self.mode == "direct" else self.current_G
            reference_c = ca if self.mode == "direct" else source_c
            missing_counts = old_counts - replay_counts
            if (missing_counts < -1e-8).any():
                raise ValueError("Replay count exceeds historical count")
            missing_g, missing_c, missing_counts, checks["missing_reference"] = (
                _missing_moments(
                    reference_g,
                    reference_c,
                    old_counts,
                    replay_g,
                    replay_c,
                    replay_counts,
                    z_ref,
                    self.reference_feature_rtol,
                )
            )
            _validate_moments(missing_g, missing_c, missing_counts, "missing reference")
            replay_count, missing_count = len(rt), float(missing_counts.sum())
            absent = [
                old_classes[j]
                for j in range(old_c)
                if missing_counts[j] > 0 and replay_counts[j] == 0
            ]
            if absent:
                raise ValueError(
                    f"No replay for non-resident historical classes: {absent}"
                )
            if missing_count:
                lookup = {c: j for j, c in enumerate(classes)}
                weight = np.array(
                    [
                        missing_counts[lookup[int(c)]] / replay_counts[lookup[int(c)]]
                        for c in ry
                    ]
                )
                active = weight > 0
                transform, map_rank = _fit_map(
                    z_ref[active], rt[active], weight[active], self.map_ridge
                )
                if self.residual == "none":
                    transported_g = transform.T @ missing_g @ transform
                    transported_c = transform.T @ missing_c
                else:
                    shifts, covariance, residual_info = self._residual_estimates(
                        z_ref,
                        rt,
                        ry,
                        weight,
                        missing_counts[:old_c],
                        transform,
                        old_classes,
                    )
                    shifts = np.vstack((shifts, np.zeros((extra_c, d))))
                    transported_g, transported_c, checks["missing_within_class"] = (
                        _conditional_moments(
                            missing_g,
                            missing_c,
                            missing_counts,
                            transform,
                            shifts,
                            covariance,
                        )
                    )
            else:
                transported_g = np.zeros((p, p))
                transported_c = np.zeros((p, len(classes)))
            next_g = transported_g + new_replay_g + gt_new
            next_c = transported_c + new_replay_c + ct_new
        else:
            if all(provided):
                first_replay = _features(replay_new, d, "replay_new", empty=True)
                first_labels = _labels(
                    replay_labels, len(first_replay), "replay_labels"
                )
                for name, value in (
                    ("replay_anchor", replay_anchor),
                    ("replay_reference", replay_reference),
                ):
                    if value is not None and len(
                        _features(value, d, name, empty=True)
                    ) != len(first_replay):
                        raise ValueError("Replay feature paths have unequal row counts")
                if len(first_labels):
                    raise ValueError("The first task cannot contain historical replay")
            next_g, next_c = gt_new, ct_new

        next_counts = old_counts + new_counts
        next_anchor_g, next_anchor_c = self.anchor_G + ga_new, ca + ca_new
        _validate_moments(next_anchor_g, next_anchor_c, next_counts, "anchor")
        _validate_moments(next_g, next_c, next_counts, "transported")
        _, checks["current_gram"] = _checked_psd(next_g, "current estimated Gram")
        head, checks["classifier_centered_gram"] = _flat_bias_ridge(
            next_g, next_c, next_counts, self.ridge
        )
        if not np.isfinite(head).all():
            raise FloatingPointError("Nonfinite transported classifier")
        diagnostics = {
            "task": self.tasks,
            "mode": self.mode,
            "residual": self.residual,
            "reference_moments": (
                "exact initial encoder"
                if self.mode == "direct"
                else "previous estimated encoder coordinates"
            ),
            "current_arrivals": len(xa),
            "historical_count_before": float(self.counts.sum()),
            "replay_items": replay_count,
            "nonresident_historical_count": missing_count,
            "statistics_count": float(next_counts.sum()),
            "classes": classes.copy(),
            "map_ridge": self.map_ridge,
            "weighted_reference_design_rank": map_rank,
            "map_bias_column_error": float(
                np.max(np.abs(transform[:, -1] - np.r_[np.zeros(d), 1.0]))
            ),
            "map_replay_r_squared": None,
            "psd_checks": checks,
            **residual_info,
            "assumptions": (
                [
                    "Missing within-class feature/residual cross covariance is modelled as zero, not proved by cross-fitting.",
                    "A pooled residual covariance approximates class-conditional uncertainty.",
                    "Replay can be model-dependent and is not asserted to be gradient-independent.",
                ]
                if self.residual != "none"
                else ["Non-resident nonlinear residual is ignored."]
            ),
        }
        if old_c and replay_count:
            observed = rt - rt.mean(axis=0)
            denominator = float(np.square(observed).sum())
            error = float(np.square(rt - z_ref @ transform[:, :-1]).sum())
            diagnostics["map_replay_r_squared"] = (
                1 - error / denominator if denominator > 0 else None
            )
        # Transactional commit: no persistent state changes before all checks pass.
        self.anchor_G, self.anchor_C = next_anchor_g, next_anchor_c
        self.current_G, self.current_C = (next_g + next_g.T) * 0.5, next_c
        self.counts, self._classes, self.head = next_counts, classes, head
        self.last_transport, self.last_diagnostics = transform, diagnostics
        self.tasks += 1
        return copy.deepcopy(diagnostics)

    def predict_scores(self, features):
        x = _features(features, self.feature_dim, "prediction features", empty=True)
        return _augment(x) @ self.head

    def resource_bytes(self):
        return int(
            sum(
                a.nbytes
                for a in (
                    self.anchor_G,
                    self.anchor_C,
                    self.current_G,
                    self.current_C,
                    self.counts,
                    self.head,
                    self.last_transport,
                )
            )
        )

    def state_dict(self):
        return {
            "version": 1,
            "config": self._config(),
            "classes": self.classes,
            "tasks": self.tasks,
            **{
                key: getattr(self, key).copy()
                for key in (
                    "anchor_G",
                    "anchor_C",
                    "current_G",
                    "current_C",
                    "counts",
                    "head",
                    "last_transport",
                )
            },
            "last_diagnostics": copy.deepcopy(self.last_diagnostics),
        }

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("config") != self._config():
            raise ValueError("Transport checkpoint configuration/version mismatch")
        classes = [
            _integer(c, "class label", minimum=np.iinfo(np.int64).min)
            for c in state["classes"]
        ]
        if len(classes) != len(set(classes)):
            raise ValueError("Duplicate checkpoint classes")
        p, k = self.feature_dim + 1, len(classes)
        expected = {
            "anchor_G": (p, p),
            "current_G": (p, p),
            "anchor_C": (p, k),
            "current_C": (p, k),
            "counts": (k,),
            "head": (p, k),
            "last_transport": (p, p),
        }
        arrays = {
            key: np.asarray(state[key], dtype=np.float64).copy() for key in expected
        }
        if any(
            a.shape != expected[key] or not np.isfinite(a).all()
            for key, a in arrays.items()
        ):
            raise ValueError("Malformed or nonfinite transport checkpoint arrays")
        tasks = _integer(state["tasks"], "tasks", minimum=0)
        if (
            bool(tasks) != bool(k)
            or tasks > k
            or (arrays["counts"] <= 0).any()
            or not np.allclose(
                arrays["counts"], np.round(arrays["counts"]), atol=1e-8, rtol=0
            )
        ):
            raise ValueError("Inconsistent checkpoint task/class counts")
        for name in ("anchor", "current"):
            _validate_moments(
                arrays[name + "_G"], arrays[name + "_C"], arrays["counts"], name
            )
            _checked_psd(arrays[name + "_G"], name + " checkpoint Gram")
            if k:
                _flat_bias_ridge(
                    arrays[name + "_G"],
                    arrays[name + "_C"],
                    arrays["counts"],
                    self.ridge,
                )
        if not np.allclose(
            arrays["last_transport"][:, -1], np.r_[np.zeros(p - 1), 1.0], atol=1e-10
        ):
            raise ValueError("Checkpoint transform does not preserve bias")
        if k:
            expected_head, _ = _flat_bias_ridge(
                arrays["current_G"], arrays["current_C"], arrays["counts"], self.ridge
            )
            if not np.allclose(arrays["head"], expected_head, atol=1e-9, rtol=1e-8):
                raise ValueError(
                    "Checkpoint head is inconsistent with its current moments"
                )
        for key, array in arrays.items():
            setattr(self, key, array)
        self._classes, self.tasks = classes, tasks
        self.last_diagnostics = copy.deepcopy(state.get("last_diagnostics", {}))
