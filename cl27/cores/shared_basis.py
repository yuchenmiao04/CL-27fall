"""A shared predictor-space prior learned from exact frozen-feature moments.

The caller supplies feature rows without an appended bias coordinate. The feature
map (including any normalization) must remain fixed across ``observe`` calls.
Only G = X.T X, C = X.T Y, feature sums, class counts and N survive a task; raw
historical features are never retained. An intercept with a flat prior is removed
by centering X and one-hot Y through these moments, then recovered analytically.

For centered moments Gc and Cc, independent class weights follow
    w_c ~ Normal(0, Sigma),  Sigma = D D.T + epsilon I,
with fixed observation noise variance s2. Their posterior is
    S = (Gc / s2 + Sigma^-1)^-1,   M = S Cc / s2.
The fixed-epsilon, rank-constrained covariance EM update diagonalizes
    Q = S + M M.T / number_of_classes
and retains at most K eigenvalues above epsilon. Loadings are
sqrt(max(q - epsilon, 0)), rather than sqrt(q). Epsilon and s2 are constructor
hyperparameters; this implementation does not tune them using future data.

These are Gaussian regression equations applied to one-hot classification
targets, a misspecified likelihood. A shared subspace and its canonical eigenbasis
are possible knowledge-sharing objects; basis columns are not uniquely identified
semantic units (especially at repeated eigenvalues). No accuracy, interpretability,
or transfer guarantee follows from an EM likelihood improvement.
"""

from __future__ import annotations

import copy
import math
import operator
from typing import Any

import numpy as np


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _symmetric(value: np.ndarray) -> np.ndarray:
    return (value + value.T) * 0.5


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result


def _canonical_columns(vectors: np.ndarray) -> np.ndarray:
    """Fix eigenvector signs; rotations inside repeated eigenspaces remain free."""
    result = vectors.copy()
    if result.shape[1]:
        pivots = np.argmax(np.abs(result), axis=0)
        signs = np.where(result[pivots, np.arange(result.shape[1])] < 0, -1.0, 1.0)
        result *= signs[None, :]
    return result


def _posterior(
    gram: np.ndarray, cross: np.ndarray, covariance: np.ndarray, noise_variance: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cholesky form avoids explicitly inverting a prior covariance."""
    factor = np.linalg.cholesky(_symmetric(covariance))
    system = _symmetric(np.eye(len(gram)) + factor.T @ gram @ factor / noise_variance)
    chol = np.linalg.cholesky(system)
    left = np.linalg.solve(chol, factor.T)
    posterior_covariance = _symmetric(left.T @ left)
    mean = posterior_covariance @ cross / noise_variance
    logdet = float(2.0 * np.log(np.diag(chol)).sum())
    if not np.isfinite(mean).all() or not np.isfinite(posterior_covariance).all():
        raise FloatingPointError("Nonfinite predictor posterior")
    return mean, posterior_covariance, logdet


def _log_likelihood(
    gram: np.ndarray,
    cross: np.ndarray,
    covariance: np.ndarray,
    noise_variance: float,
    target_scatter_trace: float,
    residual_rows: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    mean, posterior_covariance, logdet = _posterior(
        gram, cross, covariance, noise_variance
    )
    classes = cross.shape[1]
    quadratic = (target_scatter_trace - float(np.sum(cross * mean))) / noise_variance
    value = -0.5 * (
        classes * (residual_rows * math.log(2.0 * math.pi * noise_variance) + logdet)
        + quadratic
    )
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite centered marginal log likelihood")
    return value, mean, posterior_covariance


class SharedPredictorBasis:
    """Task-wise Class-IL learner on a fixed feature map.

    ``observe`` receives all fitting rows for a *new set of classes*. Repeated
    epochs must not call it again: every historical row contributes once. To test
    reuse without adapting the basis, call ``observe(..., update_basis=False)``;
    only sufficient statistics and posterior class predictors then change.

    ``rank=0`` is an isotropic-prior control. ``max_iter=0`` also skips covariance
    fitting. Numeric payload accounting excludes Python object overhead.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        feature_dim: int,
        rank: int = 32,
        epsilon: float = 1e-3,
        noise_variance: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-6,
        seed: int = 0,
    ):
        self.feature_dim = _integer(feature_dim, "feature_dim", 1)
        self.rank = _integer(rank, "rank")
        self.epsilon = _positive(epsilon, "epsilon")
        self.noise_variance = _positive(noise_variance, "noise_variance")
        self.max_iter = _integer(max_iter, "max_iter")
        self.tol = _positive(tol, "tol")
        # The fit is deterministic. Seed is retained for a common experiment API;
        # no randomized initialization or hidden validation search uses it.
        self.seed = _integer(seed, "seed")
        self._classes: list[int] = []
        self._gram = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        self._cross = np.empty((self.feature_dim, 0), dtype=np.float64)
        self._feature_sum = np.zeros(self.feature_dim, dtype=np.float64)
        self._counts = np.empty(0, dtype=np.int64)
        self._n_samples = 0
        self._D = np.empty((self.feature_dim, 0), dtype=np.float64)
        self._weights = np.empty((self.feature_dim, 0), dtype=np.float64)
        self._intercept = np.empty(0, dtype=np.float64)
        self._tasks = 0
        self._diagnostics: dict[str, Any] = {}

    @property
    def classes(self) -> list[int]:
        return list(self._classes)

    @property
    def G(self) -> np.ndarray:
        return self._gram.copy()

    @property
    def C(self) -> np.ndarray:
        return self._cross.copy()

    @property
    def feature_sum(self) -> np.ndarray:
        return self._feature_sum.copy()

    @property
    def counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def D(self) -> np.ndarray:
        return self._D.copy()

    @property
    def Sigma(self) -> np.ndarray:
        return self.epsilon * np.eye(self.feature_dim) + self._D @ self._D.T

    @property
    def covariance(self) -> np.ndarray:
        return self.Sigma

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def intercept(self) -> np.ndarray:
        return self._intercept.copy()

    @property
    def effective_rank(self) -> int:
        if not self._D.shape[1]:
            return 0
        # SVD also handles a valid externally restored factor whose columns have
        # been rotated or made linearly dependent, rather than counting columns.
        energies = np.linalg.svd(self._D, compute_uv=False) ** 2
        threshold = (
            np.finfo(np.float64).eps
            * max(1.0, float(energies.max()))
            * self.feature_dim
        )
        return int(np.sum(energies > threshold))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return copy.deepcopy(self._diagnostics)

    @property
    def class_codes(self) -> np.ndarray:
        """Posterior E[a|Y] in w=D a+epsilon-noise, not a unique semantic code."""
        return self._D.T @ np.linalg.solve(self.Sigma, self._weights)

    def _features(self, value: Any, *, empty: bool = False) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != self.feature_dim:
            raise ValueError(
                "Features must have shape [samples, feature_dim], without an appended bias"
            )
        if not empty and not len(result):
            raise ValueError("A task must contain at least one fitting sample")
        if not np.isfinite(result).all():
            raise ValueError("Features must be finite")
        return result

    @staticmethod
    def _center(
        gram: np.ndarray,
        cross: np.ndarray,
        feature_sum: np.ndarray,
        counts: np.ndarray,
        n_samples: int,
    ):
        mean_x = feature_sum / n_samples
        prior_frequency = counts.astype(np.float64) / n_samples
        centered_gram = _symmetric(gram - np.outer(feature_sum, mean_x))
        eigenvalues, eigenvectors = np.linalg.eigh(centered_gram)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if eigenvalues[0] < -1e-9 * scale:
            raise ValueError(
                "Centered Gram is not positive semidefinite; moments may be inconsistent or badly scaled"
            )
        correction = 0.0
        if eigenvalues[0] < 0:
            cleaned = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            correction = float(np.linalg.norm(cleaned - centered_gram, "fro"))
            centered_gram = _symmetric(cleaned)
        centered_cross = cross - np.outer(mean_x, counts)
        target_scatter_trace = float(
            n_samples - counts.astype(np.float64) @ prior_frequency
        )
        return (
            centered_gram,
            centered_cross,
            mean_x,
            prior_frequency,
            target_scatter_trace,
            correction,
        )

    def observe(
        self, features: Any, labels: Any, *, update_basis: bool = True
    ) -> dict[str, Any]:
        """Commit one new-class task and refit the shared prior/posterior atomically."""
        x = self._features(features)
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1 or len(labels_array) != len(x):
            raise ValueError("Labels must be one integer per feature row")
        if labels_array.dtype.kind not in "iu":
            raise TypeError(
                "Labels must be integers (boolean and floating labels are rejected)"
            )
        if not isinstance(update_basis, (bool, np.bool_)):
            raise TypeError("update_basis must be boolean")
        incoming = sorted(set(map(int, labels_array)))
        if set(incoming).intersection(self._classes):
            raise ValueError(
                "Task classes must be new; repeated epochs must not recommit historical rows"
            )
        classes = self._classes + incoming
        counts = np.pad(self._counts, (0, len(incoming)))
        cross = np.pad(self._cross, ((0, 0), (0, len(incoming))))
        with np.errstate(over="raise", invalid="raise"):
            gram = self._gram + x.T @ x
            feature_sum = self._feature_sum + x.sum(axis=0)
            for j, label in enumerate(incoming, start=len(self._classes)):
                rows = labels_array == label
                counts[j] = int(rows.sum())
                cross[:, j] = x[rows].sum(axis=0)
        n_samples = self._n_samples + len(x)
        if not all(np.isfinite(a).all() for a in (gram, cross, feature_sum)):
            raise FloatingPointError("Nonfinite accumulated sufficient statistics")
        centered = self._center(gram, cross, feature_sum, counts, n_samples)
        gc, cc, mean_x, frequencies, trace_yy, roundoff_correction = centered
        cap = min(self.rank, self.feature_dim, max(0, len(classes) - 1))
        d = self._D.copy()
        covariance = self.epsilon * np.eye(self.feature_dim) + d @ d.T
        likelihood, mean, posterior_covariance = _log_likelihood(
            gc, cc, covariance, self.noise_variance, trace_yy, n_samples - 1
        )
        history = [likelihood]
        converged = bool(not update_basis or not cap or not self.max_iter)
        reason = (
            "basis_frozen"
            if not update_basis
            else (
                "isotropic_or_no_class_contrast"
                if not cap
                else "iterations_disabled" if not self.max_iter else "iteration_limit"
            )
        )
        iterations = 0
        if update_basis and cap and self.max_iter:
            for iterations in range(1, self.max_iter + 1):
                q = _symmetric(posterior_covariance + mean @ mean.T / len(classes))
                values, vectors = np.linalg.eigh(q)
                indices = np.argsort(values, kind="stable")[::-1][:cap]
                loadings = np.sqrt(np.maximum(values[indices] - self.epsilon, 0.0))
                d = _canonical_columns(vectors[:, indices]) * loadings[None, :]
                covariance = self.epsilon * np.eye(self.feature_dim) + d @ d.T
                updated, mean, posterior_covariance = _log_likelihood(
                    gc, cc, covariance, self.noise_variance, trace_yy, n_samples - 1
                )
                # This M-step is exact for fixed epsilon/s2 and the declared rank
                # constraint. A substantial decline is a numerical/correctness
                # failure, not a reason to silently return a different model.
                if updated < likelihood - 1e-8 * (1.0 + abs(likelihood)):
                    raise FloatingPointError(
                        "Constrained EM decreased marginal log likelihood"
                    )
                change = abs(updated - likelihood) / (1.0 + abs(likelihood))
                history.append(updated)
                likelihood = updated
                if change <= self.tol:
                    converged = True
                    reason = "relative_log_likelihood_tolerance"
                    break
        intercept = frequencies - mean_x @ mean
        if not np.isfinite(intercept).all():
            raise FloatingPointError("Nonfinite recovered flat-prior intercept")
        diagnostics = {
            "method": "shared_predictor_basis",
            "task": self._tasks,
            "new_classes": incoming,
            "classes": len(classes),
            "samples": n_samples,
            "new_samples": len(x),
            "rank_limit": self.rank,
            "class_identified_rank_cap": cap,
            "epsilon": self.epsilon,
            "noise_variance": self.noise_variance,
            "iterations": iterations,
            "converged": converged,
            "stop_reason": reason,
            "basis_updated": bool(update_basis and cap and self.max_iter),
            "marginal_log_likelihood": likelihood,
            "log_likelihood_history": history,
            "centered_gram_roundoff_correction_fro": roundoff_correction,
            "bias": "flat prior, recovered from centered moments",
            "weighting": "unit weight per historical fitting row; no class reweighting",
            "history": "exact sufficient statistics for the caller's immutable feature map",
        }
        self._classes, self._gram, self._cross = classes, gram, cross
        self._feature_sum, self._counts, self._n_samples = (
            feature_sum,
            counts,
            n_samples,
        )
        self._D, self._weights, self._intercept = d, mean, intercept
        self._tasks += 1
        diagnostics["effective_rank"] = self.effective_rank
        diagnostics["numeric_state_bytes"] = self.resource_bytes()
        self._diagnostics = diagnostics
        return copy.deepcopy(diagnostics)

    def predict_scores(self, features: Any) -> np.ndarray:
        x = self._features(features, empty=True)
        return x @ self._weights + self._intercept

    def code_only_scores(self, features: Any) -> np.ndarray:
        """Restricted w=D a posterior, refitted from the same moments.

        This is an explicit code-only control, not a projection of the full
        epsilon-residual predictor with its old intercept left unchanged.
        """
        x = self._features(features, empty=True)
        if not self._n_samples:
            return np.empty((len(x), 0), dtype=np.float64)
        gc, cc, mean_x, frequencies, _trace, _correction = self._center(
            self._gram, self._cross, self._feature_sum, self._counts, self._n_samples
        )
        if not self._D.shape[1]:
            return np.broadcast_to(frequencies, (len(x), len(self._classes))).copy()
        design = self._D
        system = np.eye(design.shape[1]) + design.T @ gc @ design / self.noise_variance
        codes = np.linalg.solve(_symmetric(system), design.T @ cc / self.noise_variance)
        weights = design @ codes
        return x @ weights + frequencies - mean_x @ weights

    def resource_bytes(self) -> int:
        """Persistent NumPy numeric payload, excluding Python/temporary arrays."""
        return int(
            sum(
                a.nbytes
                for a in (
                    self._gram,
                    self._cross,
                    self._feature_sum,
                    self._counts,
                    self._D,
                    self._weights,
                    self._intercept,
                )
            )
        )

    def _config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "rank": self.rank,
            "epsilon": self.epsilon,
            "noise_variance": self.noise_variance,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "seed": self.seed,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "kind": "shared_predictor_basis",
            "config": self._config(),
            "classes": self.classes,
            "G": self.G,
            "C": self.C,
            "feature_sum": self.feature_sum,
            "counts": self.counts,
            "n_samples": self._n_samples,
            "D": self.D,
            "tasks": self._tasks,
            "diagnostics": self.diagnostics,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate and restore a state with the same constructor configuration."""
        if (
            state.get("version") != self.STATE_VERSION
            or state.get("kind") != "shared_predictor_basis"
        ):
            raise ValueError("Unsupported shared-basis state")
        if state.get("config") != self._config():
            raise ValueError("Shared-basis checkpoint configuration differs")
        labels = state.get("classes")
        if not isinstance(labels, list) or any(
            isinstance(y, bool) or not isinstance(y, (int, np.integer)) for y in labels
        ):
            raise ValueError("Invalid checkpoint class vocabulary")
        classes = list(map(int, labels))
        if len(set(classes)) != len(classes):
            raise ValueError("Duplicate checkpoint classes")
        n = _integer(state.get("n_samples"), "n_samples")
        tasks = _integer(state.get("tasks"), "tasks")
        d, c = self.feature_dim, len(classes)
        gram = _array(state.get("G"), (d, d), "G")
        cross = _array(state.get("C"), (d, c), "C")
        feature_sum = _array(state.get("feature_sum"), (d,), "feature_sum")
        raw_counts = np.asarray(state.get("counts"))
        if raw_counts.shape != (c,) or raw_counts.dtype.kind not in "iu":
            raise ValueError("Checkpoint counts must be an integer vector")
        counts = raw_counts.astype(np.int64, copy=True)
        if (
            (counts <= 0).any()
            or sum(map(int, counts)) != n
            or bool(n) != bool(c)
            or bool(tasks) != bool(c)
            or tasks > c
        ):
            raise ValueError("Checkpoint counts/classes/tasks are inconsistent")
        if not np.allclose(gram, gram.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Checkpoint Gram must be symmetric")
        if not np.allclose(cross.sum(axis=1), feature_sum, atol=1e-10, rtol=1e-10):
            raise ValueError("Class feature sums disagree with total feature sum")
        loadings = np.asarray(state.get("D"), dtype=np.float64)
        cap = min(self.rank, d, max(0, c - 1))
        if (
            loadings.ndim != 2
            or loadings.shape[0] != d
            or loadings.shape[1] > cap
            or not np.isfinite(loadings).all()
        ):
            raise ValueError("Checkpoint basis has an invalid shape or rank")
        loadings = loadings.copy()
        covariance = self.epsilon * np.eye(d) + loadings @ loadings.T
        if n:
            # This is the Schur complement of the one-hot label Gram diag(n_c).
            # It equals the sum of within-class feature scatters for genuine
            # observations, and catches corrupt C values even when their sum
            # still agrees with the total feature sum.
            within = _symmetric(gram - (cross / counts[None, :]) @ cross.T)
            values = np.linalg.eigvalsh(within)
            scale = max(1.0, float(np.linalg.norm(gram, ord=2)))
            if values[0] < -1e-9 * scale:
                raise ValueError(
                    "Checkpoint feature/label joint moments are inconsistent"
                )
            centered = self._center(gram, cross, feature_sum, counts, n)
            gc, cc, mean_x, frequencies, _trace, _correction = centered
            weights, _posterior_covariance, _logdet = _posterior(
                gc, cc, covariance, self.noise_variance
            )
            intercept = frequencies - mean_x @ weights
        else:
            if any(np.any(a) for a in (gram, cross, feature_sum)) or loadings.shape[1]:
                raise ValueError("Empty checkpoint contains fitted statistics")
            weights, intercept = np.empty((d, 0)), np.empty(0)
        self._classes, self._gram, self._cross = classes, gram, cross
        self._feature_sum, self._counts, self._n_samples = feature_sum, counts, n
        self._D, self._weights, self._intercept = loadings, weights, intercept
        self._tasks = tasks
        self._diagnostics = copy.deepcopy(state.get("diagnostics", {}))
