"""Stable all-history ridge statistics and a differentiable residual readout.

Every feature matrix has examples in rows. Ridge objectives use a SUM of
weighted squared residuals, with no implicit sample-count normalization. Only
the immutable base features enter ``StableStatistics``; mutable adapter features
must be recomputed from resident examples and passed to the stateless solvers.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _matrix(value, name):
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional tensor with examples in rows"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _positive_ridge(value, reference):
    if isinstance(value, bool):
        raise TypeError("ridge must be a positive finite scalar")
    if isinstance(value, Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError("ridge must be a floating-point scalar")
        if value.device != reference.device:
            raise ValueError("ridge and features must be on the same device")
        lam = value.to(dtype=reference.dtype)
    else:
        try:
            lam = reference.new_tensor(float(value))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("ridge must be a positive finite scalar") from error
    if not torch.isfinite(lam) or lam <= 0:
        raise ValueError("ridge must be positive and finite")
    return lam


def _common_matrices(*named):
    tensors = [_matrix(value, name) for name, value in named]
    device = tensors[0].device
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        if tensor.device != device:
            raise ValueError("All readout inputs must be on the same device")
        dtype = torch.promote_types(dtype, tensor.dtype)
    # CPU and higher-order solve kernels do not support half precision reliably.
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return [tensor.to(dtype=dtype) for tensor in tensors]


def _weights(weights, z):
    if weights is None:
        return z.new_ones(len(z))
    if not isinstance(weights, Tensor) or weights.ndim != 1 or len(weights) != len(z):
        raise ValueError("weights must have one entry per example")
    if not weights.is_floating_point():
        raise TypeError("weights must be floating point")
    if weights.device != z.device:
        raise ValueError("weights and features must be on the same device")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    converted = weights.to(dtype=z.dtype)
    if not torch.isfinite(converted).all():
        raise ValueError("weights are not finite in the solve dtype")
    return converted


def _solve(a, b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError(
            "Ridge normal equations overflowed; rescale inputs or use float64"
        )
    solution = torch.linalg.solve(a, b)
    if not torch.isfinite(solution).all():
        raise FloatingPointError("Ridge solution is nonfinite")
    return solution


def ridge(
    z: Tensor,
    y: Tensor,
    lam,
    weights: Tensor | None = None,
    prior: Tensor | None = None,
) -> Tensor:
    """Solve ``sum_i w_i ||z_i W-y_i||² + lam ||W-prior||²``.

    ``y`` is a floating-point target matrix (one-hot in classification), not a
    label vector. ``prior=None`` means zero. Inputs are promoted to a common
    floating dtype; float16/bfloat16 solves use float32. This function preserves
    autograd through features, targets, prior, weights, and a tensor ``lam``.
    When examples are fewer than features, an equivalent dual system is solved.
    """
    named = [("z", z), ("y", y)]
    if prior is not None:
        named.append(("prior", prior))
    values = _common_matrices(*named)
    z, y = values[:2]
    if z.shape[0] != y.shape[0] or z.shape[1] < 1:
        raise ValueError(
            "z and y must have equal row counts and positive feature dimension"
        )
    if prior is None:
        p = z.new_zeros((z.shape[1], y.shape[1]))
    else:
        p = values[2]
        if p.shape != (z.shape[1], y.shape[1]):
            raise ValueError("prior must have shape [feature_dim, output_dim]")
    lam = _positive_ridge(lam, z)
    weight = _weights(weights, z)
    weighted_z = weight[:, None] * z
    if len(z) < z.shape[1]:
        # P + Z^T (D Z Z^T + lambda I)^-1 D (Y - ZP).
        # Avoid sqrt(D): its derivative at zero weights is undefined, even
        # though the ridge solution itself is differentiable there. This
        # nonsymmetric system is algebraically the same Woodbury solution.
        a = weighted_z @ z.T + lam * torch.eye(len(z), dtype=z.dtype, device=z.device)
        b = weight[:, None] * (y - z @ p)
        solution = p + z.T @ _solve(a, b)
        if not torch.isfinite(solution).all():
            raise FloatingPointError("Ridge solution is nonfinite")
        return solution
    a = z.T @ weighted_z + lam * torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    b = z.T @ (weight[:, None] * y) + lam * p
    return _solve(a, b)


# The residual API uses ``ridge`` as its public regularization keyword.
_ridge_fit = ridge


def centered_residual_head(
    delta: Tensor,
    z0: Tensor,
    y: Tensor,
    W0: Tensor,
    ridge,
    weights: Tensor | None = None,
) -> Tensor:
    """Fit the residual head with the base head as its nonzero prior.

    Minimize ``||sqrt(D)(delta V - (Y-z0 W0))||² + ridge ||V-W0||²``.
    ``delta=z_adapted-z0`` must use the base feature dimension. At delta=0 the
    answer is W0: the full predictor ``z0 W0 + delta V`` equals the base but can
    have a nonzero gradient through delta. No cumulative delta statistics exist.
    """
    delta, z0, y, W0 = _common_matrices(
        ("delta", delta), ("z0", z0), ("y", y), ("W0", W0)
    )
    if delta.shape != z0.shape or len(delta) != len(y):
        raise ValueError(
            "delta and z0 must have matching [examples, features] shapes and y rows"
        )
    if W0.shape != (delta.shape[1], y.shape[1]):
        raise ValueError("W0 must have shape [feature_dim, output_dim]")
    return _ridge_fit(delta, y - z0 @ W0, ridge, weights=weights, prior=W0)


def _label_values(labels):
    if isinstance(labels, Tensor):
        if (
            labels.ndim != 1
            or labels.dtype == torch.bool
            or labels.is_floating_point()
            or labels.is_complex()
        ):
            raise ValueError("labels must be a one-dimensional integer tensor")
        return [int(value) for value in labels.detach().cpu().tolist()]
    try:
        values = list(labels)
    except TypeError as error:
        raise TypeError("labels must be an iterable of integers") from error
    result = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("labels must be integers, not booleans")
        try:
            result.append(operator.index(value))
        except TypeError as error:
            raise TypeError("labels must be integers") from error
    return result


class StableStatistics:
    """Exact sum statistics in one immutable base-feature coordinate system.

    The caller must pass every arrival exactly once to ``add``. Without sample
    IDs these moments cannot detect duplicate arrivals or verify membership of
    a requested downdate. Temporary add/remove labels must already be registered
    so the returned head always uses the persistent ``classes`` column order.
    """

    _VERSION = 1

    def __init__(
        self, feature_dim: int, ridge: float, dtype=torch.float64, device="cpu"
    ):
        self.feature_dim = _positive_integer(feature_dim, "feature_dim")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Stable statistics require float32 or float64")
        self.dtype, self.device = dtype, torch.device(device)
        try:
            self.ridge = float(ridge)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("ridge must be positive and finite") from error
        if isinstance(ridge, bool) or not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        self.G = torch.zeros(
            (self.feature_dim, self.feature_dim), dtype=dtype, device=self.device
        )
        self.C = torch.zeros((self.feature_dim, 0), dtype=dtype, device=self.device)
        self.classes: list[int] = []
        self.count = 0

    def register(self, labels):
        added = sorted(set(_label_values(labels)).difference(self.classes))
        if added:
            self.classes.extend(added)
            self.C = torch.cat(
                (self.C, self.C.new_zeros((self.feature_dim, len(added)))), dim=1
            )
        return added

    def encode_labels(self, labels):
        values = _label_values(labels)
        mapping = {label: i for i, label in enumerate(self.classes)}
        if any(label not in mapping for label in values):
            raise ValueError("Unregistered class label")
        return torch.tensor(
            [mapping[label] for label in values], dtype=torch.long, device=self.device
        )

    def _features(self, z, name):
        z = _matrix(z, name)
        if z.shape[1] != self.feature_dim:
            raise ValueError(f"{name} has the wrong feature dimension")
        result = z.to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} is not finite in statistics dtype")
        return result

    def add(self, z: Tensor, labels):
        """Atomically incorporate an arrival batch, detaching persistent moments."""
        z = self._features(z, "z").detach()
        values = _label_values(labels)
        if len(z) != len(values):
            raise ValueError("Features and labels have different row counts")
        added = sorted(set(values).difference(self.classes))
        classes = self.classes + added
        mapping = {label: i for i, label in enumerate(classes)}
        index = torch.tensor(
            [mapping[label] for label in values], dtype=torch.long, device=self.device
        )
        targets = (
            F.one_hot(index, num_classes=len(classes)).to(self.dtype)
            if classes
            else z.new_zeros((len(z), 0))
        )
        c = torch.cat((self.C, self.C.new_zeros((self.feature_dim, len(added)))), dim=1)
        g, c = self.G + z.T @ z, c + z.T @ targets
        if not torch.isfinite(g).all() or not torch.isfinite(c).all():
            raise ValueError("Stable statistics overflowed")
        self.G, self.C, self.classes = g, c, classes
        self.count += len(z)

    def head(self, remove_z=None, remove_labels=None, add_z=None, add_labels=None):
        """Solve after temporary query removal/support addition, without mutation.

        Removed examples must actually occur in these statistics, with the same
        base features and labels. Membership cannot be inferred from moments.
        Temporary features can carry autograd; stored history never does.
        """
        g, c = self.G, self.C
        for z, labels, sign, name in (
            (remove_z, remove_labels, -1, "remove_z"),
            (add_z, add_labels, 1, "add_z"),
        ):
            if (z is None) != (labels is None):
                raise ValueError(
                    "Temporary features and labels must be supplied together"
                )
            if z is None:
                continue
            z = self._features(z, name)
            index = self.encode_labels(labels)
            if len(z) != len(index):
                raise ValueError(
                    "Temporary features and labels have different row counts"
                )
            if sign < 0 and len(z) > self.count:
                raise ValueError("Cannot remove more arrivals than stored")
            targets = (
                F.one_hot(index, num_classes=len(self.classes)).to(self.dtype)
                if self.classes
                else z.new_zeros((len(z), 0))
            )
            g = g + sign * (z.T @ z)
            c = c + sign * (z.T @ targets)
        a = g + self.ridge * torch.eye(
            self.feature_dim, dtype=self.dtype, device=self.device
        )
        return _solve(a, c)

    def state_dict(self):
        return {
            "version": self._VERSION,
            "feature_dim": self.feature_dim,
            "ridge": self.ridge,
            "dtype": str(self.dtype),
            "classes": list(self.classes),
            "count": self.count,
            "G": self.G.detach().clone(),
            "C": self.C.detach().clone(),
        }

    def load_state_dict(self, state):
        """Validate first, then restore to this instance's configured device/dtype."""
        fields = {
            "version",
            "feature_dim",
            "ridge",
            "dtype",
            "classes",
            "count",
            "G",
            "C",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("Invalid stable-statistics checkpoint fields")
        if (
            state["version"] != self._VERSION
            or state["feature_dim"] != self.feature_dim
        ):
            raise ValueError("Stable-statistics checkpoint version/dimension mismatch")
        if state["ridge"] != self.ridge or state["dtype"] != str(self.dtype):
            raise ValueError("Stable-statistics checkpoint ridge/dtype mismatch")
        classes = _label_values(state["classes"])
        if len(classes) != len(set(classes)):
            raise ValueError("Checkpoint classes must be unique")
        count = state["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Checkpoint count must be a nonnegative integer")
        g, c = _matrix(state["G"], "G"), _matrix(state["C"], "C")
        if g.shape != (self.feature_dim, self.feature_dim) or c.shape != (
            self.feature_dim,
            len(classes),
        ):
            raise ValueError("Checkpoint moment dimensions do not match")
        g, c = (
            g.detach().to(device=self.device, dtype=self.dtype).clone(),
            c.detach().to(device=self.device, dtype=self.dtype).clone(),
        )
        if not torch.isfinite(g).all() or not torch.isfinite(c).all():
            raise ValueError("Checkpoint moments overflowed")
        tolerance = 100 * torch.finfo(self.dtype).eps
        if (
            not torch.allclose(g, g.T, rtol=tolerance, atol=tolerance)
            or (g.diag() < -tolerance).any()
        ):
            raise ValueError("Checkpoint G must be symmetric with nonnegative diagonal")
        if (count == 0 and (torch.count_nonzero(g) or torch.count_nonzero(c))) or (
            count > 0 and not classes
        ):
            raise ValueError("Checkpoint count/classes/moments are inconsistent")
        self.G, self.C, self.classes, self.count = g, c, classes, count
