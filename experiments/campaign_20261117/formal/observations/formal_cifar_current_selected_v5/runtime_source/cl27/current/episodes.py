"""Label-stratified support/query episodes for the anchored learner.

Indices always address ``cat([current_labels, memory_labels])``: current rows
come first, then memory rows. No features, targets, or statistics are fitted here.
In particular, a transfer episode only simulates a novel-label adaptation problem
using already available classes. It is not evidence of transfer to unseen data.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Episode:
    kind: str
    support_indices: Tensor
    query_target_indices: Tensor
    query_background_indices: Tensor
    targets: tuple[int, ...]
    skipped_reason: str | None = None

    def __post_init__(self):
        if self.kind not in {"anchored", "transfer"}:
            raise ValueError("Episode kind must be 'anchored' or 'transfer'")
        occupied: set[int] = set()
        for indices in (
            self.support_indices,
            self.query_target_indices,
            self.query_background_indices,
        ):
            if (
                indices.device.type != "cpu"
                or indices.dtype != torch.long
                or indices.ndim != 1
            ):
                raise ValueError(
                    "Episode indices must be one-dimensional CPU long tensors"
                )
            values = indices.tolist()
            if any(index < 0 for index in values):
                raise ValueError("Episode indices cannot be negative")
            if len(set(values)) != len(values) or occupied.intersection(values):
                raise ValueError(
                    "Support and query indices must be unique and disjoint"
                )
            occupied.update(values)
        if not isinstance(self.targets, tuple) or len(set(self.targets)) != len(
            self.targets
        ):
            raise ValueError("Episode targets must be a tuple of unique global labels")


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _labels(value, name):
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} must contain integer global labels")
    return value.detach().to(device="cpu", dtype=torch.long)


def sample_episode(
    current_labels: Tensor,
    memory_labels: Tensor,
    kind: str,
    generator: torch.Generator,
    support_shots=(1, 5),
    background_shots=5,
    target_classes=5,
    query_per_class=2,
) -> Episode:
    """Sample one bounded episode without overlapping support/query indices.

    Every available class gets support. A singleton is support-only; other
    classes reserve at least one row for a possible query. Target classes draw
    their support count independently from ``support_shots`` (then cap at n-1).
    Non-target classes use min(background_shots, n-1) support rows and at most
    query_per_class background queries.

    Anchored targets are sampled from eligible classes represented in *current*
    arrivals, falling back to eligible memory-only classes only if none qualify.
    Transfer targets are sampled from all eligible classes. Neither selection
    reads unavailable labels or assumes a task ID. If at least two classes have
    two or more rows, at least one eligible class is reserved for background
    queries. When only one class is eligible, a valid episode can necessarily
    have no background query (all other classes are singleton support-only).

    Empty, single-class, or entirely singleton pools return an explicitly skipped
    episode; they never silently fabricate a nontrivial classification query.
    Only the supplied CPU generator is consumed, never the global torch RNG.
    """
    if kind not in {"anchored", "transfer"}:
        raise ValueError("kind must be 'anchored' or 'transfer'")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("Sampling requires an explicit CPU torch.Generator")
    if not isinstance(support_shots, (tuple, list)) or not support_shots:
        raise ValueError("support_shots must be a nonempty sequence")
    shots = tuple(_positive_integer(value, "support_shots") for value in support_shots)
    background_shots = _positive_integer(background_shots, "background_shots")
    target_classes = _positive_integer(target_classes, "target_classes")
    query_per_class = _positive_integer(query_per_class, "query_per_class")
    current = _labels(current_labels, "current_labels")
    memory = _labels(memory_labels, "memory_labels")
    labels = torch.cat((current, memory))
    classes = sorted(set(labels.tolist()))
    members = {
        label: torch.nonzero(labels == label, as_tuple=False).flatten()
        for label in classes
    }

    def result(
        support=(), target_query=(), background_query=(), targets=(), reason=None
    ):
        return Episode(
            kind,
            torch.tensor(support, dtype=torch.long),
            torch.tensor(target_query, dtype=torch.long),
            torch.tensor(background_query, dtype=torch.long),
            tuple(targets),
            reason,
        )

    if not classes:
        return result(reason="empty_pool")
    if len(classes) == 1:
        return result(support=range(len(labels)), reason="single_class")
    eligible = [label for label in classes if len(members[label]) >= 2]
    if not eligible:
        return result(support=range(len(labels)), reason="no_eligible_target")

    current_classes = set(current.tolist())
    pool = (
        [label for label in eligible if label in current_classes]
        if kind == "anchored"
        else eligible
    )
    if not pool:
        pool = eligible  # No eligible current class: fall back to already available memory labels.
    # Preserve a genuine non-target query when enough eligible classes exist.
    count = min(target_classes, len(pool), max(1, len(eligible) - 1))
    chosen = torch.randperm(len(pool), generator=generator)[:count].tolist()
    targets = tuple(pool[index] for index in chosen)
    target_set = set(targets)
    support, target_query, background_query = [], [], []
    for label in classes:
        available = members[label]
        order = available[torch.randperm(len(available), generator=generator)].tolist()
        if len(order) == 1:
            support.extend(order)
            continue
        if label in target_set:
            shot = shots[int(torch.randint(len(shots), (), generator=generator))]
            support_count = min(shot, len(order) - 1)
        else:
            support_count = min(background_shots, len(order) - 1)
        support.extend(order[:support_count])
        queries = order[support_count : support_count + query_per_class]
        (target_query if label in target_set else background_query).extend(queries)
    return result(support, target_query, background_query, targets)
