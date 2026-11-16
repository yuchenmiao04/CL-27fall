"""Fixed-library episodes on native PILOT/timm transformer blocks.

The old window method is adapted to task-wise access: current rows enter the
immutable history exactly once, after every learning epoch for that task. The
feature-chain VJP is an execution optimization for the same differentiable
support-solve/query objective, not an additional approximation to the solver.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .episodes import sample_episode
from .readout import StableStatistics, centered_residual_head, ridge


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def class_weights(labels):
    _, inverse, counts = labels.unique(
        sorted=True, return_inverse=True, return_counts=True
    )
    return (len(labels) / len(counts)) / counts[inverse].to(torch.float64)


def class_mean_ce(logits, labels):
    values = F.cross_entropy(logits, labels, reduction="none")
    return torch.stack([values[labels == c].mean() for c in labels.unique()]).mean()


class ResidualLibrary(nn.Module):
    """Dense low-rank all-token mixture and a fixed zero-output null option."""

    def __init__(self, width, rank, count, scale=0.1):
        super().__init__()
        if min(width, rank, count) < 1:
            raise ValueError("Invalid library dimensions")
        self.norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.U = nn.Parameter(torch.zeros(count, width, rank))
        self.V = nn.Parameter(torch.randn(count, rank, width) / math.sqrt(width))
        self.gate_weight = nn.Parameter(torch.zeros(count, width))
        self.gate_bias = nn.Parameter(torch.zeros(count))
        self.register_buffer("null_logit", torch.tensor(math.log(count)))
        self.scale = float(scale)
        self.count = count

    def forward(self, tokens):
        u = self.norm(tokens)
        logit = F.linear(u, self.gate_weight, self.gate_bias)
        null = self.null_logit.expand(*logit.shape[:-1], 1)
        gate = torch.cat((null, logit), -1).softmax(-1)
        # Keep every expert active; no claim of sparse conditional compute.
        outputs = torch.stack(
            [F.linear(torch.tanh(F.linear(u, v)), a) for a, v in zip(self.U, self.V)],
            dim=-2,
        )
        return self.scale * (outputs * gate[..., 1:, None]).sum(-2), gate


class NativeDualEncoder(nn.Module):
    """Two independent native backbones; libraries hook main block outputs."""

    def __init__(self, backbone, positions=(7, 9), count=4, rank=8, scale=0.1):
        super().__init__()
        if not hasattr(backbone, "blocks") or len(backbone.blocks) < 2:
            raise ValueError("Current adaptation requires native ViT .blocks")
        width = int(getattr(backbone, "out_dim", getattr(backbone, "num_features", 0)))
        if width < 1:
            raise ValueError("Native backbone must declare out_dim or num_features")
        positions = tuple(map(int, positions))
        if (
            not positions
            or len(set(positions)) != len(positions)
            or min(positions) < 0
            or max(positions) >= len(backbone.blocks) - 1
        ):
            raise ValueError("Every library needs a subsequent attention block")
        self.anchor = copy.deepcopy(backbone).requires_grad_(False)
        self.main = copy.deepcopy(backbone).requires_grad_(True)
        self.width = width
        self.feature_dim = width + 1
        self.positions = positions
        self.libraries = nn.ModuleList(
            ResidualLibrary(width, rank, count, scale) for _ in positions
        )
        self._modules_active = False
        self._handles = []
        for index, position in enumerate(positions):
            self._handles.append(
                self.main.blocks[position].register_forward_hook(self._hook(index))
            )
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.eval()

    def _hook(self, index):
        def apply(_module, _args, output):
            if not self._modules_active:
                return output
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError("Expected native [batch,tokens,width] block output")
            correction, _ = self.libraries[index](output)
            return output + correction

        return apply

    def train(self, mode=True):
        # Parameter gradients and train/eval state are independent. Running
        # statistics and stochastic layers stay fixed for exact VJP replay.
        super().train(False)
        return self

    def raw_main(self, images, modules=True):
        self.main.eval()
        previous = self._modules_active
        self._modules_active = modules
        try:
            output = self.main(images)
        finally:
            self._modules_active = previous
        if not isinstance(output, torch.Tensor) or output.shape != (
            len(images),
            self.width,
        ):
            raise ValueError("Native backbone must return raw CLS [batch,width]")
        return output

    @staticmethod
    def append_bias(raw):
        z = F.normalize(raw, p=2, dim=-1)
        return torch.cat((z, z.new_ones(len(z), 1)), -1)

    def main_features(self, images, modules=True):
        return self.append_bias(self.raw_main(images, modules))

    def anchor_features(self, images):
        self.anchor.eval()
        with torch.no_grad():
            return self.append_bias(self.anchor(images))


DEFAULTS = {
    "base_ridge": 0.01,
    "residual_ridge": 0.01,
    "temperature": 10.0,
    "module_lr": 1e-4,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "outer_updates": 4,
    "schedule": ("modules", "modules", "backbone", "modules"),
    "support_shots": (1, 5),
    "background_shots": 5,
    "target_classes": 5,
    "query_per_class": 2,
    "background_weight": 1.0,
    "transfer_probability": 0.5,
    "feature_batch_size": 16,
    "memory_efficient": True,
    "seed": 0,
}


class FixedLibraryEngine(nn.Module):
    def __init__(
        self,
        backbone,
        *,
        positions=(7, 9),
        count=4,
        rank=8,
        scale=0.1,
        options=None,
        device="cpu",
        canonical_transform=None,
        training_transform=None,
    ):
        super().__init__()
        self.options = {**DEFAULTS, **(options or {})}
        o = self.options
        if (
            any(p not in {"modules", "backbone"} for p in o["schedule"])
            or not o["schedule"]
        ):
            raise ValueError("Invalid phase schedule")
        if (
            min(
                o["base_ridge"],
                o["residual_ridge"],
                o["feature_batch_size"],
                o["outer_updates"],
            )
            <= 0
        ):
            raise ValueError("Invalid solver or episode parameters")
        if not 0 <= o["transfer_probability"] <= 1:
            raise ValueError("Invalid episode probability")
        self.device = torch.device(device)
        self.encoder = NativeDualEncoder(backbone, positions, count, rank, scale).to(
            self.device
        )
        self.architecture = {
            "positions": tuple(self.encoder.positions),
            "count": int(count),
            "rank": int(rank),
            "scale": float(scale),
            "width": self.encoder.width,
        }
        self._anchor_digest = state_digest(self.encoder.anchor.state_dict())
        self.feature_dim = self.encoder.feature_dim
        self.statistics = StableStatistics(self.feature_dim, o["base_ridge"])
        self.W0 = torch.empty(self.feature_dim, 0, dtype=torch.float64)
        self.V = self.W0.clone()
        self.canonical_transform = canonical_transform
        self.training_transform = training_transform
        if not callable(canonical_transform) or not callable(training_transform):
            raise ValueError("Explicit canonical and training transforms are required")
        self.optimizers = {
            "modules": torch.optim.AdamW(
                self.encoder.libraries.parameters(),
                lr=o["module_lr"],
                weight_decay=o["weight_decay"],
            ),
            "backbone": torch.optim.AdamW(
                self.encoder.main.parameters(),
                lr=o["backbone_lr"],
                weight_decay=o["weight_decay"],
            ),
        }
        self.rng = torch.Generator().manual_seed(o["seed"] + 17)
        self.kind_rng = torch.Generator().manual_seed(o["seed"] + 19)
        self.memory_base = {}
        self.completed_tasks = 0
        self.task_sizes = []
        self.episode_draws = 0
        self.updates = self.module_updates = self.backbone_updates = 0
        self.counters = {
            "canonical_anchor_images": 0,
            "canonical_main_images": 0,
            "episode_anchor_images": 0,
            "episode_full_prepass_images": 0,
            "episode_common_prepass_images": 0,
            "vjp_full_images": 0,
            "vjp_common_images": 0,
            "readout_solves": 0,
            "logical_current_presentations": 0,
        }
        self._task_dirty = False

    @property
    def classes(self):
        return self.statistics.classes

    def _inc(self, key, n):
        self.counters[key] = self.counters.get(key, 0) + int(n)

    def _stack(self, items, augment=False):
        fn = self.training_transform if augment else self.canonical_transform
        values = [fn(item[0]) for item in items]
        if any(
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or x.shape[0] != 3
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
            for x in values
        ):
            raise ValueError("Invalid image transform output")
        return torch.stack(values).to(self.device)

    @torch.no_grad()
    def _canonical(self, items, anchor=False):
        values = []
        size = self.options["feature_batch_size"]
        for start in range(0, len(items), size):
            images = self._stack(items[start : start + size])
            z = (
                self.encoder.anchor_features(images)
                if anchor
                else self.encoder.main_features(images, True)
            )
            values.append(z.cpu().double())
        self._inc(
            "canonical_anchor_images" if anchor else "canonical_main_images", len(items)
        )
        return (
            torch.cat(values)
            if values
            else torch.empty(0, self.feature_dim, dtype=torch.float64)
        )

    def _targets(self, labels):
        encoded = self.statistics.encode_labels(labels)
        return encoded, F.one_hot(encoded, num_classes=len(self.classes)).double()

    @contextmanager
    def phase(self, name):
        states = [(p, p.requires_grad) for p in self.encoder.parameters()]
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        self.encoder.anchor.requires_grad_(False)
        self.encoder.main.requires_grad_(name == "backbone")
        self.encoder.libraries.requires_grad_(name == "modules")
        try:
            yield
        finally:
            for p, enabled in states:
                p.requires_grad_(enabled)

    def _temporary_base(self, episode, canonical, labels, n_current, z0_support):
        self._inc("readout_solves", 1)
        if episode.kind == "transfer":
            _, targets = self._targets(labels[episode.support_indices])
            return ridge(z0_support, targets, self.options["base_ridge"])
        q = torch.cat((episode.query_target_indices, episode.query_background_indices))
        old_q = q[q >= n_current]
        new_s = episode.support_indices[episode.support_indices < n_current]
        return self.statistics.head(
            remove_z=canonical[old_q],
            remove_labels=labels[old_q],
            add_z=canonical[new_s],
            add_labels=labels[new_s],
        )

    def _feature_views(self, views, phase, cached):
        pieces = {"base": [], "full": []}
        if phase == "backbone":
            pieces["common"] = []
        size = self.options["feature_batch_size"]
        context = torch.no_grad() if cached else torch.enable_grad()
        with context:
            for start in range(0, len(views), size):
                x = views[start : start + size].to(self.device)
                pieces["base"].append(self.encoder.anchor_features(x).cpu().double())
                pieces["full"].append(
                    self.encoder.main_features(x, True).cpu().double()
                )
                if phase == "backbone":
                    pieces["common"].append(
                        self.encoder.main_features(x, False).cpu().double()
                    )
        out = {k: torch.cat(v) for k, v in pieces.items()}
        if cached:
            out = {k: v.detach().requires_grad_(k != "base") for k, v in out.items()}
        self._inc("episode_anchor_images", len(views))
        self._inc("episode_full_prepass_images", len(views))
        if phase == "backbone":
            self._inc("episode_common_prepass_images", len(views))
        return out

    def _episode_objective(self, episode, canonical, labels, n_current, features):
        s, qt, qb = (
            episode.support_indices,
            episode.query_target_indices,
            episode.query_background_indices,
        )
        ns, nt = len(s), len(qt)
        base = features["base"]
        w0 = self._temporary_base(
            episode, canonical, labels, n_current, base[:ns]
        ).detach()
        _, ys = self._targets(labels[s])
        weights = class_weights(labels[s])

        def predictions(z):
            delta = z - base
            v = centered_residual_head(
                delta[:ns], base[:ns], ys, w0, self.options["residual_ridge"], weights
            )
            self._inc("readout_solves", 1)
            return self.options["temperature"] * (base[ns:] @ w0 + delta[ns:] @ v)

        logits = predictions(features["full"])
        query = torch.cat((qt, qb))
        yi, _ = self._targets(labels[query])
        target = class_mean_ce(logits[:nt], yi[:nt])
        background = (
            class_mean_ce(logits[nt:], yi[nt:]) if len(qb) else logits.new_zeros(())
        )
        if "common" not in features:
            return target + self.options["background_weight"] * background
        common = predictions(features["common"])
        common_target = class_mean_ce(common[:nt], yi[:nt])
        common_background = (
            class_mean_ce(common[nt:], yi[nt:]) if len(qb) else logits.new_zeros(())
        )
        old = query >= n_current
        old_full = (
            class_mean_ce(logits[old], yi[old]) if old.any() else logits.new_zeros(())
        )
        return common_target + 0.5 * self.options["background_weight"] * (
            common_background + old_full
        )

    def episode_step(
        self, episode, items, canonical, labels, n_current, phase, *, step=True
    ):
        if episode.skipped_reason:
            return {
                "skipped": episode.skipped_reason,
                "phase": phase,
                "kind": episode.kind,
            }
        with self.phase(phase):
            order = torch.cat(
                (
                    episode.support_indices,
                    episode.query_target_indices,
                    episode.query_background_indices,
                )
            )
            # Exactly one augmented CPU view per episode ID is shared by all paths.
            views = torch.stack(
                [self.training_transform(items[i][0]) for i in order.tolist()]
            ).cpu()
            cached = bool(self.options["memory_efficient"])
            features = self._feature_views(views, phase, cached)
            loss = self._episode_objective(
                episode, canonical, labels, n_current, features
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite episode objective")
            loss.backward()
            if cached:
                gradients = {
                    k: v.grad.detach()
                    for k, v in features.items()
                    if k != "base" and v.grad is not None
                }
                size = self.options["feature_batch_size"]
                for start in range(0, len(views), size):
                    images = views[start : start + size].to(self.device)
                    for path in ("common", "full"):
                        if path not in gradients:
                            continue
                        output = self.encoder.main_features(images, path == "full")
                        torch.autograd.backward(
                            output,
                            grad_tensors=gradients[path][
                                start : start + len(images)
                            ].to(output),
                        )
                        self._inc("vjp_" + path + "_images", len(images))
            trainable = [p for p in self.encoder.parameters() if p.requires_grad]
            gradients = [p.grad for p in trainable if p.grad is not None]
            if gradients:
                # Keep the original per-tensor reductions, but transfer their
                # scalar results together instead of synchronizing per tensor.
                # Float64 preserves every FP32 partial sum exactly; the final
                # Python sum retains the original parameter order and norm.
                gradient_statistics = torch.stack(
                    (
                        torch.stack([torch.isfinite(g).all() for g in gradients]).to(
                            torch.float64
                        ),
                        torch.stack(
                            [g.detach().square().sum() for g in gradients]
                        ).to(torch.float64),
                    ),
                    dim=1,
                ).cpu().tolist()
                if any(not finite for finite, _ in gradient_statistics):
                    raise FloatingPointError("Nonfinite parameter gradient")
                gradient_norm = sum(value for _, value in gradient_statistics) ** 0.5
            else:
                gradient_norm = 0.0
            if step:
                self.optimizers[phase].step()
                self.updates += 1
                self.module_updates += int(phase == "modules")
                self.backbone_updates += int(phase == "backbone")
            if trainable and not bool(
                torch.stack([torch.isfinite(p).all() for p in trainable]).all().cpu()
            ):
                raise FloatingPointError("Nonfinite parameter")
            return {
                "kind": episode.kind,
                "phase": phase,
                "skipped": None,
                "query_ce": float(loss.detach()),
                "gradient_l2": gradient_norm,
                "support": len(episode.support_indices),
                "target_query": len(episode.query_target_indices),
                "background_query": len(episode.query_background_indices),
            }

    def fit_task(
        self, current, old, *, epochs=20, batch_size=64, seed=0, on_commit=None
    ):
        if self._task_dirty:
            raise RuntimeError("Restore a complete task checkpoint before retrying")
        if state_digest(self.encoder.anchor.state_dict()) != self._anchor_digest:
            raise RuntimeError("The history anchor must remain immutable")
        current = list(current)
        old = list(old)
        if not current or min(epochs, batch_size) < 1:
            raise ValueError("A task and positive budget are required")
        labels_new = [int(i[1]) for i in current]
        ids = [int(i[2]) for i in current]
        oldids = [int(i[2]) for i in old]
        if len(ids) != len(set(ids)) or set(ids) & set(oldids):
            raise ValueError("Duplicate/current-resident IDs")
        if set(labels_new) & set(self.classes):
            raise ValueError("Task classes already committed")
        if set(oldids) != set(self.memory_base):
            raise ValueError("Resident canonical cache mismatch")
        previous_count = self.statistics.count
        self._task_dirty = True
        self.statistics.register(labels_new)
        current_base = self._canonical(current, True)
        old_base = (
            torch.stack([self.memory_base[i] for i in oldids])
            if old
            else current_base.new_empty((0, self.feature_dim))
        )
        y_old = torch.tensor([int(i[1]) for i in old], dtype=torch.long)
        rng = torch.Generator().manual_seed(seed)
        summaries = []
        skipped = 0
        before = self.updates
        original_lrs = {
            "modules": self.options["module_lr"],
            "backbone": self.options["backbone_lr"],
        }
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            epoch_updates = self.updates
            epoch_module_updates = self.module_updates
            epoch_backbone_updates = self.backbone_updates
            epoch_skipped = skipped
            factor = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            for name, opt in self.optimizers.items():
                for group in opt.param_groups:
                    group["lr"] = original_lrs[name] * factor
            order = torch.randperm(len(current), generator=rng)
            losses = []
            for start in range(0, len(current), batch_size):
                pick = order[start : start + batch_size]
                batch = [current[i] for i in pick.tolist()]
                yc = torch.tensor([i[1] for i in batch], dtype=torch.long)
                labels = torch.cat((yc, y_old))
                items = batch + old
                canonical = torch.cat((current_base[pick], old_base))
                self._inc("logical_current_presentations", len(batch))
                for _ in range(self.options["outer_updates"]):
                    phase = self.options["schedule"][
                        self.episode_draws % len(self.options["schedule"])
                    ]
                    kind = (
                        "transfer"
                        if torch.rand((), generator=self.kind_rng)
                        < self.options["transfer_probability"]
                        else "anchored"
                    )
                    self.episode_draws += 1
                    episode = sample_episode(
                        yc,
                        y_old,
                        kind,
                        self.rng,
                        self.options["support_shots"],
                        self.options["background_shots"],
                        self.options["target_classes"],
                        self.options["query_per_class"],
                    )
                    event = self.episode_step(
                        episode, items, canonical, labels, len(batch), phase
                    )
                    if event["skipped"]:
                        skipped += 1
                    else:
                        losses.append(event["query_ce"])
            summaries.append(
                {
                    "epoch": epoch,
                    "mean_query_ce": float(np.mean(losses)) if losses else None,
                    "lr_factor": factor,
                }
            )
            print(
                json.dumps(
                    {
                        "epoch_progress": True,
                        "method": "current_adapted",
                        "training": "fixed_library_episodes",
                        "task": self.completed_tasks,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "loss_mean": summaries[-1]["mean_query_ce"],
                        "optimizer_updates": self.updates - epoch_updates,
                        "optimizer_updates_total": self.updates,
                        "module_updates": self.module_updates - epoch_module_updates,
                        "backbone_updates": self.backbone_updates
                        - epoch_backbone_updates,
                        "skipped_episodes": skipped - epoch_skipped,
                        "skipped_episodes_total": skipped,
                        "wall_seconds": time.perf_counter() - epoch_started,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        # No new task contribution was present in persistent G/C during any epoch.
        self.statistics.add(current_base, torch.tensor(labels_new))
        self.W0 = self.statistics.head().detach()
        self._inc("readout_solves", 1)
        canonical = torch.cat((current_base, old_base))
        all_items = current + old
        full = self._canonical(all_items, False)
        labels = torch.tensor([i[1] for i in all_items], dtype=torch.long)
        _, targets = self._targets(labels)
        self.V = centered_residual_head(
            full - canonical,
            canonical,
            targets,
            self.W0,
            self.options["residual_ridge"],
            class_weights(labels),
        ).detach()
        self._inc("readout_solves", 1)
        cache = {
            **self.memory_base,
            **{uid: current_base[i].clone() for i, uid in enumerate(ids)},
        }
        resident_ids = (
            list(on_commit(full[: len(current), :-1].numpy())) if on_commit else []
        )
        if len(resident_ids) != len(set(resident_ids)) or not set(resident_ids) <= set(
            cache
        ):
            raise ValueError("Memory callback returned invalid resident IDs")
        self.memory_base = {i: cache[i] for i in resident_ids}
        self.completed_tasks += 1
        self.task_sizes.append(len(current))
        self._task_dirty = False
        if self.statistics.count != previous_count + len(current):
            raise RuntimeError("Repeated history contribution")
        return {
            "task": self.completed_tasks - 1,
            "new_images": len(current),
            "history_count": self.statistics.count,
            "previous_history_count": previous_count,
            "epochs": epochs,
            "updates": self.updates - before,
            "module_updates_total": self.module_updates,
            "backbone_updates_total": self.backbone_updates,
            "skipped_episodes": skipped,
            "resident_anchor_rows": len(self.memory_base),
            "epoch_summaries": summaries,
            "counters": dict(self.counters),
            "solver_history": "current task added once after all learning epochs",
            "compute_note": (
                f"{self.options['outer_updates']} episode attempts per current minibatch; "
                "not matched FLOPs to supervised methods"
            ),
        }

    def forward(self, images, base_only=False):
        z0 = self.encoder.anchor_features(images).cpu().double()
        score = z0 @ self.W0
        if not base_only:
            zf = self.encoder.main_features(images, True).cpu().double()
            score = score + (zf - z0) @ self.V
        return (self.options["temperature"] * score).to(
            images.device, dtype=images.dtype
        )

    def training_state(self):
        if self._task_dirty:
            raise RuntimeError("Checkpoint only at complete task boundaries")
        if state_digest(self.encoder.anchor.state_dict()) != self._anchor_digest:
            raise RuntimeError("The history anchor must remain immutable")
        return cpu_tree(
            {
                "encoder": self.encoder.state_dict(),
                "statistics": self.statistics.state_dict(),
                "architecture": self.architecture,
                "W0": self.W0,
                "V": self.V,
                "memory_base": self.memory_base,
                "options": self.options,
                "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
                "rng": self.rng.get_state(),
                "kind_rng": self.kind_rng.get_state(),
                "completed_tasks": self.completed_tasks,
                "task_sizes": self.task_sizes,
                "episode_draws": self.episode_draws,
                "updates": self.updates,
                "module_updates": self.module_updates,
                "backbone_updates": self.backbone_updates,
                "counters": self.counters,
            }
        )

    def load_training_state(self, state):
        if state.get("architecture") != self.architecture:
            raise ValueError("Current-method library architecture changed on restore")
        if state["options"] != self.options:
            raise ValueError("Current-method options changed on restore")
        # Validate tensor dimensions and the coordinate-system identity before
        # altering this learner's tensors or moments.
        expected = self.encoder.state_dict()
        stored = state["encoder"]
        if set(expected) != set(stored) or any(
            stored[k].shape != v.shape or not torch.isfinite(stored[k]).all()
            for k, v in expected.items()
        ):
            raise ValueError("Invalid native encoder checkpoint")
        anchor = {
            k[len("anchor.") :]: v for k, v in stored.items() if k.startswith("anchor.")
        }
        if state_digest(anchor) != self._anchor_digest:
            raise ValueError("Checkpoint changes the immutable anchor")
        probe = StableStatistics(self.feature_dim, self.options["base_ridge"])
        probe.load_state_dict(state["statistics"])
        sizes = state["task_sizes"]
        tasks = state["completed_tasks"]
        if (
            not isinstance(tasks, int)
            or tasks < 0
            or any(not isinstance(n, int) or n < 1 for n in sizes)
            or sum(sizes) != probe.count
            or len(sizes) != tasks
        ):
            raise ValueError("Task/history checkpoint mismatch")
        shape = (self.feature_dim, len(probe.classes))
        if any(
            not isinstance(state[k], torch.Tensor)
            or state[k].shape != shape
            or not torch.isfinite(state[k]).all()
            for k in ("W0", "V")
        ):
            raise ValueError("Readout checkpoint mismatch")
        if any(
            not isinstance(k, int)
            or not isinstance(v, torch.Tensor)
            or v.shape != (self.feature_dim,)
            or not torch.isfinite(v).all()
            or float(v[-1]) != 1.0
            for k, v in state["memory_base"].items()
        ):
            raise ValueError("Invalid resident anchor cache")
        if not torch.allclose(
            state["W0"].cpu().double(), probe.head(), atol=1e-10, rtol=1e-10
        ):
            raise ValueError("Frozen head does not match the historical moments")
        self.encoder.load_state_dict(stored, strict=True)
        self.statistics = probe
        for name in ("W0", "V", "memory_base", "task_sizes", "counters"):
            setattr(self, name, cpu_tree(state[name]))
        for name in (
            "completed_tasks",
            "episode_draws",
            "updates",
            "module_updates",
            "backbone_updates",
        ):
            setattr(self, name, int(state[name]))
        for k, opt in self.optimizers.items():
            opt.load_state_dict(state["optimizers"][k])
        self.rng.set_state(state["rng"])
        self.kind_rng.set_state(state["kind_rng"])
        self._task_dirty = False
        if (
            sum(self.task_sizes) != self.statistics.count
            or len(self.task_sizes) != self.completed_tasks
        ):
            raise ValueError("Task/history checkpoint mismatch")
        if (
            self.W0.shape != (self.feature_dim, len(self.classes))
            or self.V.shape != self.W0.shape
        ):
            raise ValueError("Readout checkpoint mismatch")

    def resource_bytes(self):
        return {
            "encoder_weights": tensor_bytes(self.encoder.state_dict()),
            "statistics": tensor_bytes(self.statistics.state_dict()),
            "readouts": tensor_bytes((self.W0, self.V)),
            "resident_anchor_cache": tensor_bytes(self.memory_base),
            "optimizer_state": tensor_bytes(
                {k: v.state_dict() for k, v in self.optimizers.items()}
            ),
        }
