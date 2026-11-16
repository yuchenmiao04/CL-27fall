"""Backbone adaptation for history transport, using current data and bounded replay.

The CE head is a training instrument. Deployment uses the transported analytic
head. Statistics are not updated by this loop; the caller commits each new task
once after adaptation. Optimizer state is intentionally reset at task boundaries.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def expand_head(head, feature_dim, classes, device):
    result = nn.Linear(feature_dim, classes).to(device)
    if head is not None:
        if head.out_features >= classes:
            raise ValueError("The CE vocabulary must grow at each task")
        with torch.no_grad():
            result.weight[: head.out_features].copy_(head.weight)
            result.bias[: head.out_features].copy_(head.bias)
    return result


def adapt_backbone(backbone, head, current, replay, args, device, rng):
    """Train one task without test access or persisted-feature training caches."""
    epochs = int(args.get("epochs", 20))
    batch_size = int(args.get("batch_size", 64))
    replay_batch_size = int(args.get("replay_batch_size", 64))
    if epochs < 1 or batch_size < 1 or replay_batch_size < 0:
        raise ValueError("Invalid transport adaptation budget")
    learning_rates = [
        float(args.get("backbone_lr", 1e-5)),
        float(args.get("head_lr", 1e-3)),
    ]
    if any(not np.isfinite(x) or x <= 0 for x in learning_rates):
        raise ValueError("Transport learning rates must be finite and positive")
    loader = DataLoader(
        current,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.get("num_workers", 0)),
    )
    replay_dataset = replay.dataset(current.trsf) if len(replay) else None
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": learning_rates[0]},
            {"params": head.parameters(), "lr": learning_rates[1]},
        ],
        weight_decay=float(args.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses, updates, current_visits, replay_visits = [], 0, 0, 0
    gradient_squared, maximum_gradient = 0.0, 0.0
    for _ in range(epochs):
        backbone.train()
        head.train()
        for _, images, labels in loader:
            current_visits += len(labels)
            if replay_dataset is not None and replay_batch_size:
                indices = replay.sample_indices(replay_batch_size, rng)
                _, old_images, old_labels = default_collate(
                    [replay_dataset[int(i)] for i in indices]
                )
                replay_visits += len(old_labels)
                images = torch.cat((images, old_images), dim=0)
                labels = torch.cat((labels, old_labels), dim=0)
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(head(backbone(images)), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite backbone adaptation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Accumulate on-device, with one host transfer per update. This also
            # verifies that the plastic body actually receives a finite gradient.
            gradients = [p.grad for p in backbone.parameters() if p.grad is not None]
            norm = torch.sqrt(sum(g.detach().square().sum() for g in gradients))
            gradient = float(norm)
            if not np.isfinite(gradient):
                raise FloatingPointError("Nonfinite backbone gradient")
            gradient_squared += gradient * gradient
            maximum_gradient = max(maximum_gradient, gradient)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        scheduler.step()
    backbone.eval()
    head.eval()
    return {
        "optimizer": "AdamW",
        "schedule": "cosine_per_task",
        "epochs": epochs,
        "optimizer_state_reset_per_task": True,
        "optimizer_updates": updates,
        "current_training_visits": current_visits,
        "replay_training_visits": replay_visits,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
        "backbone_gradient_l2_rms": float(np.sqrt(gradient_squared / updates)),
        "backbone_gradient_l2_max": maximum_gradient,
        "backbone_lr": learning_rates[0],
        "head_lr": learning_rates[1],
        "batch_size": batch_size,
        "replay_batch_size": replay_batch_size,
    }
