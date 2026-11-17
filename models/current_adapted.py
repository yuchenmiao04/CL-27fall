"""The anchored fixed-module candidate, using PILOT's native ViT and tasks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from cl27.common import (
    IntegratedLearner,
    build_backbone,
    feature_normalization,
    state_hash,
)
from cl27.current.engine import DEFAULTS, FixedLibraryEngine
from cl27.replay import ReplayMemory


class RawTransform:
    """Apply exactly the dataset transform, including PILOT's RGB loading rule."""

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, image):
        if self.transform is None:
            raise RuntimeError("Dataset transform has not been bound")
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Replay stores original RGB arrays or image paths")
        return self.transform(image)


class CurrentNetwork(nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.feature_dim = engine.encoder.width

    def extract_vector(self, images):
        return self.engine.encoder.raw_main(images, True)

    def forward(self, images):
        return {"logits": self.engine(images)}


class Learner(IntegratedLearner):
    def __init__(self, args):
        super().__init__(args)
        if feature_normalization(args) != "l2":
            raise ValueError(
                "Current anchored method requires l2 feature normalization"
            )
        self.memory = ReplayMemory(args)
        self.canonical_transform = RawTransform()
        self.training_transform = RawTransform()
        options = {k: args.get(k, v) for k, v in DEFAULTS.items()}
        # A separate, opt-in episode distribution. replay_batch_size is the CE
        # learner quota and has never bounded this method's episode old pool.
        options["episode_replay_batch_size"] = args.get("episode_replay_batch_size")
        options["schedule"] = tuple(options["schedule"])
        options["support_shots"] = tuple(options["support_shots"])
        backbone = build_backbone(args)
        self.provenance = copy.deepcopy(backbone.cl27_provenance)
        self.backbone_provenance = copy.deepcopy(self.provenance)
        self.core = FixedLibraryEngine(
            backbone,
            positions=args.get("module_layers", [7, 9]),
            count=int(args.get("modules_per_layer", 4)),
            rank=int(args.get("module_rank", 8)),
            scale=float(args.get("module_scale", 0.1)),
            options=options,
            device=self._device,
            canonical_transform=self.canonical_transform,
            training_transform=self.training_transform,
        )
        self._network = CurrentNetwork(self.core)

    @property
    def exemplar_size(self):
        return len(self.memory)

    def incremental_train(self, data_manager):
        current, _ = self.start_task(data_manager)
        training = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        if not np.array_equal(current.ids, training.ids):
            raise ValueError(
                "Current canonical and training rows must have the same stable order"
            )
        self.canonical_transform.transform = current.trsf
        self.training_transform.transform = training.trsf
        rows = list(zip(current.images, current.labels, current.ids))
        old = [(r.image, r.label, r.sample_id) for r in self.memory.records]
        before = state_hash(self.core.encoder.main)
        modules_before = state_hash(self.core.encoder.libraries)

        def commit(features):
            self.memory.update(current, features, list(range(self._total_classes)))
            return [r.sample_id for r in self.memory.records]

        diagnostics = self.core.fit_task(
            rows,
            old,
            epochs=int(self.args.get("epochs", 20)),
            batch_size=int(self.args.get("batch_size", 64)),
            seed=int(self.args.get("seed", 0)) + 1009 * self._cur_task,
            on_commit=commit,
        )
        diagnostics.update(
            {
                "replay_images": len(self.memory),
                "replay_payload_bytes": self.memory.payload_bytes(),
                "episode_replay_policy": (
                    "all_resident"
                    if self.core.options["episode_replay_batch_size"] is None
                    else "bounded_old_class_balanced"
                ),
                "ce_replay_batch_size_applies": False,
                "backbone_changed": before != state_hash(self.core.encoder.main),
                "backbone_sha256_before": before,
                "backbone_sha256_after": state_hash(self.core.encoder.main),
                "modules_changed": modules_before
                != state_hash(self.core.encoder.libraries),
                "anchor_sha256": state_hash(self.core.encoder.anchor),
                "numeric_payload_bytes": self.core.resource_bytes(),
                "backbone_provenance": copy.deepcopy(self.provenance),
            }
        )
        self.task_diagnostics.append(diagnostics)

    def checkpoint_core(self):
        return {
            "version": 1,
            "engine": self.core.training_state(),
            "memory": self.memory.state_dict(),
            "provenance": copy.deepcopy(self.provenance),
        }

    def restore_core(self, state):
        if state["version"] != 1 or state["provenance"] != self.provenance:
            raise ValueError("Current checkpoint backbone provenance/version mismatch")
        # Probe native encoders with fresh hooks: deepcopy of a hooked encoder
        # would retain hook closures referencing the original library object.
        checked_memory = ReplayMemory(self.args)
        checked_memory.load_state_dict(state["memory"])
        architecture = self.core.architecture
        with torch.random.fork_rng(devices=[]):
            checked = FixedLibraryEngine(
                self.core.encoder.anchor,
                positions=architecture["positions"],
                count=architecture["count"],
                rank=architecture["rank"],
                scale=architecture["scale"],
                options=self.core.options,
                device=self._device,
                canonical_transform=self.canonical_transform,
                training_transform=self.training_transform,
            )
        checked.load_training_state(state["engine"])
        if set(checked.memory_base) != {r.sample_id for r in checked_memory.records}:
            raise ValueError("Resident anchor cache and image memory differ")
        if (
            checked_memory.seen != checked.statistics.count
            or checked_memory.classes != checked.classes
        ):
            raise ValueError("History and replay checkpoint classes/count differ")
        self.core = checked
        self.memory = checked_memory
        self._network = CurrentNetwork(self.core)
