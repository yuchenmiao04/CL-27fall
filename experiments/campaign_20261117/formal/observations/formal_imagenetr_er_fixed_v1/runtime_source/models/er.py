"""Supervised experience-replay baseline with one plastic PILOT backbone.

This control deliberately shares the history-transport body's CE adaptation and
herding code. It deploys the trained CE head itself and has no anchor, analytic
historical moments, adapters or extra predictor-learning stage.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from cl27.common import (
    IntegratedLearner,
    build_backbone,
    extract_features,
    numeric_bytes,
    state_hash,
)
from cl27.replay import ReplayMemory
from cl27.supervised import CEClassifier
from cl27.transport_training import adapt_backbone, execute_adaptation, expand_head


def _cpu_state(module):
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def _finite_module(module, name):
    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
        raise ValueError(f"Nonfinite {name} state")


class Learner(IntegratedLearner):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = build_backbone(args).to(self._device)
        self.backbone_provenance = copy.deepcopy(
            getattr(self.backbone, "cl27_provenance", None)
        )
        self._initial_backbone_hash = state_hash(self.backbone)
        self.ce_head = None
        self.replay = ReplayMemory(args)
        # Same dedicated replay-minibatch generator as the history-body control.
        self.rng = np.random.default_rng(int(args.get("seed", 0)) + 113)
        self.completed_tasks = 0
        self.task_arrivals = []
        self.task_class_sizes = []
        self._task_dirty = False
        self._install_network()

    @property
    def exemplar_size(self):
        return len(self.replay)

    def _install_network(self):
        self._network = CEClassifier(self.backbone, self.ce_head)

    def _training_configuration(self):
        return {
            "epochs": int(self.args.get("epochs", 20)),
            "batch_size": int(self.args.get("batch_size", 64)),
            "replay_batch_size": int(self.args.get("replay_batch_size", 64)),
            "backbone_lr": float(self.args.get("backbone_lr", 1e-5)),
            "head_lr": float(self.args.get("head_lr", 1e-3)),
            "weight_decay": float(self.args.get("weight_decay", 1e-4)),
            "feature_batch_size": int(self.args.get("feature_batch_size", 32)),
            "seed": int(self.args.get("seed", 0)),
        }

    def incremental_train(self, data_manager):
        if self._task_dirty:
            raise RuntimeError("Restore the preceding complete task before retrying ER")
        current, _ = self.start_task(data_manager)
        self._task_dirty = True
        expected = np.arange(self._known_classes, self._total_classes)
        train = data_manager.get_dataset(expected, source="train", mode="train")
        if not np.array_equal(
            np.asarray(current.ids), np.asarray(train.ids)
        ) or not np.array_equal(np.asarray(current.labels), np.asarray(train.labels)):
            raise ValueError("ER augmented and canonical task samples differ")
        labels, ids = np.asarray(current.labels), np.asarray(current.ids)
        if (
            labels.dtype.kind not in "iu"
            or ids.dtype.kind not in "iu"
            or sorted(np.unique(labels).tolist()) != expected.tolist()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError(
                "ER requires exactly the new remapped classes and unique arrival IDs"
            )
        if set(map(int, ids)) & {record.sample_id for record in self.replay.records}:
            raise ValueError("ER current arrivals overlap resident old image IDs")
        before = state_hash(self.backbone)
        self.ce_head = expand_head(
            self.ce_head, self.backbone.out_dim, self._total_classes, self._device
        )
        training = execute_adaptation(
            self,
            adapt_backbone,
            self.backbone,
            self.ce_head,
            train,
            self.replay,
            self.args,
            self._device,
            self.rng,
        )
        _finite_module(self.backbone, "ER backbone")
        _finite_module(self.ce_head, "ER CE head")
        # Use exactly the history learner's raw canonical CLS values for native
        # herding. The selection helper itself applies its prescribed L2 scaling.
        features, encoded_labels, encoded_ids = extract_features(
            self.backbone,
            current,
            self._device,
            batch_size=int(self.args.get("feature_batch_size", 32)),
            normalization="none",
        )
        if not (
            np.array_equal(labels, encoded_labels) and np.array_equal(ids, encoded_ids)
        ):
            raise RuntimeError("ER replay-selection features lost sample alignment")
        self.replay.update(current, features, list(range(self._total_classes)))
        self.completed_tasks += 1
        self.task_arrivals.append(len(current))
        self.task_class_sizes.append(len(expected))
        self._task_dirty = False
        self._install_network()
        after = state_hash(self.backbone)
        self.task_diagnostics.append(
            {
                "task": self._cur_task,
                "method": "er",
                "training": training,
                "unique_current_arrivals": len(current),
                "unique_fitting_images_seen": self.replay.seen,
                "completed_tasks": self.completed_tasks,
                "seen_classes": self._total_classes,
                "backbone_changed": before != after,
                "backbone_sha256_before": before,
                "backbone_sha256_after": after,
                "initial_backbone_sha256": self._initial_backbone_hash,
                "backbone_provenance": copy.deepcopy(self.backbone_provenance),
                "ce_head_sha256": state_hash(self.ce_head),
                "replay_images": len(self.replay),
                "replay_selection": self.replay.selection,
                "prediction_head": "trained CE linear head on raw CLS",
                "prediction_space": "all seen classes without task ID or class-block mask",
                "historical_statistics": "none",
                "anchor_backbones": 0,
                "resources": self.resource_report(),
            }
        )

    def resource_report(self):
        return {
            "deployment_backbone_parameter_bytes": sum(
                p.numel() * p.element_size() for p in self.backbone.parameters()
            ),
            "deployment_ce_head_bytes": (
                numeric_bytes(self.ce_head.state_dict()) if self.ce_head else 0
            ),
            "training_frozen_anchor_parameter_bytes": 0,
            "historical_statistics_bytes": 0,
            "replay_raw_payload_bytes": self.replay.payload_bytes(),
            "inference_backbones": 1,
            "training_backbones": 1,
        }

    def checkpoint_core(self):
        if self._task_dirty:
            raise RuntimeError("ER checkpoints require a complete task boundary")
        return {
            "version": 1,
            "kind": "experience_replay_ce",
            "training_configuration": self._training_configuration(),
            "backbone_provenance": copy.deepcopy(self.backbone_provenance),
            "initial_backbone_hash": self._initial_backbone_hash,
            "backbone": _cpu_state(self.backbone),
            "ce_head": _cpu_state(self.ce_head) if self.ce_head is not None else None,
            "replay": self.replay.state_dict(),
            "completed_tasks": self.completed_tasks,
            "task_arrivals": list(self.task_arrivals),
            "task_class_sizes": list(self.task_class_sizes),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_core(self, state):
        if state.get("version") != 1 or state.get("kind") != "experience_replay_ce":
            raise ValueError("Unsupported ER checkpoint")
        if (
            state.get("backbone_provenance") != self.backbone_provenance
            or state.get("initial_backbone_hash") != self._initial_backbone_hash
            or state.get("training_configuration") != self._training_configuration()
        ):
            raise ValueError(
                "ER checkpoint initial provenance or training configuration differs"
            )
        # All candidates, including RNG and replay, are checked before touching
        # live tensors. No optimizer state is needed because each task resets it.
        body = copy.deepcopy(self.backbone)
        try:
            body.load_state_dict(state["backbone"], strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError("Invalid ER backbone dimensions/state") from error
        _finite_module(body, "ER backbone")
        memory = ReplayMemory(self.args)
        memory.load_state_dict(state["replay"])
        classes = memory.classes
        if classes != list(range(len(classes))):
            raise ValueError("ER checkpoint must use PILOT's remapped class vocabulary")
        tasks, arrivals, sizes = (
            state["completed_tasks"],
            state["task_arrivals"],
            state["task_class_sizes"],
        )
        if (
            isinstance(tasks, bool)
            or not isinstance(tasks, int)
            or tasks < 0
            or not isinstance(arrivals, list)
            or not isinstance(sizes, list)
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 1
                for x in arrivals + sizes
            )
            or len(arrivals) != tasks
            or len(sizes) != tasks
            or sum(arrivals) != memory.seen
            or sum(sizes) != len(classes)
            or bool(tasks) != bool(classes)
        ):
            raise ValueError(
                "ER checkpoint task, unique-arrival and class counters disagree"
            )
        head_state = state["ce_head"]
        head = None
        if classes:
            if (
                not isinstance(head_state, dict)
                or set(head_state) != {"weight", "bias"}
                or any(
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_floating_point()
                    for tensor in head_state.values()
                )
            ):
                raise ValueError("A trained ER checkpoint requires its CE head")
            if head_state["weight"].shape != (len(classes), body.out_dim) or head_state[
                "bias"
            ].shape != (len(classes),):
                raise ValueError(
                    "ER CE-head dimensions disagree with the seen vocabulary"
                )
            with torch.random.fork_rng(devices=[]):
                head = nn.Linear(body.out_dim, len(classes)).to(self._device)
            head.load_state_dict(head_state, strict=True)
            _finite_module(head, "ER CE head")
        elif head_state is not None or state_hash(body) != self._initial_backbone_hash:
            raise ValueError("An empty ER checkpoint contains trained parameters")
        generator = np.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(state["rng"])
        body.requires_grad_(True).eval()
        if head is not None:
            head.requires_grad_(True).eval()
        self.backbone, self.ce_head, self.replay, self.rng = (
            body,
            head,
            memory,
            generator,
        )
        self.completed_tasks = tasks
        self.task_arrivals, self.task_class_sizes = list(arrivals), list(sizes)
        self._task_dirty = False
        self._install_network()

    def restore_checkpoint(self, state):
        core = state.get("core", {})
        tasks = core.get("completed_tasks")
        classes = core.get("replay", {}).get("classes", [])
        if (
            isinstance(tasks, bool)
            or not isinstance(tasks, int)
            or state.get("cur_task") != tasks - 1
            or state.get("known_classes") != len(classes)
            or state.get("total_classes") != len(classes)
            or not isinstance(state.get("task_diagnostics"), list)
            or len(state["task_diagnostics"]) != tasks
        ):
            raise ValueError("ER outer lifecycle and saved training counters disagree")
        super().restore_checkpoint(state)
