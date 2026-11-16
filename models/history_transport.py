"""PILOT learner with a plastic backbone and transported historical moments."""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from cl27.common import (
    IntegratedLearner,
    ScoreNetwork,
    build_backbone,
    extract_features,
    feature_normalization,
    normalize_features,
    numeric_bytes,
    state_hash,
)
from cl27.cores.transport import TransportHistory
from cl27.replay import ReplayMemory
from cl27.transport_training import adapt_backbone, expand_head


def _cpu_state(module):
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


class Learner(IntegratedLearner):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = build_backbone(args).to(self._device)
        self.backbone_provenance = copy.deepcopy(
            getattr(self.backbone, "cl27_provenance", None)
        )
        self.anchor = copy.deepcopy(self.backbone).eval().requires_grad_(False)
        self.normalization = feature_normalization(args)
        self.core = self._build_core()
        self.replay = ReplayMemory(args)
        self.ce_head = None
        self.rng = np.random.default_rng(int(args.get("seed", 0)) + 113)
        self.anchor_hash = state_hash(self.anchor)
        self._initial_anchor_hash = self.anchor_hash
        self._install_network()

    def _transport_options(self):
        """Identical transport settings shared by ridge and EB-head variants."""
        args = self.args
        return {
            "mode": args.get("transport_mode", "direct"),
            "residual": args.get("residual_model", "class_conditional_psd"),
            "ridge": float(args.get("head_ridge", 0.01)),
            "map_ridge": float(args.get("map_ridge", 0.01)),
            "covariance_shrinkage": float(args.get("covariance_shrinkage", 0.1)),
            "mean_shrinkage": float(args.get("mean_shrinkage", 20.0)),
            "crossfit_folds": int(args.get("crossfit_folds", 5)),
            "seed": int(args.get("seed", 0)),
            "reference_feature_rtol": float(
                args.get(
                    "reference_feature_rtol",
                    64 * torch.finfo(next(self.backbone.parameters()).dtype).eps,
                )
            ),
        }

    def _build_core(self):
        """Factory seam; the original route remains a TransportHistory core."""
        return TransportHistory(self.backbone.out_dim, **self._transport_options())

    @property
    def exemplar_size(self):
        return len(self.replay)

    def _install_network(self):
        self._network = ScoreNetwork(
            self.backbone, self.core.predict_scores, self.normalization
        )

    def _encode(self, body, dataset, normalization=None):
        return extract_features(
            body,
            dataset,
            self._device,
            batch_size=int(self.args.get("feature_batch_size", 32)),
            normalization=normalization or self.normalization,
        )

    def _normalize_numpy(self, features):
        if self.normalization == "none":
            return features
        # Use the same device/dtype normalization as deployment. The raw CLS
        # cache is float64 only for transport; the encoder emitted float32.
        with torch.no_grad():
            tensor = torch.as_tensor(
                features,
                device=self._device,
                dtype=next(self.backbone.parameters()).dtype,
            )
            return normalize_features(tensor, self.normalization).cpu().double().numpy()

    def incremental_train(self, data_manager):
        if state_hash(self.anchor) != self._initial_anchor_hash:
            raise RuntimeError("The exact-history anchor changed before task access")
        current, _ = self.start_task(data_manager)
        expected = np.arange(self._known_classes, self._total_classes)
        train = data_manager.get_dataset(expected, source="train", mode="train")
        # Stable IDs prove that augmented training and canonical commits refer to
        # exactly the same arrivals. No test/future data is used for adaptation.
        if not np.array_equal(
            np.asarray(train.ids), np.asarray(current.ids)
        ) or not np.array_equal(np.asarray(train.labels), np.asarray(current.labels)):
            raise ValueError("Augmented and canonical task samples differ")
        old_dataset = self.replay.dataset(current.trsf) if len(self.replay) else None
        replay_args = {}
        if old_dataset is not None:
            old_anchor, old_labels, old_ids = self._encode(self.anchor, old_dataset)
            replay_args.update(replay_anchor=old_anchor, replay_labels=old_labels)
            if self.core.mode == "chained":
                previous, labels, ids = self._encode(self.backbone, old_dataset)
                if not (
                    np.array_equal(labels, old_labels) and np.array_equal(ids, old_ids)
                ):
                    raise RuntimeError("Previous representation lost replay alignment")
                replay_args["replay_reference"] = previous
        before = state_hash(self.backbone)
        self.ce_head = expand_head(
            self.ce_head, self.backbone.out_dim, self._total_classes, self._device
        )
        training = adapt_backbone(
            self.backbone,
            self.ce_head,
            train,
            self.replay,
            self.args,
            self._device,
            self.rng,
        )
        current_raw, labels, ids = self._encode(
            self.backbone, current, normalization="none"
        )
        current_anchor, anchor_labels, anchor_ids = self._encode(self.anchor, current)
        if not (
            np.array_equal(labels, anchor_labels) and np.array_equal(ids, anchor_ids)
        ):
            raise RuntimeError("Anchor and current features lost arrival alignment")
        if old_dataset is not None:
            new, new_labels, new_ids = self._encode(self.backbone, old_dataset)
            if not (
                np.array_equal(new_labels, old_labels)
                and np.array_equal(new_ids, old_ids)
            ):
                raise RuntimeError("Updated representation lost replay alignment")
            replay_args["replay_new"] = new
        statistics = self.core.update_task(
            current_anchor, self._normalize_numpy(current_raw), labels, **replay_args
        )
        self.replay.update(current, current_raw, self.core.classes)
        if state_hash(self.anchor) != self.anchor_hash:
            raise RuntimeError("The exact-history anchor changed")
        after = state_hash(self.backbone)
        self._install_network()
        self.task_diagnostics.append(
            {
                "task": self._cur_task,
                "method": self.args.get("model_name", "history_transport"),
                "training": training,
                "statistics": statistics,
                "unique_current_arrivals": len(current),
                "arrival_id_min": int(np.min(ids)),
                "arrival_id_max": int(np.max(ids)),
                "backbone_changed": before != after,
                "backbone_sha256_before": before,
                "backbone_sha256_after": after,
                "anchor_sha256": self.anchor_hash,
                "anchor_frozen": True,
                "replay_images": len(self.replay),
                "replay_selection": self.replay.selection,
                "transport_unbiasedness_claimed": False,
                "residual_cross_covariance_zero_is_assumption": True,
                "resources": self.resource_report(),
            }
        )

    def resource_report(self):
        body = sum(p.numel() * p.element_size() for p in self.backbone.parameters())
        anchor = sum(p.numel() * p.element_size() for p in self.anchor.parameters())
        ce = numeric_bytes(self.ce_head.state_dict()) if self.ce_head is not None else 0
        return {
            "deployment_backbone_parameter_bytes": body,
            "deployment_analytic_head_bytes": self.core.head.nbytes,
            "training_frozen_anchor_parameter_bytes": anchor,
            "training_ce_head_bytes": ce,
            "historical_statistics_bytes": self.core.resource_bytes(),
            "replay_raw_payload_bytes": self.replay.payload_bytes(),
            "inference_backbones": 1,
            "training_backbones": 2,
        }

    def checkpoint_core(self):
        return {
            "version": 1,
            "normalization": self.normalization,
            "backbone": _cpu_state(self.backbone),
            "anchor": _cpu_state(self.anchor),
            "anchor_hash": self.anchor_hash,
            "analytic": self.core.state_dict(),
            "initial_anchor_hash": self._initial_anchor_hash,
            "backbone_provenance": copy.deepcopy(self.backbone_provenance),
            "replay": self.replay.state_dict(),
            "ce_head": _cpu_state(self.ce_head) if self.ce_head else None,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_core(self, state):
        if state["version"] != 1 or state["normalization"] != self.normalization:
            raise ValueError("Transport checkpoint version/normalization differs")
        if (
            state.get("initial_anchor_hash") != self._initial_anchor_hash
            or state.get("backbone_provenance") != self.backbone_provenance
            or state.get("anchor_hash") != self._initial_anchor_hash
        ):
            raise ValueError(
                "Transport checkpoint initial-anchor hash/provenance differs"
            )
        # Validate independent candidates first. A rejected checkpoint must not
        # leave a live learner with new body weights and old statistics/replay.
        candidate_backbone = copy.deepcopy(self.backbone)
        candidate_anchor = copy.deepcopy(self.anchor)
        candidate_backbone.load_state_dict(state["backbone"])
        candidate_anchor.load_state_dict(state["anchor"])
        candidate_anchor.eval().requires_grad_(False)
        if state_hash(candidate_anchor) != state["anchor_hash"]:
            raise ValueError("Frozen anchor checkpoint hash differs")
        candidate_core = self._build_core()
        candidate_core.load_state_dict(state["analytic"])
        candidate_replay = ReplayMemory(self.args)
        candidate_replay.load_state_dict(state["replay"])
        if candidate_core.classes != list(range(len(candidate_core.classes))):
            raise ValueError(
                "Transport checkpoint must use PILOT's remapped vocabulary"
            )
        if (
            candidate_replay.classes != candidate_core.classes
            or candidate_replay.seen != int(candidate_core.counts.sum())
        ):
            raise ValueError("Replay and historical statistics disagree")
        head_state = state["ce_head"]
        candidate_head = None
        if head_state is not None:
            with torch.random.fork_rng(devices=[]):
                candidate_head = nn.Linear(
                    self.backbone.out_dim, len(candidate_core.classes)
                ).to(self._device)
            candidate_head.load_state_dict(head_state)
        elif candidate_core.classes:
            raise ValueError("A trained checkpoint requires its adaptation head")
        candidate_rng = np.random.default_rng()
        candidate_rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.backbone, self.anchor = candidate_backbone, candidate_anchor
        self.core, self.replay, self.ce_head = (
            candidate_core,
            candidate_replay,
            candidate_head,
        )
        self.anchor_hash, self.rng = state["anchor_hash"], candidate_rng
        self.backbone.eval()
        self._install_network()
