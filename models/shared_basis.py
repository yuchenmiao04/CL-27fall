"""A frozen PILOT backbone with a learned shared prior over class predictors.

This route has no historical image buffer. Current task features are read once,
then discarded; the numerical core revisits all history through sufficient
statistics. The backbone and feature normalization must stay immutable.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np
from cl27.common import (
    IntegratedLearner,
    ScoreNetwork,
    build_backbone,
    extract_features,
    feature_normalization,
    state_hash,
)
from cl27.cores.shared_basis import SharedPredictorBasis


class Learner(IntegratedLearner):
    def __init__(self, args):
        super().__init__(args)
        self.normalization = feature_normalization(args)
        self.backbone = build_backbone(args).to(self._device)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.core = SharedPredictorBasis(
            feature_dim=self.backbone.out_dim,
            rank=args.get("basis_rank", 32),
            epsilon=args.get("basis_epsilon", 1e-3),
            noise_variance=args.get("basis_noise_variance", 1.0),
            max_iter=args.get("basis_max_iter", 20),
            tol=args.get("basis_tolerance", 1e-6),
            seed=args.get("seed", 0),
        )
        self._network = ScoreNetwork(
            self.backbone, self.core.predict_scores, self.normalization
        ).to(self._device)
        self._frozen_hash = state_hash(self.backbone)

    def incremental_train(self, data_manager):
        if state_hash(self.backbone) != self._frozen_hash:
            raise RuntimeError(
                "Shared-basis historical statistics require an immutable backbone"
            )
        current, self.test_loader = self.start_task(data_manager)
        x, labels, ids = extract_features(
            self.backbone,
            current,
            self._device,
            batch_size=int(
                self.args.get("feature_batch_size", self.args.get("batch_size", 64))
            ),
            normalization=self.normalization,
        )
        expected = list(range(self._known_classes, self._total_classes))
        if sorted(np.unique(labels).tolist()) != expected:
            raise ValueError(
                "The current dataset must contain exactly the new remapped classes"
            )
        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                "Each current fitting image must be committed exactly once"
            )
        if state_hash(self.backbone) != self._frozen_hash:
            raise RuntimeError(
                "Feature extraction changed the supposedly frozen backbone"
            )
        report = self.core.observe(x, labels)
        if self.core.classes != list(range(self._total_classes)):
            raise RuntimeError(
                "Predictor columns disagree with PILOT's remapped class vocabulary"
            )
        report.update(
            {
                "backbone_state_sha256": self._frozen_hash,
                "feature_normalization": self.normalization,
                "current_fitting_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "raw_replay_images": 0,
                "feature_map_frozen": True,
                "training_source": "current task only, deterministic test transform",
            }
        )
        self.task_diagnostics.append(report)

    def checkpoint_core(self):
        return {
            "normalization": self.normalization,
            "backbone": {
                key: value.detach().cpu().clone()
                for key, value in self.backbone.state_dict().items()
            },
            "backbone_state_sha256": self._frozen_hash,
            "backbone_provenance": copy.deepcopy(
                getattr(self.backbone, "cl27_provenance", None)
            ),
            "predictor": self.core.state_dict(),
        }

    def restore_core(self, state):
        if state.get("normalization") != self.normalization:
            raise ValueError("Shared-basis checkpoint feature normalization differs")
        # Validate the numerical history before touching the live model. The core
        # validates its moments and reconstructs its posterior without raw data.
        checked_core = SharedPredictorBasis(**self.core._config())
        checked_core.load_state_dict(state["predictor"])
        checked_backbone = copy.deepcopy(self.backbone)
        checked_backbone.load_state_dict(state["backbone"], strict=True)
        if state_hash(checked_backbone) != state.get("backbone_state_sha256"):
            raise ValueError("Shared-basis checkpoint backbone hash is inconsistent")
        self.backbone.load_state_dict(checked_backbone.state_dict(), strict=True)
        self.backbone.cl27_provenance = copy.deepcopy(state.get("backbone_provenance"))
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self._frozen_hash = state["backbone_state_sha256"]
        self.core.load_state_dict(checked_core.state_dict())
