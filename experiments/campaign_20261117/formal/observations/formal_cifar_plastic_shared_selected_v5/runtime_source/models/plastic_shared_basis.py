"""PILOT plastic body + historical moment transport + shared predictor prior.

Backbone adaptation is exactly the history-transport CE/replay loop. The shared
prior is fitted afterward from reconstructed all-history moments, with an
isotropic reset in the new feature coordinates. No EB gradient is propagated
through backbone training or through the moment estimator in this version.
"""

from __future__ import annotations

from cl27.cores.plastic_shared_basis import PlasticSharedPredictorBasis
from models.history_transport import Learner as HistoryLearner


class Learner(HistoryLearner):
    def __init__(self, args):
        if args.get("basis_reset_each_task", True) is not True:
            raise ValueError(
                "Plastic shared basis currently requires basis_reset_each_task=True"
            )
        super().__init__(args)

    def _build_core(self):
        args = self.args
        return PlasticSharedPredictorBasis(
            self.backbone.out_dim,
            **self._transport_options(),
            basis_rank=args.get("basis_rank", 32),
            basis_epsilon=args.get("basis_epsilon", 1e-3),
            basis_noise_variance=args.get("basis_noise_variance", 1.0),
            basis_max_iter=args.get("basis_max_iter", 20),
            basis_tolerance=args.get("basis_tolerance", 1e-6),
        )
