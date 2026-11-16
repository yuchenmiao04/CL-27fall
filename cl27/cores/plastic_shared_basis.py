"""Approximate transported history with a reset, learned shared predictor prior.

Transport and shared-prior fitting are a single transactional update. The
transported full snapshot replaces the previous basis snapshot; it is never
passed as new rows to cumulative ``observe``. The learned covariance is fit in
the current feature coordinates from epsilon I every task, not warm-started in
an obsolete nonlinear representation.
"""

from __future__ import annotations

import copy

import numpy as np

from .shared_basis import SharedPredictorBasis
from .transport import TransportHistory


class PlasticSharedPredictorBasis:
    STATE_VERSION = 1

    def __init__(
        self,
        feature_dim,
        mode="direct",
        residual="class_conditional_psd",
        ridge=0.01,
        map_ridge=0.01,
        covariance_shrinkage=0.1,
        mean_shrinkage=20.0,
        crossfit_folds=5,
        seed=0,
        reference_feature_rtol=0.0,
        basis_rank=32,
        basis_epsilon=1e-3,
        basis_noise_variance=1.0,
        basis_max_iter=20,
        basis_tolerance=1e-6,
    ):
        self.transport = TransportHistory(
            feature_dim,
            mode=mode,
            residual=residual,
            ridge=ridge,
            map_ridge=map_ridge,
            covariance_shrinkage=covariance_shrinkage,
            mean_shrinkage=mean_shrinkage,
            crossfit_folds=crossfit_folds,
            seed=seed,
            reference_feature_rtol=reference_feature_rtol,
        )
        self.basis = SharedPredictorBasis(
            feature_dim,
            rank=basis_rank,
            epsilon=basis_epsilon,
            noise_variance=basis_noise_variance,
            max_iter=basis_max_iter,
            tol=basis_tolerance,
            seed=seed,
        )
        self.last_diagnostics = {}

    @property
    def feature_dim(self):
        return self.transport.feature_dim

    @property
    def mode(self):
        return self.transport.mode

    @property
    def classes(self):
        return self.transport.classes

    @property
    def counts(self):
        return self.transport.counts.copy()

    @property
    def tasks(self):
        return self.transport.tasks

    @property
    def head(self):
        return np.vstack((self.basis.weights, self.basis.intercept))

    @property
    def current_G(self):
        return self.transport.current_G.copy()

    @property
    def current_C(self):
        return self.transport.current_C.copy()

    @property
    def anchor_G(self):
        return self.transport.anchor_G.copy()

    @property
    def anchor_C(self):
        return self.transport.anchor_C.copy()

    @property
    def last_transport(self):
        return self.transport.last_transport.copy()

    def _config(self):
        return {
            "transport": self.transport._config(),
            "basis": self.basis._config(),
            "prior_coordinate_policy": "reset_isotropic_each_task",
        }

    def update_task(
        self,
        current_anchor,
        current_new,
        current_labels,
        *,
        replay_anchor=None,
        replay_reference=None,
        replay_new=None,
        replay_labels=None,
    ):
        candidate_transport = copy.deepcopy(self.transport)
        transport_report = candidate_transport.update_task(
            current_anchor,
            current_new,
            current_labels,
            replay_anchor=replay_anchor,
            replay_reference=replay_reference,
            replay_new=replay_new,
            replay_labels=replay_labels,
        )
        candidate_basis = SharedPredictorBasis(**self.basis._config())
        g, c = candidate_transport.current_G, candidate_transport.current_C
        counts = candidate_transport.counts
        rounded = np.round(counts)
        if not np.allclose(counts, rounded, atol=1e-8, rtol=0):
            raise ValueError("Transported class masses must remain integer counts")
        basis_report = candidate_basis.fit_moments(
            g[:-1, :-1],
            c[:-1],
            g[:-1, -1],
            rounded,
            int(sum(map(int, rounded))),
            candidate_transport.classes,
            candidate_transport.tasks,
            moment_kind="surrogate",
        )
        report = copy.deepcopy(transport_report)
        report.update(
            {
                "method": "plastic_shared_predictor_basis",
                "shared_basis": basis_report,
                "prior_coordinate_policy": "reset_isotropic_each_task",
                "history_snapshot_operation": "replace; never accumulate a transported snapshot twice",
                "classification_readout": "shared empirical-Bayes predictor with flat intercept",
                "objective_interpretation": "transported-moment surrogate, not true-history likelihood",
                "historical_objective": "transported_moment_surrogate",
                "basis_reset_each_task": True,
                "anchor_history_exact": True,
                "predictive_history_exact": False,
                "transport_ridge_role": "auxiliary transport readout only; deployed isotropic penalty is noise_variance/epsilon",
            }
        )
        self.transport, self.basis = candidate_transport, candidate_basis
        self.last_diagnostics = report
        return copy.deepcopy(report)

    def predict_scores(self, features):
        return self.basis.predict_scores(features)

    def resource_bytes(self):
        # Both transport's auxiliary ridge head and duplicated basis snapshot
        # are actually retained and therefore must both be charged.
        return self.transport.resource_bytes() + self.basis.resource_bytes()

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "kind": "plastic_shared_predictor_basis",
            "config": self._config(),
            "transport": self.transport.state_dict(),
            "basis": self.basis.state_dict(),
            "last_diagnostics": copy.deepcopy(self.last_diagnostics),
        }

    def load_state_dict(self, state):
        if (
            state.get("version") != self.STATE_VERSION
            or state.get("kind") != "plastic_shared_predictor_basis"
            or state.get("config") != self._config()
        ):
            raise ValueError(
                "Plastic shared-basis checkpoint configuration/version differs"
            )
        candidate_transport = TransportHistory(**self.transport._config())
        candidate_transport.load_state_dict(state["transport"])
        candidate_basis = SharedPredictorBasis(**self.basis._config())
        candidate_basis.load_state_dict(state["basis"])
        if (
            candidate_transport.classes != candidate_basis.classes
            or candidate_transport.tasks != state["basis"]["tasks"]
            or not np.array_equal(candidate_transport.counts, candidate_basis.counts)
            or candidate_basis.n_samples != int(candidate_transport.counts.sum())
        ):
            raise ValueError(
                "Transport and shared-basis checkpoint vocabulary/counts differ"
            )
        g, c = candidate_transport.current_G, candidate_transport.current_C
        for provided, expected in (
            (candidate_basis.G, g[:-1, :-1]),
            (candidate_basis.C, c[:-1]),
            (candidate_basis.feature_sum, g[:-1, -1]),
        ):
            if not np.allclose(provided, expected, atol=1e-10, rtol=1e-10):
                raise ValueError(
                    "Transport and shared-basis checkpoint moment snapshots differ"
                )
        self.transport, self.basis = candidate_transport, candidate_basis
        self.last_diagnostics = copy.deepcopy(state.get("last_diagnostics", {}))
