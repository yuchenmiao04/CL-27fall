# Plastic shared predictor basis: native learner integration

Implementation: `models/plastic_shared_basis.py`. Numerical composition:
`cl27/cores/plastic_shared_basis.py`. Complete component derivations are in
`history_transport.md` and `shared_basis.md`; this document specifies their
coupling and the implemented training boundary.

## What is shared and what is plastic

The predictive ViT body is plastic. The initial pretrained ViT anchor is frozen
and supplies exact historical moments in its original coordinates. Resident old
images supply before/after coordinate pairs. At a task boundary, the transport
estimator reconstructs one approximate full-history predictive moment snapshot.
The shared empirical-Bayes predictor then fits to that snapshot in the current
body's coordinates. It replaces the ordinary isotropic ridge prediction head.

The learned covariance is `Sigma=D D^T+epsilon I`. Columns of `D` are shared
predictor directions and each seen class has posterior weights under that prior.
This is an abstract predictive decomposition, with a residual outside the shared
subspace; it is not a library of independently routed neural modules, a semantic
decomposition, or an external-model architecture. All class posterior weights are
refitted when the snapshot or shared covariance changes.

The native wrapper subclasses the history-transport learner. Its only numerical
substitution is the core factory. Backbone optimizer, expanded CE training head,
current/replay sampling, task access, herding, normalization, unique-arrival
accounting, frozen anchor checks and checkpoint handling are inherited. The
original `history_transport` still constructs the same `TransportHistory` core,
uses its ordinary ridge head, and keeps its existing checkpoint schema.

## Exact task order and the stagewise coupling

1. Obtain only current-task training images and the bounded resident old images.
   Extract previous/anchor replay coordinates needed by direct or chained mode.
2. Update the predictive body and expanded training CE head on current images
   plus sampled replay, using exactly the history-transport training budget.
3. Reencode accessible images in the updated coordinates. Estimate missing old
   moments and replace residents with their exact updated-coordinate moments.
   Commit current arrivals once, independent of training epoch count.
4. Pass the reconstructed full snapshot to the shared-prior fitting API:

   ```text
   G_basis = G_transport[:-1,:-1]
   C_basis = C_transport[:-1,:]
   feature_sum = G_transport[:-1,-1]
   class_counts = transport.counts
   N = sum(class_counts)
   ```

   This is a **replacement** snapshot. Passing it to cumulative row-observation
   code would incorrectly count older tasks repeatedly.
5. Reset the prior covariance to `epsilon I` in the new coordinate system;
   perform the configured constrained EM updates and recover the flat intercept.
   There is no warm start with a basis learned in obsolete body coordinates.
6. Update the raw-image memory once, using the same herding selection features
   and quota as history transport. Deploy the updated body plus EB posterior
   scores `u W+b` over all seen classes, with no task-ID input.

This is a stagewise method. No empirical-Bayes objective or posterior gradient
flows into the body optimizer. Thus, with the same seed and data order, its
body-gradient trajectory, replay selection and transported moment snapshot match
`history_transport`; its deployed readout differs. Tests verify that equivalence
over three small tasks. Fitting a shared head after body training does not by
itself train a body to become more reusable. A future differentiated or
alternating objective would be a separate algorithmic change.

Resetting `D` does not discard the historical supervised information: the
reconstructed full-history `G`, `C`, sums and counts participate in each fit.
The fitted basis may nevertheless change across stages. Its gauge, signs and
coordinates do not establish semantic component identity across body updates.
Carrying an old covariance through a general nonlinear feature change would
require additional justified transformations; this version does not assume one.

## Configuration and reporting

Method name: `plastic_shared_basis`. Default `basis_rank=32`,
`basis_epsilon=1e-3`, `basis_noise_variance=1`, `basis_max_iter=20`, and
`basis_tolerance=1e-6`. `basis_reset_each_task` must be true; false is rejected.
All transport and CE/replay parameters retain the history-transport defaults.
Default pretrained backbone and dataset protocol remain the common pinned PILOT
configuration; small tests explicitly use a random tiny ViT and synthetic RGB
images. No official test data selects these hyperparameters.

In the rank-zero/isotropic case, the deployed slope ridge coefficient is
`noise_variance/epsilon`, with a flat intercept. It is not the transport core's
`head_ridge`. The transport core currently still computes and stores its
auxiliary ordinary ridge head for validation/inspection; it is not deployed.
That extra solve and state are real overhead and are counted, along with the
shared-prior snapshot. The implementation currently retains duplicated moment
snapshots rather than claiming a deduplicated memory footprint.

The inherited diagnostics keep body gradients, parameter hashes, current/replay
training visits, exact unique-arrival counts, anchor invariance and raw-memory
payloads. Nested statistics additionally report the shared-prior fit, reset
policy and transported-moment surrogate interpretation. Exact anchor history
does not make the predictive history exact. With full resident coverage there
is no missing-population approximation; ordinary bounded memory generally has
discarded examples whose moments are estimated.

Inference needs one predictive body and its posterior slope/intercept. The
live training object still retains the frozen anchor, replay and full numerical
core; a separately stripped deployment artifact is not implemented here.
Resource fields describe those pieces explicitly, not an unmeasured peak-memory
or runtime advantage.

## Reading and adaptation decisions

The implementation builds on the detailed primary-source reviews in the two
component documents. Additional coupling checks included the following.

| Inspected source/code | Consequence for this implementation |
| --- | --- |
| [DPCR, Sections 3.1-3.3](https://arxiv.org/html/2503.05423v2), and [LDC, Section 3.3](https://arxiv.org/html/2407.08536v1) | Representation learning and coordinate compensation are separate stages. Their existing drift/moment methods prevent describing this stage order as a new invention. This variant changes the post-transport predictor prior and inherits the bounded-replay conditional residual estimator. |
| [SLCA, Section 3](https://arxiv.org/html/2303.05118v2), local `models/slca.py`; local `models/cofima.py` and `models/duct.py` | Inspected slow body/head learning, subsequent alignment, checkpoint interpolation/Fisher weighting, and model merging followed by classifier transport. These distinguish body adaptation from later readout changes. This wrapper preserves the existing CE trajectory for comparison; no model merging, Fisher interpolation, or SLCA Gaussian alignment is added. |
| [ELLA, Sections 2-3](https://proceedings.mlr.press/v28/ruvolo13.pdf) | Shared predictor components and compressed past supervision motivate reuse at the predictor level. This is not ELLA's sparse-code/Hessian update: every class posterior is refit from the full reconstructed moment snapshot. |
| [Tipping and Bishop, PPCA Sections 3 and Appendix A](https://www.miketipping.com/papers/met-mppca.pdf) | Low-rank-plus-isotropic covariance and its rotational ambiguity inform the EB core's constrained covariance update. The modeled objects here are class-predictor posteriors, not feature-density mixture components. Resetting at coordinate changes is an explicit implementation choice. |
| [R2-D2, Sections 3.1-3.2 and Appendix B](https://arxiv.org/html/1805.08136v2), [official solver source](https://github.com/bertinetto/r2d2/blob/master/fewshots/models/r2d2.py) | Inspected support/query encoding, primal/Woodbury ridge and query CE. Differentiating through the fitted predictor trains features for that rule. That coupling is absent here: the CE body stage is unchanged and NumPy EB/transport fitting is detached. Merely attaching the EB head is not meta-learning through the solver. |

## Limits to test subsequently

- Fitting anisotropic predictor shrinkage to approximate transported moments can
  learn transport artifacts as shared directions. EM ascent refers to the
  declared surrogate, not the inaccessible true-history likelihood or accuracy.
- The feature-learning objective remains CE with bounded replay. Improved
  cross-class sharing at the head does not prove improved future-class feature
  learning or positive transfer.
- Resetting the covariance avoids a stale-coordinate prior but may require more
  iterations than a justified warm start. Fixed finite EM steps, strong isotropic
  shrinkage, rank and residual modeling can all affect results.
- The fixed observation noise is consequential, not an arbitrary harmless scale.
  With one-hot outputs, a sufficiently large noise relative to per-class sample
  counts can correctly make the evidence optimum isotropic (`D=0`). A separate
  small synthetic test uses noise `0.01` to exercise a genuinely nonzero learned
  basis and verify deployment uses that head; this does not change formal
  defaults or establish their suitability for a dataset.
- Current implementation cost includes body training, anchor encoding, moment
  transport, the auxiliary ridge solve and repeated EB matrix decompositions.
  It must be measured rather than called lightweight from architecture alone.

## Small validation

`tests/test_pilot_plastic_shared_basis.py` checks real body updates, immutable
anchor, once-only full snapshots despite repeated epochs, direct/chained modes,
both raw-image budget rules, EB deployment scores with the correct intercept,
stagewise equivalence to the original transport body, deterministic checkpoint
continuation, mismatched-state rejection, fully resident joint-moment recovery,
and rejection of the unsupported stale-basis warm-start setting. Mathematical
composition tests live with the numerical core. These establish implementation
properties only; no full dataset experiment or improvement claim is included.
