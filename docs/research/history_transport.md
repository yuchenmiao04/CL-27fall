# Plastic backbone and historical-statistics transport

## Implemented scope and relationship to the preceding design

The fourth learner [plastic_shared_basis](plastic_shared_basis.md) reuses this
route's exact CE/backbone/replay trajectory and moment estimator, then deploys a
learned predictor covariance prior in place of this route's isotropic ridge
head. The original `history_transport` algorithm and checkpoint schema remain
available. The factory seam is described in [plastic_shared_wrapper.md](plastic_shared_wrapper.md).

The learner is `models/history_transport.py`, registered through PILOT's learner
factory. The numerical estimator is `cl27/cores/transport.py`; the current-task
gradient loop is `cl27/transport_training.py`. It implements task-based
class-incremental training and evaluation: training knows task boundaries;
prediction receives neither a task ID nor a class-block mask.

The objective is to retain the earlier V4 design's all-arrival sufficient
statistics while permitting the predictive representation to change. V4's exact
fixed-representation history is preserved as an auxiliary frozen-anchor
coordinate system. The deployed predictive branch is a separately trained
backbone plus a reconstructed analytic head. The anchor is not an inference
ensemble member or an immutable teacher target imposed on every feature.

This route does not implement a fixed library of learned modules, semantic
subquestions, an external language model, or an explicit reuse objective. It is
the plastic-representation/statistical-history alternative to the shared-basis
and current-adapted routes. Shared body parameters can transfer across classes;
whether they actually transfer useful knowledge is an empirical question.

The direct estimator starts from exact initial-anchor history at every task.
An optional chained estimator starts from the previous estimated predictive
history. Both are implemented; the direct estimator is the default. The
unidentified joint-anchor/previous-coordinate estimator called V9 in earlier
discussion is deferred: the required missing-population cross moments have not
been identified, and the implementation does not fabricate zero cross blocks.

## Backbone adaptation and image access

Let `a(x)` be the initial frozen backbone and `f_t(x)` the body after task `t`.
Both have the same feature dimension `d`. Main-protocol backbone creation uses
PILOT's pinned timm ViT-B/16 and its exact AugReg IN21K-to-IN1K weight provenance;
tests use an explicitly labeled random tiny ViT. An optional common L2
normalization is applied to all analytic features, including anchor features,
replay features, commits, and inference. It is not a class-dependent operation.

During task `t`, a CE linear head expands to all seen classes, preserving its old
rows. Each minibatch contains current augmented images and a bounded uniformly
sampled subset of the resident replay buffer. A mean CE loss over the combined
batch updates both `f_t` and that training head. There is no separate old/new loss
weight, logit distillation, test-based early stopping, or hidden feature
distillation. The CE head is not deployed.

Method defaults are 20 epochs/task, current batch 64, replay batch at most 64,
AdamW, backbone LR `1e-5`, training-head LR `1e-3`, weight decay `1e-4`, and a
per-task cosine schedule. The optimizer resets each task. These are explicit
initial method hyperparameters, not validated optimum values and not copied
wholesale from an existing method. PILOT data, class order, memory rules, and
evaluation protocol do not imply identical optimizers for all methods.

The only accessible old images are the current resident buffer. Native PILOT
herding selects the new exemplars after the task; old per-class herding prefixes
shrink when their quota falls. CIFAR-100 has fixed total capacity 2,000;
ImageNet-R has fixed 20/class, reaching 4,000. A quota cannot drop below one per
seen class in this estimator. Full current-task images may be revisited for
epochs; this is not a single-pass online setting.

Task sequence:

1. Acquire only the new-class train split and the all-seen-class test loader.
   Test samples are not used by the gradient or transport routines.
2. Encode resident old images under the anchor. In chained mode, encode them
   under the previous predictive body before any adaptation. Store those feature
   arrays only transiently for this task, not as an all-history feature cache.
3. Adapt the predictive body and CE head on current images plus resident replay.
4. Encode current images and resident replay under the updated body. Encode
   current images under the frozen anchor. Stable sample IDs and labels must
   agree between the relevant feature paths.
5. Reconstruct historical predictive statistics; commit new arrivals exactly
   once, independently of epoch count or replay frequency.
6. Update the bounded image memory once using the updated raw CLS features and
   native herding. Install the one-body analytic deployment scorer.

## Exact state, source coordinates, and the affine estimator

All arrays below use examples as rows. Append an exact constant coordinate:
`z=[u,1]`, where `u` is the analytic feature. Define `G=sum(z^T z)` and
`C[:,c]=sum_{i:y_i=c} z_i^T`; retain class counts `n_c`. State contains an exact
anchor pair `(G_a,C_a)`, an approximate current pair `(G_t,C_t)`, and exact counts.
Each new-class task adds to anchor statistics once, with no decay. Thus the
anchor statistics are exact for the fixed `a`, up to floating-point error;
predictive old-history statistics generally are not exact.

For the old population, let `R` be the resident old images and `M` the missing
old images. The source is either anchor coordinates (direct) or previous
predictive coordinates (chained). Subtract the exact resident source moments:

```text
G_M = G_old_source - Z_R^T Z_R
C_M = C_old_source - Z_R^T Y_R
n_M,c = n_old,c - n_R,c
```

Chained subtraction is valid for this implementation's memory lineage: next
task's residents are a subset of the previous residents plus previous current
arrivals, both represented exactly in the preceding commit. Arbitrary
reselection of discarded old samples is not permitted. For direct mode the
source anchor moments are exact independently of this lineage.

Fit an affine map `u_new = u_source A + b + e` on paired resident old images.
The full `(d+1)x(d+1)` map `T` has exact last column `[0,...,0,1]^T`.
Its nonconstant coefficients minimize

```text
sum_{i in R} w_i ||[u_i,1] [A;b] - v_i||^2 + gamma ||A-I||_F^2
w_i = n_M,y_i / n_R,y_i
```

The offset has a flat prior; `gamma=0` uses least squares, otherwise a linear
solve with the identity slope prior. Weighting allocates equal weight per
missing sample within each class; it is not an inverse inclusion-probability
correction for herding. A missing class with no resident sample is rejected,
instead of extrapolating its residual statistics from new classes silently.
The first task needs no old-population map.

The `residual_model=none` option sets `G_hat_M=T^T G_M T` and
`C_hat_M=T^T C_M`. Under one exact affine drift map with identifiable resident
design, this recovers the true moments. This identity is a deterministic
algebraic result, not a claim that real neural-network drift is affine.

## Coherent conditional residual correction

The default `class_conditional_psd` option models class-dependent residual means
and a pooled PSD residual covariance. It uses a single common affine map, so
aggregate within-class scatter is sufficient: separate per-class Gram matrices
are not required.

Write the missing source mean as `m_c=C_M[:,c]^T/n_M,c`. The aggregate missing
within-class scatter is

```text
S_M = G_M - sum_c n_M,c m_c^T m_c .
```

Its constant row and column are zero. Estimate each residual shift `delta_c`
from resident residuals, and reconstruct new means as
`m_hat_c=m_c T+[delta_c,0]`. With a pooled nonconstant covariance `Sigma_e`, use

```text
G_hat_M = T^T S_M T
          + sum_c n_M,c m_hat_c^T m_hat_c
          + N_M blockdiag(Sigma_e,0)
C_hat_M[:,c] = n_M,c m_hat_c^T .
```

This construction retains all terms introduced by changing class means,
including their interaction with mapped source means. It guarantees a coherent
PSD second moment when `S_M` and `Sigma_e` are PSD; its `G` and `C` use the same
means. It does not add an arbitrary covariance to `G` while leaving a
contradictory `C` unchanged.

The approximation assumes *within-class centered* predictor/residual cross
covariance is zero for missing data. In the true expansion, terms of the form
`T^T sum((z-m_c)^T(e-delta_c))` and their transposes generally exist. They are
unobserved and are modeled away here, not proved to vanish. Ordinary
in-sample LS orthogonality, especially with shrinkage and class conditioning,
does not establish this assumption for missing old data.

Residual estimation uses stratified out-of-fold map fits on resident old
images, up to five folds and no more than the minimum active per-class replay
count. The final map itself uses all resident pairs. If fewer than two folds
are feasible, diagnostics explicitly record an in-sample fallback. Fold maps
use the same fixed `gamma`, with no hidden validation search.

For class `c` with `r_c` residents, shrink its residual mean by
`r_c/(r_c+kappa)`, default `kappa=20`. Pool centered sample residual covariances
with missing-class-count weights, using `r_c-1` denominators where `r_c>=2`.
Classes with fewer than two residents cannot estimate their covariance and are
reported; they receive the pooled estimate from estimable classes. If no class
is estimable, the correction covariance is zero and is not described as an
identified zero variance. Shrink covariance toward its own diagonal with
coefficient 0.1. None of these choices is proven optimal.

Cross-fitting reduces direct fitting of each residual observation by its own
map fit. It does not make replay independent of backbone training; does not
make herding random; does not eliminate old/nonresident distribution shift; and
does not imply zero conditional cross covariance or unbiasedness. The code and
task diagnostics make no such claims.

Finally, add the *exact updated-coordinate* moments of the residents and new
arrivals to the estimated missing-population moments:

```text
G_t = G_hat_M + V_R^T V_R + V_current^T V_current
C_t = C_hat_M + V_R^T Y_R + V_current^T Y_current .
```

The resident subtraction and replacement prevent double counting. Counts
remain counts of unique arrivals, not gradient visits.

## Analytic head and numerical contracts

The final classifier minimizes aggregate squared error with ridge on slopes and
an unpenalized intercept. If `s` is the nonconstant feature sum, `N` the total
arrival count, and `n` the row vector of class counts:

```text
H = G_t[:-1,:-1] - s s^T/N
B = C_t[:-1,:] - s n/N
W = solve(H + lambda I, B)
b = n/N - (s/N)^T W
score(u) = u W + b .
```

No classwise score normalization or task-specific masking is applied. The CE
training head and this analytic head optimize different objectives, a deliberate
separation whose downstream effectiveness still requires measurement.

All sufficient statistics and solves use NumPy float64. Reencoding a resident
image with an unchanged float32 backbone but a different image-batch shape can
still change its features slightly. An integration test exposed this when a
fully resident population produced a cancellation remainder with a negative
eigenvalue near `-2.55e-7`. Promoting those inputs to float64 does not restore
the original coordinates.

The numerical core therefore distinguishes an encoder reencoding budget from
its strict `1e-9` relative float64 PSD tolerance. Core default
`reference_feature_rtol=0` preserves strict synthetic/exact-coordinate checks;
the native wrapper declares `64 * finfo(backbone_dtype).eps` by default
(`7.63e-6` for float32). This multiplier is a conservative numerical working
assumption, not a theorem bounding an arbitrary neural network. It is stored
in state and reported explicitly, and can be configured before experiments.

If a resident-feature perturbation has Frobenius budget `E=r*||U_R||_F`, the
resident Gram perturbation is bounded by
`2*||Z_R||_F*E+E^2`; the class-sum budget is at most `sqrt(|R|)*E`. The code
propagates the latter through the missing class-mean outer products as well.
For zero missing count, the remainder is exactly zero only after its Gram and
class-sum subtraction discrepancies pass those bounds plus float64 tolerance.
For a partial population, zero-count class sums are set exactly to zero within
the declared budget; the missing within-class scatter is PSD-checked/repaired
and recombined with its class means. The repair keeps `G`, `C`, and the exact
bias counts coherent. All correction norms and budgets are diagnosed; negative
eigenvalues beyond the explicit budget still fail. This is not a global
relaxation of every covariance check or evidence that model error is harmless.

Remaining PSD checks reject material negative eigenvalues; only rounding
negatives are clipped and quantified. They verify counts, symmetry, constant
coordinate identities, finite values, and centered classifier feasibility.
Statistical updates commit transactionally after checks. A failed adaptation
task must be restarted from its previous task-boundary checkpoint; the neural
optimizer itself is not a transactional rollback system.

Checkpoint state includes body, immutable anchor, both moment pairs, counts,
transport configuration, analytic head, replay images/IDs/lineage, expanded CE
head and replay-sampling RNG. It also binds the exact backbone provenance and
initial anchor hash to those independently obtained from the trusted local
initial model when constructing the learner. A self-consistent hash supplied by
the checkpoint alone cannot substitute a different initial coordinate system.
The anchor hash is checked before starting each task or requesting its datasets,
and again after adaptation. Parent driver preserves global RNG state.
Optimizers reset between tasks, so no optimizer state is needed for the
supported task-boundary resume. Wrapper restoration validates independent
backbone, anchor, statistics, replay, head and RNG candidates before swapping
live state; a rejected checkpoint does not partly alter the learner. Mid-task
resume is not implemented.

## Primary literature and code inspection informing this adaptation

PILOT source inspected at upstream commit
`7a6e904c5bc5cb7a4e1823b3434020be27469b63`. These are mechanism comparisons,
not claims that this implementation reproduces those methods.

| Source | Inspected mechanism and concrete decision |
| --- | --- |
| [SDC, CVPR 2020](https://arxiv.org/html/2004.00440v1), Section 4 | Interpolates drift measured on current images to update old prototypes. Motivates treating coordinate misalignment separately from representation quality. Here bounded old-image pairs replace extrapolation from new classes; both first and second moments are reconstructed. The Gaussian-kernel drift field is not copied. |
| [LDC, ECCV 2024](https://arxiv.org/html/2407.08536v1), Sections 3.2-3.3 | Fits a forward old-to-new map after representation learning, then moves old prototypes. This supports a distinct post-adaptation coordinate-estimation phase. Here an affine map has an identity prior and a flat offset, uses paired resident old images, and transports missing-population second moments as well as means. No claim is made that prototype transport alone is new. |
| [DPCR, ICML 2025](https://arxiv.org/html/2503.05423v2), Sections 3.1-3.3, Eqs. 5, 16-18 | Already combines a plastic backbone, moment transport, and ridge classifier reconstruction. This is a direct methodological predecessor. Its cold-start exemplar-free dual projection uses current data and class row-space projectors. Here the source can instead be an exact initial anchor, bounded old-image residents are refreshed exactly, and residual means/scatter are modeled coherently. We do not reproduce its row-space projector, LwF training, per-class full-Gram storage, or classifier normalization. Novelty cannot be asserted from the three-part pipeline alone. |
| [PFR, CVPR Workshops 2022](https://arxiv.org/html/2112.15022v2), Section 3.2 | A learned current-to-previous projection relaxes feature distillation while retaining information. The relevant lesson is that coordinate change and information loss differ. Our forward post-hoc moment estimator has a different direction and role; the PFR loss is not implemented. |
| [SLCA, ICCV 2023](https://arxiv.org/html/2303.05118v2); local `models/slca.py` | Inspected backbone/head LR groups, current-class CE, class Gaussian estimates and later classifier alignment. This motivates a lower body LR and separating representation adaptation from head correction. We use CE over all seen classes with real bounded replay, then a closed-form head; SLCA's Gaussian feature sampling and classifier-alignment loop are not copied. |
| [FeCAM, NeurIPS 2023](https://arxiv.org/html/2309.14062v2); local `models/fecam.py` | Inspected covariance estimation, shrinkage and pseudo-inverse Mahalanobis scoring. This reinforces preserving covariance geometry and explicitly handling singular statistics. Here residual uncertainty affects global ridge moments; inference is not FeCAM/Mahalanobis and classwise covariance normalization is not copied. |
| [RanPAC, NeurIPS 2023](https://arxiv.org/html/2307.02251v3); local `models/ranpac.py` | Inspected first-session adaptation, fixed nonlinear random features, accumulated Gram/cross moments, ridge solve and current-training-only lambda selection. We preserve once-only accumulation, but change the predictive coordinates and must estimate missing history. No random-projection layer, fixed-backbone guarantee, or exact RanPAC equivalence is claimed. Our lambda is fixed in config. |
| [FOSTER, ECCV 2022](https://arxiv.org/abs/2204.04662); local `models/foster.py` | Inspected expansion, old-body freezing, boosted residual learning, compression/distillation and bounded native memory. It supplies an alternative stability/plasticity mechanism and the need to distinguish training from deployed resources. Our body does not expand; no FOSTER boosting/compression rule is copied. |
| [Andrew Ng, CS229 generative-learning notes](https://cs229.stanford.edu/notes_archive/cs229-notes-all/cs229-notes2.pdf), Gaussian discriminant analysis | The non-CL reference separates class means from PSD centered covariance. Our conditional residual model uses that representation only as a coherent moment construction. Gaussian density correctness is not required for the algebra; Gaussian-discriminant optimality is not claimed for the resulting ridge classifier. |
| [Cosma Shalizi, linear filtering notes](https://www.stat.cmu.edu/~cshalizi/dst/18/lectures/09/lecture-09.html) | Explicit signal-plus-noise covariance expansion keeps cross-covariance terms unless an uncorrelatedness assumption is stated. This informs the documented approximation: residual cross terms are assumptions, not identities supplied by least squares or cross-fitting. |

Also inspected PILOT `models/base.py` exemplar shrinking/herding and evaluation
lifecycle, and `utils/data_manager.py` dataset transforms/label mapping. Shared
protocol code supplies the stable sample IDs required for auditability.

## Resource accounting and unresolved questions

The deployment path has one predictive backbone and a `(d+1) x K` analytic
head. Training also stores one frozen anchor and an expanding CE training head.
Persistent numerical history is `O(d^2+dK)` for the two coordinate systems, not
an unbounded per-example feature cache. The bounded raw-image budget is
separate from these statistics and model parameters; reporting only image
count would hide auxiliary memory. Resource diagnostics count numerical/model
payloads; they do not measure Python overhead, optimizer peak memory or runtime.
Several dense `d x d` decompositions and foldwise solves run at task boundaries;
this code makes no demonstrated speed advantage claim.

Still uncertain and requiring experiments:

- Does full-body slow adaptation improve representations enough to outweigh
  drift-estimation error on strong pretrained CIFAR-100 features? Better features
  cannot be assumed from nonzero gradients or training loss reduction.
- Can an affine map and pooled residual covariance extrapolate herded residents
  to discarded old examples? OOF error on herding residents is only a diagnostic
  and is not a calibrated nonresident-risk estimate.
- Does direct transport avoid chained-error accumulation enough to compensate
  for a possibly increasing gap from the initial anchor? Chained transport makes
  shorter maps but recursively reuses approximate history.
- Are class residual means sufficiently estimable from 20 resident images/class,
  and do the default shrinkage strengths suppress useful corrections or retain
  noise? Classes with one resident cannot estimate class covariance.
- Are ignored centered predictor/residual cross terms negligible? A PSD result
  establishes mathematical consistency, not closeness to the true moments.
- L2-normalized real features lie on a sphere; affine/Gaussian residual moment
  estimates need not respect every sphere constraint. No post-hoc trace clamp
  is introduced, as it would alter the declared model.
- The training CE head can favor current classes despite replay, while the final
  analytic objective differs. Uniform replay, loss balancing, distillation,
  projected functional constraints and representation reuse losses remain
  candidate modifications, not unimplemented properties of this version.
- The teacher's requested reusable decomposition is explicit in other routes;
  this route alone is not evidence of learned reusable subproblems. It is a
  comparison mechanism for preserving historical supervision under plasticity.

## Correctness validation and limits

`tests/test_transport.py`: 29 numerical tests cover exact affine oracle
equivalence under direct/chained transport, class-mean cross terms, nonlinear
PSD consistency, shrinking replay, counts/bias, transactionality and malformed
state rejection, bounded float32 reencoding repair for full/partial resident
coverage, and rejection of mismatches beyond that budget. All-history data
appears only in explicitly synthetic oracle tests, never in the learner.

`tests/test_pilot_transport.py`: 13 parameterized synthetic lifecycle tests cover
real tiny-ViT body updates with nonzero gradients, unchanged frozen anchor,
three-task exact arrival counts despite repeated epochs, both memory rules,
direct/chained mode, source-ID alignment, deployed-coordinate consistency,
task-boundary state restoration, atomic rejection of corrupt state, conflicting
normalization-key rejection and deterministic continuation. They additionally
verify trusted initial provenance and anchor-mutation rejection before accessing
a new task. These are correctness tests, not accuracy experiments or pretrained
ViT validation.

Executed with the local Python environment:

```sh
python -m pytest tests/test_transport.py tests/test_pilot_transport.py -q
```

The transport tests plus the common runner's task-boundary resume integration
test passed 43 tests in 1.56 seconds. Main CIFAR-100/ImageNet-R training and
performance claims are intentionally not part of this validation.
