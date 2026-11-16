# Current fixed-library candidate: native PILOT adaptation

Status: implemented and checked on small synthetic images with a randomly initialized token ViT. This document specifies `current_adapted`; it contains no CIFAR-100 or ImageNet-R performance claim. Code: `models/current_adapted.py`, `cl27/current/{engine,episodes,readout}.py`. It is the existing anchored episodic method adapted to the common taskwise protocol, rather than a fourth replacement research route.

## 1. Problem and relation to the research idea

The research hypothesis is that a fixed set of learned functions can decompose and improve prediction in ways reusable across successive classes. A gate composes abstract low-rank token transformations; neither expert semantics nor an external question-generating model is prescribed. The functions and gate are shared across tasks. The plastic backbone can also acquire shared features, rather than confining all learning to newly attached parameters.

The difficulty is that changing a representation invalidates previously accumulated feature moments. This candidate keeps the V4 advantage in one immutable coordinate system and learns a bounded-data correction in a separate plastic system. **It does not transport full-history statistics to the plastic representation.** Full-history exactness applies to the anchor ridge predictor only. This separates the existing candidate from `history_transport`, which explicitly estimates such a coordinate change, and `shared_basis`, which constrains the predictor family in another way.

## 2. Architecture and exact equations

### 2.1 Native backbones and module libraries

Start from two independent copies of the same native PILOT/timm ViT checkpoint: frozen anchor `f_0` and plastic main `f_theta`. No parameter-name conversion to another ViT implementation occurs. The original native `.blocks` execute unchanged except for forward hooks that add the library output after selected blocks. Both models use evaluation-mode stochastic/running-state behavior; evaluation mode does **not** freeze the main parameters. In a backbone phase, all original main parameters, including patch embedding, positional/CLS tokens, attention, MLP and normalization parameters, are eligible for gradients.

For block output token h of width d, let u=LN(h), where LN has no affine parameters. Library j contains K low-rank functions

`m_jk(h) = U_jk tanh(V_jk u)`, with `V_jk ∈ R^(r×d)`, `U_jk ∈ R^(d×r)`.

Gate logits are `a_jk(h)=g_jk^T u+b_jk`. A fixed null logit `log(K)` accompanies a zero-output null function. The block output becomes

`h + s Σ_(k=1..K) softmax([log(K),a_j1,...,a_jK])_k m_jk(h)`.

The same gate and functions operate on every image and every token, including CLS. This is a dense mixture: every function is evaluated. There is no top-k speedup, task-ID input, task-specific expert, explicit specialization loss, expansion or splitting. Initialization: U=0, V sampled normal/√d, gate weights/biases=0. Initially the null receives probability 1/2 and each expert receives 1/(2K); the correction is zero to floating-point tolerance. U has a useful initial gradient; V/gate start receiving gradients once U departs from zero.

Default: blocks 7 and 9, zero-based; K=4, r=8, s=0.1. A later attention block is required after each hook, so transformed patch tokens can affect CLS. For d=768 these two libraries contain 104,456 trainable parameters, excluding the two fixed null-logit scalars. The two full ViT copies and main optimizer remain substantial costs.

### 2.2 Canonical features and exact historical moments

For a raw CLS vector v, `a(v)=[v/||v||_2;1]`; normalization follows PyTorch `F.normalize` with its default epsilon. Set `z0(x)=a(f0(x))`, `zf(x)=a(f_theta,library(x))`, `zc(x)=a(f_theta,no_library(x))`; p=d+1. Canonical features use the PILOT test transform on **training** images. Learning episodes use a single sampled training augmentation shared by all three paths for that episode ID. There is no additional pixel mean/std normalization beyond the selected PILOT transform.

Persistent CPU float64 moments at the end of task t are

`G_t=Σ_(i in D_1∪...∪D_t) z0_i z0_i^T`,

`C_t=Σ_i z0_i onehot(y_i)^T`, `N_t=Σ_(s≤t)|D_s|`,

`W0_t=(G_t+λ0 I_p)^(-1) C_t`.

The code solves linear systems and never constructs an inverse. The bias coordinate is regularized, exactly as in the previous implementation. New class columns are registered before task learning, but the complete current task is added to G/C **once, after all epochs**. Replay never increments G/C. Thus multiple epochs do not inflate current-class moments. The saved `G[-1,-1]` equals the number of distinct committed training rows.

Canonical anchor vectors are cached only for resident replay IDs, so exact old-query downdates need no repeated anchor forward pass on these canonical images. The cache is pruned whenever the replay budget removes images. The aggregate G/C retains all committed rows, including rows no longer resident.

### 2.3 Residual prediction and nonzero prior

Let Δ=Zf−Z0. Given support features, one-hot Y, and diagonal positive class-balanced Ω, solve

`V*=argmin_V ||Ω^(1/2)(ΔV−(Y−Z0 W0))||_F² + λr||V−W0||_F²`,

`V*=(Δ^T ΩΔ+λr I)^(-1)[Δ^T Ω(Y−Z0 W0)+λr W0]`.

Prediction is `T [z0(x)^T W0 + (zf(x)−z0(x))^T V*]`. The prior is W0, not zero. At Δ=0, V*=W0 and the correction is zero, while the query-feature derivative is `dΔ_query W0`. A zero prior would instead produce V=0 at identical branches and remove this first-order route for representation learning. This is an algebraic property, not a claim of superior empirical performance.

Weights are `Ω_ii=n/(number_of_represented_classes × n_(y_i))`, with mean one. Ridge uses a **sum** of weighted errors, so silently averaging that loss would change its effective regularization. When the support has n<p rows, the solver uses the equivalent differentiable dual expression

`V = P + Δ^T (Ω ΔΔ^T + λr I_n)^(-1) Ω(R−ΔP)`,

where P=W0 and R=Y−Z0W0. Otherwise it uses the primal system. The non-symmetric weighted dual avoids differentiating `sqrt(weight)` at zero. All ridge/moment solves use CPU float64; network computation remains the backbone dtype. The feature map's appended constant cancels in Δ, so the residual does not introduce a separate learned bias correction.

## 3. Episodes and alternating learning

For each shuffled current minibatch, the available pool is that minibatch plus the entire pre-task resident replay memory. Every available class supplies support, including background classes. Target classes are selected from eligible current classes for an anchored episode, or all eligible classes for a transfer episode. At least one background class is reserved when possible. Support/query indices are disjoint. Default target support count is drawn from {1,5} then capped at class-count−1; background support≤5; target/background query≤2 per class; target classes≤5. Single-class or all-singleton episodes are explicitly skipped. The sampler uses dedicated checkpointed CPU generators.

There are two base-head constructions:

1. **Anchored:** start from previous-task G/C. Temporarily subtract canonical vectors and labels of resident old **queries**, then add canonical vectors and labels of current **supports**. Old supports were already in the history and are not re-added. Current queries have not entered history. This removes direct query-label inclusion in the temporary ridge head without mutating persistent G/C. Other available historical information remains in the head.
2. **Transfer:** fit a fresh base ridge head from the episode's augmented anchor support vectors, with no historical moments. This tests support-based adaptation using already available labels; it is not access to or proof of transfer to future classes.

Temporary W0 is detached because anchor features do not depend on any learned parameters. In both cases the residual support solve is differentiated fully, including its normal equations. CE means below average samples within class and then represented classes equally.

Module phase: freeze the entire main backbone; update library functions and gates using

`L_module = CE_target(full) + β CE_background(full)`.

Backbone phase: freeze libraries, but keep their Jacobian with respect to tokens; update the entire main backbone using

`L_backbone = CE_target(common) + (β/2)[CE_background(common)+CE_old_queries(full)]`.

The common term requires useful changes even with modules bypassed; the old full-query term supplies a bounded-data retention signal. This objective encourages shared representation learning, but does not mathematically enforce future-class reuse or prevent interference. The gate and experts are updated together in module phases; there is no separate gate-only assimilation stage.

Defaults: anchored/transfer probability 0.5/0.5; β=1; T=10; λ0=λr=0.01. Four attempted episode updates per current minibatch with repeating phase order `[modules, modules, backbone, modules]`; skipped episodes consume their schedule position. AdamW with main LR 1e−5, module LR 1e−4, weight decay1e−4. LR resets to the configured value each task and follows epoch-level `0.5(1+cos(π epoch/epochs))`; AdamW moment state persists across tasks. No test accuracy is consulted for updates, LR or acceptance.

## 4. Exact bounded-activation VJP execution

Direct autograd retains every support/query backbone activation until the support solve/query loss finishes. The default execution preserves its gradient while reducing peak network activation memory:

1. Sample augmentations once; keep their tensors on CPU.
2. Microbatch no-grad passes compute full/common features. Transfer those features to float64 CPU leaf tensors. Anchor features are detached constants.
3. Build the identical residual solve and query CE graph on these leaves; backpropagate to obtain `g_i^path=∂L/∂z_i^path`.
4. Recompute one path and one image microbatch at a time, then accumulate `J_(theta,i,path)^T g_i^path` into the active parameter gradients. Call AdamW once after all pieces.

This follows the chain rule `∇theta L=Σ_(i,path) J^T g`. It is exact up to arithmetic/gradient accumulation order when weights, sampled views and stochastic state are identical. Evaluation-mode dropout/drop-path and fixed running-state behavior are necessary; arbitrary stochastic training-mode layers would violate that equality. Support solves still have differentiable CPU graphs. This is activation-memory control, not a FLOP reduction: it adds recomputation. `feature_batch_size` controls network microbatching, not the logical episode or statistical objective. Tests compare it to direct autograd for both update phases.

## 5. End-of-task order, deployment and replay

After the prescribed current-data epochs:

1. Commit all current canonical anchor rows once; solve the all-history W0.
2. Recompute canonical full features for the complete current task plus the **pre-task** replay memory. Fit deployment V with the same centered/class-balanced ridge equation.
3. Select new-class exemplars by PILOT-style herding on normalized full main features and shrink old exemplar prefixes to the new quota. Prune the canonical anchor cache to retained IDs.
4. Save a complete task checkpoint: two native backbones, libraries and their hook positions/scale/rank/count, G/C/count/classes, W0/V, retained images and anchor cache, both AdamW states, episode generators, schedule position and counters. The common driver also handles experiment/global RNG state. Restore validates fresh native encoders with fresh hook closures and a temporary replay object before committing anything; this temporarily requires an additional pair of backbones. It rejects changed anchor coordinates, architecture, dimensions, options, count/task inconsistencies and a W0 inconsistent with G/C. The restored main explicitly keeps `requires_grad=True`; copying a frozen anchor without restoring that flag can change native kernel dispatch even when numerical parameter states match.

The deployment V is fitted before memory selection and remains fixed for evaluation. It can use the entire currently available task before those images are discarded, as allowed by taskwise CIL. It does not include inaccessible old images. Inference uses both native backbones and only outputs all seen class columns; no task ID is supplied. Fixed-module count does not grow with task count. Class head columns, C, W0/V and possibly replay do grow according to the protocol.

Selected primary protocol: CIFAR-100 (`cifar224`), ten tasks of ten classes, fixed total 2,000 original-image exemplars; ImageNet-R, ten tasks of twenty classes, fixed twenty exemplars per seen class (final4,000). Use native PILOT transforms, specified class orders and the pinned ViT-B/16 IN21K→IN1K checkpoint. This method's replay is real original-image access and must not be described as exemplar-free. G/C, anchor cache, the second backbone and optimizer states are additional storage beyond the image quota.

## 6. Sources read and concrete adaptations

The following are influences/comparators, not claims that this code reproduces the cited methods. PILOT code references bind to upstream commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`.

| Source | Material examined | Implemented connection and difference |
|---|---|---|
| [AdaptFormer](https://arxiv.org/abs/2205.13535), §3.2; [official implementation](https://github.com/ShoufaChen/AdaptFormer) | Token bottleneck residual with small scale; frozen pretrained branch | Preserve low-rank residual token functions. Here tanh, input gate/null and post-block hooks differ from parallel-FFN ReLU adapters; main weights also learn in separate phases. No AdaptFormer result is inherited. |
| [R2-D2](https://arxiv.org/abs/1805.08136), §3.1–3.2; [official repository](https://github.com/bertinetto/r2d2) | Query objective differentiated through support ridge; Woodbury solve | Keep the full support-solve gradient and choose primal/dual by dimensions. Add nonzero W0 prior, anchor history and class weights. Ridge/temperature are configured constants here, unlike learnable calibration options there. |
| [Lifelong Learning of Compositional Structures](https://arxiv.org/abs/2007.07732), §4; [official code](https://github.com/Lifelong-ML/Mendez2020Compositional) | Shared functions, learned structure, separated assimilation/accommodation | Motivate separate roles of composition and shared learning. This candidate has input-conditioned gates and alternates main/library learning; it does not reproduce their task-specific structure, joint multi-task initialization or expansion. |
| [SLCA](https://arxiv.org/abs/2303.05118), §3.2–3.3; [PILOT code](../../models/slca.py) | Slow representation learning; distinction between feature quality and classifier alignment | Main LR is deliberately below module LR; keep feature-learning quality separate from deployed-readout mismatch. We do not copy SLCA Gaussian classifier alignment or its reported LR values. |
| [RanPAC](https://arxiv.org/abs/2307.02251); [PILOT code](../../models/ranpac.py) | Fixed-feature G/Q sums and linear-system readout; first-task adaptation | Use persistent all-history sums only in immutable coordinates and commit each arrival once. We do not use its random projection, ridge sweep or first-task-only representation freeze. |
| [PILOT APER/ADAM code](../../models/aper_adapter.py) | Native first-task adaptation, dual backbone construction and prototypes | Native checkpoint/token flow is retained. Our main continues learning after task1, predictions use centered ridge rather than concatenated cosine prototypes. |
| [PILOT LAE code](../../models/lae.py) | PET parameter selection, freezing schedule, EMA evaluation, lifecycle | Make parameter selection and phase freezing explicit. We do not add LAE EMA, max-of-softmax inference or current-class masking; full seen-class episodic logits remain active. |

This implementation round improves framework fidelity, task-level data accounting, checkpoint integrity and bounded activation memory. It does **not** add an unvalidated specialization penalty or redesign the current candidate based on small-test accuracy. Such changes would define a new hypothesis and need separate experiments.

## 7. Unresolved research questions and experiment interpretation

- Shared parameters and multi-task episodes permit reuse but do not establish beneficial cross-task reuse. Future-class adaptation, task1-only adaptation controls and module removal/replacement tests will be needed to establish it.
- Full-history moments remain exact only for the anchor representation. V depends on available current data plus bounded replay; there is no guarantee its old-data correction generalizes to discarded images.
- Both branches cost full ViT inference, plus modules. Dense gates do not save expert computation. Four episodes per current minibatch, support/query passes and VJP recomputation can be much more expensive than a supervised baseline epoch. Same epoch count is a data-access protocol, not a FLOP-matched comparison; counters and time/memory must be reported.
- Library positions, count/rank, residual scale, two ridges, temperature, phase schedule, shots and LRs are hyperparameters. Defaults preserve the existing candidate; they are not dataset-optimal values or literature-certified settings.
- At initialization, common and anchor are identical mathematically; once backbone learning proceeds the supposedly common path can forget. The old-query loss offers a bounded-memory tradeoff without a preservation theorem.
- A full-history anchored head is stronger side information than an ordinary support-only meta-learner. Query removal prevents direct self-fitting in that head but does not make historical tasks independent of representation learning.
- There is no global held-out future-class objective, domain-change guarantee, module diversity/identifiability result, or dynamic module growth in this candidate.
- Native tiny-ViT checks establish implementation properties only. No complete real-data protocol was launched in this round; those runs are required to estimate benefit, forgetting, speed and practical VJP memory savings.

## 8. Small correctness validation

Command from repository root: `../.venv/bin/python -m pytest -q tests/test_pilot_current.py`.

Eleven checks pass on CPU: three tasks/two epochs under both quota rules; frozen-branch historical ridge equals a joint fit and each training row counts once; main/library parameters change while anchor is unchanged; native block hooks transmit token corrections; memory-efficient and direct gradients agree in module and backbone phases; old-query exact downdate does not mutate history; task-boundary resumed training matches continuous training; hook positions/scale and feature normalization are bound; a rejected restore leaves live state unchanged. Exact VJP comparisons use an FP64 tiny backbone with strict gradient tolerance, because native CPU kernels may round differently under grad/no-grad in FP32. FP32 task learning and resume are tested separately. The image/test rows are synthetic and no pretrained checkpoint is downloaded by these unit checks. The common runner's current-method task-boundary resume integration check also passes with Python3.10/PyTorch2.0.1.

`scripts/verify_pretrained.py` prepares or executes a separate genuine native ViT check through `main.py`. Its default per-method workload is two CIFAR-100 tasks of ten classes, three fit images/class and three train-holdout images/class, one epoch, logical batch8 and feature/VJP microbatch2. It uses the already downloaded official **training** pickle, never official test, and permits only the pretrained weight download. Example: `python scripts/verify_pretrained.py --data-root /existing/data --output-dir /new/verification --device -1`. `--prepare-only` writes configs without running; `--methods` filters routes, and `--python` chooses the pinned environment. This bounded pipeline is not a performance comparison. Protocol buffer2000 retains every old subset image, so non-resident historical transport is not exercised by it; separate transport-core tests cover that branch. No genuine pretrained run is claimed by the preparation-only check.
