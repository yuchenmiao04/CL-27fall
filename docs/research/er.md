# Experience replay baseline

Implementation: `models/er.py`. The common raw-CLS network is
`cl27/supervised.py::CEClassifier`; the existing CE adaptation and head-expansion
functions in `cl27/transport_training.py` are reused directly. This is an
independent supervised control, not the proposed method with a hidden anchor.

## Model and training

Start from the same pinned PILOT/timm pretrained ViT-B/16 as the comparison
methods. Train all backbone parameters and one linear CE head. At task `t`,
expand that head from the old vocabulary to all seen classes, retaining old
weight and bias rows and randomly initializing only the additional rows.
Prediction is `head(backbone(image))` on raw CLS vectors, with no analytic L2
normalization, cosine head, task-ID input or task-block mask. Testing always
compares all seen class columns.

Every epoch shuffles the complete current-task training set. Each current
minibatch draws up to `replay_batch_size` resident old images, without replacement
within that draw. Apply PILOT's ordinary training transform to both groups and
minimize mean CE over their concatenation. Different minibatches can revisit
the same resident image. There is no logit distillation, gradient projection,
module library, second backbone or historical sufficient-statistic head.

The optimizer, per-task cosine schedule, separate backbone/head LRs, weight
decay, current/replay minibatch sizes and epoch count use exactly the same
`adapt_backbone` call as `history_transport` and `plastic_shared_basis`. Default
values remain 20 epochs, 64 current plus at most 64 replay images per update,
AdamW with body LR `1e-5`, head LR `1e-3`, weight decay `1e-4`; optimizer state
resets each task. The trained CE head is deployed rather than discarded.

## Bounded memory and permitted data access

After the task, encode its current training images with the deterministic PILOT
test transform and the updated raw CLS body. The common replay helper applies
its native feature normalization for herding. Add selected new-class original
images; shrink old herding prefixes if their quota falls. CIFAR-100 uses fixed
total capacity 2,000; ImageNet-R uses fixed 20/class. Stable IDs, remapped labels
and current canonical/training rows must agree. No old-class split is requested
from the data manager for training: previous images come only from the bounded
resident buffer.

The stored `seen` count and `unique_fitting_images_seen` count arrivals once,
independently of epochs and replay visits. They are audit counters rather than
feature moments. No all-history raw-image or feature cache is retained.
Diagnostics separately report current/replay training visits, optimizer updates,
body gradients and hashes, CE-head hash, memory image count and storage payloads.

## Fair body-training control

ER and the transport routes share the same native initial body, CE training
head, head expansion, train data, augmentations, replay generator, herding
features and memory rule. Their body/head/replay trajectories can therefore be
matched, even though their deployed classification heads differ.

Read-only canonical feature extraction uses a dedicated DataLoader generator.
Without this detail, even `shuffle=False` iteration consumes the global Torch
RNG for a worker/base seed; extra anchor-encoding passes would change later CE
head initialization or shuffled training order. The common helper now isolates
that read-only RNG use. This preserves the declared training equations while
removing an incidental source of different training randomness. Synthetic tests
with stochastic image flips check exact body, CE-head, replay and global-RNG
equality over three tasks, for both direct and chained transport controls.

This control separates the effect of the final transported/EB predictor from
the effect of changing the backbone's CE training recipe. It does not claim
matching FLOPs: ER does not execute anchor encoding, transport or EB fitting.

## State and resource accounting

Task-boundary checkpoints contain the plastic body, expanded CE head, bounded
raw memory, replay RNGs, unique-arrival/task/class counters, training settings,
and initial backbone provenance/hash. The parent driver saves global RNG state
and dataset/configuration identity. Initial provenance is validated against the
independently constructed trusted starting model, not just a hash supplied by
the checkpoint. Restore validates candidate tensors, CE dimensions/finiteness,
vocabulary, counts, buffer and RNG before changing the live learner. Outer
lifecycle metadata must agree with those saved counters. Mid-task resume is not
implemented; a failed task must restart from the preceding complete checkpoint.

Inference and training each use one backbone. Resource reports assign zero
anchor-parameter bytes and zero historical-statistics bytes, alongside the
actual body/head parameter bytes and bounded raw-image payload. No inference
speed or accuracy advantage is claimed from these counts alone.

## Source and adaptation

[Chaudhry et al., On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486)
provides the relevant simple ER mechanism: directly train on current examples
together with a small episodic memory. Its online protocol and reported results
are not reproduced here. This baseline follows the selected multi-epoch PILOT
Class-IL protocol, pretrained ViT and herding budget instead.

Inspected PILOT `models/finetune.py` for backbone/CE task lifecycle,
`models/icarl.py` for current-plus-memory dataset access, and `models/base.py`
for exemplar herding/shrinking. The ER control omits iCaRL's distillation and NCM
prediction. Implementation reuse comes from the already validated local CE and
memory helpers, keeping this baseline's body training aligned with the proposed
transport routes rather than claiming equivalence to a published table row.

## Small validation

`tests/test_er.py` covers three-task training under both memory rules, real body
gradients/parameter changes, all-seen raw-CE predictions, no test fitting or
old-data-manager training access, exact matched body/head/replay trajectories,
old CE-row preservation at expansion, stochastic continuation, atomic rejection
of corrupt state/provenance/counters, and RNG-isolated canonical encoding.
The tests use synthetic RGB images and an explicitly random tiny ViT; they do
not constitute a full CIFAR-100/ImageNet-R experiment or an accuracy comparison.
