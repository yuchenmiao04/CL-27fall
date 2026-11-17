# Four additional formal experiments

This extension adds four independent, fresh runs. It preserves the original
twelve formal results and their checkpoints. The 20-epoch rows supplement the
existing 5-epoch fixed-library rows; they do not replace them according to test
accuracy. There is no hyperparameter search in this extension.

## Fixed job assignment

| Host partition | Physical GPU | Job |
| --- | --- | --- |
| `current20`: original two RTX 5090 GPUs | 0 | `formal_cifar_current_epochs20_v1` |
| `current20`: original two RTX 5090 GPUs | 1 | `formal_imagenetr_current_epochs20_v1` |
| `tuna`: added two RTX 4090 D GPUs | 0 | `formal_cifar_tuna_native_v1` |
| `tuna`: added two RTX 4090 D GPUs | 1 | `formal_imagenetr_tuna_native_v1` |

Each job uses one independent GPU. The user authorized the added server so the
TUNA jobs can run concurrently with the fixed-library jobs. The original
fixed-library workers continue unchanged; their controller is adoption-only.
Unknown GPU occupants block admission. The original sequential dispatcher is
not used for this partitioned execution.

## Common protocol

- Class-incremental learning, with training-stage boundaries available and no
  task identity at evaluation. At each stage, evaluate over every seen class.
- CIFAR-100: `cifar224`, ten stages of ten classes; original complete training
  and test sets and PILOT 224-pixel transformations.
- ImageNet-R: `imagenetr`, ten stages of twenty classes; the already verified
  PILOT-prepared train/test files. Preserve the selected raw split and its
  explicit `preserve_verified_pilot` policy. Do not clean or resplit the data.
- Seed 1993, the established NumPy MT19937 class order, and the same structured
  evaluation as the original twelve runs. Report Final, AIA, and both explicitly
  named forgetting definitions from `docs/PROTOCOL.md`.
- ViT-B/16, input 224, `timm==0.6.12`, the existing AugReg IN21K-to-IN1K NPZ.
  Its SHA-256 must be
  `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`.
  `TORCH_HOME` is explicitly bound to the admitted runtime cache. Missing or
  changed weights block preparation/admission; no model download is performed.
- PILOT upstream snapshot:
  `7a6e904c5bc5cb7a4e1823b3434020be27469b63`.

## Method-specific settings

### Fixed module library, 20 epochs

Use the selected `current_adapted` configuration from each dataset's original
formal 5-epoch run. Change only epochs to 20 and the resulting cosine-schedule
horizon; administrative output paths and variant labels are distinct.

The backbone, module locations/count/rank, learning rates, reference branch,
episodic sampling, microbatches, replay selection, and outer-update schedule
remain unchanged. CIFAR-100 retains the two-update M/B schedule. ImageNet-R
retains the four-update M/M/B/M schedule. Historical-image budgets remain
2,000 total for CIFAR-100 and 20 per class for ImageNet-R. Reuse only the
mathematically frozen backbone prefix through
`CL27_CURRENT_FROZEN_PREFIX_REUSE=1`; trainable computations are not cached
across model updates.

Preparation compares all scientific configuration fields with the original
formal recipes and guards the exact original current-method source bytes.
Twenty epochs is a user-selected budget comparison, not a newly tuned optimum.

### TUNA, native PILOT recipe

Use `vit_base_patch16_224_tuna` with the admitted shared checkpoint and the
native pinned PILOT algorithm. Only common data, class split, checkpoint, and
structured evaluation are adapted.

| Setting | CIFAR-100 | ImageNet-R |
| --- | --- | --- |
| Image-training epochs per stage | 15 | 10 |
| Optimizer | SGD | SGD |
| Initial learning rate | 0.01 | 0.02 |
| Batch size | 16 | 32 |
| Weight decay | 0.0005 | 0.0005 |
| Scheduler | Cosine | Cosine |
| Classifier-alignment epochs | 30 | 30 |
| Classifier-alignment learning rate | 0.005 | 0.005 |
| Orthogonality regularization enabled | false | true |
| Stored historical images | 0 | 0 |

The complete templates, including every other native hyperparameter, are in
`configs/formal_cifar_tuna_native_v1.json` and
`configs/formal_imagenetr_tuna_native_v1.json`. Preparation compares the native
recipe with the pinned `exps/tuna_cifar.json` / `exps/tuna_inr.json` files.

Do not repair or reinterpret native behavior for this reproduction: historical
adapters retain the original gradient eligibility; classifier alignment runs
30 epochs with the original cosine period of 5; ImageNet-R mean correction
keeps the native negative task-size term; the inference path retains the
adapter actually selected by the source. Protocol accounting and checkpoint
support must not change the original training/inference calculation.

## Preparation and execution

Use explicit absolute server-local paths. The following names are examples;
the operator must substitute the inspected runtime, checkout, environment,
and a new campaign directory.

```bash
/absolute/env/bin/python /absolute/repo/scripts/prepare_followup_campaign.py \
  --runtime-root /absolute/runtime \
  --repo /absolute/repo \
  --python /absolute/env/bin/python \
  --campaign-root /absolute/runtime/campaigns/extension_fresh \
  --source-commit ACTUAL_COMMIT

/absolute/env/bin/python /absolute/control/run_followup_partition.py \
  --manifest /absolute/runtime/campaigns/extension_fresh/manifest.json \
  --partition current20 --plan

/absolute/env/bin/python /absolute/control/run_followup_partition.py \
  --manifest /absolute/runtime/campaigns/extension_fresh/manifest.json \
  --partition current20 --execute
```

The `current20` partition requires both original launch receipts and never
launches a replacement. On the added host, use the identical complete prepared
manifest, configurations, source paths, Python path, data paths and weight
cache, then run the same controller with `--partition tuna`. It independently
admits only the two TUNA jobs. Deploy the standalone controller outside the
immutable training source. It imports and hash-checks the original
manifest-bound queue and preparation modules.

Preparation never trains, never overwrites an existing campaign, and retains a
virtualenv Python path without resolving it to the base interpreter. Its
manifest binds the four expanded configurations, execution source-file hashes,
source commit, cache identity, runtime environment, commands, and absolute
data/output paths. No previous result directory is reused.

The operator must inspect the cloned server and confirm its physical GPU
inventory, dataset manifests, Python environment, and disk capacity before
execution. There must initially be at least 13 GiB free on the campaign
filesystem. This reserves approximate capacity for the four final models,
server-local backup copies, atomic-save overlap, and a safety margin; it is a
floor rather than permission to fill the disk. A further 20 GB of data-disk
capacity was recommended. Before each subsequent launch, the queue also
requires at least 4 GiB free and sufficient free GPU memory.

Run the dispatcher detached with its output preserved by the operator's
server-side process manager. Its training children use separate sessions and
individual append-free logs. Queue observation can be interrupted without
killing those children. Keep model backups on the server; do not initiate a
large model download to the local Mac.

## Ownership, recovery, and completion

- A nonblocking exclusive lock permits one active dispatcher per host campaign.
  Default execution is `--plan`, which reads state without creating logs,
  launch intents, receipts, locks, or completion seals.
- Before spawning, an exclusive launch-intent file is durably written. Each
  detached child's receipt records PID, Linux process start ticks, exact
  command, working directory, GPU, environment, config hash, and execution
  source-inventory hash.
- A restarted partition dispatcher adopts matching receipts. It never launches another
  copy merely because a remote observation timed out. An unresolved intent,
  unreceipted output/log, mismatched identity, or recorded failure requires
  operator review and blocks automatic admission. There are no retries or
  worker-killing rules.
- A matching, nonterminal PID with an empty command line remains reserved;
  Linux can clear argv during shutdown. A complete result JSON alone does not
  release the GPU. Terminal or absent original processes must additionally
  have valid complete artifacts before being sealed as complete.
- Completion requires ten ordered stages, the complete final 100-/200-class
  evaluation, the expected 10,000/6,000 final evaluation images, the exact saved
  formal configuration, a data manifest, and a final checkpoint. A partial
  stream, missing result, or observed nonzero worker exit is not completion.
- Original evidence is not deleted. Completion seals bind result, configuration,
  and data-manifest hashes. The queue does not infer method quality from test
  accuracy and does not select a replacement for the original 5-epoch row.
- Formal checkpoint backup and native CPU restore run on the host that owns
  each job. Partition audits reopen those files locally. The global merge binds
  both remote audit hashes and revalidates their exported metadata and metrics;
  it does not claim that model files were reopened on the local Mac.

## Runtime estimate and local checks

Using the original measured 5-epoch epoch-loop time plus measured non-epoch
overhead gives approximate 20-epoch durations of 10.30 hours for CIFAR-100 and
12.24 hours for ImageNet-R on the former hardware. These are extrapolations,
not measurements on the cloned server; do not assume a fourfold factor for
the entire run. Native TUNA runtime has not yet been measured. Re-estimate
using actual stage/epoch timing after launch, without changing the recipes.

Local queue validation uses fake processes, NVML snapshots, checkpoints, and
disk accounting; it does not train or allocate CUDA memory:

```bash
../.venv-pilot/bin/python -m pytest -q tests/test_followup_queue.py
```

The contracts cover read-only planning, restart adoption, empty-argv process
reservation, terminal/completion gating, failed/partial/ambiguous launch
blocking, cache/source/config admission, isolated GPU occupancy, safe
observation interruption, and no-overwrite preparation with a virtualenv
Python symlink. Actual remote runtime admission remains a separate preflight.
