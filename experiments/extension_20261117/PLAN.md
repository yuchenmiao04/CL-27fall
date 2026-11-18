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

### TUNA, native algorithm with the user-selected 20-epoch image budget

Use `vit_base_patch16_224_tuna` with the admitted shared checkpoint and the
native pinned PILOT algorithm. Common data, class split, checkpoint and
structured evaluation are adapted; the user additionally selected 20
image-training epochs per stage for both datasets.

| Setting | CIFAR-100 | ImageNet-R |
| --- | --- | --- |
| Selected image-training epochs per stage | 20 | 20 |
| Original PILOT image-training epochs, archival reference | 15 | 10 |
| Optimizer | SGD | SGD |
| Initial learning rate | 0.01 | 0.02 |
| Batch size | 16 | 32 |
| Weight decay | 0.0005 | 0.0005 |
| Scheduler | Cosine | Cosine |
| Classifier-alignment epochs, stages 2–10 | 30 | 30 |
| Classifier-alignment learning rate | 0.005 | 0.005 |
| Orthogonality regularization enabled | false | true |
| Stored historical images | 0 | 0 |

The original templates remain in `configs/formal_cifar_tuna_native_v1.json`
and `configs/formal_imagenetr_tuna_native_v1.json`. Selected 20-epoch
configurations are in `execution/tuna_epochs20_configs/`.
`prepare_tuna_epochs20.py` first admits the exact original manifest and native
recipes, then permits only the requested image-epoch change, budget marker,
prefix and new output paths. Classifier alignment is absent in stage 1 and
remains 30 epochs in stages 2–10 with the native cosine period of 5. The 20
image epochs do not imply matched FLOPs.

The 15-/10-epoch runs were stopped at the user's explicit request. Their partial
records remain in the original campaign and do not count as final results.
Both selected 20-epoch runs start afresh from the original cached checkpoint.

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
launches a replacement. On the added host, preserve the original complete
manifest, source paths, Python path, data paths and weight cache. Prepare the
selected TUNA20 overlay and use its manifest with `--partition tuna`:

```bash
/absolute/env/bin/python /absolute/control/prepare_tuna_epochs20.py \
  --base-manifest /absolute/runtime/campaign/manifest.json \
  --campaign-root /absolute/runtime/campaign_tuna20

/absolute/env/bin/python /absolute/control/run_followup_partition.py \
  --manifest /absolute/runtime/campaign_tuna20/manifest.json \
  --partition tuna --execute
```

The controller independently admits only the two TUNA jobs. The overlay retains
unused current20 configuration copies for the original controller API; they are
never launched on the added host. Deploy the standalone controller outside the
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

### Task-boundary recovery after the server restart

The user reported an accidental shutdown of both hosts. After verifying that
the original workers were terminal, the retained native checkpoints were
admitted for recovery: two completed stages in each fixed-library run, four
in CIFAR-100 TUNA, and six in ImageNet-R TUNA. Each source checkpoint was copied
to a separate server-local `resume_sources` directory and its SHA-256 retained.
New outputs use `recovery_20261118_v1/runs/JOB`; original outputs are preserved.

`scripts/resume_followup_after_reboot.py` prepares and controls this explicit
recovery. The recovered config may change only `resume` and `output_dir`.
Preparation validates source/config/data identities, complete saved stage
histories, checkpoint fingerprints and RNG policy. Execution retains the
existing launch-ownership and terminal-process rules. Recovered completion
requires the saved prefix, including its recorded durations, to match exactly.
No active training source or scientific hyperparameter changes are permitted.

For a recovered job, pass the original recipe manifest and the additional
recovery manifest to both postprocessing tools:

```bash
/absolute/env/bin/python /absolute/control/backup_followup_checkpoint.py \
  --manifest /absolute/original_recipe/manifest.json \
  --recovery-manifest /absolute/recovery/manifest.json --job JOB \
  --backup-root /absolute/server_backups \
  --evidence-root /absolute/recovery/checkpoint_evidence \
  --state-loader /absolute/runtime/scripts/compare_checkpoint_state.py

/absolute/env/bin/python /absolute/control/audit_followup_campaign.py \
  --manifest /absolute/original_recipe/manifest.json \
  --recovery-manifest /absolute/recovery/manifest.json \
  --partition current20 --host-alias current20_server \
  --host-key-fingerprint PINNED_FINGERPRINT \
  --checkpoint-evidence-root /absolute/recovery/checkpoint_evidence \
  --output-dir /absolute/recovery/summary_current20
```

Use the TUNA20 overlay as the original recipe manifest on the TUNA host and
substitute partition `tuna` and its own host alias. Restoration is CPU-only;
models and backup payloads remain on their owning server. Do not clear
`CUDA_VISIBLE_DEVICES` before admission; the backup utility disables CUDA
internally after validating the original launch environment.

After recovery, `execution.wall_seconds` and the CUDA memory peak describe the
resumed process only. Summing all ten `stage.wall_seconds` values gives the
retained-plus-resumed stage-loop duration, excluding checkpoint I/O and lost
partial-stage work. Report those scopes explicitly. Neither that sum nor a
resumed peak is an uninterrupted full-run measurement. Retained logs can
support lower bounds on discarded work; unmeasured work remains unknown.

## Initial runtime estimate and local checks

Using the original measured 5-epoch epoch-loop time plus measured non-epoch
overhead gives approximate 20-epoch durations of 10.30 hours for CIFAR-100 and
12.24 hours for ImageNet-R on the former hardware. These are extrapolations,
not measurements on the cloned server; do not assume a fourfold factor for
the entire run. TUNA had not been measured when these initial estimates were
written. Its observed stage durations and resumed-process measurements are now
recorded in [partitions/tuna/summary.json](partitions/tuna/summary.json); complete
uninterrupted-run wall time remains unknown after recovery. Use the recorded
timing scopes without changing the recipes.

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
