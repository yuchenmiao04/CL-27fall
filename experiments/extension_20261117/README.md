# Four follow-up experiments

This campaign adds the fixed-library 20-epoch variant and native PILOT TUNA on
CIFAR-100 and ImageNet-R. It preserves the original twelve results and adds four
fresh runs. Seed 1993, the selected data, class order, pretrained checkpoint,
and all-seen-class evaluation are unchanged. No parameter search is performed.

The training source snapshot is commit
`a006566923b1c9e5c981c58c615cdcd7c9dac3fa`. The prepared server manifest has
SHA-256 `8a70c7e5a6011bb293c833387d71046173503d4ba94efca3c3ee2587b82e4c43`.
The plan and complete method recipes are in [PLAN.md](PLAN.md).

## Execution

The new server has two RTX 5090 GPUs. Two independent fixed-library runs started
at approximately 2026-11-17 16:31 UTC. GPU 0 runs CIFAR-100, followed by TUNA on
CIFAR-100; GPU 1 runs ImageNet-R, followed by TUNA on ImageNet-R. The dispatcher
requires the original process to terminate and its ten-stage result to be
complete before admitting the next job. A timeout never triggers a replacement.

The prepared manifest binds the expanded configurations, source hashes, exact
cached weights, runtime environment, commands, and output locations. Training
uses an immutable source directory. Checkpoints remain on the server.

## Correctness evidence

- `correctness/local_checks.json` records local native-fidelity, lifecycle,
  protocol, execution-reuse, queue, and metric-audit checks.
- `correctness/genuine_pretrained_validation.json` records two-stage native
  TUNA checks on the real ViT-B/16 initialization for both dataset recipes.
  These use training holdouts, reduced epoch counts, and limited examples;
  they are implementation checks rather than formal performance results.
- `correctness/tuna_cifar_native_restore.json` and
  `correctness/tuna_imagenetr_native_restore.json` record exact CPU restoration
  of those genuine checkpoints: 7,583 and 8,233 compared nodes, respectively,
  with no ignored fields. No additional training or test evaluation was used.

## Final verification

After a formal worker terminates, run `scripts/backup_followup_checkpoint.py`
with the prepared manifest and job name. It writes an atomic server-local copy,
rebuilds the data identity, restores the native learner on CPU, checks every
persisted learner field, and binds the evidence to source and metadata hashes.
It does not apply CUDA RNG snapshots or transfer models to another machine.

Then run `scripts/audit_followup_campaign.py` with the manifest and checkpoint
evidence directory. It recomputes Final, AIA, and both forgetting definitions
from the complete stage matrices and sample counts. Passing result metrics
alone are insufficient for a completed audit. The optional `--base-summary`
argument imports the original twelve results unchanged and retains separate
5-epoch and 20-epoch fixed-library rows.

Formal results will be published only after those checks pass.
