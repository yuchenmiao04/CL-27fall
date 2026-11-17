# Four follow-up experiments

This campaign adds the fixed-library 20-epoch variant and native PILOT TUNA on
CIFAR-100 and ImageNet-R. It preserves the original twelve results and adds four
fresh runs. Seed 1993, the selected data, class order, pretrained checkpoint,
and all-seen-class evaluation are unchanged. No parameter search is performed.

The training source snapshot is commit
`a006566923b1c9e5c981c58c615cdcd7c9dac3fa`. The prepared server manifest has
SHA-256 `8a70c7e5a6011bb293c833387d71046173503d4ba94efca3c3ee2587b82e4c43`.
The plan and complete method recipes are in [PLAN.md](PLAN.md).

The user subsequently selected **20 image-training epochs for both TUNA
datasets**, retaining classifier alignment at 30 epochs and all other native
settings. The selected TUNA overlay manifest has SHA-256
`04fda502b11e670bb3f13b35b31fef7545d2d47728b4947220210c6b3bd0c2c6`.
It embeds and binds the original manifest. Selected configurations are in
[execution/tuna_epochs20_configs](execution/tuna_epochs20_configs/).

## Execution

The fixed-library runs started on two RTX 5090 GPUs at approximately
2026-11-17 16:31 UTC. A second server with two RTX 4090 D GPUs is assigned the
two TUNA runs, one per GPU. The user authorized this four-GPU partition; TUNA
can start independently of the fixed-library runs after deployment checks.

`run_followup_partition.py --partition current20` adopts only the two original
fixed-library workers and never restarts them. The `tuna` partition launches
only the two TUNA jobs on the added server. The previous sequential dispatcher
was stopped without stopping either original training process. A timeout never
triggers a replacement worker.

The original manifest remains authoritative for the fixed-library runs. TUNA
uses the linked 20-epoch overlay in a fresh `campaign_tuna20` directory. The
source, data, pretrained weights and package versions are unchanged. Both TUNA
runs restarted from the same pretrained initialization because changing the
image-training horizon also changes the cosine learning-rate schedule. The
earlier 15-/10-epoch partial runs are preserved and excluded from final results.
Training uses immutable source directories. Checkpoints and their backup copies
remain on their respective servers.

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

Run `scripts/audit_followup_campaign.py` on each host with its partition,
manifest, checkpoint evidence, host alias and pinned SSH-key fingerprint. The
auditor independently derives a runtime identity from the boot ID and GPU
UUIDs, exporting only their hash and GPU model names. It reopens the actual
backup and native-restore report files, verifies their hashes, and recomputes
Final, AIA and both forgetting definitions from complete stage matrices and
sample counts. Passing metrics alone are insufficient for a completed audit.

Merge the two partition summaries using their SHA-256 values captured on the
servers and the original manifest. The merge verifies complementary job sets,
the exact permitted TUNA20 overlay, common source identity, distinct server
runtime identities, and exact metadata exports. It
recomputes the metrics without downloading or reopening remote model files.
The optional `--base-summary` imports the original twelve results unchanged,
retaining separate 5-epoch and 20-epoch fixed-library rows.

Formal results will be published only after those checks pass.
