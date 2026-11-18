# Additional formal experiments

Seed 1993; 2/2 fully verified follow-up results.
Accuracy: %. Forgetting: percentage points. Pending checkpoint evidence does not constitute a completed audit.
The fixed-library 20-epoch rows are additional budget variants; the original 5-epoch results remain separate.

Audit partition: `current20`.

Jobs assigned to the other host partition: formal_cifar_tuna_native_v1, formal_imagenetr_tuna_native_v1.

Checkpoint recovery preserves the exact earlier stage records. Raw execution wall time and GPU peak cover only the resumed process; full-lineage runtime/peak are unknown. Stage-time totals and any discarded-work lower bounds are reported separately.

## CIFAR-100

| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| current_adapted | epochs20 | 89.47 | 93.82 | 8.44 | 8.44 | verified_complete |

## ImageNet-R

| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| current_adapted | epochs20 | 81.72 | 85.88 | 9.20 | 9.20 | verified_complete |

F_previous excludes the final evaluation from each historical maximum; F_including_final includes it.
Resource values are the measured runner/method records in the JSON; training costs are not inferred from epoch counts.
