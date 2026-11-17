# Additional formal experiments

Seed 1993; 2/2 fully verified follow-up results.
Accuracy: %. Forgetting: percentage points. Pending checkpoint evidence does not constitute a completed audit.
The fixed-library 20-epoch rows are additional budget variants; the original 5-epoch results remain separate.

Audit partition: `tuna`.

Jobs assigned to the other host partition: formal_cifar_current_epochs20_v1, formal_imagenetr_current_epochs20_v1.

TUNA retains the native mechanism and 30-epoch classifier alignment, with the user-requested 20 image epochs per stage. This is not reproduction of the original native image-training budget.

Checkpoint recovery preserves the exact earlier stage records. Raw execution wall time and GPU peak cover only the resumed process; full-lineage runtime/peak are unknown. Stage-time totals and any discarded-work lower bounds are reported separately.

## CIFAR-100

| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tuna | epochs20; native mechanism | 91.23 | 94.60 | 3.29 | 3.33 | verified_complete |

## ImageNet-R

| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tuna | epochs20; native mechanism | 79.27 | 84.84 | 8.69 | 8.69 | verified_complete |

F_previous excludes the final evaluation from each historical maximum; F_including_final includes it.
Resource values are the measured runner/method records in the JSON; training costs are not inferred from epoch counts.
