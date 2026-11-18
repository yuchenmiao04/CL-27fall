# Additional formal experiments

Seed 1993; 4/4 fully verified follow-up results.
Accuracy: %. Forgetting: percentage points. Pending checkpoint evidence does not constitute a completed audit.
The fixed-library 20-epoch rows are additional budget variants; the original 5-epoch results remain separate.

Audit partition: `merged`.

Checkpoint files were verified on their recorded hosts. This merge rechecks exported metadata and does not reopen remote model files.

TUNA retains the native mechanism and 30-epoch classifier alignment, with the user-requested 20 image epochs per stage. This is not reproduction of the original native image-training budget.

Checkpoint recovery preserves the exact earlier stage records. Raw execution wall time and GPU peak cover only the resumed process; full-lineage runtime/peak are unknown. Stage-time totals and any discarded-work lower bounds are reported separately.

## CIFAR-100

| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| shared_basis | original | 84.68 | 90.23 | 6.01 | 6.01 | prior campaign reference |
| history_transport | original | 90.94 | 95.01 | 6.83 | 6.83 | prior campaign reference |
| current_adapted | epochs5 | 90.53 | 94.30 | 5.96 | 5.96 | prior campaign reference |
| plastic_shared_basis | original | 90.95 | 95.02 | 6.86 | 6.86 | prior campaign reference |
| er | original | 81.42 | 91.32 | 18.87 | 18.87 | prior campaign reference |
| joint | original | 93.48 | — | — | — | prior campaign reference |
| current_adapted | epochs20 | 89.47 | 93.82 | 8.44 | 8.44 | verified_complete |
| tuna | epochs20; native mechanism | 91.23 | 94.60 | 3.29 | 3.33 | verified_complete |

## ImageNet-R

| Method | Variant | Final | AIA | F_previous | F_including_final | Audit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| shared_basis | original | 67.60 | 74.30 | 7.30 | 7.30 | prior campaign reference |
| history_transport | original | 80.70 | 86.08 | 8.18 | 8.18 | prior campaign reference |
| current_adapted | epochs5 | 80.30 | 84.87 | 7.71 | 7.71 | prior campaign reference |
| plastic_shared_basis | original | 80.68 | 86.05 | 8.13 | 8.13 | prior campaign reference |
| er | original | 70.00 | 78.67 | 23.76 | 23.76 | prior campaign reference |
| joint | original | 83.87 | — | — | — | prior campaign reference |
| current_adapted | epochs20 | 81.72 | 85.88 | 9.20 | 9.20 | verified_complete |
| tuna | epochs20; native mechanism | 79.27 | 84.84 | 8.69 | 8.69 | verified_complete |

F_previous excludes the final evaluation from each historical maximum; F_including_final includes it.
Resource values are the measured runner/method records in the JSON; training costs are not inferred from epoch counts.
