# Campaign main results

Seed 1993; 11/12 eligible final results. Missing results are —.
Accuracy: %. Forgetting: percentage points. Joint is an offline reference; its AIA and forgetting are not applicable.
Selections use explicit run paths when duplicates exist, never the highest reported test score.

## CIFAR-100

| Method | Historical image permission | Final ↑ | AIA ↑ | F_previous ↓ | F_including_final ↓ |
| --- | --- | --- | --- | --- | --- |
| shared_basis | 0 | 84.68 | 90.23 | 6.01 | 6.01 |
| history_transport | 2000 total | 90.94 | 95.01 | 6.83 | 6.83 |
| current_adapted | — | — | — | — | — |
| plastic_shared_basis | 2000 total | 90.95 | 95.02 | 6.86 | 6.86 |
| er | 2000 total | 81.42 | 91.32 | 18.87 | 18.87 |
| joint | offline / all training data | 93.48 | — | — | — |

## ImageNet-R

| Method | Historical image permission | Final ↑ | AIA ↑ | F_previous ↓ | F_including_final ↓ |
| --- | --- | --- | --- | --- | --- |
| shared_basis | 0 | 67.60 | 74.30 | 7.30 | 7.30 |
| history_transport | 20/class (4000 final) | 80.70 | 86.08 | 8.18 | 8.18 |
| current_adapted | 20/class (4000 final) | 80.30 | 84.87 | 7.71 | 7.71 |
| plastic_shared_basis | 20/class (4000 final) | 80.68 | 86.05 | 8.13 | 8.13 |
| er | 20/class (4000 final) | 70.00 | 78.67 | 23.76 | 23.76 |
| joint | offline / all training data | 83.87 | — | — | — |

F_previous excludes the final evaluation when taking each old task's historical maximum; F_including_final includes it. Raw metrics, resource records, selected sources and excluded-run reasons are retained in summary.json.
