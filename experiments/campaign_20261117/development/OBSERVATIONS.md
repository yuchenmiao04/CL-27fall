# Development observations

Fixed internal training-holdout results only. Different stage/class counts are not directly comparable. Internal CE drivers are not the formal ER baseline; rank0 runs are mechanism diagnostics.

| Trial | Dataset | Stages | Last-stage validation (%) | AIA (%) |
| --- | --- | --- | --- | --- |
| cifar_ce_screen_v1_er | cifar224 | 3 | 96.87 | 98.16 |
| cifar_ce_screen_v1_history_transport | cifar224 | 3 | 96.60 | 98.03 |
| cifar_ce_screen_v1_plastic_shared_basis | cifar224 | 3 | 96.60 | 98.03 |
| cifar_current_bg1_screen_v2 | cifar224 | 2 | 97.30 | 98.25 |
| cifar_current_initial_v1 | cifar224 | 2 | 98.00 | 98.70 |
| cifar_er_initial_v1 | cifar224 | 2 | 97.80 | 98.80 |
| cifar_shared_em100_screen_v4 | cifar224 | 3 | 92.80 | 94.73 |
| cifar_shared_initial_v1 | cifar224 | 10 | 64.28 | 76.59 |
| cifar_shared_isotropic_diagnostic_v6 | cifar224 | 10 | 84.28 | 89.60 |
| cifar_shared_noise0001_full_v3 | cifar224 | 10 | 84.38 | 89.72 |
| cifar_shared_noise0001_screen_v3 | cifar224 | 3 | 92.80 | 94.73 |
| cifar_shared_noise001_v2 | cifar224 | 10 | 83.86 | 89.34 |
| cifar_shared_rank64_full_v7 | cifar224 | 10 | 84.44 | 89.77 |
| cifar_shared_strict_full_v5 | cifar224 | 10 | 84.30 | 89.70 |
| cifar_shared_strict_screen_v5 | cifar224 | 3 | 92.80 | 94.73 |
| cifar_shared_stronger_full_v8 | cifar224 | 10 | 83.98 | 89.69 |
| cifar_shared_stronger_isotropic_diagnostic_v8 | cifar224 | 10 | 83.32 | 88.71 |
| cifar_shared_stronger_screen_v8 | cifar224 | 3 | 92.80 | 94.73 |
| imagenetr_shared_initial_v1 | imagenetr | 10 | 43.31 | 44.58 |
| imagenetr_shared_isotropic_diagnostic_v6 | imagenetr | 10 | 67.59 | 74.61 |
| imagenetr_shared_noise0001_screen_v3 | imagenetr | 3 | 78.82 | 81.98 |
| imagenetr_shared_noise001_full_v2 | imagenetr | 10 | 64.08 | 72.03 |
| imagenetr_shared_noise001_screen_v2 | imagenetr | 3 | 76.81 | 81.80 |
| imagenetr_shared_rank64_full_v7 | imagenetr | 10 | 67.55 | 74.37 |
| imagenetr_shared_rank64_screen_v7 | imagenetr | 3 | 79.22 | 82.11 |
| imagenetr_shared_strict_full_v5 | imagenetr | 10 | 67.42 | 74.27 |
| imagenetr_shared_strict_screen_v5 | imagenetr | 3 | 78.69 | 81.87 |
| imagenetr_shared_stronger_full_v8 | imagenetr | 10 | 64.34 | 72.87 |
| imagenetr_shared_stronger_isotropic_diagnostic_v8 | imagenetr | 10 | 65.43 | 72.13 |
| imagenetr_shared_stronger_screen_v8 | imagenetr | 3 | 79.22 | 82.30 |

D0023 selects the completed nonzero-rank K64/noise0.001/epsilon0.001/max100/tol1e-8 trial for fresh full-data formal refits. Shared-prior gains over a same-scale isotropic ridge are not established.
D0024 retains background5 for the module full flow after background1 lost0.7point on the two-stage prefix. Both same-pool module diagnostics give common=full at this stage; no isolated module gain has been established.

Raw exported records and their original/exported hashes are retained. Formal test outputs are excluded.
