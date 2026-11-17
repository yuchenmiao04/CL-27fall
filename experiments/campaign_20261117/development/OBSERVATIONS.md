# Development observations

These are fixed internal training-holdout results, not formal test results. Different stage/class counts are not directly comparable. Internal CE drivers are excluded from the formal ER baseline; rank0 runs are mechanism diagnostics.

| Trial | Dataset | Stages | Last-stage validation (%) | AIA (%) |
| --- | --- | --- | --- | --- |
| cifar_ce_screen_v1_er | cifar224 | 3 | 96.87 | 98.16 |
| cifar_ce_screen_v1_history_transport | cifar224 | 3 | 96.60 | 98.03 |
| cifar_ce_screen_v1_plastic_shared_basis | cifar224 | 3 | 96.60 | 98.03 |
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
| imagenetr_shared_initial_v1 | imagenetr | 10 | 43.31 | 44.58 |
| imagenetr_shared_isotropic_diagnostic_v6 | imagenetr | 10 | 67.59 | 74.61 |
| imagenetr_shared_noise0001_screen_v3 | imagenetr | 3 | 78.82 | 81.98 |
| imagenetr_shared_noise001_full_v2 | imagenetr | 10 | 64.08 | 72.03 |
| imagenetr_shared_noise001_screen_v2 | imagenetr | 3 | 76.81 | 81.80 |
| imagenetr_shared_rank64_full_v7 | imagenetr | 10 | 67.55 | 74.37 |
| imagenetr_shared_rank64_screen_v7 | imagenetr | 3 | 79.22 | 82.11 |
| imagenetr_shared_strict_full_v5 | imagenetr | 10 | 67.42 | 74.27 |
| imagenetr_shared_strict_screen_v5 | imagenetr | 3 | 78.69 | 81.87 |

The same-pool second-stage module diagnostic was anchor93.8%, common98.0%, full98.0%; this does not establish an isolated module benefit. Shared-prior gains should be compared with the corresponding isotropic ridge scale, not only the original noise1 setting.

Source records and export hashes are retained in observations/ and observations_manifest.json. The snapshot deliberately contains no formal test outputs.
