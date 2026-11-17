# Development observations

Internal training-holdout scores only. Different stage/class counts are not directly comparable. Internal CE drivers and rank0 controls are not formal baselines. Partial/stopped branches without completed result files are not counted as completed trials.

| Trial | Dataset | Stages | Last-stage validation (%) | AIA (%) |
| --- | --- | --- | --- | --- |
| cifar_ce_full_v1_er | cifar224 | 10 | 81.50 | 91.30 |
| cifar_ce_full_v1_history_transport | cifar224 | 10 | 89.74 | 93.90 |
| cifar_ce_screen_v1_er | cifar224 | 3 | 96.87 | 98.16 |
| cifar_ce_screen_v1_history_transport | cifar224 | 3 | 96.60 | 98.03 |
| cifar_ce_screen_v1_plastic_shared_basis | cifar224 | 3 | 96.60 | 98.03 |
| cifar_current_bg1_screen_v2 | cifar224 | 2 | 97.30 | 98.25 |
| cifar_current_bounded64_screen_v4 | cifar224 | 3 | 96.87 | 97.99 |
| cifar_current_initial_v1 | cifar224 | 2 | 98.00 | 98.70 |
| cifar_er_initial_v1 | cifar224 | 2 | 97.80 | 98.80 |
| cifar_psb_chained_screen_v4 | cifar224 | 3 | 97.60 | 98.40 |
| cifar_psb_noise1e3_screen_v2 | cifar224 | 3 | 96.60 | 98.03 |
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
| cifar_transport_chained_full_v2 | cifar224 | 10 | 90.80 | 94.81 |
| cifar_transport_chained_screen_v2 | cifar224 | 3 | 97.53 | 98.38 |
| imagenetr_ce_full_v1_er | imagenetr | 10 | 69.80 | 78.34 |
| imagenetr_ce_full_v1_history_transport | imagenetr | 10 | 79.66 | 84.47 |
| imagenetr_ce_full_v1_plastic_shared_basis | imagenetr | 10 | 78.06 | 83.17 |
| imagenetr_ce_screen_v1_er | imagenetr | 3 | 82.71 | 88.73 |
| imagenetr_ce_screen_v1_history_transport | imagenetr | 3 | 86.19 | 90.02 |
| imagenetr_ce_screen_v1_plastic_shared_basis | imagenetr | 3 | 84.72 | 89.64 |
| imagenetr_current_bounded64_screen_v5 | imagenetr | 3 | 87.40 | 90.48 |
| imagenetr_psb_chained_full_v4 | imagenetr | 10 | 80.27 | 85.76 |
| imagenetr_psb_chained_screen_v4 | imagenetr | 3 | 88.47 | 91.16 |
| imagenetr_psb_noise0001_screen_v2 | imagenetr | 3 | 85.92 | 89.93 |
| imagenetr_psb_noise1e3_full_v2 | imagenetr | 10 | 79.49 | 84.33 |
| imagenetr_psb_noise1e4_full_v3 | imagenetr | 10 | 79.66 | 84.47 |
| imagenetr_psb_noise1e4_screen_v3 | imagenetr | 3 | 86.19 | 90.02 |
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
| imagenetr_transport_chained_full_v2 | imagenetr | 10 | 80.31 | 85.77 |
| imagenetr_transport_chained_map01_screen_v3 | imagenetr | 3 | 88.20 | 91.20 |
| imagenetr_transport_chained_screen_v2 | imagenetr | 3 | 88.47 | 91.16 |

Selection reasons, candidate changes and promotion requirements are recorded in decisions/. Every final selected candidate must complete all10 development stages before a fresh full-data formal refit. Prepared configurations are not evidence that a task ran.

Native replay-only execution reuses exact CE transitions and independently fits moments/herding/readouts. Physical-device reordering, where used, is an explicitly recorded execution experiment and requires its own numerical evidence.

Source records and original/export hashes are retained. Formal results are separate in ../formal/PARTIAL_RESULTS.md and are not used in subsequent tuning decisions.
