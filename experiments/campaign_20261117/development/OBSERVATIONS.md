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
| cifar_transport_chained_screen_v2 | cifar224 | 3 | 97.53 | 98.38 |
| imagenetr_ce_full_v1_er | imagenetr | 10 | 69.80 | 78.34 |
| imagenetr_ce_full_v1_history_transport | imagenetr | 10 | 79.66 | 84.47 |
| imagenetr_ce_full_v1_plastic_shared_basis | imagenetr | 10 | 78.06 | 83.17 |
| imagenetr_ce_screen_v1_er | imagenetr | 3 | 82.71 | 88.73 |
| imagenetr_ce_screen_v1_history_transport | imagenetr | 3 | 86.19 | 90.02 |
| imagenetr_ce_screen_v1_plastic_shared_basis | imagenetr | 3 | 84.72 | 89.64 |
| imagenetr_current_bounded64_screen_v5 | imagenetr | 3 | 87.40 | 90.48 |
| imagenetr_psb_chained_screen_v4 | imagenetr | 3 | 88.47 | 91.16 |
| imagenetr_psb_noise0001_screen_v2 | imagenetr | 3 | 85.92 | 89.93 |
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
| imagenetr_transport_chained_screen_v2 | imagenetr | 3 | 88.47 | 91.16 |

D0023 fixed the frozen-basis parameters before its formal runs. D0026/D0027 introduced a fresh bounded64 module candidate after auditing the legacy all-buffer sampling cost. D0030 corrected an inherited epoch setting before any ImageNet-R module task completed.
D0031–D0034 use only development evidence to test lower plastic-basis prior noise. D0035–D0038 test chained coordinates and combine the surviving prior with that reference strategy. D0036/D0039 extend bounded module candidates from their own prefixes. D0041 extends the surviving chained PSB prefixes without changing their parameters.
Any final selected candidate must complete all10 development stages before fresh full-data formal refit. Native replay-only execution reuses exact CE transitions and still independently fits moments/herding/readouts. D0040 releases completed CE driver reservations without changing training or head parameters.

Source records and original/export hashes are retained. Completed formal results are separate in ../formal/PARTIAL_RESULTS.md; they are not used in subsequent tuning decisions.
