# Additional measured resources

Four verified additions; original twelve results, resource records and checkpoint coverage retained unchanged.

The original twelve resource records remain unchanged in [the prior report](../../campaign_20261117/resources/RESOURCE_REPORT.md).

| Dataset | Method / image epochs | Resident NN (M) | Auxiliary excluding optimizers (MiB) | Persistent optimizers (MiB) | Checkpoint (MiB) | GPU peak* (GiB) | Runner wall* (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar224 | current_adapted /20 | 171.702 | 18.02 | 655.39 | 1337.96 | 4.55 | 132.22 |
| imagenetr | current_adapted /20 | 171.702 | 31.51 | 655.39 | 1343.69 | 4.55 | 206.58 |
| cifar224 | tuna /20 | 89.604 | 225.67 | 0.00 | 568.09 | 1.91 | 50.75 |
| imagenetr | tuna /20 | 89.758 | 450.77 | 0.00 | 793.80 | 3.44 | 26.99 |

*Raw runner wall and GPU peak cover only the final training process. The complete wall time and peak across all attempts are unknown. Auxiliary sizes describe logical numeric storage.

| Dataset / method | Optimizer-qualified NN (M) | Restored requires_grad NN (M) | Ten completed stages (min) | Completed stage ranges and minutes by attempt | Discarded work lower bound (min) |
| --- | ---: | ---: | ---: | --- | ---: |
| cifar224 / current_adapted | 85.903112 | 85.903112 | 619.61 | 1–2 (91.25), 3–8 (396.23), 9–10 (132.13) | 45.80 |
| imagenetr / current_adapted | 85.903112 | 85.903112 | 656.03 | 1–2 (113.04), 3–7 (336.53), 8–10 (206.46) | 83.92 |
| cifar224 / tuna | 3.424320 | 3.805540 | 76.39 | 1–4 (25.88), 5–10 (50.51) | 0.00 |
| imagenetr / tuna | 3.501120 | 3.959240 | 52.79 | 1–6 (26.01), 7–10 (26.78) | 0.00 |

Optimizer-family qualification and terminal requires_grad are separate definitions. Neither asserts that every listed parameter received gradients. Completed stage timers exclude checkpoint-write regions and do not recover interrupted work.

## Restored components

| Dataset / method | Component | NN parameters | Parameter MiB |
| --- | --- | ---: | ---: |
| cifar224 / current_adapted | frozen_anchor | 85798656 | 327.30 |
| cifar224 / current_adapted | predictive_backbone | 85798656 | 327.30 |
| cifar224 / current_adapted | module_libraries | 104456 | 0.40 |
| imagenetr / current_adapted | frozen_anchor | 85798656 | 327.30 |
| imagenetr / current_adapted | predictive_backbone | 85798656 | 327.30 |
| imagenetr / current_adapted | module_libraries | 104456 | 0.40 |
| cifar224 / tuna | original_backbone_body | 85798656 | 327.30 |
| cifar224 / tuna | native_backbone_heads | 76900 | 0.29 |
| cifar224 / tuna | historical_adapters | 3043200 | 11.61 |
| cifar224 / tuna | current_adapters | 304320 | 1.16 |
| cifar224 / tuna | merged_adapters | 304320 | 1.16 |
| cifar224 / tuna | class_predictor_heads | 76800 | 0.29 |
| imagenetr / tuna | original_backbone_body | 85798656 | 327.30 |
| imagenetr / tuna | native_backbone_heads | 153800 | 0.59 |
| imagenetr / tuna | historical_adapters | 3043200 | 11.61 |
| imagenetr / tuna | current_adapters | 304320 | 1.16 |
| imagenetr / tuna | merged_adapters | 304320 | 1.16 |
| imagenetr / tuna | class_predictor_heads | 153600 | 0.59 |

The JSON preserves separate historical statistics, analytic readouts, resident caches, arrival IDs, replay representation, and optimizer tensors. Nested subtotals are not summed twice. TUNA has one original backbone instance but eleven derived native backbone invocations per final evaluation batch; zero persisted optimizers does not mean zero transient training optimizer memory.

## CPU verification cost

| Dataset / method | Audit wall (s) | Maximum child RSS (GiB) | Exit |
| --- | ---: | ---: | ---: |
| cifar224 / current_adapted | 55.30 | 4.84 | 0 |
| imagenetr / current_adapted | 61.60 | 4.62 | 0 |
| cifar224 / tuna | 27.55 | 2.48 | 0 |
| imagenetr / tuna | 36.06 | 2.98 | 0 |

CPU audit measurements are separate operations, excluded from all training timings and GPU peaks. Every sidecar is matched to its owning-server SHA capture; checkpoint evidence and restore report hashes additionally match the sealed partition proof.
