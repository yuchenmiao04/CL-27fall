# Measured resources (11/12 formal cells)

Counts use SHA-bound terminal CPU checkpoint tensors and native parameter-name/shape registries. Pending cells receive no estimates. M = 10^6 parameters; MiB = 2^20 bytes; GiB = 2^30 bytes. NN counts exclude analytic coefficients and moment arrays.

| Dataset | Method | SGD-eligible NN (M) | Training NN (M) | Inference backbones | Auxiliary numeric state (MiB) | Checkpoint (MiB) | Allocated GPU peak (GiB) | Actual runner wall (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | Frozen shared basis | 0.000 | 85.799 | 1 | 6.05 | 335.40 | 0.83 | 8.36 |
| CIFAR-100 | History transport | 85.876 | 171.674 | 1 | 15.30 | 686.45 | 1.15 | 27.81 |
| CIFAR-100 | Fixed module library | — | — | — | — | — | — | — |
| CIFAR-100 | Plastic shared basis | 85.876 | 171.674 | 1 | 21.16 | 694.25 | 1.15 | 27.99 |
| CIFAR-100 | ER | 85.876 | 85.876 | 1 | 0.00 | 336.40 | 18.46 | 252.45 |
| CIFAR-100 | Joint | 85.876 | 85.876 | 1 | 0.00 | 327.65 | 9.92 | 73.24 |
| ImageNet-R | Frozen shared basis | 0.000 | 85.799 | 1 | 7.23 | 336.37 | 0.83 | 9.05 |
| ImageNet-R | History transport | 85.952 | 171.751 | 1 | 17.06 | 681.33 | 1.15 | 21.23 |
| ImageNet-R | Fixed module library | 85.903 | 171.702 | 2 | 686.89 | 1343.66 | 3.92 | 193.52 |
| ImageNet-R | Plastic shared basis | 85.952 | 171.751 | 1 | 24.10 | 690.26 | 1.15 | 21.55 |
| ImageNet-R | ER | 85.952 | 85.952 | 1 | 0.00 | 328.40 | 18.46 | 158.76 |
| ImageNet-R | Joint | 85.952 | 85.952 | 1 | 0.00 | 327.94 | 9.92 | 125.45 |

SGD eligibility is the union of native optimizer parameter families, independent of the final phase-specific requires_grad flags. Frozen shared basis has no SGD parameters but learns its covariance factor and posterior numerically. HT/PSB eligibility includes the shared CE-training head; deployment uses an analytic head. Current has no CE head and uses both anchor and main in full inference. Dense expert gates execute every expert.

Auxiliary numeric state is logical tensor payload, not RSS or CUDA residency. It includes current’s two persistent AdamW states; separate current components below prevent confusing optimizer memory with historical statistics. Encoder payload already contains both backbones, module Parameters and buffers; it is not added again to auxiliary state. HT statistics already include its analytic head; PSB additionally retains its basis arrays/posterior. Do not sum nested totals and components twice.

| Current dataset | Encoder including NN/buffers (MiB) | History G/C (MiB) | W0/V (MiB) | Resident anchor cache (MiB) | Persistent optimizers (MiB) | Successful module / body steps | Attempts / skips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| imagenetr | 654.99 | 5.69 | 2.35 | 23.47 | 655.39 | 5685 / 1895 | 7580 / 0 |

## Timing and shared work

The 11 completed selected runners contain 55165.248372 seconds of measured timer regions in total. Count each actual runner once: the ER donor already includes native CE training, while HT/PSB rows are their actual reader workflows. These concurrent timer-region sums are neither elapsed campaign time nor GPU-hours.

- `formal_cifar_er_fixed_v1`: 14653.037004 seconds of native CE, already contained in its ER runner; 10 cache files, 3434381358 bytes, counted once.
- `formal_imagenetr_er_fixed_v1`: 9029.934395 seconds of native CE, already contained in its ER runner; 10 cache files, 3436073326 bytes, counted once.

HT/PSB reader wall and peak exclude donor gradient training and cannot represent independent full-method cost or a controlled speedup. Reader execution also includes feature encoding, moment/readout fitting, evaluation and checkpoint writes. Never add copied donor-work counters again. Current’s successful module/body steps and attempted/skipped episodes are observed counters, not a no-skip schedule estimate.

Runner timing starts after imports/model/data/resume setup and includes task work, evaluation, checkpoint writes and final numeric accounting before final result serialization. Stage timers exclude checkpoint writes. Resumed development runner wall excludes the inherited prefix; retained stage totals include it. Per-process PyTorch allocated peak excludes reserved/context/other-process memory and is not an independent method peak for a CE reader.

Search costs are separate in [HPO_COST_LEDGER.md](HPO_COST_LEDGER.md): 97,775.777922 seconds is a documented lower bound of development/diagnostic timer regions, with unmeasured stopped tails left unknown. The HPO ledger’s formal_partial section is its historical 10-cell snapshot, not the final formal total; use the measured row set above for the current formal total.

## Storage and evidence

Replay is an image budget: CIFAR-100 uses 2,000 final images; ImageNet-R uses 20/class, 4,000 final. Frozen shared basis retains none; Joint has offline access to the full training union. Current, HT, PSB and ER retain their own legal selected records. CIFAR payload is RGB arrays plus identity accounting. ImageNet-R payload is decoded RGB equivalent plus identity accounting, not compressed JPEG disk bytes, path-string memory, or simultaneously resident decoded images. R checkpoints require the external original image files.

Checkpoint file size differs from runner numeric_checkpoint_payload_bytes: the latter recursively counts tensors/arrays and excludes Python/RawRecord image contents and strings without alias deduplication. CE cache disk, replay/source images, model checkpoints, neural weights, analytic arrays and persistent optimizer tensors are separate quantities.

Exact values, tensor dtype/shape/numel and source hashes are retained in [PARTIAL_RESOURCE_EVIDENCE.json](PARTIAL_RESOURCE_EVIDENCE.json) and its [export manifest](RESOURCE_EXPORT_MANIFEST.json). Checkpoint locations and actual audit coverage are indexed in [recovery](../recovery/RECOVERY.md).
