# Partial formal results

8 of 12 formal dataset/method cells completed and audited. Full designated training data, selected test splits and seed 1993 are used. Remaining configuration selection uses only its fixed training holdout.

| Dataset | Method | Final (%) | AIA (%) | Forgetting previous (pp) |
| --- | --- | ---: | ---: | ---: |
| CIFAR-100 | Joint | 93.48 | — | — |
| CIFAR-100 | Frozen shared predictor basis | 84.68 | 90.23 | 6.01 |
| CIFAR-100 | ER | 81.42 | 91.32 | 18.87 |
| CIFAR-100 | History-statistics transport | 90.94 | 95.01 | 6.83 |
| CIFAR-100 | Plastic shared predictor basis | 90.95 | 95.02 | 6.86 |
| ImageNet-R | Frozen shared predictor basis | 67.60 | 74.30 | 7.30 |
| ImageNet-R | Joint | 83.87 | — | — |
| ImageNet-R | ER | 70.00 | 78.67 | 23.76 |

Joint is an offline full-class reference reporting Final only. Frozen shared basis retains zero historical images. Replay methods use 2,000 final images on CIFAR-100 and 20/class (4,000 final) on ImageNet-R. Native CPU restoration, actual source-data membership, arrivals, resident counts and all-class finite zero-input output are checked per completed replay cell without extra training or accuracy evaluation. Model backups remain server-local.

HT and plastic shared basis reuse their matching fixed formal ER gradient trajectory. The ten request/body/head identities and actual sealed-cache SHA256 values are verified. Shared native CE seconds and reader process wall (`execution.wall_seconds`) are separate costs; reader-only time/memory is not independent full-method training cost. See raw execution/resource records.

On CIFAR-100, plastic shared basis predicts one additional example correctly among 10,000 compared with HT. This single-seed result does not establish a clear extra benefit of the shared prior; regularization also differs. Remaining predeclared module candidates are selected from development results only.

Both forgetting definitions, stage matrices/counts, source hashes, original audit attempts and scope limits are retained in the linked JSON/receipt/observations. CPU restoration does not prove exact CUDA continuation or completion of all 12 cells.
