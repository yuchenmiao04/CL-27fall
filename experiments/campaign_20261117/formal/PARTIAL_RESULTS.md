# Partial formal results

7 of 12 formal dataset/method cells completed and audited. All use full designated training data, the selected test splits and seed 1993. Completed scores are reported at the user's request; the remaining module selection uses only its fixed training holdout.

| Dataset | Method | Final (%) | AIA (%) | Forgetting previous (pp) |
| --- | --- | ---: | ---: | ---: |
| CIFAR-100 | Joint | 93.48 | — | — |
| CIFAR-100 | Frozen shared predictor basis | 84.68 | 90.23 | 6.01 |
| CIFAR-100 | ER | 81.42 | 91.32 | 18.87 |
| CIFAR-100 | History-statistics transport | 90.94 | 95.01 | 6.83 |
| CIFAR-100 | Plastic shared predictor basis | 90.95 | 95.02 | 6.86 |
| ImageNet-R | Frozen shared predictor basis | 67.60 | 74.30 | 7.30 |
| ImageNet-R | Joint | 83.87 | — | — |

Joint is the offline full-class reference and only has Final. Frozen shared basis stores no historical images. The three completed replay-based CIFAR-100 methods each retain 2,000 final historical training images. Their actual data membership, 50,000 arrivals, checkpoint state and finite 100-class CPU deployment output were checked without extra training or accuracy evaluation. Their verified model backups remain on the server.

HT and plastic shared basis reused the exact same fixed formal ER gradient trajectory. All ten requests/body/head identities and actual sealed-cache SHA256 values were checked. Shared native CE training took 14,653.037 seconds under the observed concurrent workload; reader process wall (`execution.wall_seconds`) was 1,668.672 / 1,679.604 seconds, respectively. Reader wall and memory do not represent independent full-method training costs and must not replace or be confused with shared gradient-training cost.

Plastic shared basis classified one additional example correctly among 10,000 compared with HT. This single-seed result does not establish a clear extra benefit of the shared predictor prior; their regularization settings also differ. No formal score is used to change the remaining predeclared module candidates.

Other cells remain incomplete. Raw metrics include both forgetting definitions, stage matrices/counts, resource fields and source hashes. The first ER CPU audit had a source-inventory error involving a virtual Torch module; the fixed audit reran all checks. Both attempts are retained. CPU restore does not certify exact CUDA continuation or completion of the whole campaign.
