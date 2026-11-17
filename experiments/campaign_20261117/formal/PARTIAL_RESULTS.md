# Partial formal results

5 of 12 formal dataset/method cells completed and audited. These are full-training refits evaluated on the selected test splits, with seed 1993. Completed-method scores are reported at the user's request; remaining configuration selection uses only the fixed internal training holdout.

| Dataset | Method | Final (%) | AIA (%) | Forgetting previous (pp) |
| --- | --- | --- | --- | --- |
| CIFAR-100 | Joint | 93.48 | — | — |
| CIFAR-100 | Frozen shared predictor basis | 84.68 | 90.23 | 6.01 |
| CIFAR-100 | ER | 81.42 | 91.32 | 18.87 |
| ImageNet-R | Frozen shared predictor basis | 67.60 | 74.30 | 7.30 |
| ImageNet-R | Joint | 83.87 | — | — |

Joint is the offline full-class reference and only has Final. Frozen shared basis stores no historical images. ER uses the fixed total 2,000-image CIFAR-100 replay budget. Its final checkpoint was restored numerically on CPU against the executed snapshot; actual data membership, 50,000 arrivals and 2,000 resident training images were checked. A finite 100-class zero-input forward was performed without additional training or accuracy evaluation. The model and its verified backup remain on the server.

Other formal cells remain incomplete. Development scores are not entered here. Both forgetting definitions and raw source result hashes are retained in completion_receipts/ and PARTIAL_RESULTS.json. The first ER audit reached its final source-inventory step but encountered a virtual Torch module path; the fixed auditor reran all checks. Both audit attempts are retained in observations/formal_cifar_er_fixed_v1/. The CPU auditor does not by itself certify completion of all 12 cells or exact CUDA continuation.
