# Partial formal results

4 of12 formal dataset/method cells completed. These are full-training refits evaluated on the selected test splits, with seed1993. Completed-method scores were disclosed at the user's request for reporting; subsequent tuning still uses only the fixed internal training holdout.

| Dataset | Method | Final (%) | AIA (%) | Forgetting previous (pp) |
| --- | --- | --- | --- | --- |
| CIFAR-100 | Joint | 93.48 | — | — |
| CIFAR-100 | Frozen shared predictor basis | 84.68 | 90.23 | 6.01 |
| ImageNet-R | Frozen shared predictor basis | 67.60 | 74.30 | 7.30 |
| ImageNet-R | Joint | 83.87 | — | — |

Joint is the offline full-class reference and only has Final. Frozen shared basis stores no historical images. Other formal cells are incomplete; development prefix scores are not entered here. Complete metric closure and source result hashes are retained in completion_receipts/ and PARTIAL_RESULTS.json.
