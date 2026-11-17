# Fixed-library execution validation

The prefix optimization changes execution only and remains opt-in. Episode-local tokens before the first library are reused only during the modules/memory_efficient phase. Backbone updates retain the original path. Existing live training source snapshots were not changed.

Local full suite: 260 tests passed in16.89s using .venv-pilot.

Both CUDA probes used the pinned pretrained ViT-B/16, libraries at7/9 with4 modules/rank8, identical augmented synthetic RGB views, two successive updates (transfer and anchored episodes), FP32 with TF32/autocast disabled. Fixed loss/gradient/parameter tolerances, RNG consumption and counters were checked. This is numerical execution verification, not a dataset accuracy experiment.

| CUDA probe | Feature microbatch | Correctness | Max gradient/parameter difference | Peak reserved memory | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| current_prefix_cuda_shared_v1.json | 2 | passed | 0/0 | 1.557 GiB | 16.22 |
| current_prefix_cuda_prod_shared_v3.json | 16 | passed | 0/0 | 2.207 GiB | 9.56 |

A Joint process shared GPU0 during both probes. Observed timing ratios are descriptive; no isolated speed claim or production enable recommendation follows from these measurements. Repeat the production microbatch probe in an isolated GPU slot before enabling it in subsequent module runs. Report JSONs retain exact numerical outputs; operational path prefixes are exported with original hashes.

## Isolated production-microbatch gate

After Joint terminated and GPU0 was verified empty, the microbatch16 probe ran two correctness updates, two warmup pairs and five timing pairs on that GPU. Correctness, resource and speed gates passed. The observed median module-fragment speedup was 1.098923x; this is not the speedup of the entire training run. D0025 enables the flag for subsequent fixed-library training. See current_prefix_cuda_prod_isolated_v4.json.
