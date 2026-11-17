# Resource evidence: ten completed formal cells

The two fixed-library formal fits are still running and are not filled with estimates. Counts below come from SHA-bound completed CPU checkpoints matched to native parameter-name/shape registries. M means10^6 parameters; MiB means2^20 bytes. Numerical readouts and moment arrays are charged separately from neural-network parameters.

| Dataset | Method | SGD-eligible NN (M) | Resident training NN (M) | Forward backbones | Auxiliary numeric state (MiB) | Checkpoint (MiB) | Actual process wall (min; scope below) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | Joint | 85.876 | 85.876 | 1 | 0.00 | 327.65 | 73.24 |
| CIFAR-100 | Frozen shared basis | 0.000 | 85.799 | 1 | 6.05 | 335.40 | 8.36 |
| CIFAR-100 | ER | 85.876 | 85.876 | 1 | 0.00 | 336.40 | 252.45 |
| CIFAR-100 | History transport | 85.876 | 171.674 | 1 | 15.30 | 686.45 | 27.81 |
| CIFAR-100 | Plastic shared basis | 85.876 | 171.674 | 1 | 21.16 | 694.25 | 27.99 |
| ImageNet-R | Frozen shared basis | 0.000 | 85.799 | 1 | 7.23 | 336.37 | 9.05 |
| ImageNet-R | Joint | 85.952 | 85.952 | 1 | 0.00 | 327.94 | 125.45 |
| ImageNet-R | ER | 85.952 | 85.952 | 1 | 0.00 | 328.40 | 158.76 |
| ImageNet-R | History transport | 85.952 | 171.751 | 1 | 17.06 | 681.33 | 21.23 |
| ImageNet-R | Plastic shared basis | 85.952 | 171.751 | 1 | 24.10 | 690.26 | 21.55 |

SGD-eligible counts describe the union of the native optimizer parameter families. Frozen shared basis has0 SGD parameters but learns its latent covariance factor and posterior readout through numerical fitting. HT/PSB eligibility includes the auxiliary CE head used during the shared backbone training; that CE head is not their deployed analytic head. The NN-only counts do not include numeric analytic coefficients.

HT/PSB process wall in this table is the actual reader workflow, not independent full-method training time. Their matching CIFAR-100 and ImageNet-R ER donors respectively performed14,653.037 and9,029.934 seconds of shared native CE work, counted once per dataset in campaign totals. The CE-trajectory reader also encodes features, fits/validates moments and readouts, evaluates, and saves states. Do not add each copied donor-work record again to physical campaign totals or infer a controlled speedup from these concurrent wall times.

Recorded GPU peaks are per-process PyTorch allocated peaks. Reader peaks exclude the donor gradient-training peak and do not describe full-method peak memory. The raw evidence retains the values and scope rather than silently replacing them with baseline peaks. CUDA context/reserved memory and other processes are outside this counter.

Transport numeric-state bytes already include its analytic head. Plastic shared basis adds its basis arrays and reconstructed posterior to transport state; the analytic coefficients must not be added again. Frozen shared-basis posterior weights/intercept are materialized resident values reconstructed on restore and absent from its serialized predictor arrays. Checkpoint bytes and runner numeric payload are different measures.

Replay quantity is an image budget: CIFAR-100 final2,000 images; ImageNet-R final4,000 images. CIFAR raw pixels plus16-byte identity accounting total6,176,000 bytes. The reported ImageNet-R3,613,566,457 bytes are decoded RGB equivalent plus identity accounting, not compressed JPEG disk bytes, path-string memory, or simultaneously resident decoded images. R checkpoints store references and require the original source files for replay.

The fixed-library full head calls anchor plus main at inference. Its eventual rows must include both backbones, libraries, W0/V, historical matrices, resident-anchor cache and two persistent AdamW states. Phase-specific requires_grad flags alone are not an optimization-eligibility count.

Full definitions, per-array dtype/shape/numel, resource overlap notes, exact unrounded values and source hashes are in PARTIAL_RESOURCE_EVIDENCE.json. Initial/model/data setup is outside runner wall; stage wall excludes checkpoint writes, while fresh run wall includes them. Resumed development process wall excludes its inherited prefix.
