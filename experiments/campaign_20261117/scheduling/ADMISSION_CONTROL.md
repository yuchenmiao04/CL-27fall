# Selected formal job admission control

Five previously selected formal configurations are queued: CIFAR-100 HT and PSB, ImageNet-R HT and PSB, and ImageNet-R fixed modules. The CIFAR-100 module selection remains open until both full ten-stage development candidates finish; it is not part of this queue.

The tool does not select parameters or alter training configurations. HT/PSB readers wait for the corresponding fixed ER to finish all ten stages, its original process to terminate successfully, and all ten native CE trajectory files to be present. The module job waits for CIFAR-100 ER to release its large training slot. The native runtime then checks the exact reuse identity; a mismatch is an error rather than a request to silently retrain.

Configuration, launcher and receipt hashes are pinned. Actual PID, process start ticks and command are checked. Before each launch the tool refreshes NVML observations and reserves 24 GiB for each live ER, 6,440 MiB per module job, 4,096 MiB per analytic reader, plus a 1,024 MiB margin. An unknown GPU process blocks admission. These are admission budgets, not claims of independently measured method peak memory. The tool never kills or restarts training and refuses duplicate output/receipt paths.

Validation on 2026-11-17: 76 CPU fake-process/NVML cases passed. A real server read-only plan validated all five jobs and returned `waiting_dependency` for each. The detached operational process was then verified against its receipt, start ticks, command, source SHA and initial event; no failure file existed at the 10:21 UTC check. This is evidence of a live dispatcher, not evidence that any queued training has completed.

The read-only observer package `nvidia-ml-py==13.615.71` was installed with `--no-deps` from the official PyPI index. Torch 2.7.1+cu128, NumPy 1.26.4 and timm 0.6.12 remained unchanged. NVML does not create a CUDA context. The compatibility test includes missing unavailable-memory constants and `None`/uint64 sentinel values.

Dispatcher source SHA256: `e091e0cabb2d01a10ec0414cba961d5ba3a3a811c2df674e8ffc0d35cdaa95f1`.
Queue source SHA256: `494b68fc6916b4c2149e3e678042b643a29b24d09d9acc27985b68b3786d0d18`.

The initial evidence export preserves raw source hashes and separately records hashes after operational path replacement. Original bytes are retained privately. Shared CE training cost must be reported separately from each reader's executor time and memory; a reader-only measurement is not an independent full-method training cost.

## Reviewed continuation after the first controller stopped

At 11:25 UTC, v1 stopped with `Original process command differs`. The old failure omitted the offending PID and observed command, so the exact cause cannot be established from it. At the subsequent direct check, the original CF reader PIDs had terminated with complete ten-stage results; all four still-running training processes matched their original PID/start ticks/commands. No training was restarted. The two R readers had never been launched.

The observer now treats a known same-start-ticks nonterminal process with an empty cmdline as still live and reserved until actual termination. It does not release VRAM or pass a dependency on that observation. A nonempty different command still fails, now with identifying evidence. This guards a plausible process-shutdown window without asserting that it was proven to cause the old failure. Eighty CPU tests pass.

A new queue/state contains only the two unstarted R readers, with their original config/spec hashes. The real plan returned `waiting_dependency` for both, and the new operational PID was verified live with no failure file. GPU1 may remain blocked by the independently running R module; the new queue only dispatches on GPU0. Old failure/log/receipt evidence is preserved under `recovery_v2/`.

V2 source SHA256: `fcda7ae5ef4e9a4588431ede2e0f4b6f5b89c8d023f36773331133c0034f9749`.
V2 queue SHA256: `81e6a90cf5b38d15e8b1b96a187ca03f53bda95750e8ccb3dfd03831db6ef856`.
