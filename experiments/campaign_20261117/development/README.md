# Development records

Only the four proposed methods are tuned. New candidates first run two or three stages; final candidates complete all ten development stages before a full-training-data formal refit. ER and Joint use fixed conventional settings. An ER-shaped internal CE driver can supply exact reusable body trajectories to the two analytic candidates; it does not define a tuned ER benchmark row.

`decisions/` records the evidence and intended changes before each candidate. `configs/` stores portable parameter copies; dataset and output paths are expanded privately for execution. Original run configurations, outputs, trajectory states and process receipts remain with their corresponding runs. Development outputs are never counted as formal results.
