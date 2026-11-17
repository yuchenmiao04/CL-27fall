# 开发集配置选择

已锁定 **8/8 个方法×数据集单元**。下表全部为完整十阶段开发流程的训练内留出集成绩，不是正式测试结果。共同使用 seed 1993、固定训练内留出（validation_fraction=0.1、split_seed=0）；选择以未舍入 Final 为主，具体晋级及并列规则见事先记录的决策，AIA、遗忘与开销同时观察。CIFAR-100 模块候选沿用 D0045/D0053：Final 精确相同时再考虑成本。表内准确率单位为 %，显示四位小数。

| 数据集 | 方法 | 开发 Final | 开发 AIA | 关键参数 | 决策 | 正式配置 |
| --- | --- | ---: | ---: | --- | --- | --- |
| CIFAR-100 | 冻结共享预测基 | 84.4400 | 89.7692 | K=64；ε=1e-3；s²=1e-3；EM≤100；tol=1e-8 | [D0023](decisions/D0023.json) | [配置](../formal/configs/formal_cifar_shared_selected_v7.json) |
| ImageNet-R | 冻结共享预测基 | 67.5465 | 74.3666 | K=64；ε=1e-3；s²=1e-3；EM≤100；tol=1e-8 | [D0023](decisions/D0023.json) | [配置](../formal/configs/formal_imagenetr_shared_selected_v7.json) |
| CIFAR-100 | 历史统计迁移 | 90.8000 | 94.8100 | chained；map_ridge=0.01；head_ridge=0.01；20 epochs | [D0043](decisions/D0043.json) | [配置](../formal/configs/formal_cifar_transport_selected_v2.json) |
| ImageNet-R | 历史统计迁移 | 80.5712 | 85.9185 | chained；map_ridge=0.1；head_ridge=0.01；20 epochs | [D0048](decisions/D0048.json) | [配置](../formal/configs/formal_imagenetr_transport_selected_v3.json) |
| CIFAR-100 | 可塑共享预测基 | 90.7800 | 94.8297 | chained；map_ridge=0.01；K=32；ε=1e-3；s²=0.001；EM≤20 | [D0052](decisions/D0052.json) | [配置](../formal/configs/formal_cifar_plastic_shared_selected_v5.json) |
| ImageNet-R | 可塑共享预测基 | 80.5279 | 85.9341 | chained；map_ridge=0.1；K=32；ε=1e-3；s²=0.0001；EM≤20 | [D0051](decisions/D0051.json) | [配置](../formal/configs/formal_imagenetr_plastic_shared_selected_v5.json) |
| CIFAR-100 | 固定模块库 | 89.8400 | 94.3754 | 层7/9，各4模块、rank8；5 epochs；M/B；episode replay≤64 | [D0055](decisions/D0055.json) | [配置](../formal/configs/formal_cifar_current_selected_v5.json) |
| ImageNet-R | 固定模块库 | 79.9221 | 84.7538 | 层7/9，各4模块、rank8；5 epochs；M/M/B/M；episode replay≤64 | [D0054](decisions/D0054.json) | [配置](../formal/configs/formal_imagenetr_current_selected_v5.json) |

CIFAR-100 固定模块库按 D0045/D0053 的事先规则，在两个完整十阶段开发结果之间选择：M/B 的 Final 为89.84%（4492/5000），M/M/B/M 为89.58%（4479/5000）。两者的骨干更新均为3550次，模块更新分别为3550/10650次。不同并发条件下的 wall time 不解释为受控加速比；这也不是模块更新作用的独立消融。所选检查点通过45000拟合/5000留出、2000合法resident、两个AdamW与原生CPU恢复核验。正式运行从预训练重新开始，不延续该开发状态。

冻结共享预测基的完整开发指标来源：[CIFAR-100](observations/runs/cifar_shared_rank64_full_v7/results.json)、[ImageNet-R](observations/runs/imagenetr_shared_rank64_full_v7/results.json)。其余指标取自对应决策中的已选完整候选。K、ε、s²分别对应 basis_rank、basis_epsilon、basis_noise_variance；M/B表示模块/骨干更新。未列参数以链接配置为准。

ER 与 Joint 采用事先固定的常规配置，未参与调参；Joint 正式实验只报告 Final。正式重训从同一预训练起点使用官方完整训练集，重新初始化学习状态，不继承开发 checkpoint、统计量或留出缓存。HT/PSB 只可复用匹配的**正式**固定 ER 训练轨迹，且须通过原生逐阶段守卫；开发数据与正式训练、评测记录分开。
