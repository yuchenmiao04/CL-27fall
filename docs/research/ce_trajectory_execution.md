# CE 训练轨迹复用：执行协议

本机制属于执行层，只减少重复计算，不改变 ER、`history_transport`、`plastic_shared_basis` 的学习目标或数据权限。三个方法当前的骨干和 CE 辅助头均由相同 `adapt_backbone` 训练，历史统计和共享先验不反传到该训练；当训练参数一致时，可共享这一完整状态转移。

按当前调参规则，只有四个候选方法调参，ER/Joint baseline 使用预先固定配置。cohort 中名为 ER 的对象只是实现共享 CE 的内部执行器，配置和报告均标记 `execution_role=internal_shared_ce_driver`，不能将它当作调过参数的 ER baseline 或填入正式 ER 行。baseline 对照需另行使用固定配置的原生运行。

## 执行形态

`scripts/run_ce_cohort.py` 先通过原生 `cl27.runner.run` 独立运行 ER，逐阶段记录 CE 适配后的骨干、CE 头、专用回放采样 RNG 和全部全局 RNG；随后分别独立运行另外的方法，仅在 CE 调用处复用完全匹配的转移。每个方法仍执行自己的样本身份检查、确定性编码、历史矩更新、回放选样、部署头和测试。

不在同一任务循环中交错三个 learner。每次原生 `run` 均重新设置种子、构建 DataManager 和模型，并且每阶段执行一次自身的评测 DataLoader。因此既不叠加多次 timm 初始化的随机数消耗，也不把三次评测 iterator 的 base-seed 消耗带入一条训练流。缓存读者将 CE 后的 RNG 精确恢复，再继续自己的原生后处理和评测。

独立 `main.py` 入口及无执行器的 `runner.run` 保持原行为。最小接点是 `execute_adaptation`；默认直接调用原生适配函数。checkpoint 格式与已有随机流策略不变，可从复用运行的任务边界 checkpoint 恢复为独立原生运行。

## 复用条件

每次调用绑定：数据 manifest 指纹、阶段和已见类计数、训练前骨干与扩展后 CE 头的精确状态、骨干结构和可训练标记、当前原始样本的有序 ID/标签/内容摘要与变换、回放顺序/内容/配额/选样 RNG、独立的回放 minibatch RNG、全部 Python/NumPy/Torch/CUDA 随机流、有效超参数、实际逻辑 batch 大小序列，以及函数源码和数值执行策略。

其中 batch 序列包含尾批和实际可抽取回放数量；原始样本内容、变换和随机流共同约束随机增强及抽样。不会持久保存增强图像。数值策略绑定 Torch/NumPy/torchvision、逻辑设备、CPU 线程、确定性与 TF32 开关；初版保守要求同一个逻辑 CUDA 设备，不做跨设备 RNG 映射或近似权重复用。CLI 还会在记录训练前检查配置共同项，避免明显不匹配时先浪费一次完整训练。

任一条件不符、缓存阶段缺失或缓存权重身份不符均显式报错，不静默沿用，也不静默改为重新训练。文件通过临时文件原子替换发布。仅加载本地可信执行器生成的 Torch 文件，不接受不可信来源的 pickle。

## 缓存与内存边界

缓存只保存每阶段训练后的参数状态和随机流，不保存特征。不跨模型版本复用表示；当前与允许的回放图像仍由各方法自己的特征管线读取。全历史信息仍仅按各算法原有方式进入统计量或回放。

每个 ViT-B/16 骨干快照约 0.34 GB，十阶段单个轨迹约 3.4 GB，另加很小的 CE 头和随机状态。它是实验执行资产，不是算法部署状态；需记录实际磁盘成本。未选中的调参轨迹可按实验归档策略处理，不能把额外执行资产伪装成算法零存储。

## 使用

正式固定 ER baseline 可以在保留原生训练流程的同时录制轨迹：

```sh
python scripts/run_fixed_er_record.py \
  --config /path/fixed_er_formal.json \
  --trajectory-dir /path/fixed_er_ce_trajectory
```

该薄入口只接受正式完整流、seed 1993 和两个选定 PILOT 数据集，拒绝开发、子集和内部 driver 配置。它不改变所提供的固定训练参数，也不搜索；配置及执行报告均标记 `execution_role=fixed_baseline`。实际适配与缓存检查仍委托原生 runner 和 `CEStageCache(record)`，代理仅修正报告角色。之后只有已选 HT/PSB 与该固定 ER 训练前提完全相同时才可复用；不匹配仍拒绝，不能为复用而改变 baseline 固定配置。

为三份配置分别指定新输出目录，并保持数据、训练和设备设置一致：

```sh
python scripts/run_ce_cohort.py \
  --configs /path/er.json /path/history_transport.json /path/plastic_shared_basis.json \
  --trajectory-dir /path/ce_trajectory \
  --report /path/cohort_report.json
```

共享预测基的噪声、秩、EM 轮数和历史迁移参数不参与 CE 梯度，允许在已完成轨迹上单独调整，仍须完整经过各个原生阶段并受实时身份检查：

```sh
python scripts/run_ce_cohort.py \
  --configs /path/new_analytic_head.json \
  --trajectory-dir /path/ce_trajectory \
  --report /path/new_head_report.json --replay-only
```

改变骨干/CE 学习率、训练轮数、训练 batch、回放比例、数据划分或数值策略后不能复用旧训练。`replay-only` 需要所需全部阶段已经存在；不能从只有最终任务边界的旧 checkpoint 推导缺失的早期训练轨迹。

### 从初筛阶段继续完整开发流

可以先使用三份 `phase=development, validation_only=true, max_tasks=3` 配置初筛；最终候选继续完整十阶段开发验证时，分别为每个方法配置其自身的阶段 2 checkpoint、一个新的 `output_dir`、`resume` 和 `max_tasks=10`，其余训练与方法设置保持不变，再以同一个 `--trajectory-dir` 调用普通 cohort 模式。内部 CE driver 从其自身 checkpoint 继续记录阶段 3–9；两个候选分别从自身统计状态继续读取新记录。已有阶段 0–2 文件不会重写；任何试图从阶段 0 重记同一路径的调用仍报错。

CLI 的静态预检要求本次三份扩展配置的数据、训练、设备及目标阶段范围一致，但不要求扩展的 `max_tasks` 等于旧初筛范围。实时缓存条件绑定实际阶段和训练前状态而非流的停止上限，因而不放松实际 CE 状态匹配。只有 `output_dir`、`resume`、`max_tasks` 等原生允许的恢复控制发生改变时，任务边界 checkpoint 可继续。若改变候选学习规则、训练超参或解析头配置，不能用不匹配的旧算法 checkpoint 继续，须按其原生校验重新运行所需开发流。

## 验证与成本解释

`tests/test_ce_trajectory.py` 在随机裁剪/翻转训练及真实评测 DataLoader 上逐阶段比较独立原生运行与记录/复用运行。direct 和 chained 两种历史模式均比较骨干、CE 头、回放、完整核心状态/矩、部署结果和评测后的完整 RNG；同时模拟真实 timm 初始化的额外 RNG 消耗。另验证输入/超参/划分/原始像素改变、损坏缓存、缺失阶段、CLI 和独立断点恢复。

每个结果的 `execution.ce_training_reuse` 标明记录或复用、共享 CE 耗时、缓存身份与执行器耗时。`execution.wall_seconds` 是该次实际执行时间：读者没有实际执行共享 CE，不能直接把其较短时间称为独立算法训练速度。cohort 总实际时间单独报告；若报告独立等价分项成本，需明确共享训练只实际计算一次，不重复计为三次实测 GPU 工作。

该验证不证明任意自定义骨干、隐藏的外部随机源或修改后的训练函数都可复用。当前机制只应用于经过一致性检查的三个原生 CE 体方法；真实 CUDA 上还应完成短程核对后再用于正式运行。

## 本次双卡物理落卡验证

2026-11-17 的开发调度在两张同型号 RTX5090 上完成了一次完整十阶段验证。原运行与验证运行均使用逻辑 `cuda:1`、两个可见设备和相同随机流索引；验证运行设置 `CUDA_VISIBLE_DEVICES=1,0`，使逻辑卡1落到另一张物理卡。原生训练缓存条件、模型、数据、解析头参数和数值开关保持不变，没有改写缓存中的 CUDA RNG 状态。

验证覆盖完整 CIFAR-100 chained HT 开发流。最终 checkpoint 的33,081个状态节点逐位一致，包含历史矩、读出、回放原图及顺序、模型和全部随机状态；唯一排除项是十个阶段各自的 `wall_seconds`。完整阶段结果摘要、样本数与指标一致，十个原生 CE request fingerprints 相同，十个缓存文件的 SHA256和大小均未变化。序列化文件自身的 SHA可以不同；比较工具另给出排除指定时间字段后的状态 SHA。原始逐样本阶段预测向量未保存，因此此证据使用已保存的完整阶段结果摘要。

完整证据见 [换卡验证记录](../../experiments/campaign_20261117/correctness/cifar_chained_physical_swap_gate_v1.json)，决定见 D0044与D0047。`scripts/compare_checkpoint_state.py` 比较整个可信 checkpoint，仅允许忽略 root `stages[i].wall_seconds`；13项CPU测试覆盖浮点签名零、tensor单bit、对象数组、原图dataclass、随机状态和未知元数据。

本轮仅将这项实测结果用于明确标记的开发 HT/PSB解析读出调度，所有后续阶段仍保留原生缓存检查。证据针对上述硬件与完整HT工作负载；其他方法或硬件的数值等价性需另有证据。正式 CIFAR读出继续使用原物理卡1。记录物理落卡、实际进程耗时和共享计算成本，调度收益与算法独立训练成本分开报告。
