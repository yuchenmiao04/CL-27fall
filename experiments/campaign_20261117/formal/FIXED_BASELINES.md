# 待运行的固定 baseline 配置

本次新增三份正式配置：`configs/formal_cifar_er_fixed_v1.json`、`configs/formal_imagenetr_er_fixed_v1.json`、`configs/formal_imagenetr_joint_fixed_v1.json`。它们分别复制 `exps/pilot_figure2/baseline_er_cifar224.json`、`baseline_er_imagenetr.json` 和 `baseline_joint_imagenetr.json` 的事先固定训练设置，不进行参数搜索。均为完整正式流、seed 1993、20 epochs、原生 AdamW、骨干 LR 1e-5、CE 头 LR 1e-3、weight decay 1e-4、当前 batch 64；ER 每批最多回放 64 张，Joint 不使用回放池。CIFAR-100 ER 固定总容量 2,000 张；ImageNet-R ER 固定每类 20 张。Joint 一次访问全部训练类别并集，只报告 Final，AIA 和遗忘为不适用。

配置使用 `data/cifar100`、`data/imagenetr` 和 `runs/formal/...` 可移植路径，不含服务器地址或访问资料。初始设备分别为 CIFAR-100 ER 卡 1、ImageNet-R ER 卡 0、ImageNet-R Joint 卡 0；实际调度可以调整设备和展开路径，这不属于训练参数搜索。没有开发留出或样本截断开关，也没有恢复路径。文件存在不表示任务已启动或完成，结果以各正式运行记录为准。

ER 使用 `scripts/run_fixed_er_record.py` 原生正式入口，配置与执行报告均标为 `execution_role=fixed_baseline`。该入口在固定 ER 训练时附带记录精确 CE 轨迹；之后只有已选定 HT/PSB 的数据、设备、训练参数、初始／阶段状态和随机流等全部匹配时，才可通过已验证缓存复用相同骨干训练。若条件不匹配，候选方法仍需单独训练；不能为复用而改变固定 baseline 配置。内部候选 CE driver 不替代这两个正式 ER 对照。

## 正式 HT/PSB 的轨迹复用条件

当前 HT 与可塑共享预测基 PSB 的骨干适配均调用与 ER 相同的原生 CE＋回放训练函数。`transport_mode=direct/chained`、迁移估计参数以及共享基的秩、noise、迭代次数只作用于训练前的只读参考编码或训练后的统计量与解析头，不向这段 CE 训练反传。因此，在完整前置条件一致时，可复用固定正式 ER 的同一梯度状态迁移；无需为了获得相同骨干再次执行该段梯度计算。这个结论不适用于固定模块库方法，也不适用于未来改变骨干训练目标、回放选择或随机流的实现。

正式候选必须已经通过完整十阶段开发流程选定。由该候选配置创建新的正式配置时，设置 `phase=final`、`validation_only=false`，删除 `max_tasks`、样本截断选项及开发 `resume`，使用新的正式输出目录，从相同预训练模型初始化。保留选定的解析头参数以及开发时选定的梯度训练参数。正式运行使用官方完整训练数据，不继承开发模型、开发历史统计量或开发轨迹。

复用源只能是对应数据集的 `formal/ce_trajectories/{cifar,imagenetr}_fixed_er_v1`。启动前确认十个 `stage_000.pt` 至 `stage_009.pt` 均已原子写入完成；开发轨迹不能作为正式轨迹使用。逻辑设备也属于缓存身份：已记录的 CIFAR-100 卡 1、ImageNet-R 卡 0 轨迹不能直接改映射到另一张卡重放。设备调整若导致身份不匹配，应另行原生训练，不能跳过检查。

实际复用以每阶段 `CEStageCache` 的检查为准：数据清单、phase、阶段与类别计数、当前样本原始内容／ID／顺序及变换、回放内容及选择 RNG、梯度超参及有效 batch、训练前骨干和 CE 头权重、可训练标志、原生训练函数 SHA、全局和回放采样 RNG、设备与数值策略均须匹配；缓存训练后权重也须通过摘要检查。任一条件不符即停止复用。重放安装的是该阶段训练后的骨干／CE 头和 RNG；每个候选仍独立执行自身合法样本编码、历史矩更新、herding、解析头拟合及全部已见类别评测，缓存不提供历史图像或历史特征训练集。

使用 `scripts/run_ce_cohort.py --replay-only --configs <正式 HT 配置> <正式 PSB 配置> --trajectory-dir <正式 ER 轨迹> --report <新的正式报告路径>`。该入口支持 `phase=final`；此处不要再次传入 ER 配置，以免将固定 baseline 标为内部驱动。候选执行报告中的 `execution_role=candidate_method` 与正式 phase 并不冲突。主结果汇总器只排除配置中标为 `internal_shared_ce_driver` 的 ER；HT/PSB 只要满足完整正式协议、十阶段矩阵和指标闭合等检查，仍可纳入。存在同一方法的多个正式结果时，必须提供预先确定的显式路径选择，不能按正式测试成绩挑选。

## 共享执行时的资源成本口径

重放候选结果的 `execution.wall_seconds` 和 `gpu_peak_allocated_bytes` 只测量该重放进程，包含独立编码、统计估计、读出、评测及缓存加载／检查，不包含已经由 ER 执行的梯度计算。它们不能直接作为该方法独立完整训练的耗时或峰值显存，不能据此宣称方法训练成本下降。

资源报告分别保留：固定 ER 实际梯度计算时间；每个候选 `ce_training_reuse.stages` 记录的 `shared_native_ce_seconds` 与 `executor_wall_seconds`；候选实际重放总时间；固定 ER 梯度执行和候选重放各自测得的峰值显存。共享梯度时间在整个实验活动的实际支出中只计一次，同时明确每个候选依赖这段计算。若估算单方法独立运行时间，应说明替换缓存调用时间的计算方式及并发测量误差，并与直接测量区分。没有独立完整运行测量时，其真实峰值显存标为未测，不以 reader 峰值代替，也不将分阶段峰值简单相加。

逐阶段完整权重与 RNG 轨迹属于实验执行缓存，其磁盘容量单独记录，不并入学习器的算法历史统计量，也不从实验活动实际存储支出中隐去。方法自身的部署参数、训练参考骨干、CE 头、历史统计量和合法回放存储仍按各方法的 `resource_report()` 独立列出。固定 ER 的训练设置、容量和正式结果资格不因计算复用改变。
