# 实验设计

## 1. 对齐对象

目标是 [PILOT 技术报告 v1](https://arxiv.org/pdf/2309.07117v1)和[上游 README 的 Reproduced Results](https://github.com/LAMDA-CL/LAMDA-PILOT/tree/7a6e904c5bc5cb7a4e1823b3434020be27469b63)。固定源码为 `7a6e904c5bc5cb7a4e1823b3434020be27469b63`；它是本次固定的快照，不声称是历史论文当时的提交。

协议一致是比较的必要条件，不保证复测值等于文献值。ImageNet-R 准备后划分的身份、具体部署头、权重和训练超参数仍需与引用表核对。2026-11-17 主实验的实际结果与审核状态见[主实验目录](../experiments/campaign_20261117/README.md)；本项目测量与 PILOT 文献报告分别标注。

## 2. 数据、任务及存储

| 字段 | CIFAR-100 | ImageNet-R |
| --- | --- | --- |
| PILOT 名称 | `cifar224` | `imagenetr` |
| 类别与任务 | 100 类，10 阶段，每次 10 类 | 200 类，10 阶段，每次 20 类 |
| 首阶段 | 10 类，无额外大基类阶段 | 20 类，无额外大基类阶段 |
| 类别顺序 | NumPy MT19937，`np.random.seed(1993); np.random.permutation(C)` | 同左 |
| 正式训练 | 官方 50000 张训练图像全部使用 | PILOT 准备后的全部 train 图像 |
| 正式测试 | 官方 10000 张 test 图像 | 同一准备版本的 test 图像 |
| 回放规则 | `fixed_memory=false, memory_size=2000` | `fixed_memory=true, memory_per_class=20` |
| 阶段配额 | 每类 `floor(2000/C_t)`，总量不超过 2000 | 每类 20，最终上限 4000 |
| 类别信息 | 训练知道阶段边界；测试不给任务 ID | 同左 |

原始旧图像预算与 batch 内回放量、epoch 数、全历史矩存储是不同维度。`shared_basis` 实际图像回放为 0；其余三条路线使用表中规则。当前阶段新类训练集不受历史图像配额限制。选样采用 PILOT 的归一化特征 herding，旧类按原选样顺序保留配额前缀。

ImageNet-R 需采用 PILOT 链接的准备后划分，不能对原始数据任意重新划分。2026-11-17 从固定 README 的 OneDrive 分享链接取得 `imagenet-r.zip`，SHA256 为 `c2720edb18da7173b6cb6d9e9a65e18fb6f949f65d8cd10ac99df9003d8b055a`；train 24,000 张、test 6,000 张，各 200 类。压缩包保留原始 train/test 目录，逐图核验结果见 `experiments/campaign_20261117/imagenetr_source.json`。

这个官方准备包本身包含 23 对跨 train/test 的字节相同图像，其中 3 对同类别、20 对不同类别。为保留选定 PILOT 协议，不去重或重新划分；正式结果需披露此源数据缺陷。默认策略仍拒绝重叠；六个主实验的 ImageNet-R 配置显式使用 `prepared_split_overlap_policy=preserve_verified_pilot`，只有实际读取的数据树中完整相对路径与文件 SHA 清单摘要、样本数量及已知重叠全部匹配 `cl27/pilot_imagenetr.py` 的核验值，才允许该例外。用户填写参考压缩包 SHA 不能绕过树身份检查。manifest 记录实际树摘要、策略、重复量与未去重事实；开发模式只读取并核验 train，跨划分观测量为 null。这证明使用了本次取得的官方链接准备包，不保证重建了每篇历史论文作者当时的输入。

## 3. 骨干、权重与预处理

- ViT-B/16，224×224 输入，最终 CLS 为 768 维；通过原生 `utils.inc_net.get_backbone` 加载，不做 torchvision 名称转换。
- `timm==0.6.12` 的 `vit_base_patch16_224` 选中 AugReg ImageNet-21K 预训练再 ImageNet-1K 微调权重，不能仅标成含义不明的 IN1K 或 IN21K。
- 精确源文件名为 `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`，对应 [timm 0.6.12 模型定义](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py)。加载时记录公共 URL、权重文件 SHA-256、初始状态 SHA-256。
- 训练原图变换直接复用 PILOT：`RandomResizedCrop(224, scale=(0.05,1.0))`、随机水平翻转、`ToTensor`。
- 确定性统计及评测变换：bicubic `Resize(256)`、`CenterCrop(224)`、`ToTensor`。
- 选定 PILOT 变换不加像素 mean/std normalization。额外添加标准 ImageNet normalization 会改变协议。
- 新方法解析特征默认逐样本 L2 normalization；它属于方法定义，与像素 normalization 不同。配置键为 `feature_normalization`，旧别名 `normalization` 冲突时拒绝。current 路线固定要求 L2。

本协议的结构化 runner 对所有真实数据运行强制 `timm==0.6.12`；四个新方法还检查骨干名称并绑定实际权重文件摘要。其他未纳入本轮六方法主实验的 PILOT 原生 baseline 加载器保留原样，其完整权重来源不在本轮核验范围。这里接入的 ER、Joint 和四个候选方法统一使用绑定 NPZ 摘要的骨干构造器。合成测试使用独立小 token ViT，明确标成 `smoke`。

## 4. 训练及开发访问规则

正式评测：每阶段可多次训练该阶段全部当前图像及允许的回放；不访问未来类别训练图像，不用 test 标签选超参数或停止。任务边界允许已知。Epoch 数不等于统计累加次数；新方法均在任务结束后只为每个新到样本提交一次持久矩。

开发模式：每个原始训练类别留出 10%，由 Python `random.Random(split_seed=0)` 固定；更换类别顺序不改变留出身份。开发 loader 只读取 train，不打开指定 test。配置选定后，正式阶段用完整训练集重新拟合。真实预训练验证脚本还缩减每类样本，标为 `validation_only=true`，最多两个任务；独立合成 smoke 为三个任务。

各方法的初始化配置不代表 PILOT 所有方法统一的训练轮数。2026-11-17 开发选择已锁定：统计迁移与可塑共享基每阶段20 epochs；固定模块库每阶段5 epochs，CIFAR-100 用 M/B 两次尝试，ImageNet-R 用 M/M/B/M 四次尝试；ER 与 Joint 保留事先固定的20 epochs，未参与调参。冻结共享预测基只确定性编码一次，再进行 EM。完整超参数、选择依据与实际正式配置见 [开发选择记录](../experiments/campaign_20261117/development/SELECTIONS.md)；优化器与目标见各方法推导。

模块方法的5 epochs是为开发成本设定并沿用的初始预算，不是通过5对20 epochs的有效比较选出；本轮未系统检验其训练长度。不同方法的epoch、更新次数与单步计算均不同。当前统一数据和评测协议、分别报告实际成本，不解释为等计算预算或各方法最优收敛性能的比较。后续训练长度选择须继续使用训练内留出集。

新增 `plastic_shared_basis` 完全复用 `history_transport` 的 CE 骨干训练、数据 / 回放与矩迁移；随后在当前坐标的完整重建矩上替换拟合共享先验。实现中的 `basis_*` 默认值与冻结版相同，但本轮已选配置不同，见开发选择记录；每任务从 `epsilon I` 重新拟合，`basis_reset_each_task=false` 当前拒绝。共享基目标不反传到骨干。本轮不把它描述成端到端联合学习。

比较边界：冻结共享基实际旧图像量为 0，可塑版本使用上表预算，因此二者是完整方法对照，差异不能只归因于解冻 ViT。第四方法与原统计迁移使用相同骨干训练轨迹，但各向同性头的实现默认正则化也不同。实现默认的共享基退化值为 `lambda=noise_variance/epsilon=1000`；本轮已选冻结配置的退化值为1，可塑共享基为1（CIFAR-100）/0.1（ImageNet-R），统计迁移的 `head_ridge=0.01`。这些只是 D=0 时的尺度对照，非零共享基对应各向异性先验。后者在第四方法中只控制内部辅助头，不参与部署。测试中使用相同 lambda 验证退化等价；正式性能比较需明确先验 / 正则化配置，不能把所有差异归因于低秩共享。

真实 ViT 的极小样本检查为第四方法预先使用 `basis_noise_variance=0.01`，其余正式默认不动。原因是从各向同性重置时，若 `noise_variance>=max(n_c)/C`，EM 无法产生非零共享方向；每类三张、C=10/20、默认噪声 1 恰满足该充分条件。较小噪声仅用于覆盖非零分支，条件不成立也不保证非零基或更高准确率。该检查不属于统一性能表，不能直接比较其中的小样本准确率。

已有方法保留原生目标、学习率、优化器及轮数。派生配置不会把所有算法强制改成同一种训练预算。比较应单独报告训练成本、总参数和额外历史状态，不能仅用图像配额代表总资源。

## 5. Baseline 配置与差异

### 5.1 本项目新增的 ER 与 Joint

当前主实验包含四个候选方法、ER 和 Joint，每个数据集六行，共12组；已完成的正式训练、配置及审核见[结果记录](../experiments/campaign_20261117/formal/PARTIAL_RESULTS.md)。其他 PILOT 方法的原生代码及派生配置继续保留，公开结果可另列文献参考，不冒充本地复测值。

ER：`models/er.py`，单个可塑预训练 ViT-B/16＋普通 raw-CLS 线性 CE 头。每阶段扩大到全部已见类别，保留旧权重 / bias 行；使用当前新类与允许的回放训练，损失和测试都不按任务编号屏蔽输出。没有 anchor、模块、解析历史矩或统计迁移。回放 CIFAR 固定总量2000，ImageNet-R 固定20/类，herding 与逐类缩减规则沿用共同实现。直接部署训练 CE 头。

其训练完全复用可塑统计路线的 `adapt_backbone` 与 `expand_head`：20 epochs / 阶段，AdamW，骨干 LR1e−5、头 LR1e−3、weight decay1e−4，当前 batch64＋最多64张回放，逐阶段 cosine 并重置优化器。测试每阶段全部已见类，报告 Final、AIA 和两种遗忘。这是本项目在选定多 epoch Class-IL 协议下的 ER 适配，不是原 Tiny Episodic Memories 单遍 / 任务编号评测的复现，也不等同 PILOT FineTune。

Joint：`models/joint.py`，从相同预训练起点重新创建一个模型，一次获得 CIFAR-100 全部100类或 ImageNet-R 全部200类的训练并集，初始化完整 CE 头，训练一次。默认同样20 epochs、AdamW、两组 LR及weight decay；一个完整 cosine 周期，没有逐任务头扩展、回放、anchor 或统计矩。训练结束后才请求评估视图并执行一次最终评估。报告 Final；AIA、遗忘均为 null，不能把 Final 当 AIA、把遗忘写成0，也不产生伪十阶段曲线。

Joint 的历史数据权限是 `offline/all training data`，不受2000或20/类限制。配置中的两个 memory 字段设0表示没有独立回放池，不表示零历史数据或 exemplar-free CIL。它是离线联合训练参考，不是已证明的准确率上界；相同 epochs 也不是相同 FLOPs。主表须将其数据权限和不适用指标标清。

正式配置：`baseline_er_{cifar224,imagenetr}.json` 与 `baseline_joint_{cifar224,imagenetr}.json`。Joint 复用 PILOT 类别顺序 / 标签及划分，但十阶段仅作为类别协议元数据，不触发十次拟合。开发检查可显式设 `joint_max_classes=20` 等完整阶段前缀；正式配置禁止该字段。Joint 一律拒绝 `max_tasks`，其 checkpoint 只支持完整拟合后恢复已有结果，不支持中途 epoch 继续。

为使 ER 与统计迁移路线公平匹配，确定性提特征 DataLoader 使用独立 seed0 的 CPU generator。即便 `shuffle=false`，原 DataLoader 迭代也会消耗全局训练 RNG；多一次 anchor 编码会改变头初始化或数据顺序。该修复隔离了只读编码与训练随机流，没有修改训练数据、类别顺序、损失、回放配额、epoch 或优化器。它改变了早期提交的偶然随机轨迹，所以旧训练验证记录按历史提交解释；当前版本另行核对匹配轨迹。

新的公共 runner checkpoint 显式绑定 `random_stream_policy=canonical_loader_isolated_v1`。缺少或不匹配此标记的旧 runner 断点拒绝在新随机流策略下继续，须使用其原代码版本或重新开始；不是对旧权重文件的自动迁移。各数值核心的独立 state 接口及既有格式保留。

### 5.2 PILOT 原生 baseline 派生配置

除新增 ER / Joint 外，其余 `exps/pilot_figure2/baseline_*.json` 来源于未修改的 `exps/*.json`，统一数据名、十阶段、seed、路径、评测及存储规则，记录 `cl27_source_config`。现提供 FineTune、SimpleCIL、RanPAC、EASE、SLCA、iCaRL、Coil、DER、MEMO、FOSTER 的两个数据集配置，未在本轮完整复测。

1. 源 `exps/foster.json` 设 CIFAR 每类 20 张；历史 README 报告固定总量 2000。本次派生配置按选定历史规则改为后者，原文件保留。
2. 源 EASE CIFAR 配置每次增加 5 类，派生配置改为 10 类，其余方法超参数保留。EASE 也不是历史 v1 报告中的原始行。
3. 提供两种读出时，保存 CNN 与 NME 各自的原始矩阵，禁止测试后逐阶段选较好者。派生配置预先指定 iCaRL 用 NME，其余用 CNN；与具体文献表比较前仍须核对作者所报告的头。指定头不可用时拒绝。
4. DER 是动态扩展表示，不是 Dark Experience Replay / DER++。本轮未新增 DER++，未将旧协议实验成绩搬入此协议。

## 6. 指标与资源

记 `R[t,j]` 为学完任务 t 后对任务 j 的测试准确率，t≥j。每个测试样本都在全部已见类别中选择。`a[t]` 为该阶段全部已见测试样本的样本加权准确率。

```
Final = a[T-1]
AIA = (1/T) sum_t a[t]
F_previous = (1/(T-1)) sum_{j=0}^{T-2} (max_{t=j,...,T-2} R[t,j] - R[T-1,j])
F_including_final = (1/(T-1)) sum_{j=0}^{T-2} (max_{t=j,...,T-1} R[t,j] - R[T-1,j])
```

前者允许负遗忘，后者非负；两者都保存，跨论文不可混用。T=1 时遗忘为 null。JSON 准确率为 0–1，表格乘 100 后为百分数，遗忘乘 100 后为百分点。AIA 不是矩阵所有元素的简单平均；ImageNet-R 每类样本数不同，尤其需保留样本权重。

输出包含阶段准确率矩阵、各单元样本数、CNN/NME 规则、阶段耗时及新方法诊断。Checkpoint 包含模型、历史矩、受限回放、阶段计数与随机状态。辅助矩、冻结 anchor、训练 CE 头和部署模型分别计数；numeric payload 不是总 RSS 或峰值显存。

## 7. 本轮边界

本轮包括数值与生命周期检查、真实 ViT 小规模验证、四个候选的训练内留出集调参，以及选定配置的完整训练集正式重训。2026-11-17 原定12组主实验均已完成并核验，只使用 seed1993。其他 PILOT 方法的公开表仍是文献参考，本轮未复现整张表。单种子波动、模块复用机制与训练长度充分性不能由当前表格单独确认。用户拟追加的模块20-epoch变体和TUNA处于协议讨论阶段，尚未启动；不自动增加多种子或全套消融。
