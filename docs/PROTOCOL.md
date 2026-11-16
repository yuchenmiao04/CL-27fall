# 实验设计

## 1. 对齐对象

目标是 [PILOT 技术报告 v1](https://arxiv.org/pdf/2309.07117v1)和[上游 README 的 Reproduced Results](https://github.com/LAMDA-CL/LAMDA-PILOT/tree/7a6e904c5bc5cb7a4e1823b3434020be27469b63)。固定源码为 `7a6e904c5bc5cb7a4e1823b3434020be27469b63`；它是本次固定的快照，不声称是历史论文当时的提交。

协议一致是比较的必要条件，不保证复测值等于文献值。ImageNet-R 准备后划分的身份、具体部署头、权重和训练超参数仍需与引用表核对。本轮提供实现与正确性证据，没有正式性能复现结果。

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

原始旧图像预算与 batch 内回放量、epoch 数、全历史矩存储是不同维度。`shared_basis` 实际图像回放为 0；另两条路线使用表中规则。当前阶段新类训练集不受历史图像配额限制。选样采用 PILOT 的归一化特征 herding，旧类按原选样顺序保留配额前缀。

ImageNet-R 需采用 PILOT 链接的准备后划分，不能对原始数据任意重新划分。实现记录逐文件摘要与类名映射，检查 train/test 是否有内容完全相同的文件。这些可绑定本次输入，不能单独证明它与历史作者输入相同。尚未取得、运行或认证该准备版本时，须保留此限制。

## 3. 骨干、权重与预处理

- ViT-B/16，224×224 输入，最终 CLS 为 768 维；通过原生 `utils.inc_net.get_backbone` 加载，不做 torchvision 名称转换。
- `timm==0.6.12` 的 `vit_base_patch16_224` 选中 AugReg ImageNet-21K 预训练再 ImageNet-1K 微调权重，不能仅标成含义不明的 IN1K 或 IN21K。
- 精确源文件名为 `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`，对应 [timm 0.6.12 模型定义](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py)。加载时记录公共 URL、权重文件 SHA-256、初始状态 SHA-256。
- 训练原图变换直接复用 PILOT：`RandomResizedCrop(224, scale=(0.05,1.0))`、随机水平翻转、`ToTensor`。
- 确定性统计及评测变换：bicubic `Resize(256)`、`CenterCrop(224)`、`ToTensor`。
- 选定 PILOT 变换不加像素 mean/std normalization。额外添加标准 ImageNet normalization 会改变协议。
- 新方法解析特征默认逐样本 L2 normalization；它属于方法定义，与像素 normalization 不同。配置键为 `feature_normalization`，旧别名 `normalization` 冲突时拒绝。current 路线固定要求 L2。

本协议的结构化 runner 对所有真实数据运行强制 `timm==0.6.12`；三个新方法还检查骨干名称并绑定实际权重文件摘要。原生 baseline 的加载器保留原样，本轮未验证它们实际部署时的完整权重来源。合成测试使用独立小 token ViT，明确标成 `smoke`。

## 4. 训练及开发访问规则

正式评测：每阶段可多次训练该阶段全部当前图像及允许的回放；不访问未来类别训练图像，不用 test 标签选超参数或停止。任务边界允许已知。Epoch 数不等于统计累加次数；新方法均在任务结束后只为每个新到样本提交一次持久矩。

开发模式：每个原始训练类别留出 10%，由 Python `random.Random(split_seed=0)` 固定；更换类别顺序不改变留出身份。开发 loader 只读取 train，不打开指定 test。配置选定后，正式阶段用完整训练集重新拟合。真实预训练验证脚本还缩减每类样本，标为 `validation_only=true`，最多两个任务；独立合成 smoke 为三个任务。

两条可塑路线配置的 20 epochs 是未调优起点，不是 PILOT 所有方法统一的轮数。共享预测基只确定性编码一次，再进行 EM。各路线的优化器与目标详见方法推导。

已有方法保留原生目标、学习率、优化器及轮数。派生配置不会把所有算法强制改成同一种训练预算。比较应单独报告训练成本、总参数和额外历史状态，不能仅用图像配额代表总资源。

## 5. Baseline 配置与差异

`exps/pilot_figure2/baseline_*.json` 来源于未修改的 `exps/*.json`，统一数据名、十阶段、seed、路径、评测及存储规则，记录 `cl27_source_config`。现提供 FineTune、SimpleCIL、RanPAC、EASE、SLCA、iCaRL、Coil、DER、MEMO、FOSTER 的两个数据集配置，未在本轮完整复测。

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

只进行数值、生命周期、原生入口及限定样本验证；没有运行十阶段完整实验或复现文献表，没有向 Agent B 交接。正式性能、ImageNet-R 实际数据流水线和多种子波动待后续验证。
