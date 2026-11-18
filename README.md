# CL-27fall

在固定版本 LAMDA-PILOT 中接入四个类别增量学习候选方法，并提供 ER 与离线 Joint 两个基础对照。2026-11-17 原定的12组正式实验均已完成并核验。结果见[原12组正式结果](experiments/campaign_20261117/summary/summary.md)，最终配置及选择依据见[开发记录](experiments/campaign_20261117/development/SELECTIONS.md)。小规模正确性验证与完整性能实验分开保存。用户随后在两个数据集各追加固定模块库20-epoch变体和TUNA，四组均已启动；已核验结果与完成状态见[追加实验目录](experiments/extension_20261117/README.md)。原5-epoch结果保留。TUNA采用原生算法机制、每阶段20个图像训练epochs，首阶段不做分类器对齐，后九阶段各做30个对齐epochs。

| 方法名 | 学习对象 | 历史信息 | 骨干更新 |
| --- | --- | --- | --- |
| `shared_basis` | 类别预测器共享的低秩协方差先验 | 固定特征中的精确全历史矩，无旧图像 | 冻结 |
| `history_transport` | 可塑表征及旧矩到当前坐标的迁移 | 精确 anchor 全历史矩、近似当前矩、受限图像回放 | 主骨干允许更新，anchor 固定 |
| `current_adapted` | 固定模块库、主骨干与基于少样本的复用目标 | 精确 anchor 全历史矩、受限回放及其残差修正 | 主骨干允许更新，anchor 固定 |
| `plastic_shared_basis` | 可塑表征、历史矩迁移及共享预测器先验 | 精确 anchor 矩、当前坐标近似历史矩、受限回放 | 主骨干允许更新，anchor 固定 |

原生调用链保持为 `main.py → trainer.train → utils.factory.get_model → BaseLearner`。新方法继承 PILOT 的 `BaseLearner`，数据管理器复用原始任务构建、`DummyDataset` 与图像预处理。冻结 `shared_basis` 保留作对照，第四方法新增为独立候选，不替换前三种方法。

| 基础对照 | 数据访问 / 模型 | 主要指标 |
| --- | --- | --- |
| `er` | 单个可塑 ViT＋普通 CE 头，当前数据＋PILOT 配额的图像回放 | Final、AIA、遗忘 |
| `joint` | 相同预训练起点，一次训练全部类别并集，不受历史图像配额限制 | 只报 Final；AIA、遗忘为 null |
| `tuna`（追加对照） | 相同预训练骨干＋原生Adapter及类别统计对齐，无旧图像回放 | Final、AIA、两种遗忘；预算和结果见追加实验目录 |

## 来源与文档

- 上游：[LAMDA-PILOT，固定提交 7a6e904](https://github.com/LAMDA-CL/LAMDA-PILOT/tree/7a6e904c5bc5cb7a4e1823b3434020be27469b63)。原 Python / JSON / Markdown 源文件摘要在 [UPSTREAM_PIN.json](UPSTREAM_PIN.json)，原 README 在 [UPSTREAM_README.md](UPSTREAM_README.md)。上游算法文件保留；接入修改位于 `trainer.py` 和 `utils/factory.py`。
- [方法设计总说明](docs/METHODS.md)：四条路线的关系、V4 继承范围、文献依据、未验证问题。
- [统一实验协议](docs/PROTOCOL.md)：具体权重、存储预算、划分、评测公式及比较边界。
- 详细推导：[共享预测基](docs/research/shared_basis.md)、[可塑骨干与统计迁移](docs/research/history_transport.md)、[当前模块方法适配](docs/research/current_adapted.md)。
- 第四方法：[数学组合与噪声条件](docs/research/plastic_shared_basis.md)、[原生训练与恢复](docs/research/plastic_shared_wrapper.md)、[文献及开源代码复核](docs/research/plastic_shared_literature_review.md)。
- 基础对照：[ER](docs/research/er.md)、[Joint](docs/research/joint.md)。ER 直接部署训练 CE 头，没有 anchor 或统计矩；Joint 是独立数据访问条件的离线参考。
- [验证报告](validation/REPORT.md)：实际执行的检查与限制；不把合成准确率作为正式成绩。

## 环境

历史环境使用 Python 3.10、PyTorch 2.0.1、torchvision 0.15.2、timm 0.6.12。2026-11-17 实际服务器为 Python 3.10、torch 2.7.1+cu128、torchvision 0.22.1+cu128、timm 0.6.12，完整版本见[执行环境](experiments/campaign_20261117/environment.freeze.txt)。不建议在 Python 3.12 中直接安装旧版 timm，其其他模型定义存在 dataclass 兼容问题。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## 小规模验证

无需下载数据的合成检查通过原生 PILOT 入口运行，持续学习配置为三任务、每任务两类；Joint 对六类并集只训练一次。`tiny` 只用于合成验证。

```bash
python main.py --config exps/smoke/shared_basis.json
python main.py --config exps/smoke/history_transport.json
python main.py --config exps/smoke/current_adapted.json
python main.py --config exps/smoke/plastic_shared_basis.json
python main.py --config exps/smoke/er.json
python main.py --config exps/smoke/joint.json
```

输出目录必须为空；重复执行前指定新的 `output_dir`。持续学习checkpoint在任务完成后保存。恢复时保留原配置，添加 `resume` 指向任务边界checkpoint，并更换 `output_dir`。四个候选、ER及追加TUNA支持这种恢复；正式追加实验还使用绑定manifest、历史前缀和RNG身份的[恢复与核验流程](experiments/extension_20261117/PLAN.md)。Joint只保存完整联合训练checkpoint，恢复保留已有Final，不重新训练/评估，也不支持中途epoch恢复。多种子恢复同一个checkpoint会拒绝。

真实预训练 ViT 的限定检查只用 CIFAR-100 训练集内部划分，每类三个拟合样本与三个留出样本、两任务、每任务十类、一个 epoch。脚本不下载数据；缺少权重时 timm 会下载公开权重。是否成功以验证报告为准。

```bash
python scripts/verify_pretrained.py \
  --data-root /path/to/cifar-root \
  --output-dir validation/runtime/pretrained_check \
  --device -1
```

`--device -1` 使用 CPU，`--device 0` 使用一张 CUDA 卡。可加 `--prepare-only` 仅生成限定配置。默认不保存较大的真实 ViT checkpoint。

仅验证第四方法可加 `--methods plastic_shared_basis`。其极小样本检查预先指定 `basis_noise_variance=0.01`，目的是覆盖非零共享基；正式配置仍为 1.0。该差异源于明确的各向同性退化条件，不是按留出准确率选参，详见数学说明与验证报告。

仅验证新基础对照可加 `--methods er joint`：ER 运行两个增量阶段，Joint 对相同20类并集只拟合一次。两者仅使用每类三张训练及三张 train 留出，一个 epoch。Joint 用 `joint_max_classes=20` 限制开发检查；正式 Joint 禁止这种限制，也不接受 `max_tasks`。

## 正式配置

初始配置模板在 `exps/pilot_figure2/`；原12组实际采用的配置、源码身份及审核记录见[主实验目录](experiments/campaign_20261117/README.md)。模块20-epoch和TUNA的配置、预算变更及已核验结果见[追加实验目录](experiments/extension_20261117/README.md)。ER与使用回放的候选采用CIFAR-100 10×10类、固定总回放容量2000；ImageNet-R 10×20类、每类20张。冻结共享基与TUNA实际旧图像量为0；Joint一次访问全部训练类并集，只报Final。统一使用历史timm的IN21K→IN1K ViT-B/16、seed1993，测试不给任务编号。

设置 `data_root` 后才可运行。CIFAR 根目录包含官方 `cifar-100-python/train` 和 `test`；ImageNet-R 使用 PILOT 准备后 `train/<class>/...`、`test/<class>/...`，不能任意重新划分后仍声称是原论文协议。

`baseline_er_*.json` 和 `baseline_joint_*.json` 是新增对照；其余 `baseline_*.json` 保留 PILOT 原方法超参数并显式统一协议，未在本轮完整复测。原 `exps/*.json` 保留原样。FOSTER 的 CIFAR 存储规则和 EASE 的增量步长存在源配置与选定历史报告协议的差异，详见协议说明。已有 DER 指 Dynamically Expandable Representation，不是 DER++。

确定性提特征使用独立的 DataLoader 随机数生成器，避免额外 anchor 编码影响后续 CE 头初始化或训练顺序。这修复了早期代码的一项随机流耦合；旧提交的验证记录保留为历史证据，不能期待与修复后训练权重逐位相同。新版本另外核对 ER 与统计迁移路线的匹配训练轨迹。

本仓库不包含数据集、权重、运行 checkpoint 或服务器访问资料。上游作者归属与参考文献保留于原 README；使用相应代码时需引用 PILOT 及所用方法。
