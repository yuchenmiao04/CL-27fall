# CL-27fall

在固定版本 LAMDA-PILOT 中接入三个类别增量学习候选方法。当前交付范围是实现与小规模正确性验证，尚未运行完整 CIFAR-100 / ImageNet-R 实验，不包含性能提升结论。

| 方法名 | 学习对象 | 历史信息 | 骨干更新 |
| --- | --- | --- | --- |
| `shared_basis` | 类别预测器共享的低秩协方差先验 | 固定特征中的精确全历史矩，无旧图像 | 冻结 |
| `history_transport` | 可塑表征及旧矩到当前坐标的迁移 | 精确 anchor 全历史矩、近似当前矩、受限图像回放 | 主骨干允许更新，anchor 固定 |
| `current_adapted` | 固定模块库、主骨干与基于少样本的复用目标 | 精确 anchor 全历史矩、受限回放及其残差修正 | 主骨干允许更新，anchor 固定 |

原生调用链保持为 `main.py → trainer.train → utils.factory.get_model → BaseLearner`。新方法继承 PILOT 的 `BaseLearner`，数据管理器复用原始任务构建、`DummyDataset` 与图像预处理。三种方法是并列候选路线。

## 来源与文档

- 上游：[LAMDA-PILOT，固定提交 7a6e904](https://github.com/LAMDA-CL/LAMDA-PILOT/tree/7a6e904c5bc5cb7a4e1823b3434020be27469b63)。原 Python / JSON / Markdown 源文件摘要在 [UPSTREAM_PIN.json](UPSTREAM_PIN.json)，原 README 在 [UPSTREAM_README.md](UPSTREAM_README.md)。上游算法文件保留；接入修改位于 `trainer.py` 和 `utils/factory.py`。
- [方法设计总说明](docs/METHODS.md)：三条路线的关系、V4 继承范围、文献依据、未验证问题。
- [统一实验协议](docs/PROTOCOL.md)：具体权重、存储预算、划分、评测公式及比较边界。
- 详细推导：[共享预测基](docs/research/shared_basis.md)、[可塑骨干与统计迁移](docs/research/history_transport.md)、[当前模块方法适配](docs/research/current_adapted.md)。
- [验证报告](validation/REPORT.md)：实际执行的检查与限制；不把合成准确率作为正式成绩。

## 环境

历史环境使用 Python 3.10、PyTorch 2.0.1、torchvision 0.15.2、timm 0.6.12。不建议在 Python 3.12 中直接安装旧版 timm，其其他模型定义存在 dataclass 兼容问题。CUDA 环境应安装相同版本且匹配 CUDA 的 PyTorch 构建。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## 小规模验证

无需下载数据的合成检查通过原生 PILOT 入口运行，每个配置为三任务、每任务两类。`tiny` 只用于合成验证。

```bash
python main.py --config exps/smoke/shared_basis.json
python main.py --config exps/smoke/history_transport.json
python main.py --config exps/smoke/current_adapted.json
```

输出目录必须为空；重复执行前指定新的 `output_dir`。完整 checkpoint 只在任务完成后保存。恢复时保留原配置，添加 `resume` 指向任务边界 checkpoint，并更换 `output_dir`。该恢复协议仅适用于三个新增 learner；多种子恢复同一个 checkpoint 会拒绝。

真实预训练 ViT 的限定检查只用 CIFAR-100 训练集内部划分，每类三个拟合样本与三个留出样本、两任务、每任务十类、一个 epoch。脚本不下载数据；缺少权重时 timm 会下载公开权重。是否成功以验证报告为准。

```bash
python scripts/verify_pretrained.py \
  --data-root /path/to/cifar-root \
  --output-dir validation/runtime/pretrained_check \
  --device -1
```

`--device -1` 使用 CPU，`--device 0` 使用一张 CUDA 卡。可加 `--prepare-only` 仅生成限定配置。默认不保存较大的真实 ViT checkpoint。

## 正式配置

正式配置在 `exps/pilot_figure2/`，本轮不启动。CIFAR-100 为 10×10 类、固定总回放容量 2000；ImageNet-R 为 10×20 类、每类 20 张。共享预测基实际旧图像量为 0。统一使用历史 timm 的 IN21K→IN1K ViT-B/16、seed1993，测试不给任务编号。

设置 `data_root` 后才可运行。CIFAR 根目录包含官方 `cifar-100-python/train` 和 `test`；ImageNet-R 使用 PILOT 准备后 `train/<class>/...`、`test/<class>/...`，不能任意重新划分后仍声称是原论文协议。

`baseline_*.json` 保留原方法超参数并显式统一协议，未在本轮完整复测。原 `exps/*.json` 保留原样。FOSTER 的 CIFAR 存储规则和 EASE 的增量步长存在源配置与选定历史报告协议的差异，详见协议说明。已有 DER 指 Dynamically Expandable Representation，不是 DER++。

本仓库不包含数据集、权重、运行 checkpoint 或服务器访问资料。上游作者归属与参考文献保留于原 README；使用相应代码时需引用 PILOT 及所用方法。
