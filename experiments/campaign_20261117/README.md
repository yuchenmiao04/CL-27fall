# 2026-11-17 主实验

## 当前状态

服务器环境、两个数据集、预训练权重及小规模正确性验证均已准备完成。正式 12 组实验尚未开始；已有小规模正确性验证不计入主实验结果。用户已确认训练集内部固定留出集选参、完整训练集重训、正式测试集只用于最终报告；随后向 Agent B 交接代码审查；最新要求已恢复实验，不等待 B。后续状态以运行日志和正式结果文件为证据。

## 固定范围

- CIFAR-100：10 个阶段，每阶段新增 10 类。
- ImageNet-R：PILOT 准备后的 train/test 划分，10 个阶段，每阶段新增 20 类。
- 方法：`shared_basis`、`history_transport`、`current_adapted`、`plastic_shared_basis`、`er`、`joint`；每个数据集六行，共 12 个最终结果。
- seed 仅为 1993。ViT-B/16 与预训练权重、预处理、类别顺序遵循 `docs/PROTOCOL.md`。
- 有回放方法：CIFAR-100 固定总容量 2,000；ImageNet-R 固定每类 20，最终 4,000。冻结共享基不保存旧图像。Joint 访问全部训练并集，属于离线参考，不受增量回放容量限制。
- 五个增量方法报告 Final、AIA 和明确区分的两种遗忘；Joint 仅报告 Final。
- 不做网格搜索；依据诊断动态调整，记录理由、版本和配置。只对四个候选方法调参；前2–3阶段初筛，最终候选完整跑十阶段开发流程，选定后完整训练集重训。ER、Joint 使用预先固定的常规配置，不进行参数搜索。搜索成本与正式训练成本分开。

## 正式结果汇总

```sh
python scripts/summarize_campaign.py --runs-root runs/campaign_20261117 --output-dir experiments/campaign_20261117/summary
```

工具只接受对应配置为 seed 1993、`phase=final` 且结果 `status=complete` 的已完成运行；增量方法必须有完整十阶段矩阵与样本计数，Joint 必须是全部类别的一次离线拟合。开发、smoke、validation-only、partial 结果不会计为正式完成。目录名为 `scripts`、`smoke`、`validation`、`validation_only`、`development` 或 `dev` 的下级结果也排除。结果目录需要保留同级 `config.json` 和 `results.json`。

若一个方法/数据集有多个符合条件的正式目录，工具报错，不按测试分数自动选最好。由实验记录明确确定采用哪个运行，再用 JSON 指定目录，例如：

```json
{
  "cifar224": {
    "er": "cifar224/er_selected"
  },
  "imagenetr": {
    "joint": "imagenetr/joint_selected"
  }
}
```

目录相对于 `--runs-root`，也可为绝对路径；传入 `--selection path/to/selected_runs.json`。未指定且只有一个合法候选的单元直接汇总，缺失单元保留为 —。显式选择必须对应一个合法候选。

生成 `summary.md`：两张固定六行主表；生成 `summary.json`：原始未四舍五入指标、资源记录、所选结果/配置/数据清单/检查点路径、指纹、候选目录和排除原因。两个遗忘定义分别报告，不在汇总阶段根据成绩切换。该工具不加载检查点或认证数据身份，汇总通过不替代完整的实验审计；未测量或未写入结果的资源量不会被补造。

## 本次服务器执行环境与数据核验

- 两张 RTX 5090（各 32GB），独立运行一进程一卡；当前硬件核验及环境准备已完成。
- RTX 5090 使用 Python 3.10、torch 2.7.1+cu128、torchvision 0.22.1+cu128；timm 仍固定 0.6.12，其余精确版本见 environment.freeze.txt。这是相对历史 PILOT 环境的显式差异，不能将复测数值视为文献数值的逐位复现。PyTorch 官方从 2.7 开始支持 Blackwell：https://pytorch.org/blog/pytorch-2-7/ 。
- 预训练 NPZ SHA256：ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124，与本机已验证文件一致。
- CIFAR-100 使用公共缓存中的官方压缩包，解压后 train/test MD5 均与 torchvision 官方值相符。
- ImageNet-R 从 PILOT 固定 README 的 OneDrive 分享链接取得准备包，压缩包 SHA256 为 c2720edb18da7173b6cb6d9e9a65e18fb6f949f65d8cd10ac99df9003d8b055a。原始划分为 train 24,000 张、test 6,000 张，各 200 类；不重新划分或去重。数据来源与完整核验见 imagenetr_source.json。
- 官方准备包含 23 对跨 train/test 的相同内容（3 对同类别、20 对不同类别）。这是源数据性质，需随正式结果报告。只为完整文件清单摘要匹配的官方准备划分开放显式例外；未知重叠仍拒绝。不能把源包保留策略称为经过清洗的无重叠测试集。
- 运行观测改进增加逐 epoch JSON 日志，并合并模块训练逐参数检查的主机同步。保留原逐张量有限性与平方和运算，norm 按原顺序求和；FP32/FP64 连续四步事件及参数与原实现完全相等，NaN/Inf 报错行为保留。新增数据身份测试后，本地及服务器均为 204 项测试通过。六方法 CIFAR-100 真实 ViT 小规模检查均已通过；纯同步优化后另复核 current_adapted。ImageNet-R 检查和容量探测以保存的实际结果文件为准。

## 已完成的准备验证

- `correctness/gpu_check_0/verification_report.json`、`correctness/gpu_check_1/verification_report.json`：六方法 CIFAR-100 小规模检查，两个 10 类阶段、每类三张拟合和三张训练内留出图像；Joint 对相同 20 类并集拟合一次。
- `correctness/gpu_check_optimized_current/verification_report.json`：模块训练同步优化后的实际 GPU 复核。
- `correctness/imagenetr_mechanics/verification_report.json`：六方法 ImageNet-R 小规模检查通过，两个 20 类阶段、每类三张拟合和三张训练内留出图像；Joint 对相同 40 类并集拟合一次。
- 上述检查使用预先声明的机械验证配置，未据准确率选参数；训练计数、支路更新及输出空间均已检查，正式测试集没有用于这些小规模检查。极小数据可全部装入回放池，非驻留历史分支仍由单元测试覆盖。
- `correctness/imagenetr_full_data_manifest.json`：服务器实际完整官方数据树通过摘要核验，原始 train/test 仍为 24,000/6,000 张，观测到预期 23 对相同内容。
- `correctness/pytest_postfix.txt`：服务器新环境上的 204 项测试通过。
- `correctness/capacity_gpu_{0,1}.json`：两张卡的 FP32/TF32 关闭容量检查通过；单个可塑 ViT＋CE 头，逻辑 batch 64 约 0.226–0.228 秒/步、峰值分配 9.74 GiB；batch 128 约 0.455–0.462 秒/步、峰值分配 18.14 GiB。纯 GPU 合成数据测量不含加载、双支路、模块或统计求解，不能直接视为端到端速度或完整实验 ETA。

这些记录均不替代正式十阶段实验；正式完成数仍为 0/12。HPO 来源已经确认，实验已恢复。首轮在两张卡上分别运行 ER 两阶段20 epochs 与固定模块库两阶段5 epochs 诊断；训练配置与事前调整理由位于 development/。这些开发试验不计入正式结果，也不是等预算性能比较。
