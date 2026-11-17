# 正式 checkpoint 恢复与审计入口（12/12，checkpoint 索引）

本目录对应 seed 1993 的十二个预选主实验，身份来自 [`SELECTED_JOBS.json`](../formal/SELECTED_JOBS.json)。[索引](formal_checkpoint_index_partial_v1.json) 当前登记十二个已完成且有文件校验证据的 checkpoint，`obtained_verified_checkpoints=12`、`pending_cells=0`。索引文件名保留旧入口兼容；这份完整 checkpoint 登记不等于 GPU 续跑或额外准确率复现证书。本次建立索引只读了已有 JSON、日志和源文件，未连接服务器、打开 `.pt`、导出模型或启动训练。

## 查找与核对

在仓库根执行以下命令只显示索引，不加载模型：

```bash
python experiments/campaign_20261117/recovery/checkpoint_entry.py \
  --job formal_cifar_transport_selected_v2 --full
```

索引每格同时给出 checkpoint 的字节数/SHA256、原始 metadata SHA、备份位置、实际 runtime 或已保留的来源证据、完成回执及审计覆盖。公开 metadata 的操作路径已替换，因此原始文件 SHA 与发布副本 SHA 可能不同；按每格 `public_export_manifest`/`source_export_manifest` 对照，不能拿发布副本去要求匹配原始 SHA。

四个早期模型保存在现有私有工作区归档中，八个后续模型只保存在服务器内部备份。`checkpoint_entry.py --verify-file` 可在模型已有的存储位置核对一个**已完成、不可变**备份，仅读字节，不使用 PyTorch：

```bash
python experiments/campaign_20261117/recovery/checkpoint_entry.py \
  --job formal_cifar_transport_selected_v2 \
  --verify-file "${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_cifar_transport_selected_v2/checkpoint.pt"
```

路径变量由持有既有存储的人设置：`RUNTIME_ROOT` 是原服务器 runtime/data 根，`ROOT_BACKUP_ROOT` 是另一分区的 formal-backup 根，`PRIVATE_WORKSPACE_ROOT` 是已有私有归档根。它们不是服务器地址。脚本不会连接、下载、复制、覆盖、调度或自动选择模型；未完成索引条目会在读取任何模型前被拒绝；当前十二格均为 `obtained_verified_checkpoint`。已取消的模型下载分支保持取消，后续八格模型继续使用 server-local 备份。

## 必须保留的恢复依赖

- 使用每格保存的原始运行 `config.json`、`data_manifest.json`、`results.json`、launch/completion receipt 与 runtime/source SHA。配置里的 `implementation_commit` 只是保留字段，不能单独证明实际执行源，也不能用当前 HEAD 替换执行版本。后续八格的实际导入源 SHA 来自 native CPU audit，训练入口另由 `source_correspondence_training_entries_v1.json` 绑定。CIF ER 的源哈希已保留，但其旧 metadata 包没有导出源文件副本；使用原 `runtime_fixed_er_v5` 或与该记录逐文件匹配的源。早三格的具体证据覆盖及缺少 launch-cwd 记录处已在索引明确标出。
- 使用原环境及其记录的核心版本：`torch==2.7.1+cu128`、`numpy==1.26.4`、`timm==0.6.12`。这三个版本不是完整环境 lock。PILOT 上游身份为 `7a6e904c5bc5cb7a4e1823b3434020be27469b63`。
- 预训练 NPZ 为索引 `environment.pretrained_npz_relative_path`，文件 SHA256 必须为 `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`；初始 backbone state SHA 为 `1ece119a218036ffacbd3b576e1eb00a483915c88b21d81980c5e4e4a52e66e9`。使用现有文件，不触发下载。
- 数据必须重建为索引绑定的 fingerprint、源清单 SHA、seed 1993 类序和原 PILOT 变换。CIF 正式 train/test 为 50000/10000，ImageNet-R 为 24000/6000。保留已核验 PILOT 准备包原划分及其 23 对已知跨划分重复内容；不自行去重或重新划分。
- ImageNet-R 的 replay checkpoint 保存原图像路径引用；模型文件不包含一套独立 JPEG 数据集。必须保留或按已验证数据身份映射原图像路径。CIF replay 是原始数组。参考支路、辅助统计量、回放记录、方法局部 RNG 和外层 RNG 都属于 native state；不能只装载 backbone。
- ER 产生的每数据集十个 CE trajectory 文件由 sealed cache inventory 绑定。HT/PSB 的 replay-mode 继续执行依赖这些缓存、request/body/head SHA 和 RNG 前置条件；不能重新生成一份缓存冒充原复用。已完成模型的恢复/部署不需要再次训练 ER。current 不使用这份 CE trajectory。

## 已实际执行的 CPU 检查

早三格有协议/矩阵、checkpoint schema、行数、NPZ/冻结身份和文件 SHA 核验；没有在本索引中声称已做 native restore 或 zero-input forward。ImageNet-R Joint 有独立完成的 CPU checkpoint/data/训练日志/指标一致性核验，包含实际有限权重和 RNG payload 检查；它也没有被改写成新式 native zero-forward 审计。上述四个 legacy 模型均不计入后续八次 native restore。

后续八格 ER、HT、PSB 及两格 current 已使用各自终止后的 SHA-bound server-local 备份执行 native CPU restore、原始/恢复后 state 数值比较、有限零图像输出检查和实际数据/replay 核验，未增加训练或准确率评测。CIF ER 使用完整重跑通过的 **v2** audit；v1 在 source inventory 阶段失败的材料保留为历史，不充当 PASS。helper 的 `final_audit_complete=false` 表示它不签发整场完成证书，不否定其明确记录的单 checkpoint PASS。

这八格的 audit command/工具 SHA 已保存在各格 audit receipt。需要在已有存储上复核时，使用每格源身份匹配的工具和 runtime；下例仅展示同一 CPU 入口，不要求重复已通过审计，也不启动 GPU：

```bash
CUDA_VISIBLE_DEVICES='' "${PYTHON}" "${AUDIT_TOOLS}/native_cpu_auditor.py" \
  --repo "${EXECUTED_RUNTIME}" \
  --state-loader "${AUDIT_TOOLS}/state_loader.py" \
  --checkpoint-backup "${FINAL_BACKUP}/checkpoint.pt" \
  --expected-sha256 "${CHECKPOINT_SHA256}" \
  --metadata-dir "${FINAL_BACKUP}" \
  --pretrained-npz "${PINNED_NPZ}" \
  --confirmed-final-backup
```

可用的同 SHA 工具副本例如 `../formal/observations/formal_cifar_transport_selected_v2/audit_tools/`。`--state-loader` 独立指定，不能假定它位于原训练 snapshot。`--repo` 必须指向实际执行的模型/data runtime，不修改 snapshot；相对数据路径必要时显式传 `--runtime-cwd`，等价数据映射用 `--data-root` 并要求重建 fingerprint 匹配。该 helper 覆盖 ER/HT/PSB/current，不能套用它给早期 shared/Joint 伪造审计覆盖。

## 恢复能力边界

现有模型 checkpoint 是已完成阶段边界；ER/HT/PSB 每阶段重新建立 AdamW，不提供阶段内部优化器恢复。Joint 仅支持 completed-fit 恢复，不支持中途 epoch 续训。current checkpoint 保存模块与 backbone 两套 AdamW、更新/阶段进度、episode/replay/global RNG；两格正式最终状态均已完成自然终止后的 SHA-bound 备份和既定 native CPU 审计；CIFAR 的两次 episode skip 按实际成功更新数保留，未用无跳过的理论步数替代。

实际 native CPU restore 证明 native learner 能恢复该保存状态并产生有限、正确维度输出，**没有证明 GPU RNG 恢复、GPU 后续训练轨迹或额外准确率复现**。CPU 审计保留并检查 CUDA RNG payload，但不应用它。两格 current 的登记均来自原 PID/start-time 身份终止后的最终 SHA-bound 备份与实际审计证据，不读取或覆盖活跃或部分 checkpoint。恢复入口不自动重新启动任何训练。
