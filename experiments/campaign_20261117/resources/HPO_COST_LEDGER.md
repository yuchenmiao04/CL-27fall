# 已闭合HPO成本账本

已按65份启动receipt整理真实执行进程，覆盖8/8已闭合开发选择。可定位的计时区间合计 **97,775.778 秒**，为开发搜索及既有诊断的已记录成本下界；完整实际成本未知。它是多进程计时区间之和，不是日历耗时或GPU-hours。正式训练成本完全另列。

| 开发类别 | 进程数 | 已记录区间秒数 |
| --- | ---: | ---: |
| 模块候选runner | 8 | 59,010.281 |
| 初始CE训练诊断 | 1 | 1,231.322 |
| 共享冻结特征提取 | 1 | 407.443 |
| 冻结基数值候选 | 20 | 989.486 |
| 共享CE cohort（含读者） | 23 | 33,943.802（其余未知） |
| 模块只读诊断 | 5 | 517.560 |
| 冻结基各向同性诊断 | 4 | 11.779 |
| 冻结快照只读诊断 | 1 | 96.063 |
| 中止模块续跑的已知epoch区间 | 1 | 1,568.043（其余未知） |
| 中止错误预算任务（未知） | 1 | 未知，未计入已记录小计 |

去重口径：23个CE cohort采用外层计时，其内部ER/HT/PSB result.wall只作分项证据；不再次相加。20个记录模式CE阶段的原生训练秒数只是其中已包含的子项；读者携带的donor秒数不重复收费。24个冻结数值运行各为真实重拟合，两份不可变特征缓存的提取分别只记一次。续跑只记本次execution.wall，不再次计入保存的历史阶段。

已记录CUDA任务进程区间：96,678.450秒；CPU任务进程区间：1,097.327秒。两者都包含各自计时边界内的CPU/IO/等待；不能解释为设备利用时长。

尚缺的实际开销：

- `cifar_current_full_v1`：D0027保留两段第三阶段epoch，共1,568.043秒；完整中止进程及其他未记录区间未知。
- `imagenetr_current_bounded64_screen_v4`：D0030证明在首个任务完成前纠正20→5epochs；没有留存wall，实际成本未知。
- `cifar_ce_full_v1`：外层报告只计到已完成ER和HT，9,788.757秒；其后至中止的未完成PSB工作未知。分拆后的PSB进程按其真实新计时另记。
- 各计时器之外的启动、恢复或输出尾部仍未统一测量；没有据相邻启动时间、epoch数或设备预留补算。

来源：输入快照282JSON，已核验tar SHA256 `2777eaf5627ebb842149ad503ae63dc3e32154c8f2e4d4b212980c120337ea88`；JSON清单另算SHA。另纳入6份既有诊断报告并逐份绑定SHA。每个进程的PID/start ticks、来源文件与字段、续跑前缀、计时边界、成本及未知项见同目录`hpo_cost_ledger_v1.json`。未连接服务器、读取模型、运行GPU/测试或提交Git。

## 单独的正式训练成本观察（10/12）

现有10个已完成正式run的runner计时区间合计 **43,553.829秒**。此数与前述HPO下界分别记录；两个current正式run仍待测。它同样不是日历耗时或GPU-hours。

| 正式run | runner秒数 |
| --- | ---: |
| `formal_cifar_joint_fixed_v1` | 4,394.107 |
| `formal_cifar_shared_selected_v7` | 501.585 |
| `formal_imagenetr_shared_selected_v7` | 542.794 |
| `formal_imagenetr_joint_fixed_v1` | 7,527.062 |
| `formal_cifar_er_fixed_v1` | 15,147.143 |
| `formal_cifar_transport_selected_v2` | 1,668.672 |
| `formal_cifar_plastic_shared_selected_v5` | 1,679.604 |
| `formal_imagenetr_er_fixed_v1` | 9,525.853 |
| `formal_imagenetr_transport_selected_v3` | 1,273.735 |
| `formal_imagenetr_plastic_shared_selected_v5` | 1,293.273 |

ER正式行已经包含实际CE donor训练，不再加shared_ce_assets秒数。HT/PSB各读者的实际runner区间只加一次；读者耗时不包含donor梯度训练，不能作为该方法独立完整训练耗时。来源为SHA绑定的`PARTIAL_RESOURCE_EVIDENCE.json.completed_rows[*].execution.actual_run_wall_seconds`；本节未读取正式准确率用于任何选择。
