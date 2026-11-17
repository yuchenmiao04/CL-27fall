# 待运行的固定 baseline 配置

本次新增三份正式配置：`configs/formal_cifar_er_fixed_v1.json`、`configs/formal_imagenetr_er_fixed_v1.json`、`configs/formal_imagenetr_joint_fixed_v1.json`。它们分别复制 `exps/pilot_figure2/baseline_er_cifar224.json`、`baseline_er_imagenetr.json` 和 `baseline_joint_imagenetr.json` 的事先固定训练设置，不进行参数搜索。均为完整正式流、seed 1993、20 epochs、原生 AdamW、骨干 LR 1e-5、CE 头 LR 1e-3、weight decay 1e-4、当前 batch 64；ER 每批最多回放 64 张，Joint 不使用回放池。CIFAR-100 ER 固定总容量 2,000 张；ImageNet-R ER 固定每类 20 张。Joint 一次访问全部训练类别并集，只报告 Final，AIA 和遗忘为不适用。

配置使用 `data/cifar100`、`data/imagenetr` 和 `runs/formal/...` 可移植路径，不含服务器地址或访问资料。初始设备分别为 CIFAR-100 ER 卡 1、ImageNet-R ER 卡 0、ImageNet-R Joint 卡 0；实际调度可以调整设备和展开路径，这不属于训练参数搜索。没有开发留出或样本截断开关，也没有恢复路径。文件存在不表示任务已启动或完成，结果以各正式运行记录为准。

ER 使用 `scripts/run_fixed_er_record.py` 原生正式入口，配置与执行报告均标为 `execution_role=fixed_baseline`。该入口在固定 ER 训练时附带记录精确 CE 轨迹；之后只有已选定 HT/PSB 的数据、设备、训练参数、初始／阶段状态和随机流等全部匹配时，才可通过已验证缓存复用相同骨干训练。若条件不匹配，候选方法仍需单独训练；不能为复用而改变固定 baseline 配置。内部候选 CE driver 不替代这两个正式 ER 对照。
