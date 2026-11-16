# 可塑骨干、历史统计迁移与共享预测基

本文件对应第四条候选路线 `plastic_shared_basis`，核心为 `cl27/cores/plastic_shared_basis.py`。前三条路线保留。本路线将已有可塑表征及矩迁移流程，与共享预测器经验贝叶斯读出组合；不是简单取消冻结标志，也没有新增模块增长。完整实验尚未运行，以下不包含准确率或新颖性结论。

## 1. 学习流程与接口边界

固定 anchor a(x) 提供不可变历史坐标；主 ViT h_t(x) 用当前任务和规定的旧图像回放进行 CE 更新。训练 CE 头与部署的共享预测头是两个对象。骨干训练规则、回放选择、容量、变换及一次性历史提交直接复用 `history_transport`；详细过程见 [history_transport.md](history_transport.md)。

每个任务完成后：

1. 在固定 anchor 中为当前新样本累积一次历史矩。
2. 对旧历史使用已有 direct / chained 迁移规则；先扣除驻留样本，在新坐标中精确重新加入这些样本，缺失部分通过类均值、散度和残差模型近似。
3. 得到当前坐标中的全历史矩快照后，**替换**共享预测器的上一份矩，而不是把整份快照再次累加。
4. 在当前坐标从 `Sigma=epsilon I` 开始重新拟合共享先验，再重求全部已见类的预测器与平坦截距。
5. 仅当迁移、矩一致性验证及经验贝叶斯拟合全部成功时，同时提交两个核心。共享拟合失败不会提前提交本轮 anchor 或迁移矩。

这不是端到端对共享边际目标反传到 ViT 的实现。当前主骨干由 CE 驱动，共享结构只在任务结束后根据矩估计。它可以检验可塑表征与共享先验组合是否有价值，但不能自动证明共享先验在指导骨干学习，更不能声称已经形成对未知未来类有益的表示。

## 2. 从增广矩到共享模型

迁移核心保存带最后一个常数坐标的矩。令当前估计特征为 h∈R^d，增广行 z=[h,1]：

```
Ghat_z = [ Ghat_h   s_hat ]
         [ s_hat^T    N   ]
Chat_z = [ Chat_h ]
         [  n^T   ]
```

n_c 保留真实整数类别计数；在缺失历史上，Ghat_h、Chat_h 是当前非线性表示下的估计值。把 `Ghat_z[:-1,:-1]`、`Chat_z[:-1,:]`、`Ghat_z[:-1,-1]`、n、N、类词表及任务数传入 `SharedPredictorBasis.fit_moments`。

该公共接口专门表达全快照替换，不调用累计到达接口 `observe`。重复传入同一快照不会让历史数变为两倍。原冻结路线仍用 `observe`，保持其固定坐标、任务到达累计和同坐标暖启动行为；已有 v1 checkpoint 字段没有改变。

中心化为

```
Gc = Ghat_h - s_hat s_hat^T/N
Cc = Chat_h - (s_hat/N) n^T
tr(Yc^T Yc) = N - n^T n/N
```

共享模型仍是 `w_c~N(0,Sigma)`，`Sigma=DD^T+epsilon I`；截距不正则化。给定当前快照：

```
S = (Gc/s² + Sigma^(-1))^(-1)
M = S Cc/s²
b = n/N - (s_hat/N)^T M
score(h) = h^T M + b
```

不做预测器类别行归一化。部署头为 `[M;b]`，只有主 ViT 与该头参与预测；anchor、CE 头和迁移状态属于训练辅助状态。

### 2.1 矩的代数一致性不等于经验样本真实性

替换 API 验证有限数值、维度、对称性、`sum_c Chat_h[:,c]=s_hat`、正整数计数、计数总量和任务/类词表关系。还检验联合矩的 Schur 补：

```
Ghat_h - Chat_h diag(n)^(-1) Chat_h^T >= 0
```

仅检查 Ghat_h 为 PSD 不足以保证标签交叉矩一致。上述检查可防止不可能的类均值能量和错误双计数；输入容差仍是严格 float64 矩检查，原迁移模型单独处理明确声明的编码器重编码误差预算。

对于真实 N 行、C 类的数据，类内散度秩至多为 N−C。但缺失历史的总体残差协方差近似可能构造出超过此秩的 PSD 散度，因此当前快照不一定对应某个真实 N 行样本表。`moment_kind='surrogate'` 允许这种总体矩代理，并明确标注；`exact_fixed_features` 会拒绝违反 N−C 必要秩约束的快照。不能因希望保留真实数据解释而静默删掉总体噪声方向，也不能因 PSD 检查通过就称为真实历史似然。

## 3. 为什么每个阶段重置先验

Sigma 描述的是**当前特征坐标中类别权重的先验协方差**，不是特征样本协方差，也不是迁移残差协方差。骨干变化后，直接沿用 D 的数值意味着默认旧、新预测坐标相同，这个假设一般不成立。

即使变化恰好是可逆仿射变换 `h_new=h_old A+u`，为保持相同预测函数也需要

```
w_new = A^(-1) w_old
b_new = b_old - u w_new
Sigma_new = A^(-1) Sigma_old A^(-T)
```

特征矩向前变换和权重先验变换方向不同，不能用 `A^T Sigma A` 运输分类器先验。非线性漂移更不具备一个全局可逆 A。附加地，`A^(-1)(epsilon I)A^(-T)` 在非正交变换下通常不是 epsilon I，甚至不属于当前“固定 epsilon 噪声底加低秩项”的模型族。

因此当前明确采取 `basis_reset_each_task=true`：每次从各向同性 epsilon I 出发，利用当前坐标中的**全历史快照**重学 D。重置的是优化初值，旧类并没有从学习目标中消失。它避免错误搬运旧先验，但放弃了跨阶段 D 方向的直接连续跟踪，增加了 EM 重拟合成本。不同阶段的基向量不能因列号相同就被解释成同一个语义单元；相同阶段内 D 的旋转也仍不可识别。当前未实现先验运输、可学习全秩底噪或对旧 D 的额外正则。

## 4. 经验贝叶斯代理目标及受约束 EM

共享基的完整推导在 [shared_basis.md](shared_basis.md)。将其中 Gc、Cc 换成当前代理矩，定义

```
ell_tilde(Sigma) = -1/2 [
 C ((N-1) log(2 pi s²) + log|I + Sigma^(1/2) Gc Sigma^(1/2)/s²|)
 + (N - n^T n/N - tr(Cc^T M))/s²
]
```

固定本阶段代理矩时，仍使用

```
Q = S + M M^T/C
D_new = U_k diag(sqrt(max(q_i-epsilon,0)))
k = min(basis_rank,d,C-1)
```

其中 M-step 对固定 epsilon 的低秩加噪声底协方差约束求解。代码沿用同一数值函数，并检查每次迭代 ell_tilde 非下降；没有把真实历史图像拿出来计算证据。代理矩被固定时的代数目标改善，不代表真实旧样本的似然、分类准确率或跨任务迁移改善。跨阶段骨干、类别和代理矩都变化，也不应比较两个阶段的目标值并宣布学习单调变好。

### 4.1 各向同性重置可能让共享基完全不生长

这一现象需要公开记录，而不是通过观察测试准确率调整噪声。令 `n_max=max_c n_c`。对于中心化一热标签，

```
Cc Cc^T <= n_max Gc
```

对经验样本可由 `||Yc||²<=n_max` 得到；对经过联合 PSD 与类质量约束的代理矩，也可从标签 Schur 补得到同样上界。首轮各向同性后验满足 `S=(Gc/s²+epsilon^(-1)I)^(-1)<=epsilon I`，且 S 与 Gc 对易。因此

```
Q-epsilon I
 <= -(epsilon/s²) Gc S + n_max/(C s^4) S Gc S
 <= (epsilon/s²) Gc S * (n_max/(C s²)-1)
```

所以当 `s²>=n_max/C` 时，该 EM 从各向同性初值无法产生高于 epsilon 的方向，D 保持零（数值舍入除外）。这是**充分条件，非必要条件**；不满足它仍可能得到零基。诊断记录 `reset_collapse_sufficient_condition`、`reset_collapse_threshold_noise_variance` 与实际 `effective_rank`。

例如每类只有 3 个拟合样本、已见 10 类时，上界为 0.3；默认 `s²=1` 会满足这个退化条件。正确性验证若要实际覆盖非零共享基，可以依据该代数条件预先指定较小噪声，并单独记录验证配置；这不是按官方测试准确率调优，也不构成该噪声在正式实验中更优的证据。当前正式起点保留冻结共享基的 `s²=1`，未引入自动噪声选择。

### 4.2 rank=0 的可核对退化

若 `basis_rank=0`，则

```
M = (Gc + lambda I)^(-1) Cc,   lambda=s²/epsilon
b = n/N - (s_hat/N)^T M
```

这与同一代理矩上的平坦截距 ridge 完全一致。与 `history_transport` 比较这一退化时，必须令其 `ridge=s²/epsilon`。当前正式起点 epsilon=0.001、s²=1，对应 lambda=1000；历史迁移路线默认 ridge=0.01，二者并未自动匹配。组合核心仍保留迁移核心内部的辅助 ridge 头及其配置，它不用于组合方法的预测；辅助状态与计算照实计入，不能把它的 ridge 解释成部署共享头的正则强度。

## 5. 文献与实际代码依据

| 来源与阅读位置 | 与本路线的关系和具体适配 |
| --- | --- |
| [DPCR，ICML 2025](https://proceedings.mlr.press/v267/he25d.html)；[官方 models/lwf.py](https://github.com/RHe502/ICML25-DPCR/blob/main/models/lwf.py)，`cov_prime/Q_prime` 重构段与 `get_projector_svd` | 已存在可塑表征、漂移估计、类统计运输和 ridge 重构的直接相关方案。代码中运输旧类 Gram/原型后加入当前统计并求解读出；这里沿用已有迁移候选，改用受限驻留回放、残差近似及共享经验贝叶斯先验。DPCR 的 exemplar-free 配置、按类投影和输出权重归一化未被复制。不能仅因改用 ViT 或组合 ridge 就声称有贡献。 |
| [Rasmussen 与 Williams，GPML，2006，§2.1、式(2.8)、(2.15)](https://gaussianprocess.org/gpml/chapters/RW2.pdf) | 权重空间高斯先验决定函数核 `k(x,x')=phi(x)^T Sigma phi(x')`。用于检查“改变表示但直接保留先验”会改变函数空间先验，明确特征与权重变换的对偶方向；本实现仍是有限维线性高斯工作模型，没有声称实现无限维 GP。 |
| [Roweis 与 Ghahramani，1999，§5.3](https://cs.nyu.edu/home/people/in_memoriam/roweis/papers/NC110201.pdf) | 线性高斯模型中的坐标变换、等价参数化及各向同性噪声对尺度的依赖，帮助识别固定 epsilon 模型族并非任意重参数化不变。这里没有复制 Kalman 状态方程，也没有假设特征漂移天然满足线性高斯转移。 |
| [Tipping 与 Bishop，Mixtures of PPCA，1999，§3.2、Appendix A](https://www.miketipping.com/papers/met-mppca.pdf) | 核对固定噪声底下的协方差载荷以及旋转非唯一性。此处 M-step 的 Q 来自**类别预测器后验二阶矩**，不是把图像特征的 PCA 作为共享知识；未采用原文的混合成分。 |
| [MacKay，Bayesian Interpolation，1992，§3–4](https://cs.uwaterloo.ca/~mannr/cs886-w10/mackay-bayesian.pdf) | 先积分权重，再用证据学习正则化先验；此处将证据的矩形式应用到明确标记的近似快照。点估计 D、忽略运输映射不确定性及一热高斯失配都保留为限制，不能用经验贝叶斯名称消除它们。 |
| [ACIL，NeurIPS 2022，Theorem 3.1](https://papers.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf)；[PILOT RanPAC](https://github.com/LAMDA-CL/LAMDA-PILOT/blob/7a6e904c5bc5cb7a4e1823b3434020be27469b63/models/ranpac.py) | 固定特征累积矩是精确联合线性读出的依据；其等价性前提不能直接扩展到非线性可塑骨干。本路线仅在 anchor 中保留这一精确性，在当前坐标中明确降为近似矩，且避免重新累计全快照。 |

本轮还直接核对本仓库的 `TransportHistory` 事务更新、平坦截距 ridge、resident 扣除与回填、重编码精度修复，以及 `SharedPredictorBasis` 的后验、EM 和恢复实现。组合使用现有公开数值接口，没有复制另一份随时间分叉的迁移算法。

## 6. 状态、验证和未确定之处

持久状态包括完整迁移核心，以及共享基自己的 G、C、特征和、计数、D、M、b。当前为保持清晰接口会重复保存两套当前矩，还保留未部署的辅助 ridge 头；`resource_bytes` 同时计入这些数组。事务拟合期间还存在候选副本和矩分解暂存，峰值内存会高于持久载荷；不能将其忽略。Checkpoint 同时校验两个核心的任务数、类顺序、样本数、类别计数和全快照是否一致，再原子恢复数值核心。

新增数值验证覆盖：快照替换及幂等、与显式全历史拟合等价、旧先验重置、错误类质量/交叉矩拒绝、总体散度与真实样本秩界的区分、非正交仿射变换的权重/先验方向、重置退化充分条件、direct/chained 的可辨识仿射迁移、匹配正则的 rank0 ridge 退化、拟合失败回滚、双状态不一致拒绝以及 checkpoint 继续一致。原冻结共享基的测试继续通过。

仍需检验的研究命题包括：可塑表示是否带来额外可分性；共享先验是否超越相同表征与相同正则强度的 ridge；估计运输误差会否被经验贝叶斯放大；重置是否丢失有用的优化连续性；样本质量不平衡、共享秩和噪声尺度是否导致负迁移；以及实际计算与存储成本。原固定模块库的学习、跨任务复用和其他可能的表示更新规则仍是研究空间的一部分，本路线只是新增可检验实例。
