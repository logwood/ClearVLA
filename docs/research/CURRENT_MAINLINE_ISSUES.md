# ClearVLA 当前纯问题账本

更新：2026-08-14

本文件只保留尚未解决的问题。已经由源码确认并修复的活跃 observation 参数误冻结、
Teacher 跨相机坐标代数、global-K public-key 二次注入以及 validation 采样/消融口径
已从问题账本删除；其实现边界记录在
`00_CURRENT_ARCHITECTURE_CONTRACT.md`。本轮闭环审查没有发现第五个与这四项同级、
可由现有源码和日志独立证明的主路故障。这个结论不等于剩余行为问题已经消失。

## P0：修正后的 Schema24 尚未通过 fresh 行为恢复验收

旧 Schema24 epoch-1 checkpoint 产生于错误图，不能 exact-resume 到当前源码。当前版本
必须使用新的空输出目录完成：

- BF16 smoke、五步部署与 endpoint heads；
- batch 8 总进程显存不超过 22 GiB；
- 对齐 batch 2200 的 V120/旧 Schema24/当前 fresh-run 比较；
- 八个 epoch 的最终点和完整均值比较，不能只取 best checkpoint。

早期恢复至少同时检查：G3 parent L1、object pair cosine、typed posterior L1、P1
protected detail/spatial variation、Teacher transport/covariance、P2 null/effect、
consequence 与 P3 effect。动作端必须比较 physical/native、first/tail、三个 horizon、
arm/gripper、decoded event、event head 和 motion head。

## P1：W→action 在旧 Schema24 中是严重衰减，fresh run 后复核

旧日志并非 autograd 断线：P2 与 consequence action-only 梯度非零；但错误 Teacher
几何使 P2 合法地偏向 null，validation 中 effect/consequence/P3-effect 接近消失，策略
退回 P1+temporal。当前修复去除了已确认的上游原因，但尚无新实验能证明动作端已恢复。

关闭条件：在相同数据、seed、batch 和对齐 iter 上，Teacher transport 回到合理量级，
P2 null 下降、effect/consequence/P3-effect 恢复且 action 指标同步改善。仅有 future loss
下降、W 梯度非零或 representation 改变不算关闭。若修正后 W zero 仍改善 action，才把
它升级为新的 W/P2 结构任务；此前禁止重写 P2/P3/bottom、取消合法 null、增加 gain、
quota、硬门控或人工梯度。

## P1：S typed innovation 的部署条件依赖仍未判定

旧 Schema24 训练末段 typed innovation 约 `0.08–0.16`，但首四个 validation 诊断 batch
约 `1.29e-4`；goal/history/object innovation 同时仍非零。这是旧子集上的真实边界输出，
但旧采样只覆盖验证集开头约 2.23%，不能外推到全验证集。

当前验证已改为在全 loader 上均匀抽取 16 个 sampling diagnostic batches。fresh run 后需
按完整 goal/history、各自 dropout 和 train-mask/eval-mask 对同一固定 batch 做无梯度
对照，并记录 typed null/source mass。若完整条件下仍稳定归零，才确认
condition-dependent routing shortcut。不得通过 typed gain、熵目标、非零配额或删除
null 制造使用率。

## P1：P1 protected detail 的 train/eval 衰减仍未判定

旧 Schema24 训练末段 `p1_protected_detail_rms≈0.03–0.04`，首四个 validation 诊断
batch 约 `0.0104`，而 dynamic delta 仍约 `0.385`。静态源码复核已经确认当前 P1 仍
完整执行 24 factual queries、四种 glimpse、N=49 posterior 和真实 3×3
RGB/detail/coordinate microgrid；`FactualPrecisionDock` 只是参数自由的已计算结果边界，
不是粗暴替代 reader。

先在恢复 observation trainability、移除 public-key 注入并采用均匀 validation 采样的
fresh run 上复核。如果同一输入的 train-mask/eval-mask 仍产生异常 detail 衰减，再定位
mask/selector 输入分布；此前禁止改 P1、缩小 microgrid、增加 detail gain 或再造 reader。

## P1：G/S/W/P3 的可识别性只能由修正后的完整实验判定

旧日志中的 K 槽公共化、三类 typed posterior 相似、W 比 Teacher 更公共、P3
temporal/history 偏置，可能由本轮已修的冻结/key/Teacher 问题造成，也可能包含独立的
数据可识别性问题。静态 closure audit 已再次核对当前 P2、P3、RoleDeltaAttnRes、
transition 与 Evidence bottom 的代数，没有发现新的断线或替代实现，因此本轮不修改
这些成熟模块。

只有在 G1/G2/G3、P1、Teacher 和对象几何边界全部恢复，且八轮仍复现公共化时，才把
它归类为监督/可识别性任务。届时必须基于完整轨迹和 causal intervention，而不是靠
额外 loss 或接口命名判断所有权。

## P2：保留但不进入当前修复的债务

- `FutureObjectDynamics.future_address [B,4,K,C,8,8]` 当前只有诊断消费者；不把它
  接回 P2 来制造相机轴或空间使用率。
- P1 learned null 明确延期。若未来实现，protected base 必须在竞争外、null value
  必须代数零、且只能抑制 optional detail innovation。
- HistoryActionProposal 仍是监督辅助项；future proposal token 不进入主路，真实 executed
  history 仍通过共享 V120 seed 条件化策略。只有完整消融证明该辅助项无益且昂贵时再处理。

## 失败归类

- forward/shape/dtype 失败：源码边界问题，停止长跑；
- non-finite：使用 `gradient_failure` 首个参数记录归因，不先增加 clip；
- 显存/速度失败：按静态/动态调用次数和模块清单归因；
- 边界测试正确但八轮性能失败：数据、监督或可识别性问题；
- 不把“梯度存在”“张量非空”或“辅助 loss 下降”当作主路有效证据。
