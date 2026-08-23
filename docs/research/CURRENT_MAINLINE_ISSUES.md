# ClearVLA 当前主线纯问题账本

当前源码身份：Schema35 `object_intent_dynamics_323`。行为比较锚点是 V120
`long` 与提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`；Schema34 只用于识别
新回归，不是正确性目标。

本文件只保留尚未被源码代数和回归测试关闭的问题。已修复的 G 重复计票、Teacher
dustbin/status 混淆、camera 几何压平、S 绝对均值目标、W far->near、动态 P1 事实污染、
P3 temporal 事实旁路、在线 uncertainty 无消费者以及静态 P1 日志别名均已删除，不作为
“历史问题”继续堆放。

## 记账规则

- 源码可证明的问题与实验可识别性问题分开。
- “有梯度、张量非空、接口有名字”不等于策略使用。
- 只记录当前问题；历史版本只作为归因证据。
- 问题关闭后从本文件删除。未完成八轮或冻结干预时不得声称净收益。

## O-01：G3 条件-K 校正的实际贡献尚未完成因果确认

**级别：P1/P2。类型：实验可识别性风险，不是当前确定性断路。**

Schema35 保留了 G3 bounded conditional-K residual，并从构造上保证 residual 为零时精确继承
G2、不能改变 object-vs-null mass。结构测试已证明 owner/fact 图连通，但现有训练日志仍可能
出现很小的 G3 parent L1。

关闭边界：在 Schema35 checkpoint 上分别测 G3 owner、最终 fact 与 action JVP/zero-shuffle；
只有 owner 改变却 fact/action 不变时，才把 G3 归类为实际冗余。不得为追求幅度放大 residual、
增加配额或额外 loss。

## O-02：learned flow 仍可能只是低置信几何先验

**级别：P1/P2。类型：数据与几何可识别性风险。**

Schema34 末期 flow confidence 虽升至约 `0.23`、correlation entropy 降至约 `0.73`，但历史
zero/shuffle 探针曾显示 flow 对 action 的影响很小。Schema35 已让 Teacher/W/P2 全程保留真实
camera 轴，并让低匹配置信度通过 identity-null/covariance 扩大几何不确定性；这解决了错误
放大，不证明 learned flow 已提供动作增益。

关闭边界：完成 learned-flow zero、spatial shuffle、DINO-key shuffle 的分离探针，同时观察
Teacher transport、P2 coordinate posterior、consequence 和 action。零流合法；不得加入非零流
配额、gain 或硬使用约束。

## O-03：S/W 的字段区分与策略净使用仍需长跑验证

**级别：P1。类型：结构已闭环后的可识别性风险。**

Schema35 将 S 改为相邻区间增量目标，将 W 改为 W1 near、W2 far 的单向因果字段，并让现有
generic W 容量只通过受监督 typed owner 出口进入 P2。静态测试可证明 `d(near)/d(far)=0`、
`d(far)/d(near)!=0`、neutral W 的 P3 effect/temporal 为零；尚不能证明真实数据会学出足够的
interval/object variation，也不能证明 W effect 对 action 为净正收益。

关闭边界：完整比较四区间 prediction/target adjacent cosine、common/residual、object variation，
并做 S public/typed、W semantic/geometry/status、W common/residual 的 matched interventions。
W zero 不得改善动作误差；若边界全部正确但 action 无净收益，归类为数据可识别性，而不是继续
增加 W gain、容量、门控或辅助 loss。

## O-04：Schema35 完整性能、后期反弹与资源门尚未验证

**级别：P0 放行门。类型：实验未完成。**

本地回归只能证明实现边界、有限性和生命周期，不能代替生产 batch-8 训练。需要 fresh smoke
和八轮长跑验证：

- epoch-1 physical RMSE 不高于约 `0.1006`；
- 最终 physical RMSE 不高于 V120 `0.0814`，最佳点不差于 `0.0793`；
- first/tail、arm/gripper、event/motion 与 horizon bands 不互相牺牲；
- epoch 6-8 不出现比 V120 更严重的反弹；
- P3 precision、static/dynamic P1、S/W JVP 可分别定位；
- batch 8 总显存不超过 22 GB，且相对 Schema34 约 11.9 GB 增量不超过 1.5 GB；
- 中位吞吐不慢于 2.2 s/batch。

完成 smoke、epoch-1 和八轮 gate 后，再删除或深化本条。
