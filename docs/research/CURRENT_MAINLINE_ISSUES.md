# ClearVLA 当前未解决问题

更新：2026-08-13

本文件只记录当前尚未解决的问题；完成的源码修复直接删除，不在这里建立
版本契约。恢复参考是 V120 `long`，commit
`0b92d359a2889a0a1b1eba256007c00ccbc54f3c`。Schema 22 已被实验拒绝，
不得 exact resume；当前源码身份是 Schema 23。

## P0：Schema 23 尚未完成生产实验验收

Schema 23 已在源码中完成以下受控恢复：mirrored flow-time、五个 V120
action-update nodes、clean endpoint event/motion heads、`1+4=5x` event boost、
无 entropy/reliability 收缩的唯一 Teacher 目标、`current_loss_support` 与
`future_selector_validity` 分离，以及首个 non-finite named-parameter 哨兵。
这些修复没有改变 G/S/W/P block 数、外部 loss 权重、P2 gain、bottom 或参数量。

仍需完成：

- fresh CUDA BF16 smoke；
- batch 8 峰值显存不超过 22 GiB；
- 完整八个 epoch，与 V120 比较训练及全部验证点；
- 同时检查 action/native、first/tail、三个 horizon band、arm/gripper、
  event/motion、Teacher target scale、G/S/W/P、梯度、速度和显存；
- 禁止用最好 RMSE 掩盖第三轮以后反弹或 gripper 保守化。

Schema 22 的拒绝证据仍是恢复 gate：E1/E2 first RMSE 约为 V120 的
`3.51x/3.14x`，tail 约为 `2.91x/2.86x`；尾部 G content pair cosine
约 `0.742`，W2 adjacent cosine 约 `0.965`，P2 geometry mass 约 `0.025`，
consequence effect 约 `0.0215`。Schema 23 至少需要显著离开这些退化状态。

## P0：首个 non-finite owner 仍需由下一次真实失败给出

- Schema 22 只能证明 non-finite 出现在 backward 后、optimizer 前，不能
  证明 QR、W 或任何单一模块是根因。
- Schema 23 已在全局裁剪前记录首个异常参数、role、optimizer group、
  shape/dtype、finite fraction/max-abs 以及 NaN/+Inf/-Inf 数量；失败批次
  不执行 optimizer、scheduler 或 global-step 更新。
- 若不再复现，此项自然关闭；若复现，只沿报告出的 owner 做局部 backward
  定位。不得提前增加 per-owner clip、人工梯度或幅度门控。

## P1：恢复后再判断 G common-mode 与 W/P2 动作效益

- Schema 22 的 G object common-mode、W 对象/区间公共化、P2 status 主导和
  consequence effect 偏小是真实现象，但此前与错误 Teacher 目标及失效
  future selector 混在一起。
- Schema 23 长跑后先看 slot separation、typed intervention、W target/pred、
  P2 type mix、consequence JVP 和 action 因果效应；边界变化和 action 变化
  必须同时成立，才可称 W 被策略使用。
- 若这些问题仍在，再进入下一轮结构设计。目前不增加 gain、quota、hard
  gate、额外 loss、block 或容量，也不重写 G/S/P1/P3。

## P1：保留但不混入本轮的诊断债务

- `future_address` 当前无 loss/P2/action consumer，是 dead diagnostic compute；
  它不是 Schema 22 崩溃根因。本轮保留接口以避免扩大行为变更。
- `proposal_keep` 仍是 future-proposal no-op 诊断；真实 observable history
  condition 由 action-history 路径承担。本轮不清理。
- 这两项以后可单独删除或重命名，但必须证明数值与 checkpoint 边界不变。

## P1：P1 learned null 仍然延期

learned null 可以表达当前证据不足，但不能成为关闭精细读取的 shortcut：

- protected factual base 位于竞争外；
- null value 精确 identity/zero，不携带 learnable payload；
- 只抑制 optional detail innovation；
- prior 只来自当前可观测 support/conflict；
- bounded correction 最多读取 clean goal/action-basis query，不能读取 noisy
  action、future teacher 或直接 policy carrier；
- 不使用固定 null mass、entropy target、硬选择或人工梯度。

Schema 23 不实现新 P1 learned null。只有 recovery gate 通过且 P1 仍有明确
证据需求时，才作为独立研究项。

## 下一次运行

先用空目录跑 smoke；通过后才开 batch 8 长跑。Schema 22 checkpoint 不得
resume，正式实验也不使用 bottom migration。
