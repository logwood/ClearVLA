# ClearVLA 当前未解决问题

更新：2026-08-12

本文件只保留当前主线尚未解决的问题。已完成的源码修复不在这里保留
“完成记录”；其历史证据见
[`TOP_ARCHITECTURE_ISSUE_LEDGER.md`](TOP_ARCHITECTURE_ISSUE_LEDGER.md)。

当前默认参考是 V120 `long` 的真实执行数据流。Schema 21 已恢复这条
源码路径，但尚未经过服务器 smoke 和完整长跑，因此下面的项目仍不能
仅凭本地单测判定为解决。

## P0：V120 行为恢复尚未得到实验验证

- 需要 fresh BF16 smoke、五步部署和 batch 8 正式训练。
- 早期至少对照 batch 20/100/200/340 的 action/native/arm/gripper、
  global preclip、transition owner、S interval/temporal、W、P1 与 P3。
- 必须完成全部 8 epoch，并比较 E1、每个验证点、最好点和 E8；不能用
  一个 best RMSE 掩盖 tail、gripper 或后期反弹。
- 若源码路径恢复后仍从 V120 曲线分叉，先按真实 producer/consumer
  定位，不再新增 contract、gain、gate、loss 或版本分支。

## P1：global clip 是否仍被 transition 挤占

- Schema 20 的 transition 曾消耗约 79% 的 postclip 平方预算；这是错误
  静态 transition 和 source-bank 漂移的连锁结果。
- Schema 21 已恢复 per-ODE noisy-action transition、learned neutral 和 V120
  bottom source bank，但是否回到 V120 同量级只能由正式日志确认。
- 禁止用 per-owner clip、人工梯度或任意缩放提前掩盖这个问题。

## P1：W 的实际动作效益仍待恢复后判断

- 历史日志证明 W 不是完全断开，但其对象/区间差异弱于 teacher，且最终
  action 效益不足。
- Schema 20 中这个现象被错误 transition、P3 公共旁路和 P1 压缩混在
  一起；这些消费者现已恢复到 V120 语义。
- 新日志需要同时看 W target/prediction、P2 effect、consequence、P3 effect
  与 action 梯度/干预。只有边界和 action 都变化，才能声称 W 有效。
- 在此之前不增加 W loss、effect gain、route quota、hard gate 或强制差异。

## P1：public observation 的固定混合仍未独立验证

- 当前 `public_scene_base` 固定组合 current content、flow-aligned visual
  innovation、recent motion 与 earlier motion。
- 三帧、两段 flow、方向和单位的静态边界成立，但尚无独立证据证明这种
  固定混合优于 V120 的 source 组织方式。
- 它不是 Schema 20 早期退化的首因，当前不继续改。若正式恢复仍失败，
  先补 source-wise RMS/JVP 和分源消融，再决定是否调整。

## P1：生产显存与吞吐尚未验收

- 本地 CPU/BF16 和静态调用频率不能替代 CUDA 测量。
- batch 8 总进程峰值要求不超过 22 GiB；P1 的 49 候选必须保持分块
  checkpoint，teacher 每 batch 一次且部署为零次。
- 同卡、同 batch 的窗口中位数/p90 需与 V120 对比。明显加速若来自少算
  active 分支，不算优化；明显变慢则先定位 teacher、P1、ODE 动态分支和
  validation intervention 的实际耗时。
