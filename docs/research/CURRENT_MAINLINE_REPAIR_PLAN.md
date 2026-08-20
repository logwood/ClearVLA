# ClearVLA Schema26 G3/S 边界受控修复

状态：**源码实施与本地回归完成；CUDA smoke、batch-2200 和八轮行为 gate 待执行**
更新：2026-08-20

本计划只处理 Schema25 源码审计已经确定的四个边界问题：最终 G3 anchor 在 transition 前被伪轴替换、S typed 区间被公共分量淹没、同一 typed evidence 两次进入 W、以及无消费者诊断路径继续消耗计算并制造错误归因。行为基准始终是 V120 `long`、提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c` 与 `.audit/v120_exact_source_0b92d359/`。

## 一、已实施范围

### 1. 精确 G3 rollout 进入 transition

- P1 与 `ControlledTransitionSource` 读取同一个 completed-G3 rollout `[B,4*C*8*8,H]`。
- 删除 transition 内部由 public chart 复制四次再添加 `interval_identity` 的伪 anchor 构造。
- transition core、动态 noisy-action 读取、CVAE、workspace、Evidence MMDiT 与 execution 均未修改。
- 新日志区分 source spatial variation 与 anchor variation；shape=512 不再被当成语义接通证明。

### 2. S common/differential typed selector

- typed selector 保留真实 `[interval,K,type]` 轴。
- 对每个 type，使用同一组 bias-free 投影分别读取：

  ```text
  common = mean(interval carrier)
  differential = interval carrier - common
  ```

- common score 与 differential score 在原 `[-1,1]` 余量内有界组合；完全相同的四区间得到精确零 differential，不会被人工制造差异。
- differential 使用现有 0.25 variance floor 的零保持归一化，避免近零区间残差被普通 LayerNorm 放大；common 分支保持原数值分布。
- semantic、appearance、geometry 仍分别与固定零 null 比较；没有 entropy、mass、diversity、progress loss 或人工梯度。
- 日志把各类型 selector 主动 null 与共同的事实 unsupported 分开，避免再次把二者混称为 `null_mass`，也不伪造不存在的 type-specific validity。

### 3. typed evidence 只通过一条边进入 W

- `ActionIntentDock` 与 CoarseAction 只消费 public interval、public object 与 observable history。
- 完整 `[interval,K,type]` typed relevance 只通过 `WorldIntentDock` 进入 W。
- 已有的 reduced `typed_policy_components` 仍服务既有 P1/P2/P3 policy context；没有把完整 typed object value 复制给这些消费者。
- W1/W2 block、四区间预测头、FutureObjectDynamics、future loss 与 P2/P3 均未重写。

### 4. 删除无消费者路径，不删除真实能力

- 删除 Teacher/W 导出但 loss、P2、action 均不消费的 `future_address`，同时删除 W 的在线 `grid_sample` 平移计算。
- 保留 Teacher 内部真实候选 posterior、same-camera transport/covariance moment 与 object-level geometry。
- 删除 bit-exact no-op 的 proposal deployment ablation 和相应配置/日志；保留 proposal dropout 的 owned-generator 抽样节奏，以维持 V120 RNG cadence。
- history proposal auxiliary loss 与 compressed executed-history condition 均保留。

## 二、明确未改

- Observation/Pre-G、G1/G2/G3 progressive updater 和 dense grounder内部；
- V120 P1 的 24 queries、N=49、四 glimpse、3×3 microgrid；
- P2 bounded routing、zero-preserving consequence、P3 five lanes；
- transition 动态部分及全部 bottom 主路；
- action/future/flow/grounder/execution 外部 loss 权重；
- AdamW ownership、decoder-local→global clipping、五步节点与 endpoint heads；
- 数据、DINO cache、T5、batch、worker 与 normalizer 默认值。

本轮没有新增 block、head、loss、gain、quota、gate、learned null 或容量。Schema26 参数为 `169,979,847 total / 153,585,526 trainable`；相对 Schema25 唯一的 `-2,048` 参数差是删除 transition 的四行 pseudo-anchor identity。

## 三、已完成的静态验收

- P1 forward pre-hook 捕获的 G3 rollout 与 transition selector 张量完全相同。
- transition 不再拥有 `interval_identity`，并强制检查 `[B,4*C*8*8,H]`。
- identical interval carrier 的 differential score 精确为零；真实 interval 扰动能改变 typed selector，combined score 始终有界。
- differential normalization denominator 始终不低于 0.25，新增分支不存在无界近零 Jacobian。
- CoarseAction interface/source 均无 typed 字段；清零 typed relevance 时 CoarseAction bit-exact，W 输入发生变化。
- `future_address` 与 proposal ablation 在 active mainline/config/runtime 中无残余消费者。
- Schema26 日志审计不再要求已删除的 proposal ablation；Schema25 及更旧日志的历史字段仍可读取。
- manifest/checkpoint identity 升至 Schema26；Schema25 不能 exact resume。

## 四、尚未关闭的行为问题

详细证据只保存在 [`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md)：

1. Schema26 是否关闭 Schema25→V120 的早期 action/native gap；
2. S typed interval variation 是否不再被 common mode 淹没；
3. W 是否仍只保留约 34% 的 teacher interval variation；
4. epoch 7/8 gripper 与 5–24 步反弹；
5. 生产 batch-8 显存、吞吐与生命周期。

这些都需要新日志，当前源码测试不能替代。若 S 边界改善而 W 仍欠拟合，应把它归类为独立 W 可识别性问题；不得继续修改 bottom 或用增益强迫 W 变大。

## 五、放行顺序

### A. fresh smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema26_g3_s_boundary_smoke \
nohup bash scripts/smoke_mainline.sh > schema26_g3_s_boundary_smoke.log 2>&1 &
```

要求：BF16 forward/backward、finite gradient、五步部署、endpoint head、Teacher isolation、checkpoint 写入全部通过；本地微型测试小于 8 GB。

### B. batch-8 早期 gate

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema26_g3_s_boundary_b8 \
nohup bash scripts/train_mainline.sh > schema26_g3_s_boundary_b8.log 2>&1 &
```

对齐 batch 2200 比较 V120，不与 Schema24/25 自我对齐。至少同时检查：action/native/decoded、arm/gripper、G3 anchor source、S common/differential typed、W prediction/teacher interval variation、梯度、seconds/batch 与 process peak。

### C. 八轮 gate

早期 gate 通过后完成八个 epoch，比较每轮验证、最佳点与最终点。不能只报告 RMSE，也不能用最佳 checkpoint 隐藏末期反弹。

## 六、失败时的处理边界

- G3 source 未对齐：只回到 G3→transition 边界，不改 transition core/bottom。
- S differential 仍近零：先核对真实 public interval innovation 和梯度，再判断数据可识别性；不加 diversity/entropy loss。
- W 仍公共化但 S 边界健康：作为独立 W 问题重新审计，不提高 gain/loss 掩盖。
- action 仍落后而各边界均正确：转为优化/数据可识别性归因，禁止再次凭猜测重构 P1/P2/P3 或 bottom。
- 显存超标：先核对缓存生命周期与重复构建；不得以压低 P1 细节或移除 bottom 能力换显存。
