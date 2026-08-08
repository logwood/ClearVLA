# ClearVLA 顶层问题账本

更新：2026-08-09

对象：`object_intent_dynamics_323` schema 4，默认实验标签 V122。

范围：Pre-G 之后的 G / S / Teacher / W / P1 / P2 / P3、相邻 loss、静态缓存和 bottom ingress。底层 Evidence MMDiT、CVAE、workspace 与 execution controller 不在本轮重做范围。

当前目标图和不可违反的边界以
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)
为准。本账本只保留仍能指导实现或实验判断的信息；V117–V120 的已失效局部契约和重复历史已清除。

## 1. 本轮证据与结论边界

直接证据来自：

- 当前工作区源码；
- 完整 V117、V118、V119、三条 V120 日志；
- V121 完整 epoch 1、验证与 epoch 2 训练段；
- 冻结 checkpoint 的历史路径干预结论只在当前边界仍相同时引用。

V121 epoch 1 的 action RMSE 为 `0.10004`，同口径 V120 为 `0.09762`。V121 的 first `1–4` 略好（`0.04018` 对 `0.04122`），middle 近似持平，tail 更差（`0.12609` 对 `0.12219`）。因此 V121 是定位实现问题的证据 run，不是性能改进基线。

V121 最关键的非 RMSE 证据是：W prediction 的区间差异弱于 teacher，P2 null 逐渐升高，P3 temporal 的幅度/梯度压过 precision，global K 后 typed evidence 的字段差异很弱，而 P1 的 micro/detail 读取本身非零。这些量与源码共同把问题定位在 G 后的语义边界，而不是“底层没训练”或“P1 没读细节”。

## 2. 已完成的 schema-4 源码修正

| 原问题 | 源码根因 | schema-4 修正 | 当前状态 |
| --- | --- | --- | --- |
| `P2-DELTA-CALIBRATION` | 将零中心 visibility/persistence change 和正 uncertainty 直接乘候选先验，静态正确预测反而偏向 null | 只让 physical camera validity 决定合法性；status 去公共模后仅作为有界相对排序分数，null prior 不受其单向惩罚 | 源码已修，待 V122 验证 |
| `CAMERA-COORDINATE-COLLAPSE` | G 后把双相机坐标、transport 和 P1 source coordinate 压成 `[K,2]`，随后再对各相机广播 | `ObjectFactSet`、Teacher、W geometry、`ObjectFactualDock`、P2 geometry 全程保留真实 `[K,C,*]` | 源码已修，shape/等变测试通过 |
| `FUTURE-LOSS-UNIT-IMBALANCE` | content、normalized coordinate、covariance、status 用 raw unit 混合；有效优化压力几乎由 content 独占 | 七个 future 字段先按 detached variance floor 变为无量纲误差，再使用原内部系数；外部 future 总权重不增加；native-unit error 只记日志 | 源码已修，零目标测试通过 |
| `P3-PRECISION-REDUNDANCY` | 同一 aggregate P1 fact 已在 protected consequence 中，却又经 `precision_fact` 和 cumulative consequence 重投影 | precision 只读取现成 `ObjectFactualDock.fact_by_object` 的 K-centred 细节；`effect+interaction` 只条件化 query；aggregate fact 不再作为 optional value | 源码已修，代数零测试通过 |
| `P3-TEMPORAL-FUNCTIONAL-BYPASS` | learned temporal identity 与 action query 即使无 S/W innovation 也能产生 temporal value | P3 只消费 `temporal_innovations` 与 `effect+interaction`；action 只能乘性调制，不能合成 value | 源码已修，query-only 零语义测试通过 |
| `S-INTERVAL-COMMON-MODE` | online S 非 causal；online/recognizer 匹配包含 identity 的累计 state；更深一层是 CrossRead FFN 与 self-attention V 仍可从 query identity 制造“update” | online/recognizer/CoarseAction 统一 causal；Q/K identity 与 V innovation 在算子内部拆开；CrossRead FFN 只处理真实 attention update；训练只做 innovation-to-innovation matching | 源码已修，query-only 零语义测试通过 |
| `W-STATE-COMMON-MODE` | full state/action carrier 作为 additive/broadcast W value；W interval/decoder identity 也可形成 object-free 默认方向 | W 只读 S/CoarseAction innovation；state 必须与 object K/V 乘性交互；action/interval/decoder identity 只能调制当前 object-owned value | 源码已修，零条件回到 object base 测试通过 |
| `G-TYPED-UNDERIDENTIFIED` | semantic/appearance/geometry verifier 只有远端下游压力，G reconstruction 只约束 DINO content/position | 保留唯一 physical K；在原 G reconstruction 预算内加入三类 target-normalized typed-field consistency，不强迫三 posterior 不同 | 源码已修，普通 autograd/字段 loss 测试通过 |
| `COARSE-ACTION-INTERVAL-UNKNOWN` | 只有总 loss/RMS，且 query carrier 可进入 W | CoarseAction 改为 causal innovation-only；新增 interval variation、adjacent cosine、target-normalized error | 源码已修，待日志验证可识别性 |
| `ATTENTION-ENTROPY-SEMANTICS` | `object_H/semantic_H/...` 是事后 cosine-softmax audit，却被命名为真实 attention | V122 改名为 `*_sim_H`，内部 canonical key 明确为 `audit_similarity_entropy` | 日志语义已修 |

## 3. 修正后的真实张量边界

```text
G physical object:
  one K+null assignment
  semantic / appearance / geometry = typed evidence reads, not three identities
  camera_coordinates / camera_transport / camera_validity [B,K,C,*]

S / recognizer / CoarseAction:
  learned identities = Q/K/index only
  action/state/object/temporal/coarse innovations = legal downstream values
  online and training target share the same ordered causal interval structure

W:
  protected current object + object-conditioned intent/action modulation
  state can only modulate object K/V
  W1 = 4–8, 8–16; W2 = 16–32, 32–48
  only supervised FutureObjectDynamics crosses W→P

P1/P2/P3:
  P1 reads high-resolution observation once and exports one ObjectFactualDock
  P2 semantic selects [I,K]; geometry selects [I,K,C]
  consequence = protected P1 fact + effect + interaction
  P3 precision = centred unresolved K detail
  P3 temporal = S temporal innovation + consequence innovation
  protected consequence enters bottom exactly once
```

## 4. 当前日志口径

V122 需要重点观察：

- G：`typed_consistency` 与三字段 consistency、physical K pair cosine/chart overlap、三 typed posterior L1、camera coordinate variation；
- S：action/state/temporal innovation RMS 与四区间 variation；`*_sim_H` 只是 read-only similarity audit；
- CoarseAction：innovation RMS、interval variation、adjacent cosine、target-normalized error；
- W：condition interaction、state-object interaction、object K/V innovation，W1/W2 interval/object cosine；
- loss：每字段优化 loss、每区间 target-normalized error，以及单独的 native-unit error；
- P2：semantic/geometry null、`relative_status_abs/mean`、相机几何 interval mass、effect RMS；
- P3：centred-detail RMS、consequence-innovation RMS、precision null、precision/temporal/state-change RMS；
- action：完整 train/val loss、first/middle/tail、arm/gripper/event，不以单个 RMSE 下结论。

禁止再把下列现象单独当作“接通”：张量非空、梯度非零、RMS 非零、接口名字不同或 attention audit entropy 不为 1。边界干预必须先改变自己的 typed state，再改变下一消费者，最终 action 变化的置信区间脱离零后才能声称有策略收益。

## 5. 尚未由源码修正自动保证的实验问题

以下问题仍需 V122 smoke/长跑，而不能靠本次静态修正提前宣布解决：

1. **global K 是否在真实数据上形成稳定对象**：字段一致性提供近端压力，但数据可能仍不足以唯一识别四个 object slots；不使用 forced diversity、slot quota 或 entropy target。
2. **W 四区间是否达到 teacher 的可识别差异**：接口已阻断公共 carrier，但未来变化本身可能弱；比较 prediction/target variation、adjacent cosine 和四区间 normalized error。
3. **P2 是否真正把 W effect 传给 action**：修正 null 语义不等于必然产生动作增益；需要 effect zero/shuffle 的 boundary→consequence→action 因果链。
4. **tail/event 是否恢复**：V121 的时间退化与旧旁路相容，但也可能包含 gripper/event 分布因素；必须看至少三个验证点。
5. **learned flow 的 action 增益**：本轮修复了它的相机几何落点，没有设置非零流配额；仍需独立 zero/spatial-shuffle 干预。

若上述边界探针正确、W normalized error 确实下降，但 action 仍无增益，应归类为数据可识别性或任务收益问题，而不是继续增加 contract、硬门控或辅助 loss。

## 6. 明确排除与禁止回归

- 不恢复 scalar progress、phase label、completion terminal、LSTM cache 或训练期未来值到部署路径。
- 不把 local M 当 global K；不混用 prior、allocation、existence 和 validity。
- 不 reduce 后 `expand` 伪造 object、interval、camera 或 type 轴。
- 不加入 hard gate、route quota、固定 entropy、forced diversity、forced flow 或人工梯度。
- 不削弱 P1 的一次高分辨率事实读取，不重新打开第二次 RGB/DINO bank。
- 不增加 `_validate_vXXX_*` 或按版本号选择源码；manifest 只记录 capability schema。
- 不恢复未监督 public W residual、重复 P1 fact、cumulative temporal identity 或第二个 protected-consequence bottom ingress。

## 7. 验证状态与下一步

本地已通过：

- schema-4 manifest/reject；
- G mass、typed consistency、camera-axis shape、object permutation；
- Teacher FP32/no-grad 与 future-target/action isolation；
- S/CoarseAction query-only 零语义；
- W object-owned zero-condition；
- P2 common-status offset invariance、typed selector 独立性与有界 score；
- P3 centred-detail/temporal 代数零；
- 完整 capability BF16 forward/backward、optimizer ownership、teacher-forced boundary 与五步静态缓存；
- 日志解析回归。

仍需服务器 fresh smoke 和 batch-8 长跑。默认命令见当前架构契约；schema 3 的 top checkpoint 不允许 resume 到 schema 4。
