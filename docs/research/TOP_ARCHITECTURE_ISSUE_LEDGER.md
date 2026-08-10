# ClearVLA 顶层问题账本

更新：2026-08-10

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
- 完整 V122 八轮长跑（22,768 train steps，63/5 train/val episodes）；
- 冻结 checkpoint 的历史路径干预结论只在当前边界仍相同时引用。

V121 epoch 1 的 action RMSE 为 `0.10004`，同口径 V120 为 `0.09762`。V121 的 first `1–4` 略好（`0.04018` 对 `0.04122`），middle 近似持平，tail 更差（`0.12609` 对 `0.12219`）。因此 V121 是定位实现问题的证据 run，不是性能改进基线。

V121 最关键的非 RMSE 证据是：W prediction 的区间差异弱于 teacher，P2 null 逐渐升高，P3 temporal 的幅度/梯度压过 precision，global K 后 typed evidence 的字段差异很弱，而 P1 的 micro/detail 读取本身非零。这些量与源码共同把问题定位在 G 后的语义边界，而不是“底层没训练”或“P1 没读细节”。

完整 V122 把若干待验证项变成了实际结论：

- validation RMSE 为 `0.09760 -> 0.08914(best, E6) -> 0.09109(E8)`；同数据、seed、batch 的 V120 best/final 为 `0.07931/0.08145`，因此 V122 的最佳点仍差约 `12.4%`，不是仅由最后两轮反弹造成。
- `1-4 / 5-12 / 13-24` action-band RMSE 从 E1 的 `0.04064/0.07355/0.12205` 到 E8 的 `0.02672/0.06479/0.11644`。近端改善约 `34%`，远端只改善约 `4.6%`；tail/first 从 `3.31` 升到 `9.24` 主要是 near 继续拟合而 far 停滞，并非 tail 数值突然爆炸。
- decoded gripper event 从 `1424/1357` 变成 `438/1357`，precision `0.365 -> 0.591`、recall `0.383 -> 0.191`、F1 `0.374 -> 0.289`；motion F1 仍约 `0.81`。这是夹爪逐渐保守化的独立问题，不能用整体 action loss 下降掩盖。
- global-K content pair cosine `0.493 -> 0.710`，chart pair overlap `0.282 -> 0.575`；semantic/appearance/geometry posterior 的总变差距离仍只有约 `0.02-0.04`。四槽没有数值死亡，但越来越共享同一内容和空间支持，真实数据没有验证“稳定对象”假设。
- W 的 supervised content/transport/covariance/visibility/persistence/uncertainty 项均下降；prediction interval variation `0.030 -> 0.082`，但 teacher 为 `0.106 -> 0.139`，prediction adjacent cosine E8 仍约 `0.918`、teacher 约 `0.821`。因此 W 并非未训练，而是比 teacher 更公共化。
- W coarse-action innovation RMS `0.805 -> 1.595`，S interval innovation `0.573 -> 0.769`，而 condition interaction 只到约 `0.211`。源码先相加再 `tanh`，所以 action 大幅值会降低较小 S innovation 的局部灵敏度；这是一项当前候选仍需处理或由首轮日志否证的尺度风险，不等价于“coarse action 无用”。
- learned flow 的 E8 moving/static warp gain 为 `+0.0618/+0.0227`，明显优于零流；但 global-object transport prior RMS 从约 `0.0149` 降至 `0.0054`。几何 flow 有信号，旧 G 落点使用得弱。候选已经修正 flow frame/time units 与 RGB anchor，不能据此删除 flow。
- V122 的自由 uncertainty NLL 在尾部形成约 `-0.01091` 的负抵消，execution value 约 `+0.01065`；这两项几乎互相抵账。候选已经把 uncertainty 改为 detached photometric-dispersion 回归，并删除 differentiable execution candidate replay。
- 正式 V122 实际参数账为 `230,717,082 total / 168,064,059 trainable`；旧文档的 `227,466,394 / 166,360,123` 是 synthetic/default-width 口径，不能再用于主线迁移差额。

这些结论不授权加入 slot diversity、route quota、progress/entropy loss 或增大 future 外部权重。V122 没有 W-specific frozen action intervention，因此“W 对最终 action 有正收益”仍未被证明；只可把 G 槽公共化、W 条件尺度、远端动作和 gripper event 作为下一主线的明确验收项。

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
| `GRAD-AUDIT-SCHEMA-DRIFT` | P3 重构后梯度诊断仍访问已删除的 `precision_fact/precision_consequence`，且漏掉新增的 W object/camera 参数；真实训练会在第一次诊断挂钩时退出 | 梯度分组改为 schema-4 实际成员，并让完整 BF16 系统回归在 backward 后强制执行 `_attach_grad_diagnostics` | 源码与回归测试已修 |

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

## 8. 主干重构问题账

这组问题是本轮独立 `clearvla/mainline/` 重构必须消除的源码缺陷，不能再用新版本号或附加 validator 覆盖：

| 问题 | 当前源码证据 | 新主干要求 |
| --- | --- | --- |
| `LAUNCHER-INHERITANCE-GRAPH` | V122 正式入口向下继承 25 层 shell，累计约 1504 行、566 次 CLI 参数赋值和 191 个环境变量导出 | 一份可序列化的 active experiment spec；主机路径允许少量显式 override；run label 不选择源码 |
| `CONFIG-FLAG-ACCUMULATION` | V39 policy config 连同父类约 399 个字段，训练 CLI 有 415 个 `add_argument` | 新配置只声明当前图真实消费的字段，并按 data/model/teacher/train/runtime 嵌套；legacy 字段不得进入 active manifest |
| `MONOLITHIC-TOP-DISPATCH` | 当前 planner 类约 7986 行，object mainline 在初始化、八块循环、缓存、P1/P2/P3 和 bottom ingress 多处按布尔条件穿插 | 新 composer 按 G1-G3/S/W1-W2/P1-P3/bottom 的执行顺序显式拥有模块；active forward 内无版本或 competing-mainline 分支 |
| `TRAIN-TEACHER-BOOLEAN-BOUNDARY` | online 与 teacher 共用 planner forward，并依赖 `future_training_pack` 与 `allow_future_training_evidence` 组合授权 | online API 的类型签名不能接收 future；teacher 由 training engine 单独构建一次，只进入 objective target |
| `RUNTIME-MULTI-OWNERSHIP` | loss、optimizer、grad diagnostics、日志和训练循环集中在 2 万行 runtime；P3 改名曾使诊断访问已删除成员后崩溃 | objectives、optimizer、diagnostics、metrics 分离；模块发布自己的参数/诊断 group，外部不得访问私有成员 |
| `LEGACY-SOURCE-FINGERPRINT` | 当前 fingerprint 包含 V94-V118 等无关旧启动脚本，legacy 修改可改变当前实验身份 | checkpoint 只记录 active import closure、精简 spec、component ABI、Git commit 与数据/T5 identity |
| `STATIC-CACHE-VERSION-OWNERSHIP` | 部署缓存仍名为 `V115StaticEvidenceCache`，同时容纳多代 goal/effect 类型 | 新 `OnlineContext` 只包含当前 G/S/W/P1 在线产物；P2/P3/bottom 保持 ODE-step 动态 |
| `BOTTOM-COMPATIBILITY-MISLABEL` | 正式入口最终选择 `EvidenceLatentMMDiTActionDecoder`；该类是确定性 latent organizer + time-domain Evidence MMDiT，明确不运行 CVAE posterior、旧 hierarchical workspace 或 adaptive CVAE refine，但 manifest/文档仍宣称 `evidence_mmdit_cvae_workspace_v1` | ABI 按真实运行能力命名；保留三层 time-domain Evidence MMDiT、确定性 condition bottleneck、typed selector/value、低秩容量和可微 execution，禁止以历史类名前缀冒充实际图 |
| `BOTTOM-CANDIDATE-REPLAY` | 旧 `_run_dynamic_execution` 在每个 decision 上为全局 block×dwell 候选重复执行完整 action block，并保留 differentiable candidate graph；这与历史上新增 execution 后数倍吞吐/显存膨胀一致 | 训练主路每个 host block 只执行一次；controller 只给出有界连续 capacity/continue，execution cost 仅 audit，不再通过候选重放制造计算图 |
| `BOTTOM-STATE-DUPLICATION` | 数据 `state_history` 已含当前 offset，旧 S 构造又拼接 `state[:,None]`，新提取版 bottom 也再次追加 current state | `ObservableHistory` 以最后一项被当前 `state` 替换的 canonical 序列作为唯一状态历史；任何消费者不得再次 append current |
| `BOTTOM-TYPE-AS-VALUE` | 提取初稿把 learned `source_identity` 直接加到 protected/precision/temporal/state-change value；即使 optional lane 为零也会向 action 注入非零常量 | 类型、basis、位置身份只进入 Q/K/selector；所有 optional V 使用 bias-free 投影并保持 `input=0 -> value=0`；protected consequence 只经一个专用 basis read 进入 bottom |
| `BOTTOM-NOISY-EVIDENCE-WRITE` | 提取初稿前两层同时执行 `evidence_from_action`，让 noisy action 改写随后层的 evidence value，违反 action/noisy 只能作为 query 的边界 | Evidence MMDiT 的 selector/value bank 全程只读；每层只更新 action stream，controller 也只能选择计算，不得写 evidence value |
| `MASK-TARGET-ERASURE` | 新观察编译器把 `visible` 同时并入在线候选合法性和 `DenseFactChart.dino_content` 的构造；被 mask 的格子因此既不可见、target 也变成零，而且 reconstruction owner 同样为零，名义 JEPA mask 没有任何被遮事实恢复压力 | 单独保存 detached 全当前 DINO target 与 `cell_observed`；在线 G 仍只能读可见候选，masked reconstruction 只能由上下文化 public query 读取全局 K prototype，不能从被遮 candidate value 泄漏 |
| `INTERVAL-OBJECT-AXIS-ALIAS` | `FuturePlanRecognizer` 将 `[1,I,H]` 的 interval identity 直接 `expand(B,I,K,H)`，遗漏 singleton object 轴；由于 `I=K=4`，shape contract 全部通过，但原 interval 维实际对齐到了 object 维 | 所有新增轴必须先 `unsqueeze` 再 expand；增加 `I != K` 的轴语义回归，区间置换只置换 interval、对象置换只置换 K |
| `W-ADDRESS-SELF-CONFIRMATION` | W 的 `future_address` 由 `current_address + transport` 生成，P2 又用同一 transport 构造 `transported_source` 与该地址矩比较；该 coordinate 分数是代数自证。旧生成器还用“原分布乘新中心高斯”而非真正平移，所以地址几乎留在源位置 | W 使用可微 backward warp 真正平移 object chart，并以 teacher address loss 约束；P2 由 action query 独立提出 destination/transport 请求，再与 W effect 比较，禁止一个字段和自己的确定性重写互证 |
| `FUTURE-RELIABILITY-OMISSION` | Teacher 已有 null/entropy，但导出的 `FutureObjectDynamics` 没有 reliability；future loss 对低置信匹配与可靠匹配使用同一 current-validity 权重，P2/W 日志也无法区分“静态事实”与“匹配失败” | 导出 object/interval reliability；内容、delta、transport/address 由 detached teacher reliability 加权，reliability/uncertainty 自身仍在物理 validity 上校准；reliability 只作 loss/selector calibration，不作非零 effect value |
| `P3-EFFECT-BYPASS` | precision 的 action query 可在 `effect=0` 时重新选择 K-detail；temporal 可由 S temporal 单独产生 value。两条 optional lane 因而都能绕开 W consequence，和账本中宣称的“必须同时读取 consequence”不一致 | precision 与 temporal value 都以 consequence innovation 作乘性、bias-free 条件；neutral FutureEffect 时二者精确为零，P1 protected fact 仍原样保留，因此修复不删除当前事实能力 |
| `BOTTOM-NEARZERO-RENORMALIZATION` | 新 bottom 在 optional evidence V 与 controller V 上使用普通 LayerNorm；一个刚学到的微小非零 lane 会立刻被扩成单位方差，重现历史近零 residual 高增益问题 | selector 可归一化，value 必须使用固定 RMS floor 的零保持归一化；零值精确为零，近零值局部 Jacobian 有显式上界 |
| `DEPLOYMENT-CACHE-RETENTION` | `OnlinePolicyCache` 保存完整 `OnlinePolicyInput` 与 `ObservationEvidence`；五步部署只需要 history/G/S/W/P1，却让 raw RGB、DINO history、flow 特征和 local chart 一直存活 | deployment cache 只拥有 history、G/S/W/P1；训练 evidence 作为训练返回值由 objective 单独持有，进入五步采样前即可释放 |
| `RESUME-RNG-NONEXACT` | checkpoint 保存全局 RNG，却不保存 DataLoader 专用 generator；启动 preflight 和每轮随机 validation 又消费训练随机流。所谓 exact resume 会重复/改变 shuffle 与 flow/noise 序列 | train-loader、flow-matching、preflight、validation sampling 分别拥有可序列化 generator；preflight 不改变训练流，epoch-boundary resume 恢复下一轮 loader/flow 状态 |
| `UNVERIFIED-BOTTOM-MIGRATION` | `--migrate-bottom` 只按 `bottom.*` 名字和 shape 加载任意文件，不检查 mainline manifest 或 bottom ABI；历史同名张量可绕过 fresh-run 边界 | migration source 必须是带 identity 的 mainline checkpoint且 bottom ABI 完全相同；否则完整拒绝，不提供“shape 看起来一样”的静默迁移 |
| `SOURCE-CLOSURE-OMISSION` | source snapshot 只 hash `clearvla/mainline`，但 active data path仍直接导入 HDF5、split、decoded-store 和 preprocessing 基础模块；这些依赖变化不会改变 checkpoint source identity | snapshot 显式包含经审查的外部基础依赖，或把它们迁入 package；active import closure 与 source identity 必须一致 |
| `RUNTIME-CONFIG-AND-EVAL-DRIFT` | compact config 仍有未消费的 proposal/execution diagnostic 与 memory 字段；公共 `evaluate_loader` 声称接收 mainline loader，实际 loader 返回 raw mapping 而函数强制要求 `TrainingBatch` | 删除死字段并只保留一个真实 diagnostic-batch 预算；evaluation 在唯一 typed conversion 边界转换 raw batch，CLI 和库函数共享同一路径 |
| `ACTIVE-G-HOST-OMISSION` | V122 默认图的第 1–3 个 grounding `TemporalDynamicsBoundDiTBlock` 都实际执行且可训练，共 25,210,368 个参数；提取版仅保留 Pre-G 局部块和 global-K binder，没有这三个 state-conditioned self/visual/transition/FFN 宿主 | 在 typed local-chart → global-K 边界恢复三个完整 grounding host；只允许读取当前事实与当前 state，不恢复 task/history/proposal/noisy-action canvas 汤 |
| `ACTIVE-P1-HOST-OMISSION` | V122 第 6 个 policy block 实际执行且有 6,300,160 个可训练参数；它在一次 high-resolution reader 前组织 action-basis factual query。提取版直接从小线性 query 进入 micro-read | 在 `ObjectFactualReader` 内恢复一个静态、typed 的 P1 self-attention/AdaLN/FFN host；它作用于 clean action-basis query，仍不得读取 noisy action、teacher 或第二次视觉 bank |
| `ACTION-HISTORY-PROPOSAL-OMISSION` | 正式链固定使用 8 个 executed-action offsets `(-24,-16,-12,-8,-6,-4,-2,-1)`、4 recent + 3 summary token 和两层 history proposal；提取版把历史缩为 `(-8,-4,-1)`，并完全删除了 10,010,631 参数的活动 proposal 与原 proposal loss | 恢复完整 8-row causal history encoder、summary cross-attention、24-step clean proposal 和原 action proposal loss；proposal 只作为 P1 clean query 与 controlled transition 的动作条件，禁止重建 W/bottom 旁路 |
| `CONTROLLED-TRANSITION-OMISSION` | V122 的 7,894,025 参数 `ControlledResidualLatentDynamics` 在 P3 后始终执行；其 action-centred low-rank delta 与 event-context 被送入 Evidence MMDiT `transition_memory`。提取版 bottom 没有对应输入 | 在 W/P 与 bottom 之间恢复 fixed-zero、action-minus-neutral 的低秩 controlled transition；使用 W typed future state、clean proposal、current state/observable history，作为独立只读 evidence lane 进入 bottom |
| `ACTIVE-PARAMETER-INVENTORY-DRIFT` | 早期 synthetic/default-width V122 账为 227,466,394 / 166,360,123；正式 4096-wide T5 长跑实际为 230,717,082 total / 168,064,059 trainable。最初提取候选仅 117,138,019 / 117,039,715，缺口主要由活动 G hosts、P1 host、proposal、controlled transition 构成，不是冻结 ancestry | 候选参数账按“活动旧模块 → typed 新 owner → 参数量 → 调用边界”逐项闭合；比较基准使用正式长跑 inventory；仅 4/5/7/8 已冻结且跳过的 generic blocks、冻结 legacy heads 与候选重放可计入合法删除 |

重构以当前 schema-4 数学为默认 preserve 基准；只有本表或前述问题表中具有源码证据的问题允许在迁移时改变行为。每个修正必须先有最小失败回归，再实现，再做 provenance、numerics/autograd 和 runtime/performance 三轮审查。

## 9. 独立 mainline 候选的已结问题（schema 17）

这里仅记录会改变候选主干行为或实验可信度的问题；旧日志结论仍留在前文作为 V122 冻结祖先证据，不再把它们伪装成候选当前状态。

| 问题 | 源码根因 | 候选修正与验收 |
| --- | --- | --- |
| `S-POSITION-AS-VALUE` | S 将固定相对历史位置 `[-1,0]` 与 state/action/delta 一起投影成 history value；即使可观测量全零也能制造固定 interval/temporal innovation | 位置只进入 history Q/K carrier，V 只携带可观测数据 innovation；零 goal/state/action 时 S 四区间、对象、temporal 与 coarse intent 精确为零 |
| `W-SUCCESSOR-GHOST-GRADIENT` | `successor=detach(current)+delta` 却从 attached `current_reference` 相减，前向为零但向 G 注入 `-d(current)` | 两个 current baseline 使用同一 detached provenance；零 successor innovation 对 G 的梯度精确为零 |
| `FUTURE-ADDRESS-LOW-RELIABILITY` | future address loss 未使用 Teacher-G 匹配可靠度，低置信关联仍训练 W 空间地址 | address Hellinger 与 successor/delta/transport 共用 detached object/interval reliability；可靠度本身只在 physical validity 上校准 |
| `DEPLOY-BACKWARD-FLOW` | 部署构图仍计算只供 cycle loss 使用的 backward flow | backward flow 只在 geometry-supervision 训练边界构建；五步部署只计算一次 forward learned flow |
| `VALIDATION-DYNAMIC-COLLISION` | 单随机 flow-time 的 loss-forward 指标与五步部署末端指标同名后被平均 | 部署动态指标统一加 `validation_deploy_` 前缀，保持两种语义独立 |
| `METRIC-KERNEL-FANOUT` | 每批每指标分别累加 CUDA scalar，虽不立刻同步仍产生大量微小 kernel | 按稳定 key signature 向量化累加，只在日志/epoch 边界一次 materialize |
| `CHECKPOINT-PARTIAL-MUTATION` | malformed model/optimizer/scheduler/RNG 可能在验证后半段才失败，留下半恢复 live state | exact resume 在任何 mutation 前验证完整 model shape/dtype/finite、optimizer ownership/state、scheduler LR、全局 RNG 与 owned generator；失败回归验证 live model 未改变 |
| `HISTORICAL-MIGRATION-PARSER` | bottom ABI 可兼容的旧 mainline manifest 因当前 schema 强校验而无法被显式审计 | 仅 bottom migration 使用 relaxed historical-schema parser；exact resume 仍要求当前 schema，bottom state 必须全量匹配 |
| `DINO-DOUBLE-MMAP-READ` | 当前 DINO 与 12 个 future supports 分两次读取同一 episode mmap | dataset 一次 `load_batch` 读取 13 行再拆 current/future；不复制完整缓存 |
| `EVAL-QR-RECOMPUTE` | 三个低秩容量算子在每个 ODE step 重做相同 512×32 QR | frozen eval/no-grad 按 parameter version 缓存精确正交基；训练每次保持可微新 QR |
| `OUTPUT-STREAM-CONTAMINATION` | fresh run 可向已有 metrics/checkpoint 目录追加；且 preflight 前写 context 会留下无法直接重跑的半成品目录 | fresh/migration 要求空目录，resume 对齐已有 identity；fresh context 仅在 preflight 成功后写入 |
| `FLOW-FRAME-AXIS-AND-UNIT` | recurrent flow 是源网格上的受边界缩放参数；G/Teacher 却把 previous-grid 参数当成 current-grid 的真实 normalized displacement | 部署先求 current→previous 对应，再转换为真实位移并取逆，导出 current-chart 上的 previous→current transport；warp/cycle/G/Teacher 共用同一物理单位；方向、边界转换和重建有可执行回归 |
| `FLOW-HORIZON-TIME-UNIT` | raw pair 为 `-4→0`，Teacher 却用 `offset/48` 缩放：H4 只有正确先验的 `1/12`，H48 也只有一次四帧位移 | `flow_reference_frames=4` 同时拥有数据 raw-pair 与 Teacher 外推；四个未来区间按真实 `offset/4` 使用流，语义全局匹配仍可纠正错误先验 |
| `FLOW-FEATURE-COLLAPSE-SHORTCUT` | warp/advantage 只比较共享可学习 encoder 的前后特征；encoder 可通过抹平时间细节降低 loss，而无需学会运动 | 同一外部 flow 权重同时包含 feature warp 与低分辨率 literal-RGB photometric anchor；motion advantage、uncertainty 与 refinement 由不可学习 RGB 变化锚定，不加入非零流配额 |
| `FLOW-CORRELATION-KERNEL-FANOUT` | 半径 2 的 25 个邻域逐个 `grid_sample`，训练双向、三次 refinement 约 156 次小采样 kernel | 一个迭代用一次向量化邻域采样加一次 matched 采样；逐元素等价回归并锁定为每迭代两次 `grid_sample` |
| `AUDIT-PROGRESS-LEAK-OR-DEAD-FIELD` | `frame_progress` 每批搬到 GPU 后完全未消费；既增加无效路径，又无法判断 S/W 是否随真实轨迹位置变化 | metadata 固定留在 CPU，只在日志批次 detached 计算 frame-position 与 S interval-energy/state-change、W successor/interval variation 的相关性；forward/loss 类型仍不接收 progress |
| `GRADIENT-AND-LR-LOG-SEMANTICS` | 全局范数是 clip 前、owner 范数是 clip 后却同名；warmup 日志在 scheduler step 后读取，记录的是下一批 LR | 明确记录 `gradient_global_preclip_l2` 与 `gradient_postclip_<owner>_l2`；learning rate 在 optimizer update 前捕获，与当前 loss 对齐 |
| `ACTIVE-MODULE-MIGRATION-RESTORED` | schema-12 候选错误删除三个活动 G host、一个活动 P1 host、8-row history proposal 和 controlled transition，导致 trainable 参数从当时 synthetic 口径的 166,360,123 降到 117,039,715；正式 V122 长跑实际 trainable 为 168,064,059 | schema 13 按 typed owner 恢复四类算法：169,378,419 total / 169,276,019 trainable；proposal 参数精确恢复为 10,014,727 total / 10,010,631 trainable；旧 4/5/7/8 frozen+skip block 不迁移；schema 16 对正式长跑的最终差额另按 4096-wide T5 口径闭合 |
| `TRAIN-CONDITION-DROPOUT-OMISSION` | V122 正式脚本链固定启用 goal `0.05`、executed-action history `0.10` 与 proposal `0.25` 的训练期条件失活；schema-13 候选恢复 proposal 模块后仍把三路条件在每个训练样本上恒置为 keep=1，改变了既定的防捷径训练算法 | schema 14 按样本恢复三路独立 keep；完整 history proposal 仍承担原辅助监督，只有进入 S/P1/controlled-transition/bottom 的条件值 exact-null；validation/deployment 恒为 keep=1 且不消费条件 RNG；keep 比例进入日志，exact-resume 复用已序列化的训练随机流 |
| `PHYSICAL-ACTION-FIELD-OMISSION` | 正式 V122 最终参数解析固定为 `legacy_independent + legacy_handcrafted + gripper_field_dim=6 + decode_delta_blend=0.25`，即在 18 维 `[arm_abs(6), arm_delta(6), gripper_field(6)]` 中做 flow matching；schema-14 候选却直接在 7 维 normalized action 上加噪、预测和积分 | schema 15 已恢复确定性的 18 维物理动作场；P2/P3/bottom 的 noisy-action query、三块 Evidence MMDiT velocity 与五步 ODE 都位于该场，最终才解码回 7 维动作；proposal/history/controlled-transition 仍保留 7 维 clean-action 语义；逐通道编码、解码和 flow/runtime 同场回归已通过 |
| `ACTIVE-ACTION-OBJECTIVE-OMISSION` | schema-14 只保留普通 7 维 velocity MSE、decoded MSE、普通 CE 与错误阈值；正式链实际使用 `anchor_bands=(4,12,24)`、tail `0.20`、first `0.05`、raw-unit gripper event threshold `0.10`、focal positive `4.0`、normalized arm-delta motion threshold `0.02`、SmoothL1 decoded/delta consistency。Parseval 包装层最终把 transition/event-delta/magnitude/off-delta 四项覆盖为 0，不能错误恢复祖先默认值 | schema 15 已按最终 argv 恢复 semantic physical-field flow、anchor-band 加权 proposal/event/motion/decoded/delta；外部权重为 proposal `0.05`、event `0.03`、motion `0.03`、decoded `0.08`、smooth-delta `0.02`、physical-delta-consistency `0.03`；最终为 0 的四项没有迁移 |
| `INFORMATION-BALANCED-SAMPLER-OMISSION` | 正式链固定 `information_balanced_sampling=1`，uniform/event/motion 配额为 `0.50/0.125/0.375`、motion quantile `0.70`；schema-14 使用普通全局 shuffle，改变了每个 epoch 的监督分布 | schema 15 已从 normalized action delta 与 raw-unit gripper transition 预计算无图像 sampling strata；train loader 使用确定性 information-balanced batch sampler，validation/test 顺序遍历，sampler epoch、阈值和 summary 由新 runtime 显式拥有 |
| `EXECUTION-VALUE-REPLAY-BOUNDARY` | 正式链的 execution value weight 为 `0.05`，但旧实现为每个 block×dwell 候选重复运行完整动作块并保留可微 candidate graph；这是已记录的吞吐/显存结构缺陷，不能作为“重要模块”原样迁入 | schema 15 保留三块 Evidence MMDiT、32-rank/32-group 低秩容量、controller attention、bounded capacity 与 soft continue，只删除重复候选执行图；完整 action backward 已证明 capacity operators、capacity head 与 continue head 均收到非零梯度，execution cost 仍为 audit-only |
| `FORMAL-T5-WIDTH-OMISSION` | schema 15 的独立配置把 `goal_token_dim` 留在 DINO 风格的 768，但正式 `grasp_pen_embed.pt` 是 `[39,4096]`；V122 从文件动态解析 4096，因此候选会在真实语言加载边界直接失败，也少计了 S 的语言投影参数 | schema 16 将正式 T5-XXL ABI 固定为 4096，配置加载对错误宽度继续 fail-fast；top ABI 升级，新增的 1,703,936 个 trainable 参数全部来自 `4096→512` goal projection，并由真实 `.pt` 加载回归与参数 inventory 锁定 |
| `GLOBAL-K-MIXTURE-IDENTITY-SHORTCUT` | G 的主原型项计算 `MSE(sum_k owner_k * prototype_k, target)`；多个高度重叠 K 可以在混合后共同解释 DINO，而不要求任何单个、实际送入 W/P 的 prototype 解释自己读到的事实。V122 的 object-content cosine 增至约 `0.71`、chart overlap 增至约 `0.57` 与此代数捷径一致 | 保留原混合重建作为事实图诊断，但 65% 主原型项改成 soft-EM 条件失真：每个 K prototype 对其自身 competitive read posterior 下的 detached DINO target 负责；相同事实仍允许相同 slot，不加入 diversity/entropy/quota。回归锁定 overlapping read 在异质事实上有损失、在相同事实上精确为零 |
| `S-ACTION-STATE-CONTEXT-SOUP` | S 将 `interval_action_innovation + interval_state_innovation` 在同一个 `tanh` 前相加，再用同一 context 同时制造 object key/value；history 因而被重复，goal/action 可以直接合成“可观测 object value”，四区间更容易公共化 | action/goal innovation 只条件化 semantic object key，observed-state innovation 只条件化 typed object value；两者分别使用相同 RMS 上界。零 state/history 时非零 goal 可产生 key、但 object value 必须精确为零 |
| `W-INTENT-ACTION-SATURATION-COMPETITION` | W 把 S interval innovation 与 supervised coarse-action innovation 先相加再过一个 `tanh`/writer；V122 中 coarse-action RMS 增至约 `1.60` 而条件交互仅约 `0.21`，较大的 action carrier 能饱和并抹掉 intent 的局部 Jacobian | intent 与 coarse action 各自通过 bias-free object interaction，各自在 pre-tanh 与 write 边界受相同 RMS 合约，再用固定 `1/sqrt(2)` 合并；不使用 gate/quota。回归锁定 intent 的边际写入不随 action 从零增至 `1e6` 而改变，并记录两路 pre/post RMS 与 contract scale |
| `ACTION-BAND-ROW-COUNT-ALIAS` | 原 action/proposal loss 对 24 行直接求均值，因此 `1-4/5-12/13-24` 的总预算主要由行数 `4/8/12` 决定；高度相关的远端行被当成更多独立任务，日志却只显示轻微的 per-row tail emphasis。这种监督仍让 V122 形成 first 很低、tail 几乎停滞的失衡解 | 在同一 action 外部预算内按三个语义 band 分配质量，再保留既有 `1/1.1/1.2` 远端轻微强调与 first-step `0.05`，最后令 24 行权重均值精确为 1；日志同时给出每 band 未加权误差、实际平衡误差与预算 mass |
| `GRIPPER-HEAD-WITHOUT-TRAJECTORY-PRESSURE` | information sampler 只保证 event window，窗口内仍以 hold 行为主；event focal loss只训练辅助 head，真正的 18-D flow 与 decoded gripper 仍按所有行平均。另有 `1 + positive_weight(4)` 实际把正类写成 5 倍而非配置声明的 4 倍 | raw-unit event 行在 physical gripper field 和 decoded gripper SmoothL1 中使用 inverse-root-frequency event/hold 权重，并在 horizon 权重下重新归一为精确相同预算；无事件批次 bit-exact 恢复原损失。辅助 head 正类倍率修正为真实 `4.0`；事件/保持的未加权 flow、decoded error 与实际 row weight 分开记录 |

三轮静态审查结论：

1. **Provenance**：在线输入类型不含 future；Teacher/recognizer 只属于训练 target plane；P2 只消费被监督 `FutureObjectDynamics`，P3 optional lanes 必须经过 consequence innovation；active import closure 无 legacy runtime/launcher。
2. **Numerics/autograd**：在线可训练 cosine/selector 使用显式 variance/RMS floor；optional value 与 capacity 保持零值和近零 Jacobian 上界；Teacher FP32/no-grad；flow 的空间轴、normalized-coordinate 单位与四帧时间单位一致；RGB anchor 关闭可学习特征坍缩捷径；每个 trainable parameter 恰有一个 optimizer owner。
3. **Runtime/performance**：observation/G/S/W/P1 每次 observation 构建一次，Teacher 每训练 batch 一次、部署零次，P2/P3/bottom 每 ODE step 一次；flow 部署一次方向求解、训练两次方向求解，每次 refinement 仅两次采样 kernel；非日志热路径跳过诊断归约，部署 cache 不保留 raw/DINO/flow chart，audit metadata 不常驻 GPU。

schema 16 的本地生产 shape batch-one CUDA BF16 完整训练更新已通过：真实 4096 宽 T5、336 RGB、576 DINO patches 与 12 个 future supports 下 peak allocated/reserved 为 `2.674/2.785 GiB`。schema 17 新增一个 512×512 writer，不能把旧峰值冒充成新实测；当前 86 项 mainline 回归、scoped Ruff 和 production Pyright error=0 已通过。仍未关闭的是 schema 17 本地显存复测与服务器验收：生产 batch 8 `<22 GiB`、fresh smoke、五步部署和真实吞吐。未通过前不切换公开 V122 启动器。
