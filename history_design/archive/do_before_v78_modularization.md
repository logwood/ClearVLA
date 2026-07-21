# do_before_v78：模块化重构计划（不动逻辑，不改结果）

状态: 设计评审中
前置: v77 通过 E8/§36 验收并 tag（执行前置; 设计与评审可并行于 v77 后程）
后继: v78 (Exhaustion-Driven Stage-Adaptive Refinement) 在新布局上开工

## 0. 首要原则与范围三分法

**唯一原则: 这是一次零变量实验。** 搬迁后的代码与搬迁前逐位等价——同种子
同数据下每个 batch 的每个 loss 键、每个仪表键、每个参数张量完全一致。任何
"顺手改进"都是走私, 一律禁止上车。

范围按行为风险三分:

- **M1（本计划主体）**: 纯搬迁。整类移动、import 手术、门面转发。零行为。
- **M2（本计划尾声）**: 逐字节相同的重复实现去重 + config 拆分（带委托兼容）
  + console 数据化。仍零行为, 每步黄金跑复验。
- **M3（明确出范围, 归 v78 后各案）**: 一切碰逻辑的统一——normalize-then-gate
  跨类归一、角色分类法清理(CR9)、legacy 世代删除（待迁移工具）。M3 的每一件
  都是未来的 flag 门控单变量实验, 不属于重构。

## 1. 现状普查（2026-07-12 实测）

| 文件 | 行数 | 类 | 顶层函数 |
|---|---|---|---|
| policy_v39.py | 9338 | 37 | 6 |
| policy_v36_2.py | 981 | 11 | 0 |
| policy_runtime_v39.py | 2858 | 2 | 44 |
| policy_runtime_v36_3.py | 878 | 1 | 19 |
| cli/train_v40_policy.py | 993 | 0 | 9 |

policy_v39 的 37 类横跨五个世代（V37 残差族 / HierarchicalLatentMain /
CVAE+workspace 族 / adaptive 族 / owned-intent 现役族), 共享模式靠复制粘贴。
本仓已归档的四起事故（v70 修复落死父类、stage 门被内容范数绕过、wpq/wgstate
半程传播、幽灵 event 角色）均源于此结构。

## 2. 目标布局与逐类去向

```
clearvla/policy/
    config.py      V39PolicyConfig（M1 整体搬入; M2 内部分组）
    codec.py       ParsevalGripperTemporalFrame, PhysicalActionCodec,
                   PhysicalActionTokenLift, PhysicalVelocityHead   [自 v36_2]
    gauges.py      纯函数仪表库: sinusoidal_positions, _align_milestone_*,
                   _rollout_tokens_to_action_horizon, 分层统计, 分支几何
                   （M1 仅收编现有自由函数; 类内 staticmethod 的收编归 M2）
    trunk.py       TemporalMidcutWorldActionDiT, MidcutContractHeads,
                   LayerContractAdapterHeads, SharedLayerFlowActionProbe,
                   LayerRoleScheduler, UnifiedInterventionBlock,
                   RecurrentMilestoneConsequenceCell
    intent.py      PolicyConditionOrganizer, IndependentIntentFusion,
                   IntentContractCompiler
    evidence.py    EvidenceMemoryBank, OwnedEvidenceMemoryBank,
                   PreparedEvidenceMemory, HierarchicalWorkspaceManager,
                   HierarchicalEvidenceWorkspace
    decoder.py     HierarchicalMMDiTActionDecoder, OwnedHierarchicalActionBlock,
                   ConditionNeutralActionInitializer,
                   ActionOnlyPhysicalVelocityHead, MMDiTConditionLayout
    system.py      V39PolicySystem
    legacy/
        v36_system.py    V362PolicyConfig, HorizonRoleEmbedding, DiTPlannerBlock,
                         PolicyLatentDiTPlannerV362, ActionExpertBlock,
                         PlannerConditionedPhysicalActionExpert, V362PolicySystem
        v37_residual.py  V37StyleResidualActionBlock, V37/Layered...FlowDenoiser
        latent_main.py   HierarchicalLatentActionBlock,
                         HierarchicalLatentMainActionDecoder
        cvae.py          LatentCVAEActionBlock, LatentCVAEActionDecoder,
                         AdaptiveRecurrentCVAEActionDecoder,
                         AdaptiveRecurrentCVAERefinementBlock,
                         AdaptiveCVAEMicroRefineBlock, AdaptiveCVAEFunctionBank,
                         LatentCVAEMMDiTBlock
        cvae_workspace.py  SemanticEvidenceWorkspaceBlock,
                           SemanticEvidenceWorkspace, WorkspaceController
clearvla/runtime/
    losses.py      flow/decode/aux 损失组装（自 runtime_v36_3 + runtime_v39）
    evaluate.py    eval 循环、event_head_metrics、采样诊断
    trainer.py     训练主循环、优化器、checkpoint 存取、V39PolicyTrainerConfig
    console.py     键表数据 + 渲染（M2 完成数据化; M1 先整体搬）
```

分界裁决记录: LatentCVAEMMDiTBlock 归 legacy/cvae.py——现役解码器已用独立的
OwnedHierarchicalActionBlock, 二者无继承关系(实测); EvidenceMemoryBank(基类)
留 evidence.py 因 Owned 继承它; codec 是现役且最命脉, 单独成模块并最先搬。

## 3. 兼容不变量（机械规则, 违反即黄金跑失败）

1. **属性名冻结**: 任何 nn.Module 的 self.xxx 属性不改名不改嵌套——
   state_dict 键 = 属性路径, 这是 checkpoint 兼容的全部。
2. **旧 import 路径门面**: policy_v39.py / policy_v36_2.py / policy_runtime_*
   保留为纯转发门面（`from clearvla.policy.decoder import *` 式 re-export +
   __all__）。三重作用: 脚本与外部引用零改动; 任何 torch.save 整对象 pickle
   按旧路径可解析; 团队过渡期心智成本为零。门面文件头部注明 deprecation。
3. **__init__ 体不动**: 参数注册顺序 = 初始化 RNG 消耗顺序, 移动类定义不影响
   实例化次序, 但严禁在搬迁中调整 __init__ 内部语句顺序或 Sequential 组成。
4. **签名冻结**: 一切 def 的参数表、默认值、返回结构不动。
5. **config 字段冻结（M1）**: 字段名、默认值、validate 全部原样; M2 拆分组时
   以组合+__getattr__ 委托保证 `config.任意旧字段` 逐字可用, 且 asdict()
   输出键集不变（context JSON 与 checkpoint 指纹依赖它）。
6. **仪表键冻结**: 所有 loss/console 键名原样——跨版本 log 对比是本项目的
   生命线, 键名漂移等于烧史书。

## 4. 黄金跑协议（每 commit 一轮, 训练机执行）

工具 `tools/golden_check.py`（执行期第一件产出）, 四级检查:

- **G0 静态**: py_compile 全量 + unittest 全量 + import 环检测
  + 门面 __all__ 与新模块导出集一致性。
- **G1 构造等价**: 固定种子构造 V39PolicySystem, 与基线 tag 对比:
  state_dict 键集全等、逐张量 shape 全等、逐张量 allclose(atol=0)
  （纯搬迁下初始化位流必须逐位相同）。
- **G2 前向等价**: 固定种子 + 固定合成 batch, train 模式 forward+backward
  各 3 步, 逐键对比全部 loss dict 与梯度范数, 容差 0（同机同环境应逐位）。
- **G3 短程回放**: 真数据 50 batch 同种子训练, 逐批 loss 与全部仪表键
  对比基线记录, 容差 1e-12。E1 级full回放仅在 M1 全部完成后做一次终验。

基线记录在动工前从 v77 tag 生成一次, 存 runs/golden_baseline/。

## 5. M1 步骤序列（每步 = 一个 commit + G0-G2; G3 在 M1.4/M1.8 后）

- M1.0 排他窗口声明 + 从 v77 tag 切分支 `refactor/modular` + 基线生成。
      （团队编辑与重构不并行——这是本计划对协作纪律的唯一硬要求）
- M1.1 codec.py 析出 + v36_2 剩余入 legacy/v36_system.py + 门面。
- M1.2 legacy 三件（v37_residual / latent_main / cvae + cvae_workspace）。
- M1.3 gauges.py（仅自由函数）+ trunk.py。
- M1.4 现役族: intent.py / evidence.py / decoder.py / system.py / config.py。
      → G3 短程回放。
- M1.5 runtime 四分: losses / evaluate / trainer / console + 门面。
- M1.6 CLI 不动（它只 import 门面, 应零 diff 自证）。
- M1.7 tests/ 按新路径补 import 双轨断言（新旧路径各 import 一次并 assertIs）。
- M1.8 全量 G3 + E1 级终验回放 → merge → tag `modular-m1`。

## 6. M2 序列（仍零行为, 每步黄金跑）

- M2.1 逐字节重复普查: `_split_heads/_merge_heads/_modulate/_attention` 等
  在 ≥3 个类中手写; 仅将 **diff 为空** 的实现收敛为 mixin/自由函数。
  已知不同实现（如各 _attention 的 mask/bias 分支差异）保持分立并登记为
  M3 候选——**相似但不同 = 不碰**。
- M2.2 config 分组（TrunkConfig/EvidenceConfig/DecoderConfig/LegacyConfig
  组合 + 委托; asdict 兼容层）。
- M2.3 console 键表数据化（键名/格式串/来源键三元组表驱动渲染）。
- M2.4 tag `modular-m2` → 解除排他窗口 → v78 开工。

## 7. 风险与对策

- **挂载不可信**: 一切 git 操作与黄金跑在训练机/Windows 侧执行;
  沙盒侧只做设计与审查（本仓既有教训, 已三次误报）。
- **并发编辑冲突**: M1.0 的排他窗口是硬前置; 预计 M1 全程 2-3 个工作日,
  期间实验照跑（跑的是 v77 tag 的部署副本, 与重构分支无交集）。
- **隐性 pickle**: 执行期先 grep torch.save/load 全部调用点, 确认均为
  state_dict 级; 若发现整对象 pickle, 门面 re-export 已覆盖, 另加解析测试。
- **dataclass 字段顺序**: M1 原样保序; M2 拆分后以 asdict 键集测试兜底。
- **黄金跑通不过怎么办**: 该 commit 直接丢弃重做——不允许"调容差过关"。
  容差 0/1e-12 是设计值不是谈判起点。

## 8. 与既有纲领的衔接

- v76_cvae_replacement_program.md 的 CR9（配置/日志/checkpoint 清算）中
  纯删除性条目移入 M3; 其"legacy 只存在于 migration loader"原则由本计划的
  legacy/ 包 + 门面实现其结构前半。
- v78_exhaustion_adaptive_refinement.md 的全部改动将落在 decoder.py/
  evidence.py 两个文件内——这是本重构对 v78 的直接交付物: 手术区从
  9338 行缩小到 ~800 行。
- 四起历史事故的结构性根治核销表: 死父类(继承迷宫)→legacy/ 隔离 + 现役族
  单文件; 半程传播→console 数据化(M2.3)使发射/转发/渲染同源; 绕门套利与
  幽灵角色的同类风险→单一实现点(M2.1/M3)。
