# do_before_v78 函数台账（主线为纲，逐函数搬迁计划）

配套 do_before_v78_modularization.md 使用: 计划定边界与协议, 本台账定执行
粒度与覆盖证明。**可达性规则**: 以下三个入口在 v77 现役配置
(final_action_decoder=hierarchical_mmdit_action) 下的调用闭包 = 主线;
闭包之外的一切类与函数 = legacy/, 机械判定, 无需逐个讨论。

入口: ① 训练前向 V39PolicySystem.forward(planner 链)
     ② 采样 V39PolicySystem.sample
     ③ 训练循环 runtime.train_v39_policy → flow_losses → evaluate_v39_policy

列说明: 批次=modularization 计划的 M1.x; 状态列执行期填写(空=未动,
✓=已搬+黄金跑通过)。标 ✎ 的函数名需执行期核对（记忆库置信度低于 100%）。

## A. 跨代共享原语 → clearvla/policy/primitives.py（M1.1，最先搬——被所有世代引用）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| sinusoidal_positions | world_model.py | 正弦位置编码 | |
| BiasFreeFFN.forward | world_model.py | 无偏置 FFN | |
| TimeEmbedding.forward | policy.py | 流时间嵌入 | |
| RejectableHistoryProposal.forward | policy.py | 历史动作提议头 | 实测贡献≈0, M3 退役候选 |
| _align_milestone_tokens_to_horizon | policy_v39@619 | 里程碑→horizon 对齐 | 自由函数+类内包装双份, M2 收敛 |
| _rollout_tokens_to_action_horizon | policy_v39@~640 | rollout 网格→horizon | |
| _scaled_contract_view ✎ | policy_v39 | 梯度缩放视图 | |
| _zeros_like_scalar | policy_v39 | 零标量工具 | |

## B. 仪表库 → clearvla/policy/gauges.py（M1.1）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| LatentCVAEActionDecoder._time_stratified_attention | policy_v39@~4700(staticmethod) | t 分层 sum/count 统计 | **跨代边**: 现役解码器调用 legacy 类的 staticmethod——台账捕获的第 1 号必迁项 |
| OwnedHierarchicalActionBlock._branch_geometry | policy_v39@7391 | 分支正交基线几何 | classmethod→自由函数 |
| OwnedHierarchicalActionBlock._sample_rms | policy_v39@7378 | 单样本 RMS | |
| OwnedHierarchicalActionBlock._attention_stats | policy_v39@7449 | 注意力熵/max | |

## C. 编解码器 → clearvla/policy/codec.py（M1.1，e-14 金丝雀随迁）

| 函数 | 职责 | 特记 |
|---|---|---|
| PhysicalActionCodec.encode / decode | 动作↔物理场 | H3 语义所在 |
| PhysicalActionCodec.sample_noise | 流形感知噪声 | v53.5 修复区 |
| PhysicalActionCodec.project_physical / project_arm_tangent / project_arm_field | 流形投影族 | autocast 豁免区, 整体原样 |
| PhysicalActionCodec._arm_difference / _sample_native_arm_noise ✎ | 内部工具 | |
| ParsevalGripperTemporalFrame.analysis / synthesis | 紧框架分解/合成 | ΦᵀΦ=I 断言随迁 |
| PhysicalActionTokenLift.forward | 物理量→token 提升 | 主线(解码器 noisy lift 用) |

## D. 主干 → clearvla/policy/trunk.py（M1.3）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| TemporalMidcutWorldActionDiT.forward | @8039 | 主干前向总控 | 最大单函数, 只搬不拆 |
| TemporalMidcutWorldActionDiT.seed / _mod_embed ✎ | | canvas 种子/调制嵌入 | |
| MidcutContractHeads.forward | @749 | midcut 简单头 | |
| LayerContractAdapterHeads.forward | @827 | 层契约头 | |
| RecurrentMilestoneConsequenceCell.forward | @976 | 反事实里程碑单元 | 事件证据上游 |
| LayerRoleScheduler.forward | @921 | 层角色增益 | |
| UnifiedInterventionBlock.forward | @950 | 干预潜变量块 | |
| SharedLayerFlowActionProbe.forward | @867 | 层流探针 | |
| ControlledResidualLatentDynamics.forward | **policy_v38** | 受控残差动力学 | **跨代边 2 号**: 主线部件住在 v38 文件 |
| CanvasPhysicalVelocityHead(.pooled) | policy_v38 | 直接速度头/池化 | pooled 为主线(trajectory_pooled) |
| event_probe / motion_probe (Sequential) | @~5073 | 层级事件/运动探针 | 事件证据源, 0.05 CE |

## E. 条件与意图 → clearvla/policy/intent.py（M1.4）

| 函数 | 现居 | 职责 |
|---|---|---|
| PolicyConditionOrganizer.forward | @6795 | 类型化摘要+货架装配 |
| PolicyConditionOrganizer._layer_summary / _project_memory / _groups | | 层摘要门控/记忆投影 |
| IndependentIntentFusion.forward | @7104 | 单合同融合 |
| IntentContractCompiler.forward / _cosine | @7125 | 三合同编译+粘连仪表 |

## F. 证据市场 → clearvla/policy/evidence.py（M1.4）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| EvidenceMemoryBank.prepare_sources / prepare_static_memory | @2177 | 源规整+静态 KV | 基类留此, Owned 继承 |
| EvidenceMemoryBank.role_key_bias / role_token_counts / role_attention_metrics / _source_role | | 角色记账族 | 幽灵 event 注释随迁 |
| OwnedEvidenceMemoryBank.prepare_sources ✎ | @2351 | 五角色严格版 | |
| HierarchicalWorkspaceManager.forward | @2885 | 无动作检索管理 | 防火墙注释随迁 |
| HierarchicalEvidenceWorkspace.init_stage / prepare_evidence ✎ / step | @3034 | stage 初始化/证据准备/步进 | |
| HierarchicalEvidenceWorkspace._low_selector_context / _promote_stage / _stage_retain_gate / _step_state / _attention / _slot_diversity | | 选择器/晋升/保留门 | normalize-then-gate 纪律注释随迁 |

## G. 解码器 → clearvla/policy/decoder.py（M1.4）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| HierarchicalMMDiTActionDecoder.forward | @7708 | 解码总控 | |
| HierarchicalMMDiTActionDecoder._step_state / _gate_noisy_tokens / _mean_metrics / _initialize_outputs | | 步态/噪声门/指标聚合/输出初始化 | 门模式档案注释随迁 |
| OwnedHierarchicalActionBlock.forward | @7289 | 五分支串行块 | |
| OwnedHierarchicalActionBlock._cross_update / _normalize_residual | @7455/@7383 | 分支读+幅度宪法 | 宪法注释随迁 |
| OwnedHierarchicalActionBlock._attention / _row_norm / _split_heads / _merge_heads / _modulate | | 注意力原语 | 与 legacy 各版本存在近似但不同实现, **M2 不碰**, M3 候选 |
| ConditionNeutralActionInitializer.forward | @7201 | 条件中立初始化 | |
| ActionOnlyPhysicalVelocityHead.forward / output_layers | @7255 | 动作 only 输出 | |
| MMDiTConditionLayout (dataclass) | @2164 | 条件布局 | |

## H. 系统层 → clearvla/policy/system.py（M1.4）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| V39PolicySystem.forward ✎ | @9021 | 训练前向包装 | |
| V39PolicySystem.sample | @~9060 | 采样入口(积分循环) | v51 修复区+z 探针, 原样整搬 |
| V39PolicySystem._policy_forward ✎ | | planner 调用包装 | |
| V39PolicySystem.parameter_report ✎ / 参数分组函数 ✎ | | 优化器分组 | 分组名与 grad 仪表键耦合, 键名冻结 |

## I. 损失 → clearvla/runtime/losses.py（M1.5）

| 函数 | 现居 | 职责 | 特记 |
|---|---|---|---|
| 物理流损失核 ✎（arm/grip/null/事件强调全家） | runtime_v36_3@~200-560 | 单一大函数 | 只搬不拆; gfnehr/null 键冻结 |
| gripper_event_labels | runtime_v36_3@105 | 事件标签 | |
| _normalized_event_emphasis | runtime_v36_3@~206 | 事件加权 | |
| flow_losses | runtime_v39@~840+ | 总装配 | 键复制清单同文件同段落随迁 |
| aux 层契约损失族 ✎ (w_event@1437 区) | runtime_v39 | 层级 CE/流损失 | |
| _layer_contract_as_primary | runtime_v39@~1266 | 契约回退 | H3 投影修复区 |
| micro_refine_supervision_losses | runtime_v39 | 微监督 | **不可达(mmdit 排除)→ legacy 判定的试金石案例** |
| _finalize_metric_tensors ✎ | runtime_v39 | 指标终化 | |

## J. 评测与循环 → clearvla/runtime/evaluate.py + trainer.py + console.py（M1.5）

| 函数 | 职责 | 特记 |
|---|---|---|
| train_v39_policy | 训练主循环 | checkpoint 存取/恢复语义原样 |
| evaluate_v39_policy | 采样评测 | first/tail/null 率键冻结 |
| event_head_metrics | 事件头准确率 | |
| console 渲染块 | 逐批打印 | M1 整搬, M2 键表化 |
| 优化器/梯度分组函数 ✎ | grad_* 仪表 | |

## 台账使用规则

1. 执行时逐行打 ✓, 每个 M1.x 批次收尾时: 本批 ✓ 行数 = 该批计划行数,
   否则黄金跑不启动。
2. 追踪中新发现的主线函数**必须先补行再搬**——台账即覆盖证明。
3. ✎ 标记 12 处, 执行期第一小时集中核名清零。
4. 未列入且不可达的函数一律随其类进 legacy/, 不单列（约 130+ 个）。
5. 两处跨代边（B-1 号 strata staticmethod、D-9 号 v38 动力学）是本台账
   相对类级计划的净增发现, 优先搬迁并在门面中保留旧引用路径。
