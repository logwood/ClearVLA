# G 槽位身份、内容与共享绑定：审核注意点

> 2026-10-06更新：最终完整评估见第19–24节；旧探针的native动作命名与指令参考口径由第20节修正。

> 2026-10-05。此文件是持续维护的审核与验收清单，不是新架构、不替代 `00_CURRENT_ARCHITECTURE_CONTRACT.md`。本次提交只新增本文件；不修改模型、配置、loss、探针、checkpoint、工作流或训练进程。文中的“应检查/应修”均为待办，不表示已经实现或通过。

## 1. 固定来源与结论边界

| 项目 | 身份 |
|---|---|
| 目标分支 | `codex/g-slot-identity-logit-scale-20261005` |
| 审核基线 C9 | `b229003a55b64cd220dbdfebe7ec9eb1c75d071e` |
| 基线源码树 | `275eb9cfe8856dd05fbb6815fbc22e37cbb4513a` |
| 直接父提交 C8 | `d60928853fb7b8ded17a72e0cea976c7990ff2fe` |
| 更早 repair | `d931fe01f9c6eb7d0e8f74d1ef994503020ef278` |
| 当前完整 grounding.py blob | `cc2f6ff521974dc9f96053097ebbbe9050bf776d` |

用户消息中的分支后缀 `202605` 与实际远端不符；上述分支以 HEAD 对应 b229003a 确认，不创建同名近似分支。旧 `9eed73a9` 的 run_context、R9 固定 checkpoint、正式长跑后续 checkpoint 与本次源码干预必须分别记录，不能拼成一个实验。

**现阶段结论：C8/C9是身份幅度与寻址尺度候选，不是物理对象分工、语言绑定或闭环行为的完成证明。**用户报告候选特征空间差异约 0.428、语言换目标后绑定变化仍小、address logits为零；这些是待附原始记录的本地结果，本次未读取用户权重或服务器。b229附带的是程序及提交说明，不能将它们等同于完整结果文件。

C9说明记录的 K2/K4 candidate-read cosine约0.9821→0.9632、object-chart cosine约0.99828→0.99705、content cosine约0.99971→0.99958，改善层级明显不同。0.6669→0.6670是单节点velocity的RMS，不认证动作方向、差分、24行轨迹或任务质量；instruction-pair cosine>0.99999也需同时看RMSE、符号和实际接触。[S01,S02]

## 2. 优先级总表

| 编号 | 优先级 | 注意点与完成判据 |
|---|---|---|
| A1 | P0 | C9跨整个batch求gain；同一样本更换batch伙伴的结果应被单独测试，不能用B1短测验收B8训练语义。 |
| A2 | P0 | 按来源身份区分当前G、回看G、Teacher；不能用hook列表最后一项配当前S/W。 |
| A3 | P0 | 部署no_grad缓存、合成特征平方loss、正式动作训练VJP分开；补正式目标与optimizer更新。 |
| A4 | P0 | slot、candidate read、像素后验、内容、binding、最终动作分别验收；不可跳过仍相似的object-chart。 |
| B1 | P1 | identity gain应审查空间对比、null、上下限、stop-gradient、实际有效贡献，而非只看总logit RMS。 |
| B2 | P1 | task_object_score与原object_score都含任务；查实际相对logit及训练信号，不比较权重RMS后直接加大gain。 |
| B3 | P1 | post_pool_only零address属于关闭支路，不证明共享binder无语言；开关生产与消费两端都须追踪。 |
| B4 | P1 | S的K归一化共同值、P2区间求和相消和约0.02尺度分别核查，不把std比例当作动作信息保留率。 |
| C1 | P1 | 探针修正字段归属、统计轴、静态历史说明、JSON尾部和覆盖保护，随后复跑固定面板。 |
| C2 | P1 | checkpoint/配置/源码/随机状态一致；C8/C9不是等价优化，不可改身份冒充exact resume。 |
| D1 | P2 | 保留真实小目标、完整软支持与K置换；任何新监督需有独立、置换兼容的物理来源。 |

## 3. C9新增尺度必须补的检查

当前 `_competition` 在FP32概率路径中计算：[S01]

```python
main_logit_rms = logits.float().square().mean().sqrt().detach()
identity_logit_rms = identity_logits.float().square().mean().sqrt().detach()
identity_gain = (0.25 * main_logit_rms /
                 identity_logit_rms.clamp_min(1e-6)).clamp(max=8.0)
logits = logits + identity_gain * identity_logits
```

### A1：无dim的mean把B/N/K全合并了

`logits`为 `[B,N,K]`；gain是全批一个标量。同一观测单独运行、与不同观测合批、换batch大小、分成不同microbatch，都可能得到不同身份增益和候选read。batch置换不变不等于batch组合不变；DDP每个rank的本地批次也需独立考虑。这是新增的跨样本数值依赖，不是参数新增问题。

本次从blob核验的完整文件提取真实 `_competition` 及其helper，用CPU/H32/K4/N24、三个新参数种子905/906/907做了局部检查：锚样本完全相同，更换同批伙伴后read最大差约0.00175—0.00370；合批复制锚样本的对照最大差不超过2.24e-8。**未执行GRU/G3、真实checkpoint或完整策略；这些数值不等于R9或生产误差。**它只确认该代码路径的batch耦合可实际出现。

待办：固定同一模型与观测A，对比 A、AA、AB、ABC、分块/整批，记录每样本gain、owner/null、read、G值和最终命令。若目标语义要求样本独立，后续补丁应在同一真实样本的合法来源上定义尺度；不能未经数值与行为对照直接宣布按样本RMS已修好。

### B1：总RMS不是空间分工所需的对比量

两个K若在全部候选上的logit只差常数，第二次逐K读归一化会消掉该差异。设动态key为q、identity为e、候选为c，则应看：

\[
\operatorname{center}_n\log(R_{kn}/R_{jn})
=\operatorname{center}_n\{[(q_k-q_j)+g(e_k-e_j)]^Tc_n/\sqrt H\}.
\]

中心化和方差使用同一合法候选权重。分别记录动态项、identity项、合成项的空间方差与协方差；总logit大、identity占比25%均不保证这些空间log-odds有用。共同场景偏置可抬高main RMS，却不增加对象间位置区分。

边界必须记录：raw/scaled identity RMS、gain、是否触顶8、是否触及1e-6下限、real/null质量、小而合法的支持。未触下限时，实际scaled/main比例为 `min(0.25, 8*raw_identity/main)`，不是永远25%；main为零时当前gain也为零。增加相同的零填充在无下限效应时可同时缩放两个RMS而抵消，不应未经测试声称任何padding都会改变gain。

缩放只作用于真实K logits，null logit未同步改变；即便K+null轴和归一化公式没改，**real-versus-null事件概率仍可改变**。保留质量守恒、全无效、非有限隔离、小目标和极低真实质量测试。不能用predicted confidence/allocation取代物理支持，也不能强制每个K等质量。

两个RMS均detach：反向忽略gain对输入的导数，但直接logit路径仍有梯度。将其标记为明确的stop-gradient尺度策略，分别测试实现VJP及重新计算gain的有限差分；不能要求二者自然一致，也不能称为原计算的等价性能优化。

## 4. C8遗留的状态语义仍须保留

父版本更新为 `h'=u+0.5*stopgrad(max(||u||,1e-6))*e`，u是GRU与FFN之和，e为seed中心化后逐slot单位化方向。[S01,S02]

身份范数跟随u不保证其投影到候选变化方向；slot cosine下降不证明read或像素后验分开。前向完整导数含 `0.5*e*u^T/||u||`，实现detach省略这一项；不能只用非None梯度验收。

seed减K均值后再逐slot单位化、乘各自不同尺度，最终identity carrier不保证K均值为零。应记录实际carrier的公共分量，不能沿用“中心化所以不增加共同值”的绝对保证。重新中心化、删除detach或增加identity常数都是新行为修改，不是文档修正。

首轮须分别读取 initial slot、GRU输出、加FFN、加identity，再在固定candidate/prior/validity下送回原竞争函数。对每一步同时看空间log-odds、read重叠与物理区域覆盖，区分原始差异被缩小、共同分量增大以及LayerNorm后差异减弱。

低联合mass不等于低强度更新：GRU读取逐K条件期望，并未再乘总allocation。existence是对象相对null的置信，validity是观察支持，都不证明相对其他K有独占物理解释。木桌本来可见，不会被这两项自动判为错误；直接乘raw mass则可能伤害小物体。

## 5. 已有及新增探针：先修测量再据此归因

| 脚本/模式 | 当前限制 | 后续必须做的修正 |
|---|---|---|
| `probe_g_slot_identity_short.py`的`current_calls[-1]`；dataflow及`probe_second_plan_binding.py`的`facts[-1]` | 同一encode随后可能回看过去观测，再调用同一grounder；最后G可能是过去源 | 与当前`training_state.top.facts`或当前cache.content精确身份匹配，并记录source role、time和encode序号；不用allclose判断来源。 |
| `probe_g_competition_logit_scale.py`逐次记录 | 未给各次竞争绑定当前/回看作用域和iteration；记录raw比例，没有完整记录实际gain/scaled结果 | 分作用域记录三轮及G3前竞争，附B/N/K形状、支持、cap/floor和有效空间对比。 |
| 部署cache后反传 | `deployment_cache`内的no_grad已截断G/S生产图 | 分开部署前向与允许梯度的真实online encode；不得给detached cache加requires_grad冒充恢复生产者链。 |
| 独立gradient脚本的content/S/W平方和 | 合成特征目标，不是正式动作/重建训练，也未证明动态P2/整轮采样的梯度作用 | 准确保留其局部范围；补原训练loss分项、正式总loss、参数owner、原裁剪顺序和实际一步更新。 |
| “209参数通过” | 统计非None梯度的参数张量条目，不保证非零、方向有效或优化器更新 | 分别导出None/zero/finite、分解前激活VJP、参数VJP和参数更新量；不把共模抵消误判断图。 |
| 单帧`_history`再伪装四步 | 同RGB/state/命令重复，时间改为4并标observed，仅是静态接口fixture | 静态G审计可保留且明确命名；反馈、恢复与时序结论必须使用真实源历史，禁止虚构observed。 |
| 字段归属 | validity曾命名为mass；predicted_dynamics曾命名P2 | joint assignment/conditional read/existence/validity分别记录；W预测与实际P2 terminal输出分开。 |
| 统计轴 | 残差整体RMS与逐通道interval std平均不同 | 输出定义、维度、单位和分母；K、I、C、空间和动作行不可混比。 |
| JSON与复现 | 新旧若干脚本追加字面反斜线+n（`'\\n'`），不是换行；硬编码路径且可覆盖 | 改为真实换行，落盘后标准json回读；参数化checkpoint/input/device/output；默认拒绝覆盖。 |

这些限制不能推出所有短测数值都是假的；它们意味着当前脚本不足以证明某些来源和因果归属。旧报告中的误测应标为范围受限而非删除。`probe_second_plan_binding.py`的`cos(v)`辅助函数比较自身，仅为自相似且当前未调用，不得以后拿它作跨指令相似度。

本分支不包含已复核的完整本地GPU结果文件。下一批应单独交付小型JSON摘要及其hash；不要将权重、完整hidden、原始日志或大tensor写入架构文档。诊断同步开销不能用于性能测量。

## 6. 0.428为什么不能跳过G的中间证据

新binding探针的 `g_candidate_cell_variation` 将 `candidate_content`展平所有候选轴后求std再平均。它不是K2/K4输出差异，也不是独立物理区域覆盖。[S03]

必须分解以下三个映射：

\[
h_k\longrightarrow R_{kn}\longrightarrow
P_{kx}=\sum_nR_{kn}Q_{nx}\longrightarrow
f_k=\sum_nR_{kn}v_n+W_{res}h_k.
\]

Q表示局部候选对当前图像的来源映射，具体支持/概率归一化以生产代码为准。不同候选编号可映射同一像素区域；不同区域也可有相似材质和DINO共同成分。最后内容差为 `(R_k-R_j)(V-mean(V)) + W_res(h_k-h_j)`；read与value方向不匹配、期望相消、或两项抵消都可能使content继续近似。[S01]

依次记录复合candidate key、真实candidate_content、read、每相机current_image_measure、aggregated_content、slot residual、二者和，以及semantic/appearance/geometry各自的重加权read。共同typed先验可把不同parent read重新拉向同一候选，不能假定四条属性自动沿同一分工改善。

至少包含“固定V换R”“固定R换V”“聚合项/残差分别中和”的局部诊断。干预须保持支持、来源与后续缓存一致；完整策略中不能静默只替换一个字段。相似背景本身允许，目标物块是否至少被某个K可靠覆盖才是任务相关验收；禁止为降低cosine给同一区域加不同内部ID。

## 7. 追task_object_score时必须按真实调用链

当前binder直接消费 `protected_goal` 和 `objects`，发生在 `interval_goal`读取之前；objects已包含content及semantic/appearance/geometry投影。不能用下游interval-goal attention不变解释上游binding不变，也不能称当前binder在softmax前完全没见颜色/语义。[S04,S05]

两条评分分别为：

```text
object_score = score(task_read(query(obj,history), task) * tanh(compatibility(obj)))
task_object_score = Linear([obj*task_context, obj*tanh(task_context)])
```

原object_score也有语言，不是纯视觉；新增项已有真实乘性交互，不能再套用旧 `Linear([obj,task])` 的可分离性结论。权重RMS、置零整支路的输出RMSE不是可加的视觉/语言贡献百分比。[S04]

同一画面、状态、历史和噪声，只换颜色、保持方向不变，分别记录两项与总项的每K logits及：

\[
\Delta_{kj}=[\ell_k(t_a)-\ell_j(t_a)]-[\ell_k(t_b)-\ell_j(t_b)].
\]

再记录真实K条件分布、real/null总质量、目标物理区域质量及实际动作。这样区分公共偏置、真实相对排名、相互抵消和下游未消费。换方向实验单列：同一目标左/右动作不同本来合理，不能要求全部语言变化都必须改变K。

训练检查顺序：`requires_grad`与唯一optimizer owner/真实lr → 正式loss对binding相对logit的VJP → 两条评分的激活与参数VJP → 梯度裁剪前后 → 实际optimizer delta → 固定颜色面板的前向贡献。特征平方和、参数count、attention argmax相同均不是此链验收。

## 8. 零address必须区分“未选中”和“选中但失效”

S构造器仅在 `target_object_address_mode=shared_target_prior_v1`时创建 `_ZeroStartTargetAddress`；否则模块为None，`_typed_relevance`显式输出零address。P2在post_pool_only下也将address条件置零。[S05,S06]

因此，先核对checkpoint序列化的选择、实际实例类型/参数owner，以及S生产和P2消费两端；零address在关闭模式下不是训练失败。该后置address不是已先算出的binder的输入，不能把二者混称为同一条task→K路径。

shared binding模式下semantic K后验直接沿用binding；geometry只在给定K内部选相机。若新address仅给同一K各相机加同一个常数，softmax会消去。不能只打开一个开关或新增未消费参数就宣布恢复任务寻址；改动仍需维持唯一共享K+null所有权，不给P1/P2另造选择器。

没有固定K标签不自动等于bug。若真实目标没有被任何K有效覆盖，binding重加权无法凭空恢复它；先测物理区域覆盖上限。若考虑辅助监督，只能用独立可靠、K置换兼容的区域/对应来源，不以“动得最多的块”或当前G自身作为无条件真值，不直接固化K编号、颜色或等分配。

## 9. S/P2保留的串联风险

S新增任务值中存在 `b_k / sum_j(b_j)`乘同一个U_i后对K求和。在相应typed来源全有效、真实质量大于下限且固定上游query时，增量严格回到U_i；真实质量很小也可被重新归一化到完整幅度。原G事实项和上游binding依赖仍在，不能说整个S与对象无关，但新增std可能主要反映共同任务/区间模式。[S05]

P2的target尺度为 `0.02 + 0.02*tanh(parameter)`；名称含zero_start不代表有效初值为零。`source_interval_variation`和`target_value_interval_variation`都在区间汇总前，后者本来就是source乘scale。约1.97%的比值不是最终信息保留率。[S06]

必须记录真正的 `scale * sum_i(p_i U_i)`，其共同/残差、与基础效果的夹角、再到P3/最终命令的敏感性。若U_i含零均值区间残差，均匀p使其相消；放大scale不能解决严格相消。高interval cosine也不等于值幅度相同，要看中心化范数与奇异值。

S common/residual精确重组时common节点梯度可抵消，应监控分解前被消费的V；公共载体去通道均值仍大，不排除固定interval模板。C5小gain在BF16的`1+modulation`可被舍入，需检查真实forward增量，而非仅看参数更新。既有的C3/C5/C6、global和CT是不同替代路径，需逐条标清，不能一条静默就称全链断开。

## 10. 下一批最小可复核面板

| 面板 | 固定项/干预 | 关键输出 |
|---|---|---|
| 当前来源资格 | 当前/回看作用域；静态fixture和真实历史分开 | 对应facts身份、源时间、支持、命令边界；严禁串源。 |
| C8/C9配对 | 同一权重/input/noise；父/候选；B1、复制样本、混合伙伴/分批 | gain、cap/floor、当前每轮log-odds、owner/null、最终像素分布。 |
| G空间到内容 | 固定R换V、固定V换R；聚合/残差拆分 | 每相机目标/木桌覆盖、重叠、内容差异与共同模式。 |
| 语言到目标 | 同画面同方向换颜色；同色换方向作为另一组 | 两项score、真实K相对log-odds、目标区域质量。 |
| 目标到动作 | 固定其他源，干预shared binding并重算全部消费者 | S/P1/P2实际值增量、coarse、proposal/W/refined、最终24行与实际前8行。 |
| 正式训练路径 | 可微online encode、真实监督和原loss；单独一步更新 | 参数归属、finite/zero/None、裁剪、delta及真实前向贡献。 |

每条记录最少带源码提交/dirty文件hash、checkpoint hash或既有强身份、checkpoint实际配置、图ABI、state/action normalizer、DINO身份与精度、完整输入来源、当前/回看标记、RNG状态、dtype、B/N/K/C/I轴、计算模式、执行行数。服务器路径与文件名不是已读取内容；R9与正式长跑必须分开。

固定物块小面板与留出的新布局都要覆盖；slot编号不能永久映射颜色，双相机坐标未经标定不得平均成世界点。接受概率的误差阈值应先由原版重复性建立，不为了候选通过修改标准。局部velocity范数不代替完整命令、夹爪边界或闭环阶段。

## 11. 改动顺序与不可越过的边界

1. **先修探针与来源/序列化，补A1 batch独立性对照。**保留原失败和中断证据；不把不可信字段继续用于下一轮归因。
2. **再区分G的candidate→pixel→content映射。**真实目标覆盖不足时不要直接跳到绑定loss；不能通过排除木桌像素、扩大slot residual或正交化编码伪造对象分工。
3. **在已有目标覆盖下追binder真实训练和颜色相对排名。**只改一个已确认节点，保持同一共享K+null，不新增外接selector、不硬化argmax。
4. **再检查对象相关值到S/P2/最终动作。**保留必要的公共任务/方向信息，但不要让归一化共同gate冒充对象特异性。
5. **正式训练与部署验收单列。**C8/C9改变了前向与停止梯度策略；同shape旧权重仍可用于明确标注的诊断初始化，不得改checkpoint身份冒充精确恢复。先查真实加载验证覆盖哪些文件/模式，不能假定新增源码自动被全部身份规则捕获。

保持完整候选、相机、软分布、来源支持和K置换；不改变动作归一化、原生控制语义、R1/R8或ODE计划来掩盖问题。不得热改正在运行的工作树、覆盖权重/结果、强推或更改master。此文档没有修改detach、gain、mask、模式或loss，也不授权启动训练。

历史问题不得无证据搬到当前：9eed终点来源缺失只在那份运行确认；617的P1缩放回归不能自动当作当前bug；C7在兄弟分支完成不代表本checkpoint启用。Teacher自引用/未知与零变化、R1/R8、clean-arm/sampled-arm端点、W(proposal)/最终动作条件、物理穿透与控制器停滞仍应按实际图和阶段分别验证。已有层的有限梯度、loss下降或总成功率不能宣布这些全部闭合。

## 12. 源码索引与本次交付范围

以下索引均以首节的b229完整提交为准，后续变更应更新本文件结论，不默默沿用移动HEAD。

| 索引 | 固定源码位置/核对内容 |
|---|---|
| S01 | `clearvla/mainline/model/grounding.py`：`_competition`约395—487、G迭代更新、typed_reweight、aggregate、camera_aggregate与当前图像pushforward；完整blob已核验。 |
| S02 | `docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`：C8/C9尾部记录；统计为提交者报告，非本次checkpoint重跑。 |
| S03 | `probes/probe_g_competition_logit_scale.py`、`probe_second_plan_binding.py`及继承的三份`probe_g_slot_identity_*.py`；目录/源码与父报告逐项核对。 |
| S04 | `clearvla/mainline/model/task_execution.py:531—618`：TaskConditionedTargetBinder的两个任务相关分数与唯一K+null。 |
| S05 | `clearvla/mainline/model/intent.py:430—447, 575—750`及forward中的binder→interval_goal顺序；当前blob `3e82e7712d3c47ccf1f88480060bf6a5d8db8378`与此前已读版本一致。 |
| S06 | `clearvla/mainline/model/compiler.py:740—845, 1080—1426`：address选择、空间/区间消费者和target-value指标；blob `2954020ab5f2ecaf6b2ffcc30c12834347903b62`。 |
| S07 | `clearvla/mainline/model/policy.py::_replay_executed_world`、`clearvla/mainline/runtime/sampling.py::deployment_cache`、`scripts/probe_calvin_internal_layers_npz.py::_history`；当前/过去与无梯度边界。 |

本次完成：分支/父提交/变更核对，完整grounding blob核验，继承报告与当前源码对照，三种合成batch组合的真实竞争方法检查，以及文档检查。未完成也未宣称：R9/正式checkpoint重放、GPU训练、整网梯度/闭环或全仓库测试。此文档只提交决策统计、来源和验收要求，不复制权重、tensor cache、原始大日志或服务器数据。

## 13. 2026-10-05：完整目标交换的下游因果复核

本节记录的是当前分支短 checkpoint 的只读测量，不是生产修改或正式训练验收。测量使用 `probes/probe_target_binding_measurement_v2.py`，固定同一观测、同一噪声，并交换最远的两个有物理支持的 K 槽位；null 质量、真实质量多重集和支持均保持不变。新增提交 `1ea28eaf` 记录 proposal/refined 每次 velocity 调用，`8c092dbd` 增加受控的 P3、底层 P3、protected detail、P1 precision 缩放。短 checkpoint 身份为 `short-bs8-gpu3-r1/checkpoints/best.pt`，`global_step=64`，因此下列数值不能外推为训练完成结果。

### 13.1 轨迹复核结果

四条指令的 full proposal→W rebuild→refined 目标交换，平均末端差异为：P3 temporal `.0328`、底层 P3 lane 0 `.0235`、lane 1 `.0061`、P3 bridge `.0166`、protected detail `.0464`、最终 native action `.0567`。逐次 ODE 调用中，P3/bridge 的差异随时间从约 `.0045` 增长到 `.0166`，而 protected detail 一直约 `.0465`；这表示 P3 路径确实被消费，但当前短 checkpoint 的最终目标差异主要由并行的 protected-detail 载体提供。

### 13.2 受控缩放的因果证据

| 只读干预 | P3 temporal 交换差异 | P3 bridge 交换差异 | 最终 action 交换差异 |
|---|---:|---:|---:|
| baseline | `.0324` | `.0165` | `.0564` |
| coordinator semantic 输入置零 | `.0249` | `.0147` | `.0550` |
| 底层 P3 两个 lane 置零 | `.0293` | `.0105` | `.0578` |
| protected detail 置零 | `.0243` | `.0115` | `.0126` |

protected detail 置零会使最终目标差异显著下降，而 P3 置零只改变其中一小部分；这不是“P3断路”的证据，反而说明继续只放大公共 P3 载体会掩盖真正的并行消费关系。`p1-precision=0` 的最终差异会上升到约 `.138`，属于移除相消路径后的非线性重排，不能当作“precision贡献为负”的线性分解。

源码对应关系是：P1 `build_static` 将 target detail 加入 factual protected detail；P2 consequence 在此基础上加入 typed effect/interaction；P3 另外生成 temporal/state-change 两个可选 lane；bottom 在 `_read_policy_delta_bank` 中把可选 P3 与两个 protected carrier 分开读取。因此，下一步应先在同一完整 checkpoint 上复测这条分叉，而不是修改 bridge 固定缩放。

### 13.3 正式长跑状态与下一门槛

正式长跑仍使用启动时锁定的源码 `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，没有被上述探针提交改写；截至 batch `6300/11012`，运行约 `6.41 s/batch`，P2 target interval variation `.1642`、source variation `.3327`、P3 temporal RMS `.3393`、P3 routed update `.4.236`、fixed bridge scale `.25`、null mass `.0015`、source effective count `3.98`、capacity `.9995`、execution gate `1.0`。日志审计仍为 JSON/traceback 完整，只有预期的 capacity-saturated warning。正式最终 checkpoint 和同 branch target-swap 尚未产生；完成前不据此改生产网络。

## 14. 2026-10-06：protected-detail 实际生产者六源账本

本节承接 13 节的短 checkpoint 复核。报告中的关键修正已落实到探针：`protected-detail` 不是纯视觉细节，而是 P1 当前 factual lattice、P1 target read、P2 semantic effect、P2 geometry effect、P2 semantic interaction、P2 geometry interaction 六项在 P1/P2 边界合成后的 carrier。探针只读包裹真实 forward，不改变生产模型、梯度或正式长跑。

### 14.1 生产路径和闭合

- P1 `build_static` 的 factual carrier 按真实调用顺序拆为 `(updated-clean)` 与 `target_detail`；四条指令 target-swap 的 factual 闭合误差为 `1.38–1.92e-9`。
- P2 consequence 使用源码中的分组顺序 `factual_base + (semantic+geometry) + (interaction.semantic+interaction.geometry)`；闭合误差为 `0`。
- bottom `protected_detail_basis_attnres` 的 route probability 与 value contract 从同一次 reader 输入重算；重算与真实 reader 输出误差为 `0`。
- 六个来源在进入 reader 前的 carrier 重组误差为 `0`，因此没有发现来源截断、错配或漏分配。

reader 输入是先把已经合成的 carrier 转成 action-query dtype，再进入 bf16 路由归约。若把六个上游张量分别转型后再相加，会出现可重复的 dtype/归约残差：carrier RMS 约 `2.0e-4`，路由后的 residual update RMS 约 `1.9e-4`，六源账本与真实输出的未校正差约 `0.9e-3`。这属于转换和归约顺序的数值误差，不是网络路径丢失；探针已单独记录 `ledger_cast_residual`，不能把它归因给 P3 或 bridge。

### 14.2 目标交换的当前读数

v11 在同一观测、同一噪声和同一 checkpoint 上覆盖四条颜色/方向指令；每条 baseline/swap 的事件轨迹均已写入。最终 native action 的 arm RMSE 为 `.0580–.0621`，gripper RMSE 为 `0`。这些仍来自 `global_step=64` 的短 checkpoint，只说明链路可测和下游消费存在，不能代替完整训练闭环。

### 14.3 正式长跑状态

正式训练仍锁定启动提交 `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，探针提交没有改写它。最新日志到 batch `8000/11012`，最近窗口约 `6.16 s/batch`；loss、ledger gap（约 `1e-8`）、梯度和数值项均有限，没有 traceback、OOM、NaN 或异常梯度尖峰。P2 target interval variation `.1717`、P3 temporal RMS `.3429`、P3 state-change RMS `.2852`、bottom capacity `.9997`，仍属于运行中的健康窗口；最终 checkpoint 尚未生成。

下一步是在同一分支的最终 checkpoint 上重跑这组六源账本，并把 arm/gripper、P3 optional、protected carrier 三条消费链分别与完整闭环结果对齐。当前证据不足以修改 bridge scale 或 P3 拓扑。

### 14.4 六个来源的目标交换贡献

以四条指令的同观测 target-swap 为例，按真实 bottom route 的 `beta*lambda` 逐源重算（数值为底层动作 hidden 更新的 RMSE，不是 6 维原生命令或 TCP 位移，不能线性相加成“贡献百分比”）：

| 来源 | RMSE 范围 |
|---|---:|
| P1 factual lattice `(updated-clean)` | `.000031–.000033` |
| P1 target read | `.02081–.02202` |
| P2 semantic effect | `.03651–.03921` |
| P2 geometry effect | `.00584–.00634` |
| P2 semantic interaction | `.00577–.00622` |
| P2 geometry interaction | `.00070–.00075` |

P1 target 与 P2 semantic 是当前 protected-detail 目标差异的主要两条来源；factual lattice 几乎不随这组颜色/方向交换变化。P2 geometry 和 interaction 仍有可测的下游读出，但量级较小。由于这些向量会相互抵消，RMSE 仅用于定位来源，不能把它们当作独立百分比。P1 target/factual 与 W、S 形成的 semantic 在 consequence 汇合，P1 还影响查询和交互；二者不是同一信号的连续测点，不能把其 RMSE 比值称为串联放大率。下一节区分实际生产值变化、bottom 读取系数变化和 P2 内部来源；仍不支持直接改 bottom bridge 或公共载体。

## 15. 2026-10-06：八源双状态分解、W 重读与单项 baseline 钳制

本节是在源码 `785627e9b1d875bb50b7e751f23ce479afe39845` 上新增评估探针的结果，生产网络没有修改。输入仍为 `short-bs8-gpu3-r1/checkpoints/best.pt`（step 64）、同一 standard 观测、四条蓝右/红右/粉右/蓝左指令和固定噪声 12345。交换的是有支持的 K 权重；继承探针的坐标选择不构成真实颜色物块的身份认证。每条重复两次 baseline，记录 proposal/refined 的调用 0、5；调用 5 是终点 head，不是额外积分步。

### 15.1 计量契约

`probes/probe_source_delta_consumer.py` 从真实 producer/reader 返回帧获取 W 读取、S 目标补充、W×S 调制、原始共同 P2 contract、实际 bottom probability/value scale 和 reader 输出。semantic 三项共用原始 semantic+geometry contract；八个来源共用完整 carrier 的实际 bottom 系数。新观察器不另算 softmax，不给每个来源单独收缩。

`probes/source_delta_ledger.py` 离线以 float64 计算 `D_j=sum_q(beta*lambda*X_j)` 的对称双状态恒等式：

```text
delta D_j = sum(mean(beta*lambda)*delta X_j)
          + sum(mean(lambda)*delta beta*mean(X_j))
          + sum(mean(beta)*delta lambda*mean(X_j))
```

它是数值分账，不能解释成独立因果贡献百分比。NPZ 保存 source/checkpoint/input/probe 身份及 stage、integration/call、dtype、轴；不从 JSON 摘要反推张量。实际 reader 与 float64 分账的数值残差，以及残差在两状态间的变化，单独保存。

### 15.2 四指令、16 对节点的结果

| 项目 | RMSE 范围（bottom hidden 更新） |
|---|---:|
| 实际 reader 的目标交换差异 | `.04339–.04740` |
| 源值变化项之和 | `.04359–.04755` |
| basis 概率变化项之和 | `2.37e-7–3.86e-7` |
| value 收缩变化项之和 | `.000329–.000366` |
| 两状态数值残差之差 | `.000934–.001018` |
| W 读取项的路由后差异 | `.03170–.03482` |
| S 目标补充的路由后差异 | `.01817–.01986` |
| W×S 调制的路由后差异 | `.0000380–.0000413` |

包含独立数值残差后，float64 代数闭合最大绝对误差 `8.19e-16`；这不表示生产 bf16 的残差为零。P2 semantic 三项的转型/加法残差约 `.000122–.000207`。上述实际窗口以源值变化为主，不能把报告中的合成路由案例直接当作本 checkpoint 的主因。

重复性也不是全部严格为零：蓝右/蓝左 baseline 重复一致，红右/粉右的 7 维命令重复 RMSE 分别为 `.000841/.000676`，reader 重复差异最高 `.000549`。因此不能把邻近此量级的小项当成确定机制。

### 15.3 W 预测变化和读取上下文的区分

局部 P2 进行 W0/W1 与 read-context0/1 的 2×2 配对。每份 candidate world 始终携带自己的原 action condition 并通过原身份验证；不重新标记 cache。read-context 同时包含 P1 query 和 S，这一步尚未拆开两者。

W semantic 原值差异 `.03189–.03502` 中，对称 read-context 项为 `.03189–.03492`，candidate-world 项为 `.000430–.000485`。固定 baseline W、仅换读取上下文，差异仍为 `.03187–.03492`，方向与真实差异 cosine 至少 `.99985`。这些是 **P2 共同 contract 前的 hidden 值**，与 15.2 的 bottom 单位不同。候选 W 变化项接近重复数值底噪；不能据此宣称 W 没用或未来预测正确。

### 15.4 只钳制定位到的 W 读取项

蓝右完整 proposal→W rebuild→refined 评估中，在全部 12 个对应节点把 raw W 读取项换回该节点 baseline 值。S 补充、调制、geometry、joint contract 及全部后续消费者正常重算；没有整条 lane 置零。

| 原生命令差异，相对 baseline | 正常 binding swap | swap + W 读取钳制 |
|---|---:|---:|
| arm 全 24 行 RMSE | `.06237` | `.03637` |
| arm 实际执行前 8 行 RMSE | `.03422` | `.02090` |
| gripper RMSE | `0` | `0` |

该干预确认 W 读取值确实影响最终 arm；不能把非线性差异的下降率称为独立贡献比例，也不证明命令方向正确。

### 15.5 自然语言选择仍未验收

同一观测下，相对蓝右，红右/粉右的自然 binding RMSE 仅 `.000809/.000253`，arm 前 8 行差异 `.001499/.001121`；蓝左的 binding RMSE `.007583`、arm 前 8 行差异 `.003075`。而人工 binding swap 的 arm 前 8 行为 `.03212–.03438`。自然颜色响应明显更弱，部分读数接近重复底噪；不能用人工交换有效替代自然目标选择正确，也不能从 step 64 宣判训练完成后的能力。

下一步使用同分支最终 checkpoint 复测相同分解和单项钳制，并与固定 18 例、6 任务×3、seed 0、max_steps 360、execute_rows 8 的闭环逐阶段对齐。若自然颜色重新指向仍弱，先核对真实物块覆盖、binder 的相对排名和读取内容，再判断是上游选择还是下游执行；当前没有依据扩大 gain、添加差异 loss 或改 P3/固定 bridge。

### 15.6 复现与保存位置

结果根目录为 `/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/`：`source-delta-short-all4-world-v1` 保存 16 对 NPZ、完整原生命令及 summary；`source-delta-short-clamp-w-v1` 保存单项钳制；`source-delta-short-combined-check-v1` 用于组合选项回归。初次 wrapper 识别失败的 `source-delta-short-v1` 保留为失败记录。

在当前 checkout，使用模型环境 Python、`PYTHONPATH` 指向 checkout、`CUDA_VISIBLE_DEVICES=3`、HF/Transformers 离线模式运行：

```bash
python -B probes/probe_source_delta_consumer.py \
  --checkpoint /data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/short-bs8-gpu3-r1/checkpoints/best.pt \
  --t5-condition /data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt \
  --observation-dir /data/senwang/clearvla/experiments/dinov3-online-task-global-optimized-20261002/probes/standard-observations-20261003 \
  --output-dir /absolute/new/result-directory --repeats 2 \
  --cross-world --clamp-source e_semantic_w
python -B probes/source_delta_ledger.py baseline.npz swapped.npz --output paired_result.json
```

输出目录必须不存在，禁止覆盖历史结果。离线固定源/换 beta、固定 beta/换源、同号/反号残差的代数检查通过；实际 16 对输入通过轴、有限性和身份对齐检查。文件 hash 写在各 NPZ 的 metadata 中。仓库只保存探针和决策统计，不提交 checkpoint 或 NPZ。

最终权重的独立接续脚本为 `/home/sen.wang/mysh/clearvla-gslot-source-delta-final-20261006.sh PROBE_COMMIT`，只接受本次长跑的 epoch 1 / step 11012 / source `a2d597d...`。它检查生产及探针源码未漂移，等待 GPU3 空余至少 16 GB，再执行上述四指令、重复性、W/read-context 配对与 W 单项钳制。任务状态与脚本快照保存在 `probes/source-delta-formal-final-20261006-r1-job/`；与已有 GPU4 的 R8 闭环接续任务相互独立，不重启训练。

## 16. 2026-10-06：K 坐标含义与 EMA 的实际状态

当前配置为 `entity_chart_mode=current_image_support_v1`。`grounding.py:929–935` 将完整候选位置分布推到当前相机图像，`vision/entity_chart.py:215–220` 在每个相机内归一化后求二维一阶矩，输出 `[B,K,C,2]` 的 normalized xy，`align_corners` 对应 RGB 两端像素中心。它是读分布重心，不是经标定的 3-D 位置、语义物体标签或跨时跟踪的身份保证。生产保留 C 轴；旧探针 `_select_physical_pair` 对双相机 xy 加权平均仅能定义其人工交换，不能作为真实物块间距证据。

step 64、同一 standard 观测的逐相机复测：坐标与实际分布重新求矩的最大误差为 `0`。但 200×200 上方相机内 K1 中心约 `(104,117)`、空间标准差约 `(61,36)` px；K2/K4 中心约 `(92,7)/(101,6)`，主要读背板上沿；K3 中心约 `(27,183)`，主要读桌腿/地板附近。K2/K4 的条件分布重叠质量 `sum(min(p,q))` 在上方/手腕相机为 `.9095/.8435`。因此本样本有明显背景和重复区域读取，不能把“求矩正确”当作“物块定位正确”，也不能据此外推正式最终权重。没有物体分割/标定真值，本次不报告伪精确的物块定位误差。

复测目录：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/k-coordinate-ema-short-20261006-v1/`。`summary.json` 保存 checkpoint/input/source hash 和统计；`per_camera_k_coordinates_aligned.png` 的热图节点与 RGB align-corners 端点对齐；初稿热图使用普通 imshow cell extent，最终分析只使用 corrected aligned 图。每幅热图独立缩放，不按颜色亮度比较不同 K 的总质量。

当前主线 Teacher 是 `training_targets.teacher: ObjectFutureTeacher`，其 `semantic_content_key`、`appearance_content_key` 均为冻结的 `[64,768]` 正交初始化投影，实际 current-reference mode 为 `g_assignment_v1`。`observation_association.py:76–88` 冻结参数，`:342–389` 仍用当前 G 坐标构建几何匹配先验；冻结参数不代表标签独立于 G。`training/engine.py:1187–1194` 没有 EMA 更新。旧 observation encoder 虽仍实例化 decay `.995` 的 Teacher-G 投影，但其更新只位于旧 `teacher_interval_targets` 路径，从当前 adapter 调用的 encoder 方法不可达；`teacher_supports` 只调用冻结 DINO 内容变换。

指数移动平均 Teacher/权重与 Efficient Multi-Scale Attention 是不同方案，需分清用户含义。前者可作为训练目标/权重稳定性的候选，无法凭平均生成正确的对象定位或语言标签；直接平滑 K 坐标还需先解决跨时 K 身份和相机运动。后者是视觉特征注意力，需要另行定位特征分辨率/可分性不足的证据，不能由现有 EMA Teacher 代码推断效果。本次只检查源码和短 checkpoint，没有改网络、监督或现有长跑/接续探针。

## 17. 2026-10-06：完整权重复测已接续，尚未得到长跑结果

截至 12:48 UTC，`train-bs8-gpu0-r1` 的最近日志为 batch 10300/11012，窗口耗时 6.351 s/batch；最终 checkpoint 目录仍为空。这里不把 step 64 探针结果写成完整训练结论。

补齐 K 的物体覆盖测量：复用 `scripts/export_calvin_probe_observations.py:state_for_initial_condition` 的原观测生成方式，四种布局的 top/wrist RGB 均与已保存 NPZ **逐像素完全一致**。直接导出模拟器 body segmentation，按真实物块 UID 取得红/蓝/粉可见区域。粉色块在这四种布局中均不可见，不能把零可见覆盖判作定位失败。最初按官方初始状态函数重放的图像不匹配，v2 结果被拒绝；只使用精确匹配的 v3 masks。

新测量从真实 G producer 的 `camera_read` 与 `CurrentImageSupport` 读取完整 M/N 分布，在实际采样坐标上双线性读取物块 mask，再按生产相机条件概率积分。没有用 8×8 热图格心或 K 重心估算物体覆盖。standard / step 64 的脚本校验通过：完整位置支持重算中心与生产 K 坐标最大误差 `3.84e-7`，每相机概率预算误差小于 `3e-7`。它仍只是测量校验，不代表最终权重的定位结果；此计量评估的是可见图像读质量，并非 3-D 误差或跨时身份。

新增一次性接续任务：

- helper：`/home/sen.wang/mysh/clearvla-gslot-final-k-p3-20261006.sh PROBE_COMMIT COORDINATE_PROBE_SHA256`；
- job：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/final-k-p3-20261006-r1-job/`；
- 固定探针源码 `6fdb2eb63935c3244635ee6e0f71c7fb8c775e8a`，coordinate probe SHA-256 `7ccf1e9faaec96233bdec875ed3d9dd7cc2af8f9f7cfdf3c2091aed1ec246484`；
- 只接受 epoch 1 / step 11012 / training source `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，保存 checkpoint hash；等待既有 source-delta job 结束及 GPU3 至少 16 GB 空闲；
- 四布局逐相机 K/物块覆盖，然后四指令、重复 2 次、完整 proposal→W rebuild→refined 的 baseline、P3 semantic zero、bottom P3 zero、protected-detail zero；
- 保留 15.6 的八源/W 单项钳制和 GPU4 上既有的标准闭环（18 例、6 任务×3、seed 0、max_steps 360、execute_rows 8）。

helper 的 shell/嵌入 Python 语法、当前源码 guard 和短 checkpoint 的实际测量均已验证；任务已启动等待最终权重，尚未完成这些 full-checkpoint 测试。每阶段失败独立记账，不能因部分成功宣告全部通过。标准闭环与探针结果要结合分析：优先核对自然语言选择、arm 前 8 行、gripper 和重复底噪。人工 K 权重交换尚无真实物体身份保证；整条 lane 置零是上下文依赖的非线性干预，不据此计算独立贡献百分比或直接修改 P3/bridge。

外部测量源码、helper 快照和结果保存在上述实验的 `probes/` 中；不修改已被两个接续任务锁定的生产/探针源码。分割原始数据与图像不纳入仓库文档。

## 18. 2026-10-06：六份历史正式 checkpoint 的 K 空间读取

用户澄清要看历史版本，而非 step 64 短跑。本次补测六份已完成正式训练的 best.pt，均保存 epoch 1 / step 11012；训练 batch size、初始化和配置不同，因此只作行为描述，不是受控的修复收益实验。为避免用新网络解释旧权重，每份 checkpoint 使用其 git commit 的独立源码快照，逐文件 SHA-256 与 checkpoint.identity.source.files 全部核验通过。

输入统一为已保存的 standard NPZ、蓝块向右指令、与既有探针一致的重复帧/四步 executed-world history。调用各版本自身的 act_with_input，截获采样前输入，再执行原 deployment_cache；G producer 和 per-camera 条件读取均来自该版本原函数。没有在最新生产 checkout 中切换源码，也没有改权重。物块 mask 沿用第 17 节逐像素匹配的模拟器分割。

| checkpoint source | 历史阶段 | 上方相机 K 两两平均重叠 | 手腕相机 K 两两平均重叠 |
|---|---|---:|---:|
| 617fc567 | 初期优化 bs8 | 95.17% | 93.19% |
| 18750208 | bs4 r1 | 62.81% | 53.05% |
| f405c877 | W/global 合并修复 | 92.26% | 94.20% |
| 4f0a5b9b | identity/trajectory 修复 | 95.04% | 86.51% |
| eb9e62fb | deep-repair r2 | 84.59% | 85.19% |
| d931fe01 | S-interval repair 正式长跑 | 41.73% | 58.85% |

重叠定义为每相机六对 K 的 sum(min(p_i,p_j)) 均值，100% 表示同一空间读分布；它不是语义向量 cosine、对象识别率或独立实体数。早期多份权重存在强空间重复；d931fe01 分布分离增大，仍不能据此认定一槽一物。

d931fe01 的 top 像素重心依次为 (88.5,120.8)、(53.5,116.5)、(37.9,106.2)、(104.0,21.6)，wrist 为 (50.3,25.8)、(41.9,22.4)、(42.9,51.0)、(57.9,50.4)。K1 读较宽桌面操作带，K2/K3 的 top 读分布重叠 79.11%，K4 偏上方背板边缘；腕部 K3/K4 有大量下方背景质量。实际可见红/蓝物块上的读质量：top 各 K 为 0.564/0.352/0.136/0.911%，wrist 为 5.748/5.245/1.580/2.526%。此结果是采样位置对可见 mask 的概率质量；上下文视觉特征仍可能携带区域外物体信息，不能把该量直接称为语义信息占比或定位准确率。

同一蓝右指令下，4f0a5b9b/eb9e62fb 的 S real mass 给 K3 约 99.80/99.99%；d931fe01 为 K1–K4 19.39/13.70/42.52/10.12%（余量为 null）。只有一条指令，不能由此宣称所有指令均选 K3。单画面也没有验证跨视角语义身份或跨时间跟踪。

证据根目录：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/historical-k-slots-20261006-v1/`。manifest.json 固定 checkpoint/source，probe.py 为测量快照，六个提交目录各含 summary/分布 NPZ/图，comparison.json 为紧凑比较，training-audit.json 已核对六个完整 epoch 记录。复现参数保存为各提交.command.json；应在 manifest 中对应源码目录、GPU5、PYTHONPATH=该目录执行，输出必须使用新目录。仓库只存本节统计，不保存源码快照、模型或图像。

## 19. 2026-10-06：最终权重、完整验证与标准闭环

最终训练身份为 `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，`train-bs8-gpu0-r1/checkpoints/best.pt`，epoch 1 / step 11012。文件 2,468,580,588 bytes，完整反序列化通过，SHA-256 `5ca168e3f4f33772dd01aca26adcc1b00dfc029b526a950f3b534d5428d19d15`；配置 digest `bfb2257904a6d07f3e66760f667b46f33ec6b827d76980495355fc70b386ee86`。评估 checkout 的生产文件逐项匹配 checkpoint source identity；训练完成后新增的代码为评估探针。DINO 在线冻结、bs8、无 RGB/特征缓存，初始化来自本分支 step64，正式 optimizer/schedule/RNG 重新开始。

原 watcher 使用 simulator NumPy1 环境读取 NumPy2 checkpoint，因 `numpy._core` 导入失败退出。接续修复使用 model Python 验证权重，并消除预建 case 目录与 evaluator 的冲突。失败目录保留；完成结果为：

`/data/senwang/clearvla/experiments/calvin/dinov3-gslot-identity-carrier-20261005-train-bs8-gpu0-r1-closed-loop-replan8-r2`

固定 **18 例、6 任务各3、seed0、max_steps360、execute_rows8**，完成18例、成功11例（61.11%）、错误0。实际 chunk adapter 每次执行前8行；bridge health 中的单行 ABI 字段不能当成重规划周期。仅为本固定面板的成功率。

| 任务 | trial1 / trial2 / trial3；括号为执行步数 | 成功数 |
|---|---|---:|
| 蓝左 | 成功195 / 失败360 / 成功348 | 2/3 |
| 蓝右 | 成功62 / 失败360 / 成功61 | 2/3 |
| 红左 | 成功60 / 成功60 / 失败360 | 2/3 |
| 红右 | 失败360 / 失败360 / 成功56 | 1/3 |
| 粉左 | 成功61 / 失败360 / 成功352 | 2/3 |
| 粉右 | 成功57 / 失败360 / 成功58 | 2/3 |

上一份相同初始状态 manifest 的正式结果为 d931fe01、8/18：本次新增成功01/03/07/08/13，退步02/09。旧 summary 的 source_commit=4f0a5b9b 是过期派生字段；其 launcher 与 bridge checkpoint health 均确认 d931fe01，权重 SHA-256 为 `ef10a6c1da2030362e4be03ba11497198eca7142234b5c648c4df98fb7499011`。

训练审计覆盖完整 epoch、221个日志窗口；无 traceback/fatal，loss 首/尾窗口中位数 .78202→.45201，physical flow .66701→.39408。完整离线验证 **256批、2046样本、每任务341样本**，对比使用相同 action normalizer：

| 完整验证 / 运行指标 | 本次 a2d597d | 上次 d931fe01 |
|---|---:|---:|
| 全24行 source-native action RMSE | .262813 | .268038 |
| 前8行 source-native action RMSE | .232714 | .230691 |
| 全24行 arm native RMSE | .132131 | .133958 |
| gripper event F1 | .330204 | .347098 |
| step耗时中位数，s/batch | 6.22875 | 6.11251 |

8批 execution-ablation 子集的 .314071/.322129 不属于上述全量指标。显存峰值 allocated 21.53 GiB / reserved 22.34 GiB。尾段实际预算 action/execution/representation 为92.34/4.74/2.91%；账本闭合。W connectivity 220窗口均为1，delta/transport/covariance 动作损失 VJP 尾值约1.33e-4/7.32e-5/5.96e-6；P3/bridge raw gradient L2 约 .16086/.004677。模块梯度存在不认证目标语义正确。

S semantic/geometry interval variation 尾值约 .04081/.04032，公共区间载体 .12227；G content cosine 降至 .6245。这些健康度改善须与下文的自然目标选择分开验收。gripper-command-transition 的实际权重为0；配置中存在 gripper-trajectory 权重也不代表当前 binary 路径使用它，实际该项损失为0。不可把历史已实现的监督项写成本次已启用。

## 20. 三项计量修正：交换幅度、指令参考和动作单位

1. 原“最远坐标”规则在最终 checkpoint 选中 K2/K4，binding 仅 .029589/.040128，交换幅度 RMSE=.007453；短权重的交换幅度不同。两次动作差异不能直接解释为消费能力随训练降低。standard 六对全枚举后，真正反归一化的 arm前8行 RMSE 为 .000379–.014333；其中 K2/K4 接近重复底噪。强交换仍是人工 K 置换，保留总 real mass、null、熵及集中度，尚无真实物块身份保证。
2. 轨迹探针现在先以该例 state0 建立 instruction-start reference，再输入实际 t 时刻的 causal history。原 v3 在 t 重新建参考，故其下游值只能视为另一上下文；原始文件保留。最终采用 `measurement-trajectory-anchor0-20261006`、strong W/fork 的 `anchor0` 结果。probe noise固定12345，与正式 rollout 的随机序列不同。G参考不变性另验02的24/216：content/semantic/appearance/geometry/binding差异为0，坐标最大差3.58e-7，支持保留既有 goal-free K 轨迹测量。
3. **旧 `probe_target_binding_measurement_v2._final_nodes` 将 `sampled.action` 命名为 native，实际它仍是 normalizer chart。** 生产 `clearvla/simulation/clearvla_policy.py:386–405` 还会 `action_normalizer.decode`，再用 binary command 覆盖 gripper。第12–15节引用该探针的 arm 数字应重新标记为归一化 chart；正式训练 native 验证和本次实际执行 telemetry 不受此命名错误影响。本次已用保存的完整动作张量离线重算，normalizer SHA-256 为 `7e1b3c4d179c0151c6764c10caaca846291594c83547dfd71bb4dd4db4e93b2b`。下文命令数值全部使用反归一化后的 CALVIN relative_7d 原生控制量，不能称为米或真实 TCP 位移。

探针现要求显式传入 checkpoint normalizer，保留 `final_sampled_action_chart`，正确生成 `final_native_action`，输出 schema v3 和 units 元数据。两个 CPU 回归分别验证 arm affine decode+binary覆盖、continuous gripper decode，均通过；model环境无pytest，直接调用同一测试函数并保存结果。已完成的旧NPZ不覆盖，修正值另存 `native-command-correction-20261006.json`。

## 21. 18条轨迹的阶段证据

目标初始 world-x≈.05 的8例全部在56–62步成功；目标 x≈.23 的10例只有01/03/15成功，分别耗时195/348/352步。state24 时，所有左推任务TCP-x落在 .0345–.0473，所有右推落在 -.0373–-.0299。轨迹和图像显示稳定的近处操作区域偏向，部分例中该处甚至没有目标物块。

| 窗口 | 实际过程与分离位置 |
|---|---|
| 01、15早期24–40 | 01先接触粉块30–58；15先接触蓝块30–35。目标仍在较远处，错误在首次接近阶段已经形成。 |
| 04/06/07/08/12/13/16/18 | 目标恰位于近处，原有接近/推动序列能完成任务。16没有控制步末采样到物块接触，仍被官方oracle判成功，说明“未采样到接触”不能当成从未接触。 |
| 02的80–112；09的64–80 | 碰到固定桌体/按钮机构，控制器内部target继续积累、实际TCP受阻；最大分离分别15.92cm/11.70cm。解除接触后恢复跟随。错误接近先于此阻滞；不是仅由推理耗时解释。 |
| 01的160–195与02的208–232 | 01重新靠近并于174–194持续接触蓝块，176–192完成约10.6cm左移。02仅217–219短暂触蓝，整体最大左移1.87mm，随后再次离开；回到物体附近不等于重新完成任务。 |
| 05/10/11与17撤回 | 05先触红、11先触粉；目标最大右移6.38/2.52/5.25cm后停滞。17在128–136推动粉块约4.28cm，136窗口原生命令开始抬升/反向、140松爪，剩余目标未完成。 |
| 14与15的328–352 | 两者均终于到粉块附近。14在336–344先向错误方向推约3.46cm，之后内部target继续下压；354的TCP误差7.38cm且有强物块接触，最终净左移未达标。15在336–352左移约10.64cm、352成功。夹爪在接触时的状态与姿态不同，不能只看接近距离。 |

原生请求、controller target、实际TCP和关节、已执行chunk行号均保存。源码 `calvin_env/robot/robot.py:228–236` 在 `use_target_pose` 下累加内部target，解释受阻时“请求继续、实体不动”的位置；其是否被 W/恢复策略准确预测仍未认证。

按执行长度归一化，本次成功/失败组夹爪切换为 **3.14/2.78次每100步**；前8行与24行均值向量反向比例为 **8.52%/7.62%**。因此不能沿用“失败组更频繁切换、更多前缀反向”的旧版本结论来解释本次全部失败。逐例早退和夹爪异常仍可定位；大多数命令没有裁剪饱和。

## 22. G/K到S：空间分离改善，物体选择仍弱

standard最终权重的 top/wrist K两两平均重叠为19.05%/42.56%（d931为41.73%/58.85%）。但 top 的K2/K4仍主要位于图像边缘/背景；K1/K3重叠65.10%，空间标准差很宽。可见红蓝物块上的 top 读质量每K仅约 .000004–.000218。这里衡量采样位置对可见mask的质量；DINO上下文特征可能携带位置之外的物体信息，故此量不等于语义信息占比。

01/02/04的多阶段RGB经执行命令重放，双相机逐像素匹配后导出真实body mask。以02 state24蓝块为例：top可见面积 .005775，**G竞争前 producer-prior 的蓝块读质量已仅2.14e-5**；故不能把全部物体缺失归因于最后一次K竞争。wrist中K4蓝块读质量 .0670，高于K3的 .0187，但S给K4/K3的质量约 .0189/.5159。恢复时K3通常仍占主导，并未形成已认证的物体身份切换。跨相机、跨时间“一槽一物”目前仍没有通过证据。

自然指令探针直接截取生产S/binder内部张量。02 state24，蓝左改红左/粉左，与蓝左改蓝右的对照如下；相对量为差分RMSE除以基线RMS，**不作为信息保留百分比**：

| 节点 | 红左 | 粉左 | 蓝右 |
|---|---:|---:|---:|
| raw T5 relative Δ | .4333 | .3858 | .4609 |
| S goal_input relative Δ | .1930 | .1681 | .8243 |
| goal_read relative Δ | .04757 | .03672 | 1.1475 |
| protected goal relative Δ | .05246 | .03940 | 1.1537 |
| interval_goal创新 relative Δ | .008738 | .004404 | 1.7504 |
| shared binding绝对RMSE | .000221 | .001176 | .037399 |
| 原生arm前8行RMSE | .000500 | .000277 | .067081 |

该窗arm重复底噪 .000232–.000301。红色变化的绝对goal值差分 .02092→.02221→.02346，并未在这三步归零；主要可见的是方向强选择性、颜色相对共同量很弱，随后binder相对logit/排名几乎不变。不能仅由relative RMS降低断言不可逆信息截断。

02 state216/320、01恢复160，以及04成功32的原指令家族均复核了颜色响应弱。04蓝右改红右/粉右的原生arm前8差异 .000217/.000255，改蓝左 .083824。物理场景、goal输入和普通执行路径保持相同；微小颜色下游值接近重复底噪，不能据此给P3计算可靠衰减比。

源码定位：`intent.py:793–803`（masked T5→投影→4个goal queries→goal_self），`:911–922`（protected goal与pooled G object facts进入共享binder），`:971–973`（interval_goal再次读取）；`task_execution.py:532–606`包含真实 task/object 交互评分。当前没有恢复旧的可分离评分错误，但压缩后的语言、对象值及相对评分已经不产生有效颜色重指向。数据/监督偏向与该汇总接口各自的因果程度还需窄范围干预区分。

## 23. W、protected-detail、P3与最终命令

在02 state216，保留state0参考、同一噪声，交换支持K3/K2的质量 .562025/.011484。八源账本实际reader差分 .12027–.12122，源值项之和 .12034–.12127；basis变化仅7.0e-7–1.1e-6，contract项3.6e-5–7.4e-5，双状态数值残差约 .000913–.001032。含残差的float64闭合最大误差6.67e-16。本组未发现来源截断、错配或漏分配。

W raw semantic差分 .11598–.11720；2×2分解中的read-context项 .11600–.11729，candidate-world项 .000459–.001082。该干预主要改变P1 query/S如何读取W；不能据此认证W未来预测准确，或断言W不需要。全12个采样节点仅把W raw读取项钳回baseline后，原生arm前8差异 **.009926→.006394**，全24行 **.017658→.013304**；gripper保持相同。向量重排存在，不转换成独立贡献百分比。

相同strong交换的四路分叉如下。左列衡量删除lane本身对原baseline动作的影响，右列是在该lane条件内再交换binding的响应，均为原生arm前8行RMSE：

| 条件 | 删除lane对原动作的改变 | lane内目标交换响应 |
|---|---:|---:|
| baseline | 0 | .009863 |
| P3 semantic输入置零 | .002184 | .009606 |
| bottom P3两可选lane置零 | .040191 | .009959 |
| protected-detail置零 | .071189 | .002394 |

重复底噪 .000254–.000294。P3在参与动作生成；本组人工目标交换主要依赖已经融合P1/P2的protected-detail，P3不是这一响应的主要必经通路。完整权重已有消费能力，现有证据不足以把问题归为“P3断路”“固定0.25缩放导致全部失败”，也不足以由单个窗口外推所有任务。

## 24. 证据排序、下一步验收与复现

下一轮按以下顺序处理；本次完成评估和探针单位修复，尚未训练新网络。

1. **优先验证S的目标选择接口。** 在同一共享binder中检验目标相关的逐token语言值能否在全局汇总前与对象事实交互；保留单一K+null绑定、S所有权和普通梯度契约。先做局部producer/consumer替换与梯度核验，再决定结构或监督修复。验收要求同一图像换颜色确实换到对应物体并改变正确方向的原生命令；共同RMS变大不算验收。
2. **并行约束对象证据的真实性。** 以已匹配的逐相机mask和时间序列作评估标尺，追 producer-prior、上下文内容及G分工；当前分离更大仍可能只是背景分工。正式输入契约保持不引入模拟器真值。EMA可另测稳定性，其本身不能生成正确物体/语言标签；当前g_assignment Teacher仍依赖G先验。
3. **处理接触后任务维持与恢复。** 优先17的128–144、02的208–232、14/15的328–352，将剩余关系、实际执行前8行、gripper、W预测/已执行创新一并对齐。只有确认哪一消费边界对正确接触证据不响应，才修改那一边界。现有证据保留P3与protected-detail的不同作用，不先整体放大bridge。
4. 下一候选同时通过普通action-loss反向、source账本、原生命令与重复底噪检查，再跑同manifest的18例。全量验证前8行和gripper事件已有退步，须纳入验收，不能仅以24行平均误差或槽位健康度放行。

证据根目录统一为 `/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes`。紧凑总账 `final-evaluation-audit-20261006.json`；训练 `formal-final-training-audit-20261006.json`；全部阶段 `closed-loop-stage-audit-20261006`；K时间证据 `trajectory-k-20261006-v2` 及 `trajectory-k-anchor0-invariance-20261006.json`；自然指令 `measurement-trajectory-anchor0-20261006`；强W `source-delta-case02-state216-maxcontrast-anchor0-20261006`；分叉 `fork-trajectory-strong-anchor0-20261006`。原始图像、NPZ、checkpoint和失败记录保留在实验目录。

复现当前已保存结果的统计与单位修正（不启动训练或闭环）：

```bash
cd /data/senwang/clearvla/checkouts/dinov3-g-slot-logit-scale-20261005
export PYTHONPATH="$PWD"
clearvla_probe_root=/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes
/data/senwang/envs/clearvla-sim/bin/python -B "$clearvla_probe_root/decode_saved_probe_commands_20261006.py"
/data/senwang/envs/clearvla-sim/bin/python -B "$clearvla_probe_root/audit_closed_loop_trajectory_20261006.py" \
  --root /data/senwang/clearvla/experiments/calvin/dinov3-gslot-identity-carrier-20261005-train-bs8-gpu0-r1-closed-loop-replan8-r2 \
  --output "$clearvla_probe_root/closed-loop-stage-audit-20261006"
```

采样时源码为fd89fb71，外部wrapper hash与输入身份保存在结果中，队列脚本/command.json保存精确参数。重采样使用新输出目录；新v3读出直接产生正确native单位，离线修正脚本只用于上述保留的旧chart张量。标准闭环、两个原接续任务、anchor0补测及最终审计均已完成。

## 25. 2026-10-06：逐一核验七例失败与真实物体身份

本轮沿用第19节完整 checkpoint（step11012，SHA-256 `5ca168e3…19d15`）及同一18例正式闭环；没有训练或改动生产网络。当前 checkout 的 `clearvla/` 与训练提交 a2d597d 无差异。物理结论来自原始执行，模型干预来自冻结权重的观测快照，两者不混用采样随机序列。

证据根目录为 `probes/failure-object-identity-20261006-v1/`。检查全部18例、3890条真实执行命令，选取135个重规划状态（主测122，针对间接碰撞补测13），另做11窗自然指令干预及5次局部物理重放。仅保存针对性统计、图像mask与短段接触账本；没有导出全网络内部张量。

**物体真值**：模拟器 body UID 与 `environment_info` 的红/蓝/粉命名、连续位姿及 `scene_obs` 对应列逐步交叉核验。每个模型窗口的两路RGB都先按原始命令重放，逐像素匹配，再接受对应的body分割。初版同一模拟器连续reset导致02 state88像素偏离，其结果被拒绝并保留；采用每例独立进程的 `masks-r2`，18例全部通过。仓库脚本 `replay_identity_masks.py` 已固定此进程隔离契约，并通过02及补测案例验证。

读取覆盖直接积分G生产者的完整条件位置分布与可见mask；坐标重算闭合最大误差 `9.07e-7`。目标在top/wrist可见的窗口分别134/107，所有K中最大目标mask质量的中位数仅 `0.0001113 / 0.03247`。这些是可见位置的采样质量，**不是语义信息比例**；DINO区域外上下文不能由mask排除。跨视角/跨时的一槽一物身份仍未通过认证，不能给K1–K4直接冠以固定红蓝粉名称。

### 25.1 S目标重指向与实际物体证据

135/135窗口的S最大绑定项都是K3；11窗×2次换颜色的22个干预也全部仍以K3为首位。它本身不证明K3固定代表某个物体，但结合goal-free G及同画面颜色干预，不能解释为正确的语言目标切换。

| 同观测/历史/初始参考、相同采样噪声 | arm前8行原生RMSE中位数（范围） | binding最大元素差中位数 |
|---|---:|---:|
| 同指令重复 | 0.0002108（0.0001548–0.0003064） | 0 |
| 只换颜色 | 0.0003448（0.0002094–0.0006706） | 0.001818 |
| 只换左右方向 | 0.05024（0.01976–0.07483） | 0.07092 |

换颜色的22次测试，前8行夹爪命令均未改变。颜色信号并非在最初投影中归零：goal_input、goal_read、protected_goal的颜色差分RMSE中位数为 .01935/.02269/.02307；但coarse action、P1 detail仅 .002049/.000555。P3 temporal的颜色差分 .003940，重复本身已有 .003406，因此不再用此近底噪差分计算“P3衰减率”。不同节点单位/维度不同，不把它们的RMS比直接当信息守恒率。

17 state120中，wrist的K2/K4读取粉块质量为80.55%/87.91%，而S给它们的质量仅2.59%/3.23%，K3为45.96%。state136撤回前，wrist K4仍有53.33%落在粉块，粉块可见420像素；不能用“目标已完全不可见”解释撤回。另一方面，两个K在同一相机强读同一粉块，也不能算两个独立物体身份。

05 state64的S加权wrist可见读质量，红/蓝为 .10746/.02248；11 state80粉/红为 .08715/.03404。此值只是共同binding加权的逐相机mask读质量，不是物体身份后验；它与下节真实碰错物体的接触记录方向一致。

### 25.2 进一步核验：相机分工不能当对象身份

真实红/蓝/粉UID为2/3/4，对应 `scene_obs` 的xyz起始列6/12/18。追加02 state24、17 state120/136的G→S接口探针，直接取生产全局读质量，得到：

| 窗口 | K1来自top | K2来自top | K3来自wrist | K4来自top |
|---|---:|---:|---:|---:|
| 02 state24 | 92.08% | 99.998% | 99.912% | 99.998% |
| 17 state120 | 97.06% | 99.993% | 99.568% | 99.974% |
| 17 state136 | 89.43% | 99.793% | 99.714% | 99.978% |

因此，上节“wrist条件下K4读取粉块较多，但S绑定较少”**不能单独证明S放弃了更好的完整对象**：17 state120时K4的wrist全局质量仅0.0263%，其87.91%的条件粉块覆盖，经全局相机质量加权只剩约0.0231%。K3则主要读wrist。这3个窗口的K分工明显带有相机偏向，不能把K2/K4的条件读取当作独立物体，也不能把K3的首位绑定直接解释为正确颜色选择。

源码对应 `grounding.py:764–776`：content为全局视觉汇总加slot残差，typed值另有其重加权；`:988–994`才导出每相机条件值。`intent.py:854–873,911–922`用pooled content/typed构成共享binder输入；逐相机值随后进入目标证据。重构全局content的相机加权闭合RMSE约1e-7，重建S的objects最大误差0，接口测量已闭合。

追加两个**仅在静态binder输入**发生的定位干预，其他typed值及共享绑定契约不变：去掉content的slot残差；或仅将content视觉项改为有效相机等权并保留slot残差。3窗×3颜色中，两种干预都仍选K3。换颜色的binding最大元素差分别仍仅约 .0016–.0038、.00028–.0055；未恢复物体重指向。它们不是训练结果，也没有重新执行下游动作，结论仅是：不能把删除slot残差或平均相机content当成已验证的完整修复。

采信 `binding-interface-r3/summary.json`；前两次接口探针因FactSet/WorldBelief封装识别错误退出，未产生可用记录，失败日志保留。生产代码未受影响。下一步需联合定位相机/背景分工、pooled typed对象值与语言条件评分，避免再次只放大一个共同量。

## 26. 七例失败的阶段链与接触因果

官方 `calvin_env/envs/tasks.py:86–119` 的push判据是指定方向位移严格超过0.1m，并保持初始非机器人支撑body/link；本轮没有更换判据或把重规划改成1。

| 失败例 | 真实物理过程 | 已证实的分离位置 |
|---|---|---|
| 02 蓝左 | 早期进入近处操作区并撞固定桌体；208后靠近蓝块，217–219短暂接触；最大正确位移仅1.87mm | 错误接近发生在控制器阻滞之前；接近恢复后也未形成持续有效推动 |
| 05 蓝右 | 先推红块；红块在68–73、128–144附近与蓝块碰撞，蓝块最终右移约6.38cm | 目标位移主要是推错对象后的间接碰撞，不能写成已正确选择蓝块、只差一点距离 |
| 09 红左 | 先在近处撞桌，后于258–262压到红块；其中心从z≈.460降至≈.420，转到桌体内部下层碰撞面 | 晚段包含明显的下压/穿透异常，不能把之后目标可见性下降全归给G；最大左移3.71cm仍不达标 |
| 10 红右 | 首轮较早撤回；第二次132–137短暂擦到红块，只推约2.52cm | 控制步末日志没有机器人-红块接触，但物理子步补测确实有19/128步接触，不能称为“从未触到目标” |
| 11 红右 | 先推粉块；粉块在70–92附近撞红块，红块推进约5.25cm | 与05相同的错误对象→间接目标移动，随后未继续完成目标 |
| 14 粉左 | 328后到达目标；开爪下压段先把粉块推右约3.5cm，随后接触约束和控制器误差增长 | 单因素夹爪重放可逆转结果，见下；并非仅靠近距离或目标颜色未分清 |
| 17 粉右 | 128–136有效推进约4.28cm，之后明确反向、抬升，140起开爪，最终最大推进4.41cm | 当时控制器误差不足1cm、目标仍可见；这是策略发出的提前终止/撤回，尚不能等同于S已判定完成 |

05/11的物体间接触由真实body对确定；0力接触不计入有效力记录。10的事实重放在每个控制状态均保持双相机RGB与原轨迹一致，然后才采集8倍频物理子步。原telemetry只在每个控制步末取样，接触缺测属于采样粒度限制。

### 26.1 14：只改夹爪、保持原手臂命令的局部反事实

在原始状态0起完整重放，state328前RGB完全一致；328–360使用相同原始xyz/rotation命令，只将夹爪固定为闭合。该实验是**保留记录动作的局部物理干预，不是重规划策略的闭环成绩**。

| 条件 | 最终粉块向左净位移 | 官方push判据（段末） | 单接触点峰值力 |
|---|---:|---:|---:|
| 原命令事实重放 | -0.931cm（向右） | 失败 | 965.3N |
| 手臂相同、夹爪保持闭合 | +10.194cm | 通过 | 56.64N |

14在state338–342虽仍请求开爪，实际开度在接触下从约7cm缩至约1cm；`robot.py:355–369` 的开爪电机力仅为闭爪的1/100。原接触法向主要竖直向下，15成功窗口的初次接触则有显著左向分量。因而当前证据指向“夹爪状态×下压轨迹×接触几何”的联合错误，而不是夹爪符号写反。不能把固定闭爪推广到所有任务。

### 26.2 09：异常下压后，视觉证据变化是结果的一部分

事实重放的目标-手指单点峰值约1091N；state262桌面穿透深度已约1.27cm，state264开始接触桌体z=.4的下层面，而原支撑面z=.44。官方支撑检查仍看到同一个body/link，因此该布尔值不能审计出“落到同body另一表面”。图像也显示红块逐渐沉入桌面遮挡。

只把256–280的继续向下命令设为0，会消除这次接触及4cm高度下降，但目标也不再被推动、官方仍失败。这个负对照只确认异常下压的物理来源，**不是合格的修复**。应先改善接触姿态和受阻反馈消费，不以更改官方物理参数或放宽oracle来提高成功率。

## 27. 本轮后的修复优先级与证据边界

1. **目标选择链优先。** 同画面颜色响应弱已扩展到11个窗口，错误对象接触又由05/11真实碰撞链印证。必须联合检查G的真实物体证据、S的pooled对象值与逐token语言交互；不能把K空间分散或公共载体变大当作成功。候选验收是换颜色能重指向实际物体，并传到正确原生命令。
2. **夹爪与接触段单独修。** 14已经有单因素物理干预证据。下一步追binary夹爪状态/事件消费者、实际监督权重与前8行arm-gripper配合；本次已有的零权重项不能当已启用。强制闭爪只作定位对照，正式修复必须保留其他任务的开合能力。
3. **任务维持与受阻恢复。** 17剩余位移超过5.5cm仍主动离开；02/09/14存在请求-实际执行分离。需要对齐剩余任务关系、coarse/P1/P2、W已执行创新与二值夹爪输出，区分哪里先产生退出计划。现有证据尚未证明某个P3/bridge缩放是根因，不作全局放大。

所有候选都应通过实际对象身份、自然指令重指向、普通梯度、前8行原生命令/夹爪和同18例闭环验收。未查清的部分包括：G内容中区域外对象语义能保留多少、语言压缩与对象值偏置各自的因果份额、17在哪一个上游节点首次产生提前退出。报告中的已证实物理因果与这些网络根因候选分开记录。

复现入口（工作目录为本分支checkout，PYTHONPATH指向该目录）：

```bash
python probes/build_identity_phase_plan.py --root CLOSED_LOOP_ROOT --output NEW_AUDIT_ROOT
# 模拟器Python；每个case独立进程、双相机逐像素验收
python probes/replay_identity_masks.py --plan NEW_AUDIT_ROOT/probe_plan.json --output NEW_MASK_ROOT
# 模型Python；完整checkpoint；coverage脚本为实验已有 probe_k_coordinate_full_20261006.py
python probes/probe_physical_object_identity.py --plan PLAN --masks MASKS \
  --coverage-script COVERAGE_SCRIPT --checkpoint CHECKPOINT --output NEW_PROBE_ROOT
python probes/summarize_identity_audit.py --input MAIN_PROBES ADDON_PROBES --output SUMMARY
python probes/probe_contact_replay.py --case CASE14 --start 328 --end 360 \
  --variant gripper_closed --output NEW_RESULT.json
python probes/probe_binding_interface.py --plan PLAN --checkpoint CHECKPOINT --output NEW_INTERFACE_ROOT
```

上述大写占位符的本次精确实参在证据目录的 `identity-probes-command.json`、`identity-addon-command.json`、`replay-jobs.json`；mask每次必须使用新目录。补测即使第一个选中状态晚于0，也先以state0建立instruction reference，并显式断言。主测使用的原始探针文件已按hash保留；之后仅补了这个晚起点计划的保护。汇总为 `summary/`、物理干预为 `contact_intervention_summary.json`，失败重放和原始大文件仍留在实验目录。

## 28. 2026-10-06 根因追查：两处可复现的接入问题

继续使用 epoch1/step11012 的完整 checkpoint（SHA `5ca168e3...d19d15`），未改生产源码、权重或正式评估配置。证据目录：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/failure-root-cause-20261006-v2`。以下生产源码经 `git diff a2d597d... -- clearvla/...` 确认与训练时一致。§27 的候选判断以本节新增证据为准。

### 28.1 终点目标监督实际全程缺席：原始标注编号漏传

这是已闭合的源码—数据—日志链，不是“终点头可能训练不足”：

1. 旧 HDF5 缺少 `source_annotation_index`。`virtualize_calvin_cached_prefix` 已用 raw reader 严格核验 split、指令、task、source/context bounds、真实前缀长度，但 `clearvla/benchmarks/calvin_raw.py:974–986` 的 `replace` 漏传了已有的 `raw.annotation.index`。
2. `clearvla/data/annotation_endpoint.py:23–36` 因任一 provenance 字段为空而返回 `unknown-provenance`。这里拒绝凭空编造标签的保护是正确的，问题在已核验来源没有接进来。
3. `clearvla/mainline/data/dataset.py:728–750` 将 declared/state-observed 设为 false；端点视觉也按该声明屏蔽。训练2165、验证280、测试154个 episode 全部 unknown，labeled windows 均为0。
4. 配置虽为 `annotation_goal_mode=annotated_endpoint_relation_v1`、权重0.01，221个训练日志采样点的 scene/relation/robot/total、两项coverage及weighted contribution 全为0；完整离线验证亦为0。这个分支仍可通过动作损失间接学习，不能称所有参数“从未更新”，但没有获得设计中的直接终点监督。
5. 独立探针用**现有生产 loader、overlay 和 resolver** 加载全部2599个 episode；只在探针 dataclass 中填入严格匹配的 `raw.annotation.index`，全部恢复为 `annotated-end-observation`，且 endpoint 均在真实缓存前缀内。RGB、state、action、split、窗口和checkpoint均未修改。恢复的是有来源的末帧观测，**不是成功标签**。

直接修复边界已经具体：在 overlay 接入准确 annotation index，已有编号若与 raw 不一致必须报错；在正式训练 admission 增加“启用监督但整份训练集标签覆盖为0”的报错。不能靠取消 provenance 检查、从文件名猜编号、放宽label mask解决。

### 28.2 训练动作与部署控制器采用不同的相对起点

原始 CALVIN `utils/utils.py:160–171`：`rel_xyz = clip(raw_actions_xyz - measured_tcp_xyz, ±0.02) / 0.02`。跨51个训练片段抽查153帧，和原始 `rel_actions[:3]` 的最大差为 **0**。这里的raw动作标签不能直接等同示教控制器内部held setpoint；其renderer来源与本轮补查见§33.1。

当前环境 `robot/robot.py:228–242`、实际 `use_target_pose=True`：`stored_target_next = stored_target_previous + 0.02 * rel_xyz`。18条完整轨迹的控制器目标递推最大误差均为 **0**。接触受阻后两种起点分离，旧目标误差会继续被积累；模型在线观测只有实际proprioception和命令历史，不含这个控制器目标。

四个窗口各做一对新进程物理重放，事实前缀双相机RGB逐像素相同；保留全部原手臂/夹爪命令，只将干预后的起点切到实际TCP：

| 窗口 | stored-target 最大目标/TCP差 | measured-TCP 最大差 | 额外物理结果 |
|---|---:|---:|---|
| 02，32–120 | 15.92cm | 0.675cm | 两者都未接触蓝块；漂移减少不等于目标选对 |
| 09，240–280 | 1.044cm | 0.476cm | 目标最低z由0.4174恢复0.4600；1091N异常接触消失，但没有推动 |
| 14，328–360 | 7.381cm | 0.621cm | 965N异常接触消失，但没有推动 |
| 17，136–168 | 1.158cm | 0.700cm | 最终右移由4.413cm变5.211cm，仍未完成 |

这是明确的动作参考点不一致，以及它对漂移/异常接触的因果证据；**不是新策略的18例成功率**。CALVIN所带环境本身默认该累积方式，因此后续适配必须显式记录执行语义变化，在相同18例/replan8下单独复测，不能把两种执行配置混算。也不能只改 anchor 就宣称解决颜色选择或撤回。

## 29. G/S：相机分工的直接来源与训练数据中的捷径

02/state24 候选 `content_key` 的相机均值差 RMS=1.112，视角内变化RMS=0.358；完整candidate归一化后分别0.648/0.348。同一当前G输入下，仅消去投影content的相机均值差，K3腕部质量从0.9991降至0.8455；消去全部candidate的相机均值差降至0.6150。17/state136分别0.9971→0.7376→0.5859。context/history/coordinate单项同类干预影响很小。此处诊断干预保留视角内候选变化，未当作正式修复。

G的唯一直接目标仍是全图重建（`grounding.py:1029–1030`）；当前权重下，全部相机均值消去使重建误差02从0.2496升到0.3381、17从0.2648升到0.3417。说明现有重建表示确实利用了相机差异；它没有提供“跨视角同一实体”的直接验收。K分散与低cosine仍不能替代真实物体身份。

以上干预后，S在三种自然颜色指令下仍主要选择K3。故不能用简单去均值/相机平均冒充颜色绑定修复。§25的135个真实mask窗口、11个自然指令窗口、05/11错物体碰撞链仍成立。这里已定位到实际candidate-content和共享binding接口；尚未通过一次训练干预证明“某一种新身份损失就是充分修复”。

训练原始几何审计覆盖 train2165/val280/test154：训练片段起始手—目标平面距离中位数5.44cm；按每8帧采样，**90.54%**窗口目标就是离手最近物体，接近代理窗口中为94.97%。全部训练片段的正确目标净推进至少10.04cm，没有发现本轮六任务原始片段未完成推动的证据。近物捷径有明确的数据基础，但这些频率本身不是“训练因果已经证明”的替代物。

审计特别处理了ABC物体次序：A为pink/blue/red，B为red/blue/pink，C为blue/red/pink，D为red/blue/pink；由源frame范围 `scene_info.npy` 和对应scene YAML确定，不能用合并包的D `.hydra`解释ABC。首版统计因这个审计索引错误已标记REJECTED，使用 `training-geometry-r2`。生产模型没有读这些审计物体标签，训练数据未因此改写。

## 30. 撤回：实际命令复现与历史条件的因果定位

这轮按正式rollout的每个8步重规划推进同一私有噪声生成器。14个定位状态原生arm前8行重现RMSE为约1.5e-4–4.5e-4，二值夹爪逐行一致；paired repeat作为数值底噪。之前只固定局部噪声的自然语言对照仍可用，但不能用它逐行替代原始动作。

17/state128 的coarse前8行x已从正转负（-0.037），z升至+0.106；实际执行前8行此时仍x≈+0.249。state136最终变为x≈-0.201、z≈+0.384，并在第5行开始开爪。故提前撤回已在上游出现，不能只归给P3输出缩放。

保持state136当前画面、真实G/S目标来源与采样噪声，在命名消费边界替换为state120已有特征。下表x/z是原生相对动作的无量纲命令值，不是米：

| 干预 | 最终前8行平均x | 平均z | 夹爪 |
|---|---:|---:|---|
| factual/repeat | -0.201 | +0.384 | 中途打开 |
| 仅bottom executed-history | -0.026 | +0.186 | 全闭合 |
| bottom三类proprio/history | +0.017 | +0.149 | 全闭合 |
| S robot/history与bottom三类history | +0.263 | +0.030 | 全闭合 |

S层单独替换robot/history时coarse恢复x≈+0.090，但最终仍x≈-0.104/z≈+0.354；只替换coarse预测几乎不改变最终命令。因此退出趋势经S与底层的多个历史入口产生，coarse不是唯一控制入口。仅把instruction reference重置到当前帧，最终仍x≈-0.185/z≈+0.367，不能解释成只是“旧参考太久”。这些是定点因果定位，不是冻结历史作为正式策略；冻结会丢掉真实控制反馈。

第10例追加4个状态复核（56/72/120/136），原动作前8行arm RMSE为1.95e-4–5.27e-4。state72尚未接触目标，事实命令x=-0.051/z=+0.265；仅将bottom executed-history替成state56，恢复x=+0.203/z=+0.015，均保持闭爪。state136接触后的第二次撤回，事实x=-0.121/z=+0.325、末3行开爪；只替executed-history得到x=+0.007/z=+0.184，而S robot/history与bottom一起替为state120后为x=+0.315/z=+0.020、全闭爪。重置指令参考仍然撤回。这样在两条失败轨迹、接触前后均定位到了历史消费入口，而不是仅有一个窗口的偶然响应；仍不能由这些特征钳制推导真实闭环恢复成功。

## 31. 终点预测与夹爪：实测边界

在3个相同初始参考画面分别测试红/蓝/粉×左/右六条自然指令：FP32下换颜色的预测终点TCP差仅 **0.010–0.070mm**，换方向约 **0.400–0.429mm**；protected goal的方向差RMS仍约0.547，说明方向语言并未在此前完全丢失。预测场景的颜色差RMS仅3.07e-6–6.60e-6。BF16舍入会进一步掩盖差异，但FP32也极弱，不能归因于半精度一种因素。

该目标头没有直接终点监督的原因已在§28.1闭合。预测的机器人终点不能当作目标物位置或完成判据；仍需恢复有来源的终点监督后检验目标场景/关系，以及是否真正改善10/17任务维持。注意 `reference_observation_law` 是固定、完整原生chart的label metric，不能泛称本版本所有Teacher都由G自己生成。

14/state336：开爪概率前7行约0.959–0.999；绕过private gripper gate仍约0.733–0.964，仍选择开爪。仅把过去夹爪命令诊断性改成闭合、同时同步world/robot/sparse history中的重复命令，输出开合行数基本不变。因而不是简单的符号反转、private gate单独错误或纯粘连上一条夹爪命令。§26.1的同手臂闭爪物理反事实仍证明接触配合错误有实质影响；当前来源失配与缺少恢复状态监督应在重新评价这个分支之前优先处理。

## 32. 修复顺序、验收与复现

后续报告复核已调整优先级，当前执行顺序以§33.4为准：控制器状态/动作契约优先；终点标签作为独立修复，不再作为其他问题的前置条件。

7个失败的行为证据与根因边界：

| 失败例 | 已确认的阶段机制 | 已闭合/未闭合 |
|---|---|---|
| 02 | 先朝近处区域，随后控制目标与实际TCP严重分离；很晚才短暂接触蓝块 | 累积起点对漂移的作用已闭合；正确目标选择尚未恢复 |
| 05、11 | 先推动错误物体，再由物体间碰撞间接推动指令目标 | 真实物体身份/碰撞链已核验；G相机分工和S弱颜色重指向已定位，具体身份训练修复仍待验证 |
| 09 | 迟到的接触伴随异常向下挤压 | anchor物理干预去掉异常接触，但未产生合格推动 |
| 10、17 | 接触前或未推动足够距离就撤回/开爪 | S与bottom历史消费的定点因果作用已定位；缺失终点监督是实在的训练接入缺陷，其行为收益需修复后训练验证 |
| 14 | 开爪与向下动作共同形成异常接触/停滞 | 同arm闭爪物理反事实可完成目标；不是简单符号/private gate错误，策略开爪依据的充分修复仍待验证 |

1. **先闭合硬接口**：raw overlay补全严格核验的annotation index，并对“选中终点监督而整训练集0覆盖”报错；明确训练relative-action与执行器基点契约。生产修改后分别验证真实终点label、普通梯度、训练/部署一步与多步目标一致性，不能更改split或成功阈值来掩盖问题。
2. **同完整checkpoint重新评估执行适配影响**：相同18例、6任务各3例、seed0、360步、replan8，记录新的动作执行语义。局部physics replay不计入成功率。终点监督缺失无法通过只重跑旧checkpoint恢复，应在修好数据接入后训练并重新评估。
3. **继续保留身份与任务维持验收**：实际物体mask/跨视角身份、自然颜色重指向、G→S→P1/P2→原生命令、未完成时保持推进、受阻恢复与gripper接触；不能删history、固定闭爪或统一放大P3来代替修复。G目标选择捷径的具体训练修复仍需这些验收，不能称现已解决所有7个失败。

本轮仅新增/修改探针与研究文档。旧训练、权重、正式18例结果及生产行为均未改。大文件留在实验目录；本地只交付摘要和文档。失败尝试保留：`boundaries-r1`因故意换原始S state触发源身份保护；`drivers-r1`因诊断命令未同步所有重复源触发一致性保护。修正探针在投影/消费边界干预或同步完整命令来源，未关闭保护。

工作目录：`/data/senwang/clearvla/checkouts/dinov3-g-slot-logit-scale-20261005`；模型Python `/data/senwang/envs/clearvla-sim/bin/python`，物理回放Python `/home/sen.wang/.venvs/clearvla-calvin/bin/python`。所有输出目录必须新建；完整实参和有效/拒绝输出见本轮 `completion.json`。

```bash
python -m probes.audit_training_geometry --manifest MANIFEST --rollout ROLLOUT --output NEW_GEOMETRY
python -m probes.probe_annotation_provenance --manifest MANIFEST --output NEW_PROVENANCE
python -m probes.probe_rootcause_boundaries --plan PLAN --checkpoint CHECKPOINT --output NEW_BOUNDARIES
python -m probes.probe_failure_drivers --plan PLAN --checkpoint CHECKPOINT --output NEW_DRIVERS
python -m probes.probe_failure_drivers --plan PLAN --checkpoint CHECKPOINT --output NEW_CASE10 --case-ids 10
python -m probes.probe_endpoint_conditioning --plan PLAN --checkpoint CHECKPOINT --output NEW_ENDPOINT
python -m probes.audit_controller_anchor --audit AUDIT_WITH_GEOMETRY_R2 --rollout ROLLOUT
```

## 33. 2026-10-06：外部轨迹报告复核与优先级调整

用户要求同时处理三件事：重估终点监督的重要性、优先核验累计控制目标问题、吸收报告并继续原定的身份选择/任务维持工作。本节修订§32的处理顺序；没有修改正式模型、执行器或18例评估配置。

### 33.1 终点监督缺席属实，但尚不是已证明的主要失败原因

§28.1的零覆盖来自本次 **a2d597d / step11012** 自己的run_context、221个训练采样点及完整验证，不是旧9eed日志。外部报告提醒不能跨版本推断是正确的；它没有否定我们已补做的本版本核验。

但“没有直接终点监督”不能直接推出“七例失败主要因此发生”。该分支仍获动作损失的间接梯度，§31的弱目标差异也没有给出恢复标签后的行为收益。将它列为独立、边界明确的数据接入修复，暂不作为身份/控制问题的前置条件，也不靠扩大这个辅助损失推动整个方案。

对§28.2的措辞也作重要限定：`raw.actions`不能直接叫作示教控制器内部的held setpoint。当前上游 `calvin_env/datarenderer.py:138–146,258–284` 从`robot_info`取实测TCP，并明确将后续一帧姿态写为当前帧动作标签；`utils/utils.py:160–171` 再计算该标签相对当前实测TCP的截断差。

跨51个训练片段153帧补查，145帧的`actions[:6]`与下一帧实测姿态一致；其余8帧位置最大差1.534mm、Euler最大差0.005331rad，来源尚未继续追到原始pickle。不能声称153帧全部等于下一状态，也不能据这些标签恢复遥操作控制器的真实累计状态。上轮153帧`rel_actions[:3] == clip(actions[:3]-robot_obs[:3])/0.02`的最大误差0仍成立。

因此已经成立的是**训练标签、实际执行器状态和策略可读状态之间的语义缺口**。仅凭该差别，不能把切换`use_target_pose=False`自动提升为正式修复；§28.2的同命令物理重放只证明切换会改变漂移与接触，没有证明任务成功。

### 33.2 报告最有价值的新增定位：累计误差没有被现有反馈表示

使用本分支实际源码重新验证：

1. 18例各取一个观测，改变scene、模拟器接触、成功、累计目标及不准入的关节字段后，`calvin_policy_observation`输出完全一致。真实接触/物体ID仍只作诊断真值。
2. 复现有限历史的接口反例：32步中仅第2条命令相差+0.5X，之后观测和命令相同；当前`CausalHistory.snapshot()`的所有字段逐位相同，但按已验证控制器递推，累计目标相差10mm。这是**接口不可辨识性反例**，不是两段物理上同像素轨迹的声明。
3. `RobotExecutionObserver.observe`确实计算“观测feature变化减预测feature变化”；其`read`不是累计目标误差。仅在诊断副本中将response设为正确的零位移预测器，非零命令/实测不动得到loss=0、innovation=0、read=0；正对照有非零read。原权重未修改。
4. 补上外部报告未取得的完整checkpoint读数：按真实前一状态、已登记命令、当前状态，以及生产的normalizer/state-feature编码，逐个batch1、BF16执行实际observer，覆盖全部 **3890个一步转移**。这是实际节点输出，不是新闭环，也没有把任意query的read当成P3最终贡献。

| 09号状态 | 累计控制目标到TCP误差范数 | 当前一步预测残差的XYZ范数 |
|---|---:|---:|
| 56 | 33.75mm | 5.67mm |
| 64 | 73.41mm | 4.63mm |
| 72 | 116.71mm | 4.09mm |

02/09/14全程累计误差最大分别159.20/117.01/73.81mm。实际observer并非没有响应，但其响应没有编码仍欠下多少控制位移。不能把两个不同语义的范数相除，称为反馈保留率；也不能说这些真实窗口innovation为零。当前模型还缺少对累计误差的显式、可追溯输入。

10/17在state136的累计误差仅6.47/4.94mm，一步残差0.81/1.24mm；两条撤回路径仍应按§30追查S/底层历史与剩余任务，不能把所有失败都归到累计误差。

### 33.3 对报告各条建议的取舍

| 报告内容 | 复核结论与作用 |
|---|---|
| 物理step136不是训练warmup136 | 源码确认；checkpoint同步step11012，排除错用时钟解释 |
| 每8步重规划但逐步收观测；8个动作历史点是稀疏时刻 | 与本版本接口一致；不要再用“历史没更新”代替消费问题 |
| 当前观测变化、预测创新、预测终点差是三种量 | 采纳；独立记录来源与物理时钟，不能以P3总RMS代替它们 |
| 选目标与判断该目标进度共用G/binding | 源码确认；首次旁物接触前后必须联查选择与进度，不能只测K是否分散 |
| 内部K/P1/binding值尚未取得 | 是外部报告的边界；本线程已有§25、§29、§30完整权重探针，需要合并证据，不必从静态猜测重做 |
| 显式表示控制器状态，预测残差保持独立 | 作为当前第一优先修复边界；必须处理真实reset、控制器跨指令持续状态、执行确认和训练来源 |
| 直接重置累计目标/外部接触停机 | 不作为当前已通过的修复；此前TCP起点实验保留为明确标注的诊断 |

### 33.4 未丢弃的下一步：具体输入、消费者和验收

1. **执行契约优先。** 将模型动作标签、提交给控制器的命令、确认后的命令、held setpoint、实测TCP分别命名和记录。可读控制器目标须来自实际设备报告，或从已知物理reset及确认命令连续重建；只更换任务指令不能把同一控制器累计状态清零。控制目标误差与预测创新保持不同类型、单位、时间戳及validity。缺少训练控制器真值时标unknown，不能把`raw.actions`冒充它，也不能把未知填0。若选择改变动作执行语义，则以单独的ABI/配置和相同18例重新评价，不与原11/18混算。
2. **真实目标选择继续推进。** 保留01/05/11/15首次旁物接触前后的窗口，以及同观测自然颜色替换。沿`G逐相机物块覆盖 → shared_binder → TargetEvidence/task_relation → P1 target_read/factual_base → protected消费者/前8行动作`定位。分别判定“没有形成目标对象”“选对但读取错”“读对但动作不跟随”；没有权重/真实物体mask支持的节点不先改拓扑。
3. **任务维持继续推进。** 10/17的72作为无目标接触对照，128/136/144作为推动/撤回窗口。共同追踪所选物体的真实位移、绑定后的instruction progress、预测终点差、四步world残差、一步robot创新、S/底层历史消费，检查是否把机器人自身移动解释成任务完成。保留§30已获得的因果钳制；不以冻结history、强制闭爪或加大P3增益替代修复。
4. **终点provenance单独闭合。** 严格补传已核验的annotation index并增加0覆盖验收；记录梯度和行为效果，但不把它升级为上述三条已证实的共同根因。

当前完成第1项的输入缺口、实际反馈与来源核验；生产接口设计/实现和新的正式闭环仍待后续执行。用户本轮要求的报告审查与优先级调整已完成，没有把它记成网络修复已经成功。

### 33.5 证据与复现

证据目录：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/controller-contract-review-20261006-v3`。`observer-r1/summary.json`含各例统计与20个关键状态，`transitions.json`保留3890个节点读数；不下载大张量。实际checkpoint SHA仍为`5ca168e3f4f33772dd01aca26adcc1b00dfc029b526a950f3b534d5428d19d15`。

新增仓库探针`probes/probe_controller_state_semantics.py`；实际实参保存在`launch.json`，使用模型Python、GPU0、`--checkpoint`/`--rollout`/`--training-episodes`/`--raw-root`/`--output`。复跑必须使用新输出目录。源码边界为`calvin_eval.py:86–103`、`history.py:121–129,186–225`、`clearvla_policy.py:346–355`、`model/policy.py:741–747`、`model/robot_execution.py:31–66`。原始附件仅作待核验报告保存在实验目录；未收到其所列的9份脚本/结果附件，不把外部独立CPU复算声称为本线程已重复执行。

## 34. 2026-10-06：独立修复分支与完整策略诊断（进行中）

实现分支 `codex/causal-execution-object-repair-20261006` 从 `dd4a9763` 分出；原完整 checkpoint 与原闭环源保持独立。

### 34.1 已修复的数据来源边界

`benchmarks/calvin_raw.py::virtualize_calvin_cached_prefix` 在既有 split/task/text/source bounds/实帧长度核验之后补传 raw annotation index；已有非空编号不符或类型错误时拒绝。原 HDF5 和动作/视觉源不改写。训练入口在 endpoint objective > 0 且整个训练集合有零个可核验终点窗口时拒绝启动；部分未知来源仍按原合同保留 BC 窗口并掩蔽终点标签。

实际 2599 条 inventory 通过：train2165/val280/test154 经生产 overlay 后已经全部是 `annotated-end-observation`。81 项针对性回归通过（raw reader、真实 HDF5 endpoint、future/online 隔离和 endpoint 梯度合同）。这闭合来源漏传，不证明行为失败已解决。证据：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/causal-repair-20261006-v4` 的 `provenance_tests_r3.log` 和 `provenance_full_inventory`；旧 probe 的 after 是重复赋同值，决定统计使用生产 overlay 后的 before。

### 34.2 同权重完整策略控制诊断

`probes/probe_controller_policy_closed_loop.py` 显式区分 stored_target 与 measured_tcp。保持 final checkpoint、seed0、18例、360步、execute_rows8；每个新观测仍完整重新规划，输出含实际提交后的绝对目标。新增遥测以原第04例核验：初始RGB完全相同、首8行arm RMSE0.0002593，旧62步/复现63步均成功；不得称为逐位相同轨迹。

完整 measured_tcp 面板在 `/data/senwang/clearvla/experiments/calvin/gslot-final-measured-tcp-replan8-20261006-r1`。它是显式改变后端控制语义的诊断；不并入原11/18，不提前作为生产修复。

### 34.3 未遗漏的并行研究项

原长跑配置确实将 `gripper_command_transition=0`、`calvin_frame_weight_mode=uniform`；早期 trajectory repair 代码已合入但本次未启用。后续候选需明确合并/验收这两项，而非默认为已使用。同时继续追查进入全局K以前的真实物块采样质量、共享对象选择、以及10/17接触前后历史消费与剩余目标判断。

### 34.4 Exact inventory migration and merged real-data smoke

The fixed overlay restores `source_annotation_index` only after raw annotation,
task/text, bounds, split and physical-prefix checks. All 2,599 sources now resolve
measured endpoints: 2,165 train / 280 validation / 154 test. Recomputing the old
inventory from immutable HDF5 attributes reproduces the source checkpoint identity
exactly (`e3a3e672...`). Only the annotation-bearing inventory changes
(`b370c993...`); data paths, splits, file inventory, normalizers and frozen online
vision identity remain equal. Initialization admits this difference only with
that exact source proof; unrelated drift is rejected.

`dinov3_causal_repair_calvin_20261006.json` includes the full current model plus the
previously inactive `motion_event_v1` row weighting and gripper transition weight
0.05. Row weights remain normalized against the existing horizon budget. The
first real smoke (`merged-objective-smoke-bs8-r3`) performed 4 updates / 32 samples
and 6 validation batches / 48 samples across all 6 tasks. Endpoint state/visual
coverage = 1.0; weighted endpoint loss = 0.0132243, gripper transition = 0.00192329;
ledger gap = 0 and maximum contribution rounding error = 5.96e-8. After the first
warmup batch, runtime was 6.21–6.42 s/batch with bs8. This verifies labels and
backward execution, not improved task success. Its default fresh update clock
must not be confused with the mature source checkpoint phase.

For actual repair fine-tuning, the new explicit retained-clock option above
keeps the source's 11,012 completed updates and mature execution phase while
starting fresh optimizer/RNG state. Tests verify exact parameter loading,
inventory/normalizer/source rejection, shared schedule/model counters and the
restriction to the parameter-preserving migration.

### 34.5 Separate spatial and language bottlenecks

Artifacts under `probes/causal-repair-20261006-v4`:
`local_spatial_r1`, `coarse_bank_r1`, `language-chain-r1`, `language-attention-r2`.
All use the full 11,012-step checkpoint, actual causal histories, current online
producer interventions and matched initial action noise. Raw masks remain
evaluation truth and never enter policy inputs.

At 01/05/11 state 24, pre-G1 centered score RMS is 11.5–11.9 versus G1's learned
correction 1.39–1.69. Neutralizing the inherited G2 parent law raises
binding-weighted true target top-camera coverage about 29–210 times, yet natural
color action changes remain near the repeat floor. The larger geometric RMS is
not itself causal attribution: separate bank measurements show content-only
object coverage lower than geometry-only coverage, and their combined local
law compounds suppression. Source availability alone does not restore selection.

Natural color swaps and direction swaps have comparable mean T5 input deltas
(0.01793 vs 0.01857). The learned input projection produces 0.02092 vs 0.08204;
the protected goal reader produces 0.02243 vs 0.56365, and the following self block
0.02329 vs 0.54731. This is a learned direction preference before K binding, not
evidence that the input language bank omitted color. The four language tokens
are not identical; attention also reads the near-common final token heavily
(roughly 0.41–0.46 mean weight). Attention alone is not an independent contribution
percentage. Binder scoring and native actions remain weakly color-dependent
through multiple spatial interventions. Do not promote camera centering,
uniform spatial reads, or a larger common carrier from these diagnostics.

Reproduction commands and exact output paths are recorded in the v4
`*_command.json` receipts. Full controller-policy panels use the original source
checkpoint and distinct declared controller anchor contracts; they must be
reported separately from the original 11/18 standard panel.

### 34.6 Complete controller panels and independent actuator calibration

All four panels retain the same original final checkpoint, 18 initial RGB pairs,
seed 0, execute_rows 8 and max_steps 360. The controller contracts differ:

| Controller | Success | Largest held-goal/TCP error |
|---|---:|---:|
| Original stored target | 11/18 | 159.20 mm |
| Measured TCP each step | 3/18 | 9.66 mm |
| Measured TCP at each replan | 10/18 | 73.59 mm |
| Independently calibrated inverse servo | 9/18 | 22.12 mm |

Every initial RGB pair matches. Inverse-servo goal recurrence reproduces exactly
from the stored float32 robot observations, coefficients and submitted commands;
its finite goal bound did not activate. The other recurrence residuals below
3e-8 m use NPZ float32 start poses; §33's float64 environment recurrence remains
the separate near-machine-precision measurement.

The independent servo fit uses 1,152 balanced random air-motion commands,
two calibration poses, and a third held-out pose. It reads only measured TCP,
past measured displacement and known commands: no task label, object/contact
truth, checkpoint or closed-loop success enters calibration. Measured-anchor
position gains about 0.49–0.51 become 0.99–1.03 under the inverse on the held-out
pose. This recovers the free-motion response lost by naive per-step anchoring.

Even then, none of the original seven failed cases succeeds. Original successes
01 and 03 are lost; 15 still recovers late. Thus controller windup is a real
semantic/state defect, but changing its dynamics does not solve object selection
or premature withdrawal. None of these controller variants is promoted.
Actual first object contact remains wrong for 01/05/11/15 in the inverse panel.
Sources: robot.py::relative_to_absolute; repo probes
probe_calvin_servo_response.py, probe_controller_policy_closed_loop.py and
summarize_controller_policy_panel.py. Data: v4/servo-calibration-r2 and
controller_four_panel_summary.json. The failed r1 calibration serialization
record is preserved; r2 reran calibration and held-out validation completely.

### 34.7 Reconstruction does not certify object identity

The original full checkpoint reconstructs red, blue and pink top-camera regions
mainly through the same K1. At 01/state24, conditional reconstruction K1 ownership
is 0.9827 / 0.9993 / 0.9970 respectively; each object covers only about 0.36–0.52
of the 64 coarse target cells. Region reconstruction errors can remain below the
background error. Therefore low average reconstruction MSE is compatible with
several real objects sharing one value basis.

The existing G content/target chart is explicitly pooled to 8x8; full 16x16
candidate coordinate support is not native 16x16 content values. This is the
declared current chart contract, not a newly discovered format mismatch.
A coherent current-G3 native-content diagnostic, and a combined frozen-DINO
matching/native-content diagnostic, do not restore natural color selection in
01/05/11 state24 or 17 state136. Frozen matching can improve object coverage and
reconstruction while first-8 color action differences remain around 1e-4–1e-3.
Do not promote resolution increases or cosine matching alone as a behavioral fix.

Source boundaries: flow_dino_evidence.py::_teacher_content_grid and current G3
content materialization; grounding.py:927–987 (destination K reconstruction
assignment, shared position term, observed-cell MSE). Data:
v4/reconstruction-identity-r1, frozen-bank-r1, native-content-r1.

### 34.8 Formal supervision ownership and next acceptance gates

A no-update VJP probe now runs through the actual train entry, strict source/
inventory initialization, retained step11012, online DINO, formal engine forward
and weighted ledger. On one ordinary four-sample batch, action-flow gradients
reach G (L2 0.0941) and shared binder (0.0781). Reconstruction reaches G (0.000407)
but not binder; annotated endpoint loss reaches neither G nor binder directly,
while reaching the protected language encoder and endpoint predictor.
This is the source-defined supervision ownership, not an accidental detach:
annotation_goal.py::supervise_annotated_goal directly compares the start-image/
protected-language endpoint predictor to measured endpoint labels.
Nonzero action gradients alone do not establish correct semantic learning.

Probe r1 failed on an incorrect diagnostic record field; r2's retained multi-VJP
bs8 graph exceeded 24GiB. The corrected diagnostic uses bs4 and zero optimizer
updates; the actual repair training remains bs8. Focused early/non-nearest-target
training diagnostics separately map annotation age to source_start-context_start
plus age, retaining the real 24-step context prefix. Dropped language rows are
excluded from legal instruction-direction perturbations using the producer's
goal_keep, not the token mask (conditioning keeps that mask). Their zero-input
normalization derivatives must not be called color learning pressure.

The actual repair run is short-mature-bs8-r1 under
/data/senwang/clearvla/experiments/dinov3-causal-repair-20261006, immutable source
1b5c302de530494746b63dadf14ff25b696ae73e. It starts original step11012 with fresh
optimizer/RNG, keeps the mature model phase, performs 512 bs8 updates, then 256
validation batches and the original-controller standard 18-case panel.
Target checkpoint is epoch1/step11524. It is not an exact optimizer resume.

A separate registered GPU5 continuation checks the final checkpoint/source/hash,
then repeats fixed-observation language-chain, real-object reconstruction and
10/17 history/withdrawal probes. Those are new weights on original histories,
not fresh closed-loop trajectories. Fresh 18-case behavior runs separately on
GPU4 and must be analyzed before promotion. Endpoint loss improvement, numerical
health and increased S/K variance do not close the task.

Final focused diagnostic is v4/formal-object-routes-far-r4.json: four real train
windows at annotation age8, target distances 0.178–0.258m, all non-nearest.
The exact producer dropped language on row0; remaining three rows have finite
admitted goal-input action VJP RMS7.44e-6. Colored perturbation directional
derivatives are small and not uniformly positive; these are local Taylor
derivatives, not full counterfactual losses or evidence that a particular color
was physically selected. G/action L2=0.0410 and binder/action L2=0.0292.
The full model parameters receive finite gradients; enormous virtual tangents
at zeroed language inputs are not admissible instruction perturbations.

Natural wording scope was expanded with probe_language_template_geometry.py:
20 complete templates cover all six color/direction tasks (120 matched color
pairs, 60 direction pairs). On original full weights in FP32 CPU, average raw T5
mean difference is 0.01610 (color) versus 0.01478 (direction); protected goal_self
difference is 0.03613 versus 0.54798. The learned direction preference therefore
extends beyond the panel's canonical wording. This pure language diagnostic
does not assign real objects or demonstrate a policy benefit. Output/command:
v4/all-language-templates-r1.json and its command receipt.

Fresh-rollout auditing is separately registered at
short-mature-bs8-r1-fresh-trajectory-audit (GPU6, after standard panel completion):
all18 physical phase ledgers -> per-case fresh simulator RGB-exact mask replay
-> G/S/P1/P2/P3/native-action identity tracing. It rejects incomplete or wrong-
source panels and waits for a free GPU. No original masks are reused on changed
trajectories. New --match-rollout-rng in probe_physical_object_identity.py advances
the owned noise once at every skipped replan; the original default remains
explicitly unmatched for historical compatibility. On original case04 states
0/24/40, replay arm RMSE is 0.000273/0.000147/0.000135 with zero gripper mismatches
(v4/matched-rng-case04-r1). These finite BF16 discrepancies remain the repeat floor.


### 34.9 Actual sampling does not further suppress early non-nearest targets

The original final checkpoint stopped at the configured 11,012 batches
(88,096 samples); the uncapped sampler iterator contains 17,236 batches.
Calling that run a complete dataset pass would be inaccurate.

Join the unchanged information/task-balanced sampler at epoch1 with the audited
physical training geometry only at exact ages 0/8/.../56. Do not interpolate
unmeasured object identity or compare different denominators:

| Same audited age support | Nearest target | Age <16 | Early non-nearest | Distance >0.15m and non-nearest |
|---|---:|---:|---:|---:|
| Uniform source windows (17,236 admitted) | 90.54% | 25.12% | 4.73% | 2.46% |
| Actual first11,012 sampler batches (11,126 admitted draws) | 88.52% | 32.66% | 7.57% | 4.28% |
| Repair first512 sampler batches (517 admitted draws) | 88.97% | 31.33% | 6.77% | 3.09% |

Thus the tested sampler dilution hypothesis is contradicted: event/motion
sampling increases these early/hard fractions on the common support. Source
windows are still mostly nearest-target examples; this remaining correlation
alone neither proves causality nor warrants a geometry-oracle sampler repair.
The endpoint provenance repair leaves sampler scores, task balance, physical
center mapping and sampler seed unchanged. Source: data/loading.py::loader and
train.py epoch-owned set_epoch. Probe: audit_effective_training_coverage.py,
v4/effective-training-geometry-r2.json and command receipt. The first r1 full-
iterator row must not be relabeled as the original capped run.


### 34.10 Repair step11524: completed offline validation and fixed-history rejection

The 512-update candidate finished at epoch1/step11524, immutable production
source 1b5c302de530494746b63dadf14ff25b696ae73e; best.pt SHA-256 is
a61886f80317e96719965e250871e0b79e4062cc3118d422f78f4cfaab9d610b.
The normalizer fingerprint remains unchanged. Actual bs8 training averaged
6.237 s/batch without nonfinite errors, and the weighted loss ledger closes.
All 256 validation batches cover the same 2,046 samples/6 tasks.

| Same-panel native metric | Original step11012 | Repair step11524 |
|---|---:|---:|
| Full24 action RMSE | 0.26281 | 0.26631 |
| First8 action RMSE | 0.23271 | 0.23506 |
| Arm RMSE | 0.13213 | 0.13305 |
| Decoded gripper event precision | 0.28585 | 0.28309 |
| Decoded gripper event recall | 0.39085 | 0.40261 |
| Decoded gripper event F1 | 0.33020 | 0.33243 |

Endpoint supervision coverage is now 1.0, with validation weighted contribution
0.00647. Its previous zero was missing-label coverage, not superior prediction.
This proves the data/objective repair executes, not that physical behavior is
repaired. Full action error slightly worsens; do not promote the candidate.

Fixed-original-history probes completed. On all20 natural wording templates,
protected goal_self color delta changes only 0.03613 -> 0.03662, while direction
delta remains 0.54798 -> 0.55238. At 01/state24, top-camera red/blue/pink
reconstruction still belongs chiefly to K1: 0.9856/0.9991/0.9962. These remain
conditional reconstruction assignments, not object identities or global view mass.
At 10/state72 the first8 mean x/z is -0.0566/+0.2541; at 17/state136 it is
-0.2300/+0.3750. The old premature withdrawal persists on the same histories.
Earlier S/bottom history substitutions still reverse the withdrawal; they are
source diagnostics, not coherent alternate physical trajectories or a freezing
prescription.

Artifacts under dinov3-causal-repair-20261006:
short-mature-bs8-r1-log-audit.json, *-audit-decision-summary.json,
*-all-language-templates.json and *-fixed-observation-probes.
The original-controller standard18 fresh panel and its independent exact-RGB
mask/identity audit are running; fixed-history findings do not replace those.

### 34.11 Complete wrong-instruction loss sensitivity, with numeric admission

probe_matched_instruction_loss.py performs the complete production weighted
loss on the four real early/non-nearest training windows from §34.8. Change
exactly one row's natural color or direction instruction; keep its actual
images/history/recorded future and the owned flow seed fixed. No optimization,
new labels or architecture modification occurs. This is sensitivity to a wrong
instruction given one recorded future, not a valid expert action demonstration
for that wrong instruction.

Production BF16/TF32 repeats (r1/r2) differ by up to 2.9e-4 and cannot resolve
small color margins. Restoring global RNG alone does not eliminate this floor;
the eval forward consumes no global CUDA RNG. Strict deterministic FP32 policy
execution after the original strict initialization (r4-fp32) gives exact repeats
for all three flow seeds. The online DINO observations remain the same.
The failed r3 attempted a configuration precision change; the strict migration
correctly rejected it and its failure record is retained. No migration allow-list
was weakened for this diagnostic.

On accepted r4, median absolute batch-mean action-flow loss change is 2.31e-5
for 24 color replacements, versus 7.96e-3 for 12 direction replacements.
Color loss increases in 14/24 cases, with mean change -1.01e-5; correct color is
not consistently favored. Three direction replacements on the fourth early
approach example also reduce error, so this is not proof that every
wrong-direction motion should be worse at every approach step. The robust
finding is weak learned color sensitivity under this complete objective, despite
finite ordinary gradients and available early hard examples.

Artifacts: v4/matched-instruction-loss-r4-fp32.json, config and command receipt.


### 34.12 Two further narrow hypotheses rejected on measured histories

The existing conditioning producer was used to drop all executed-command
history while retaining language (same strict-restored weights and deterministic
FP32 diagnostic). Its timing masks and executed-world/robot-step sources follow
the production dropout contract. In the same four early training windows, correct
action-flow losses rise from 0.405/0.442/0.426 to 0.422/0.443/0.479 across the
three noise seeds; the median color perturbation remains only 3.18e-5, with
12/24 positive margins and exact repeats. Command-history removal alone does
not restore learned color sensitivity. This is an intervention on the trained
model, not evidence that retraining with different dropout cannot help.
Data: v4/matched-instruction-loss-r5-nohistory.json.

A second paired probe consistently removes prepared robot endpoint values,
visual endpoint values (content/image/joint), or all four components from BOTH
existing S and P3 goal readers. At old and repair 10/state72, 10/state136 and
17/state136 the backward/upward command persists in every variant. For example,
repair17/state136 x/z remains -0.230/+0.373 with robot values zero and
-0.242/+0.376 with all goal values zero. Thus a misleading robot endpoint does
not explain these withdrawal windows by itself. These removals are not
independent contribution percentages and must not be deployed as a patch.
Source: annotation_goal.py::AnnotatedGoalValueRead.prepare/forward and
probe_goal_remaining_consumption.py. Data: v4/goal-components-original-r1 and
dinov3-causal-repair-20261006/short-mature-bs8-r1-goal-components.

### 34.13 Completed fresh panel rejects the combined 512-update candidate

The standard original-controller panel completed at 7/18 versus original11/18.
Initial RGB pairs match in all18; seed0, max_steps360 and execute_rows8 are
unchanged. Original successes01/03/08/13 are lost; none of the original seven
failures is recovered. Do not promote checkpoint11524 or attribute this joint
fine-tune regression to one objective without isolating it.

Fresh trajectory auditing is complete: 140 selected states across all18, each
admitted by exact replay of both camera images; 11 natural-instruction windows.
Matched original replan noise gives median first8 arm replay RMSE0.000235,
maximum0.000664 and zero gripper mismatches. Binding argmax is K3 in140/140.
Median color arm difference0.000333 remains near repeat0.000203, while direction
difference0.05907 is much larger. This is weak color dependence on fresh failed
and successful histories, not a claim of bitwise color invariance.

05 first contacts red at45;11 first contacts pink at46; neither directly contacts
its requested target. In contrast,08/13 first contact the correct target at20/21,
then achieve only13.48/0.01mm signed progress;17 contacts pink at132 and reaches
20.36mm. These latter cases require a persistence/command explanation in addition
to target availability. Original01/03 successes were late recoveries rather
than reliable initial object selection.

At fresh05/state24 the largest target-pixel probability among all8 fixed K/view
read distributions is0.000186, and both other blocks dominate it in every basis.
At11/state24 the target maximum is0.0000357, dominated by pink in every basis.
No convex K/view reweighting of those FIXED spatial distributions reverses the
corresponding ordering. This bound does not cover P1's query-dependent spatial
reselection or all semantic information in a DINO feature. At17/state128 the
target maximum is0.897, confirming a distinct later failure with available target
pixels. Tiny absolute support with a large conditional object share is not
physical identity certification.

Decision artifact:
dinov3-causal-repair-20261006/short-mature-bs8-r1-fresh-audit-decision-summary.json.
Raw phase, masks, identity and natural-instruction evidence remain under
short-mature-bs8-r1-fresh-trajectory-audit; simulator truth remains audit-only.

### 34.14 Executed-history semantics and flow-time alternatives

The full original checkpoint was replayed on six actual10/17 windows. A causal
diagnostic replaces XYZ in ALL duplicated past action sources by observed
TCP displacement/0.02, clipped in the original native chart; orientations and
gripper remain recorded. Sparse history, action_state, four-step commands and
one-step robot commands are synchronized. It reads no observation after the
current state and does not relabel production confirmed commands.

At17/state136 mean x/z changes from-0.200/+0.384 to-0.144/+0.373; at10/state72
from-0.051/+0.266 to-0.0055/+0.210. Withdrawal is reduced but remains. The command/
realized-motion mismatch is thus not sufficient to explain these windows.
Exact old arm replay differs only at the measured BF16 repeat floor.
Source/probe: probe_realized_translation_history.py; v4/realized-translation-
history-r1. This does not establish a physically closed-loop replacement.

Strict-FP32 complete-loss diagnostics were also evaluated at explicit flow times
0.05/0.5/0.95 with the same noise seed, rebuilding the actual Q5 flow-step context
and leaving unknown-label rows as source noise. Median color loss changes remain
2.33e-5/2.25e-5/1.74e-5 with exact repeats. Weak color dependence is already present
near source noise; it is not explained solely by clean-target leakage at large t.
Source training/losses.py::sample_flow_matching owns the correctly mirrored
noise-to-data clock; no time-direction defect was found. Artifact:
v4/matched-instruction-loss-r6-times.json.

The spatial coordinate alternative was checked in actual source:
online_vision rasterizes native DINO patch centers once into the declared
full_rgb_endpoint_v1 chart; subsequent endpoint resizes preserve its alignment.
No evidence supports silently reinterpreted native patch centers or a current
flow-pixel/DINO-cell unit mismatch. Do not change these charts speculatively.


### 34.15 Camera pooling alone does not restore learned color selection

Four paired source-boundary variants were evaluated on original11012 and
repair11524 at the same original01/05/11/state24 and17/state136: remove the global
slot-content residual; supply the existing per-camera conditional values to the
already implemented single K+null binder; retain that residual with per-view
values; or average supported camera values before the same binder. All S/W/P1/
P2/P3/bottom consumers are recomputed; no mask or physical identity enters input.

Binding spreads over more K under per-view input, but color-native differences
remain around0.00022–0.00069. This does not establish a beneficial camera-pooling
repair. The K-common residual and early camera pooling are not sufficient
explanations for the learned color insensitivity on these weights. No production
switch was made. Source: intent.py:854–924; task_execution.py:561–606.
Artifacts: v4/binder-view-source-original-r1 and
short-mature-bs8-r1-binder-view-source-r1. Each contains six natural color/direction
instructions per source variant, exact matched noise and original-weight repeats.

### 34.16 The new08/13 failures begin in approach geometry, not simply opening

Across all18 identical initial states, repair first8 native X shifts left by
0.0302–0.0530, median-0.04346; Y/Z median shifts are+0.01594/-0.02393.
At08/state16, TCP is32.8mm left of the target versus18.7mm originally.
At13/state16 these distances are26.1mm versus11.7mm. Before closing, contact then
moves the blocks sideways in Y; both descend about40mm bystate32. The unchanged
later arm sequence proceeds without carrying the target.

Fresh simulator interventions retain the actual recorded arm sequence and force
only the gripper closed from16 or24 through64. None recovers the required left
motion. For08, signed progress at64 is13.31mm factual, -2.88mm when closed from16,
15.21mm from24. For13 it is-9.40/-15.97/-7.97mm. Thus these cases must not be
described as an isolated early-opening bug or solved by forcing contact.
The original successful motion also raises the objects while moving left;
oracle task success alone does not characterize the contact mechanism.

Reversible parameter-group interventions on identical initial inputs localize
the *new training drift*. Restoring all old state reproduces original first8
arm within0.00016–0.00030 RMS. At08/13, restoring bottom alone changes mean native
X by+0.03755/+0.03893; restoring P1 alone gives+0.00901/+0.00928. Restoring coarse
or language alone has much smaller effects. These hybrid sensitivity tests
neither assign additive causal percentages nor qualify a hybrid for deployment.
No hybrid checkpoint is saved.

Artifacts: *-gripper-contact-replay, *-initial-action-drift.json,
*-parameter-reversion-r1, *-parameter-reversion-summary.json and
*-regression-mechanics.png under dinov3-causal-repair-20261006.

### 34.17 Explicit named AdamW continuation for the parameter-preserving repair

On one actual bs8 production forward/backward atstep11012, the optimizer step
was intercepted before live mutation. Identical clipped gradients and current LR
were applied to two CPU AdamW copies. Fresh moments produce bottom updates
3.16–4.81 times larger in L2 than carrying the checkpoint moments; P1 is3.91,
grounder5.03, observation5.03. This establishes an initialization sensitivity,
not proof that moment reset alone explains512-update behavior or old failures.
Artifact: short-mature-bs8-r1-optimizer-restart-r1.json.

Training now offers the explicit flag --init-optimizer-state checkpoint, only
with --init-training-clock checkpoint and the parameter-preserving
calvin_endpoint_trajectory_repair_v1 initialization. Default initialization still
uses fresh moments. After the existing strict source/config/inventory/normalizer
admission, it verifies the same live weights, checkpoint source and step, exact
named group ordering, AdamW hyperparameters, finite correctly shaped moments,
nonnegative variances, owned IDs and legal per-parameter update counts.
All validation precedes optimizer mutation. The current retained schedule owns
LR; old schedule, RNG and loader position are not restored. This is declared
fine-tuning, not exact resume. Run context records actual optimizer loading.

Sixteen focused tests pass, including exact next-AdamW-update parity after a new
gradient and rejection of reordered names, wrong weights/source, hyperparameters,
bad shapes, nonfinite moments, negative variance, illegal steps and wrong clocks.
A real bs8 continuation pilot and behavioral validation remain required before
claiming a repair to training drift. The target-selection and persistence
investigations remain open; this option does not solve them by definition.

### 34.18 Reconstruction pressure favors camera/scene coding in the measured windows

The probe_reconstruction_identity_pressure.py script captures the actual complete
11012 checkpoint at 01/05/11 state 24 and 17 state 136. Parameters and policy inputs
are untouched. Actual simulator masks only partition the report. Area pooling
onto the 8x8 loss grid is an approximate region allocation, not a claim that a
DINO cell contains only that physical object. Keep it separate from the exact
bilinear K-read coverage audit.

Holding existing decoded K values and the shared position term fixed, every
visible top-view block-overlapping region prefers K1; wrist regions prefer K3.
Non-block pixels account for 94.99–97.24% of area-allocated reconstruction error
and 97.18–99.76% of the corresponding detached destination-owner-logit gradient
magnitude. The latter is an analytical local interface derivative, not a whole
network gradient share or an independent contribution percentage.

The source computes one cross-camera K content at grounding.py:764–768, then
uses it for both cameras in reconstruction at 959–967. Actual per-camera values
are exported only afterwards at 988–994. A diagnostic fit of ONE constant value
per camera, with no object identity and the same position term, attains lower
MSE than the existing K mixture in all four windows:

| case/state | actual FP32 reconstruction | one fitted value/camera |
|---|---:|---:|
| 01/24 | 0.245272 | 0.239682 |
| 05/24 | 0.267984 | 0.246437 |
| 11/24 | 0.277942 | 0.257924 |
| 17/136 | 0.264690 | 0.257740 |

This is an optimistic per-observation fit, not a trained replacement. It is a
counterexample to treating lower reconstruction error as proof of real-object
identity. Substituting existing camera-conditioned values lowers MSE in some
windows but does not certify object separation, so that substitution is not
promoted as a complete repair. The earlier natural-instruction/full-loss probes
also show weak color pressure. Current endpoint supervision has no direct
G/binder VJP; restoring endpoint provenance alone cannot establish target
binding. Do not cure this by merely increasing a common carrier.

Reproduce from the 11012 checkpoint's own checkout with:
PYTHONPATH=. python /path/to/candidate/probes/probe_reconstruction_identity_pressure.py
--checkpoint <original11012>/checkpoints/best.pt --plan <identity-v1>/probe_plan.json
--masks <identity-v1>/masks-r2 --output <new-dir>.
Exact commands/hashes and both runs are in causal-repair-20261006-v4/
reconstruction-identity-pressure-r{1,2}; no raw tensors enter this document.

### 34.19 Optimizer pilot admission and pending behavioral gate

The continued-moment 64-update bs8 pilot completed at step 11076, source ffb6c39c,
with 1249 parameter moment states admitted. New checkpoint SHA256:
b02d6e9bc58674c6199dadb86111f8018a7f795cfd9f2c766472a8af194a22f8.
Same 18 initial observations/seeds yield median mean native XYZ change
(-0.002080,-0.005394,-0.008877) versus original 11012. These are snapshot actions,
not 18 successful episodes. The matched fresh-moment 64-update pilot is separate;
do not compare 64 versus 512 updates as an optimizer causal estimate.

The 16-batch offline panel contains 126 samples, 21/task, and is NOT the standard
256-batch panel. Diagnostic limits set to 0 mean ALL available batches in this
entry point, not disabled diagnostics; serialized validation coverage is 16/16.
Training is finite at approximately 6.2–6.6s/batch. Full standard 18 closed loop
(seed 0, max 360, execute 8, stored-target) is launched separately, and exact fresh
trajectory masks/physical-identity probes are chained afterwards. Artifacts:
dinov3-causal-repair-20261006/pilot64-mature-bs8-adam-{checkpoint,fresh}-r1,
pilot64-adam-checkpoint-eval18-command.json and
pilot64-adam-checkpoint-fresh-audit-receipt.json. No candidate acceptance yet.

### 34.20 Research priority and completed 64-update diagnostic (2026-10-07)

The user clarified that 64 updates are a diagnostic of update direction and
regression, not a credible efficacy or convergence test. Do not replace a
structural diagnosis with repeated 64-update repairs. Existing work was allowed
to finish naturally; no training, closed loop or probe was stopped for this
change of emphasis. Endpoint provenance remains a correctness repair, not the
main explanation for the behavioral failures.

The retained-Adam pilot in 34.19 has now completed the standard 18-case closed
loop: **9/18**, versus the original complete model's 11/18. Cases 1, 3 and 15
were lost, and 11 was gained. The gain in 11 is indirect: the robot contacted
pink, pink contacted red, and red reached 101.858 mm signed progress. It is not
evidence of correct red-object selection. The 512-update merged candidate was
7/18; neither short continuation is promoted.

The matched 64-update fresh/retained-Adam runs share source, data, seed, batch
size and update count. Median first-eight arm snapshot change over 18 initial
observations is 0.009571 versus 0.008888 (about 7% smaller), not a large behavioral
repair. Median absolute mean-X shift is 0.008562 versus 0.008885; do not confuse
cancellation in the signed median with suppression of drift. Both 16-batch,
126-sample offline panels have decoded event F1 0.271186; these are not the full
256-batch panel. Both log audits remain finite; saturated capacity is a triage
finding, not the demonstrated cause of failure.

The retained pilot's fresh trajectory audit completed 125 exact-RGB mask states
covering all 18 cases. Recorded-command replay has median arm RMSE 0.000213,
maximum 0.000595, and no gripper disagreement. K3 is binding argmax in every
audited state; this does not establish a persistent physical identity. Natural
color changes remain weak (median arm RMS 0.000356, repeat 0.000206), while
direction changes are 0.057716. Fixed-read target coverage is poor in 05/24 and
11/24, but a target-covering read is available in 17/136. Selection and later
task maintenance remain distinct failure boundaries.

Artifacts under dinov3-causal-repair-20261006: adam-pilot64-matched-decision.json,
adam-pilot64-log-audit.json, pilot64-mature-bs8-adam-checkpoint-r1-fresh-audit-decision.json,
the corresponding fresh-trajectory-audit, and changed-cases-physical.json.

### 34.21 Ordinary backward and language-read controls

probe_objective_update_routes.py replays one actual bs8 ordinary training batch
at step 11012 with restored RNG/buffers, normal backward/clipping and CPU AdamW
proposals. Live parameters, optimizer and training clock do not change. Compared
with the formal repaired objective, restoring uniform action-row weights changes
the bottom/P1 clipped gradient by about 16.5%; removing the adjacent gripper
transition term changes it by about 7.3%. Removing endpoint supervision changes
bottom/P1 gradients by about 0.5–0.6%, comparable to the repeated-forward floor,
but changes intent gradients by 17.8%. This is one batch, not an attribution of
the 64/512-update rollout regression. The binary action-flow loss is already
arm-only; no continuous-gripper flow bug was found.

Artifact: pilot64-mature-bs8-adam-checkpoint-r1-objective-routes-r3.json.
The first two rejected probe attempts and their receipts remain preserved.

The full 11012 model was also tested at 01/05/11 state 24 and 17 state 136 with
uniform goal attention, half query scale, and masking only the last valid text
token. Actual V/output/self blocks and downstream computation remain active.
Masking the final token roughly doubles the color difference at the goal read
(0.0231 to 0.0511), but action color differences remain around 0.0004–0.0008.
Uniform attention also damages direction distinction. None is an accepted fix;
do not call the final valid token EOS without verifying token identity.
Artifact: causal-repair-20261006-v4/language-attention-interventions-r1-decision.json.

### 34.22 Arm-conditioned gripper training/deployment boundary

Source: training/engine.py:1009–1065, endpoint_supervision.py:60–74 and
runtime/sampling.py:271–285,417–455. The training endpoint head reads the labelled
arm endpoint on the coarse W cache. Deployment reads the generated arm endpoint;
the refined pass also rebuilds W. This factorization is explicit in the source,
not a newly discovered label leak or proof that every changed gripper is wrong.

probe_endpoint_condition_distribution.py crosses these two factors on the SAME
16 validation batches (128 samples, 2536 observed command rows, 958 first-eight
rows). This differs from the 126-sample short-run panel above. Labels, source
noise and row support are matched. Recomputed refined/generated logits match
the original sampler exactly, with zero command disagreement. No optimization
or checkpoint write occurs.

| W cache / endpoint arm | observed-row command agreement | first-eight agreement | mean CE |
|---|---:|---:|---:|
| coarse / labelled | 95.071% | 96.242% | 0.140907 |
| coarse / generated | 87.934% | 91.336% | 0.543074 |
| refined / labelled | 95.032% | 96.347% | 0.141321 |
| refined / generated | 88.013% | 91.545% | 0.542118 |

Changing arm field alters 274–275 commands; changing only W cache alters 4–5.
The complete-24-row sample subgroup retains the difference, so unknown tail
noise does not explain it. Exact-row/type event matching is recorded separately
and must not be conflated with the production decoded-event metric.

This establishes conditional distribution sensitivity. A generated arm can
already be physically wrong; forcing its gripper to match the expert's different
arm plan is not automatically a valid repair. Any training change must check
arm/gripper compatibility and real contact outcomes, not only increase CE
agreement. Artifact: causal-repair-20261006-v4/endpoint-condition-distribution-r2-decision.json.
The rejected r1 config-admission attempt is preserved; the gate was not weakened.

### 34.23 Auxiliary dialogue checked against factual stage records

The attached dialogue usefully separates several failure stages. Its following
claims were independently reproduced from all-18 trajectory/environment records:

* At state 40, 03 has an empty gripper (0.00662 mm opening), no block contact,
  yet requests nominal first-eight XYZ displacement (-76.451,-13.871,+21.823) mm.
  01 holds wrong pink with 41.202 mm opening and requests (-126.246,-7.842,+15.123);
  07 holds correct red with 41.220 mm opening and requests (-124.924,-6.989,+17.124).
  These are nominal command increments, not measured TCP displacement.
* At 09/201, TCP X error to target is -0.408 mm but Y error is 47.826 mm.
  The held-target Y error is 50.273 mm and target-to-TCP tracking error only
  2.447 mm. At 11/231 the corresponding Y errors are 67.779, 70.714 and 2.936 mm.
  Correcting servo accumulation alone does not align these requested contacts.
* After their last target contacts (10/139 and 17/138), both execute another
  27 replans and about 1.434 m of TCP travel. Further target progress is
  -0.019 mm and +0.414 mm. Repeated robot movement is not task progress.

These observations do not prove a hidden task clock. Artifact:
causal-repair-20261006-v4/auxiliary-dialogue-stage-verification.json.

The width-only sensor intervention updates current state and its timestamp-zero
duplicate through the actual encoder/normalizer, while retaining factual RGB,
past state/control and reference. It is not a physical counterfactual rollout.
In 13 windows it changes commands, ruling out complete absence of width input.
At 17/136 the mean first-eight Z command is +0.384, +0.395 and +0.396 for factual,
held-object and open-width inputs; the withdrawal remains. At 03/40, empty and
held-object width both retain left/up transport. Do not reduce the failure to
one missing proprioceptive scalar. Artifact: observed-opening-consumption-r1-decision.json.

### 34.24 Newly localized S correspondence failure and bounded causal controls

The actual S posterior is produced by model/instruction_posterior.py:179–192,
218–257. G's current spatial source is multiplied by a learned full-image
current-to-current kernel, then by a separate current-to-reference kernel. The
current side therefore performs another search after the G source was located.
Independent learned Q/K/content-position projections have no self-identity
constraint in this module; its geometric interpretation is not guaranteed by
its name. Policy gradients can learn a useful global comparison without learning
physical correspondence.

The complete model was traced over 14 factual windows, with simulator masks only
partitioning the report. At 17/136, the best K/view pink-overlap mass on the
native 16x16 chart is 0.599399 before current rematching; the SAME K4/wrist read
retains only 0.064683 afterwards (the best post-rematch K/view is 0.092937).
The actual binding/view-weighted target footprint is 0.06835 before and 0.03991
after rematching, with background dominating both. An available target-covering
K does not mean S selected that K. This is an area-pooled mask footprint, not the earlier exact bilinear P1
coverage measure and not proof that a DINO feature contains only that object.
K-to-K probability differences also contract substantially. Do not interpret
artificial K index swaps as real physical identities.

Five windows were then inspected at kernel level in both the original complete
checkpoint and the 64-update checkpoint on EACH model's own exact-RGB histories:

| model | current-to-itself diagonal mean | normalized entropy | self-location top-1 fraction |
|---|---:|---:|---:|
| original 11012 | 0.002823–0.003320 | 0.9725–0.9888 | 0–0.003906 |
| retained-Adam 11076 | 0.002878–0.003496 | 0.9683–0.9910 | 0–0.003906 |

There are 256 real patches (uniform real-cell probability about 0.0039).
The learned module does not behave as an identity-preserving current-image
correspondence. A fixed DINO-cosine RANKING with each row's exact probability
multiset and null retained improves object-overlap, but the original distribution
is too diffuse for ranking alone to recover a concentrated object read.

In the actual S progress output at 17/136, projected source RMS is robot 1.319,
content 0.599, joint content/position 0.336, status 0.362, image 0.0100, before the
shared RMS contract. These are source magnitudes, not independent contribution
percentages or a proof that the robot component alone causes withdrawal.

Seven matched full-sampler controls include the reset observation and the
empty/wrong/correct-contact and late-withdrawal windows. No G/binding change is
allowed. They test (a) repeated baseline, (b) DINO-cosine ranking with EXACT real
mass/null/concentration, (c) fixed cosine/sqrt(D) redistribution with original
null, and (d) zeroing only S's robot-displacement value. Both kernel alternatives
preserve zero content/image/posterior difference when current and reference are
identical at reset. Neither removes withdrawal: at 17/136 baseline X/Z is
-0.201/+0.384, ranking -0.223/+0.377, fixed metric -0.284/+0.375, robot-zero
-0.198/+0.377. A one-line matcher replacement or deleting robot evidence is NOT
an accepted behavioral repair. This narrows what the remaining repair must do.

Artifacts: causal-repair-20261006-v4/instruction-correspondence-{identity-r1,
kernel-r2,controls-r1}, instruction-correspondence-identity-decision.json, and
dinov3-causal-repair-20261006/pilot64-mature-bs8-adam-checkpoint-r1-instruction-kernel-r1.
Scripts are probe_instruction_correspondence_identity.py (--kernel-audit for the
bounded kernel panel) and probe_instruction_correspondence_controls.py. Exact
commands, hashes and source snapshots remain alongside each experiment.

### 34.25 Remaining structural repair contract and the two future trainings

The next repair must address several distinct boundaries; none of the completed
inference controls establishes a complete replacement architecture. Do not launch
two formal runs simply because the diagnostic panels finished.

1. **Observed outcome versus prediction error.** policy.py:679–688 computes the
   measured semantic/image change and subtracts W's prediction, then exports the
   innovation. ExecutedWorldPlanRead primarily consumes this innovation. A
   correctly predicted lack of motion and a correctly predicted useful motion
   can both give zero innovation. Covariance/null/status do not reconstruct the
   discarded signed outcome. RobotExecutionObserver has the same distinction.
   This is a lossy FEEDBACK interface; the whole network still has current RGB,
   state/history and instruction-reference inputs. A repair should retain typed
   observed change, predicted change and innovation separately, with source-owned
   masks. It must not label prediction accuracy as task completion.
2. **Where feedback can revise the proposal.** Current policy ordering is robot
   observer -> G -> S -> executed-W replay -> coarse -> candidate W. Explicit
   executed-world and robot-response reads enter compiler.py:1853/1873 at P3.
   They do not condition this S/coarse proposal directly. Split binding from
   interval/phase organization if necessary, so the ONE S-owned binding can read
   causal observed outcome before forming the proposal, without another G pass,
   duplicate target law, future label, task-conditioned W, or extra ODE clock.
3. **Identity and correspondence.** G reconstruction's camera/background shortcut
   (34.18), weak language-to-object pressure (34.21), and the measured S rematch
   diffusion (34.24) are separate. Merely increasing S amplitude or sharpening
   uncalibrated logits does not fix them. Camera-conditioned reconstruction must
   not reward coding camera identity as object identity. A revised correspondence
   needs self/known-transform consistency and explicit unknown handling, tested
   against actual object movement, occlusion and cross-view source support.
4. **Joint arm/gripper conditions.** Resolve the matched conditional discrepancy
   from 34.22 while retaining ordinary gradients and observed-label support.
   Detached predicted-arm endpoint conditions are a candidate training method,
   not a certified fix; expert command labels on an incompatible generated arm
   require mechanical compatibility checks. Do not accept CE improvement alone.

The requested two full trainings are reserved as:

* **A — confirmed interface repair:** changes justified by source/trajectory
  evidence, with explicit outcome/innovation ownership, proposal consumption and
  an admitted arm/gripper condition contract. Keep unproven broad changes out of
  this variant. The current one-line kernel/robot-zero controls are excluded.
* **B — structural extension on A:** additionally rebuild the object/value and
  task-phase organization where camera/background reconstruction and compressed
  language/progress signals fail. Preserve token/object/view/source-time axes
  until their owning read, instead of amplifying a common pooled carrier.

These are implementation targets, not completed model configs. Before either
launch: review all accumulated repairs together; verify forward and ordinary
backward source ownership, same-image no-change, missing/unknown support, K/view
permutation, no future/oracle leakage, and real stage/identity diagnostics. Use
the same full declared training exposure and standard 256-batch offline plus
18-case replan-8 closed loop for both; 64 updates may check numerical mechanics,
not success. Keep data/normalizer/outlet/controller choices explicit, preserve
failure records, and do not silently reuse an incompatible checkpoint or Adam
state. Simulator positions/contact/masks remain audit-only under current scope.
