# 数据量 / 表示难度扩容：文献复核（2026-09-07）

这份辅助账记录“别人如何扩容”，再映射到 ClearVLA。它不改变主架构合同，也不把旧实验的窗口数或日志现象当作扩容判据。

## 结论先行

1. 文献中的扩容不是单一动作。至少有四条路线：函数保持的离线 widening/deepening、按新任务/新域增加列或 adapter、按条件输入激活更多专家/深度、以及通过数据/模型/计算 scaling law 选择下一档模型。
2. 对 ClearVLA 最稳的组合是 **S/M/L 离线模型族 + 函数保持 checkpoint migration + 覆盖度驱动的升级判据**。第一版应只扩大内部 hidden/FFN/depth/rank/adapter capacity，保持 Schema/action codec、G/S/W/P 语义边界和 24×18 输出 ABI 不变。
3. MoE/dynamic-depth 可以作为条件容量和推理预算实验，但不能被误报为“总模型参数随数据自动成长”；它们通常在固定最大图上改变每个样本的激活计算。
4. 机器人/VLA 证据强烈支持“数据的独立轨迹、任务/物体/embodiment 覆盖比 stride-1 窗口数重要”，但公开工作大多是离线多规模对照或大数据预训练，并未给出 ClearVLA 可直接采用的在线自动扩容规则。

## 方法与证据表

| 工作 | 扩容触发 / 训练形态 | 扩的轴 | 函数保持 | checkpoint 迁移 | 对 ClearVLA 的可迁移性 |
|---|---|---|---|---|---|
| [Net2Net](https://arxiv.org/abs/1511.05641), Chen et al., 2015/ICLR 2016 | 先训小网，再按需要离线变宽/变深并继续训练 | width、depth | 是。Net2Wider 复制神经元并重分配下游权重；Net2Deeper 插入 identity 层（需满足激活/归一化条件） | 直接给出权重变换；优化器状态需按复制/零扩展处理 | 最直接的 S→M warm-start 候选。只扩内部 hidden/FFN/块数，保留 ABI；需为 Linear、LayerNorm、残差和 action head 写显式迁移测试 |
| [Network Morphism](https://arxiv.org/abs/1603.01670), Wei et al., 2016 | 架构搜索/预算变化时做 function-preserving morphism | conv/linear width、depth、kernel 等 | 是（在可行初始化条件下） | 支持不同拓扑的参数映射；不是针对 Transformer/Flow-VLA 的现成脚本 | 可借鉴“变更图但保持输出”的证明框架；ClearVLA 要自定义 MMDiT、cross-attn、低秩算子映射 |
| [Energy-efficient and Robust Cumulative Training with Net2Net](https://arxiv.org/abs/2003.01204), 2020 | 数据子集和模型容量分阶段增加；每阶段从上一 checkpoint 迁移而非重训 | width/depth + data subset | 主要依靠 Net2Net，目标是累计训练效率；不是任务语义保持证明 | 是，逐阶段扩网并继训 | 与“数据量增长→下一档模型”最贴近；可将阶段边界定义为独立 episode/覆盖度阈值，而不是窗口数 |
| [R2R: Towards a More Complete Theory of Function-Preserving Transformations](https://arxiv.org/abs/2410.11038), 2024 | 在已有图中加入 residual branch/block 或做结构变换 | residual depth、branch、宽度等 | 是；强调残差函数保持和训练可塑性 | 可通过零初始化新分支、复制/重标定旧参数迁移 | 适合在最后动作 DiT 或 P3 插入零初始化 residual block，先验证 logits/velocity 等价，再放开训练；预印本，需自行验证数值细节 |
| [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466), Yang et al., 2021/ICML 2022 | 在小宽度代理模型上调超参，迁移到更大宽度；不是数据触发的在线生长 | width，及与宽度耦合的学习率、初始化、参数化 | 不保证 checkpoint 函数等价；保证宽度极限下更新/激活尺度可迁移 | 权重可重初始化或复制后再训；关键是 μP 参数化和超参映射 | 用于 S/M/L 的学习率、初始化、weight decay、warmup 迁移；不能替代 Net2Net 的函数保持；对 512→768/1024 需做小网校准 |
| [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)；机器人 sim-to-real 版 [1610.04286](https://arxiv.org/abs/1610.04286), Rusu/Tzeng et al., 2016 | 新任务/新域到来时冻结旧列，增加新列和 lateral connection | task/domain columns、adapter | 旧列行为保持（新列初始不改变旧输出）；全图不等价 | 直接复用旧列 checkpoint；新增列随机/小初始化 | 适合新 embodiment、夹爪/灵巧手或视觉域；比把所有样本混在一个更宽 head 更可控。代价是列数和推理图增长，需 ABI 统一 |
| [DEN: Lifelong Learning with Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547), Yoon et al., 2018 | 新任务损失/梯度无法由当前容量解释时，group sparsity 选择性扩神经元；必要时 split | selective width、neuron split、任务特定连接 | 通过 selective retraining/约束尽量保持旧任务，不是严格全图等价 | 支持在旧模型上增参数并继续训 | 可用于“表示难度上升”触发候选 adapter/FFN 扩容；需要先定义 ClearVLA 的可观测触发量和回归集，不能直接移植 DEN 阈值 |
| [Subset Sampling for Progressive NN Learning](https://arxiv.org/abs/2002.07141), 2020 | 训练数据子集和拓扑逐级增长，在线选择超参 | data subset + topology/depth/width | 依赖渐进初始化；不承诺严格函数保持 | 支持阶段 checkpoint | 可把有效数据量（episode、任务/物体/接触覆盖）分桶，给每桶配 S/M/L；需防止样本窗口泄漏造成虚假“数据量” |
| [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538), Shazeer et al., 2017；[Switch Transformer](https://arxiv.org/abs/2101.03961), Fedus et al., 2021 | 固定主干和专家池；路由器按 token/input 选择少数专家 | experts、conditional FFN | 非函数保持（新专家可零/小初始化，但路由改变函数） | 可以增加专家并复制/零初始化，但路由/负载均衡需重校准 | 可放在最后动作 DiT 的 adapter/FFN 旁，按表示难度或 embodiment 路由；先做 2-expert ablation，保持 action ABI；不能宣称减少总参数，只是稀疏激活 |
| [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), Raposo et al., 2024 | 固定最大层数，router 决定每个 token 经过多少层 | conditional depth / token compute | 通常不严格保持；可用 residual skip 近似 | 从 dense checkpoint 加 router/skip 分支，需蒸馏/再训 | 适合作为后期推理预算实验（难样本走更深 DiT）；不应作为第一版扩容，因为动作 token 的跳层可能破坏时间/接触一致性 |
| Kaplan scaling laws ([2001.08361](https://arxiv.org/abs/2001.08361))、Chinchilla ([2203.15556](https://arxiv.org/abs/2203.15556)) | 通过多组独立训练拟合 loss 与参数、数据、compute 的幂律，选择 compute-optimal 档位 | model size、data tokens、compute | 否 | 不涉及 checkpoint 迁移 | 提供“何时升档”的实验设计原则：先做少量 S/M/L × 数据桶测量，再拟合 ClearVLA 的任务成功率/校准曲线；语言模型幂律不能直接移植为机器人成功率 |
| [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864), 2023 | 汇聚多机器人、多任务、多 embodiment 数据；比较单体与跨 embodiment 训练 | data diversity、embodiment coverage、model transfer | 否 | 以预训练权重/微调迁移，不是结构手术 | 证明有效数据应按独立轨迹、任务和 embodiment 覆盖核算；支持 ClearVLA 先扩数据覆盖再升 hidden/rank |
| [Data Scaling Laws in Imitation Learning for Robotic Manipulation](https://arxiv.org/abs/2410.18647), Hu et al., 2024/2026 revision | 在超过 40,000 个示范、15,000 次真实 rollout 上改变环境数、物体数和每个环境/物体的示范数 | data diversity、environment/object coverage | 否 | 不做 checkpoint 结构迁移；给出数据采集/升档依据 | 最直接的机器人扩容证据：泛化对环境/物体数近似幂律；每环境/物体达到阈值后再堆同分布示范收益很小。ClearVLA 的有效数据计数应优先覆盖新环境/物体，而不是 stride-1 窗口 |
| [The Curse of Precision: A Data Scaling Law for High-Precision Robotic Manipulation](https://arxiv.org/abs/2607.23108), Xu et al., 2026 | 固定成功率、提高目标精度时，研究所需示范数与精度的关系，并改变传感器/专家能力 | demonstrations、sensor/expert capability、precision difficulty | 否 | 不涉及 checkpoint 迁移 | 给出“表示/执行难度扩容”证据：所需示范数满足 `log N ∝ 1/(P-c)`，接近系统上限 `c` 时超指数增长；腕部相机或更强 expert 可降低 `c`。可用于把接触、遮挡、精密插入等难例单独作为升档信号 |
| [Octo](https://arxiv.org/abs/2405.12213), Ghosh et al., 2024 | 在 Open X-Embodiment 大规模轨迹上预训练，再对新机器人/任务微调 | data scale/diversity、policy backbone + action head | 否 | 预训练 checkpoint + action head/adapter 微调 | 说明共享主干、outlet-specific adapter 是可行路线；ClearVLA 可保持 G/S/W/P 语义，新增 embodiment adapter，不改 24×18 codec |
| [OpenVLA](https://arxiv.org/abs/2406.09246), Kim et al., 2024 | 大视觉语言模型 + 约百万机器人 episode 预训练；通过微调适配新任务 | backbone size、data scale、action head | 否 | 标准预训练/LoRA/全参微调；非 Net2Net | 作为 M/L 上限参考，不给出在线扩容规则；重点是数据多样性与表示容量共同增加，需在 ClearVLA 上做受控消融 |
| [RDT-1B](https://arxiv.org/abs/2410.07864), 2024 | 1B 级扩散 Transformer，多数据源/多 embodiment 训练与微调 | model width/depth、data diversity、action horizon/conditioning | 否 | checkpoint + embodiment/action adapters | 与 ClearVLA 的 diffusion/DiT 形态接近，但其 ABI 不同；可借鉴“共享 backbone + 末端 action adapter”及大模型微调流程 |
| [MoDE: Efficient Diffusion Transformer Policies with Mixture-of-Denoising Experts](https://arxiv.org/abs/2412.12953), Reuss et al., 2024 | 固定专家池和噪声条件路由；在不同去噪噪声水平选择稀疏 expert，并用缓存减少重复计算 | sparse experts、noise-conditioned routing、active compute | 否（路由改变函数） | 可加专家/复制专家后重新校准路由；需负载均衡和蒸馏/再训 | 与 ClearVLA 的扩散 DiT 最接近的条件容量证据：报告 active parameters 约降 40%、expert caching 使推理成本约降 90%，并在 CALVIN/LIBERO 134 tasks 上评估。适合后期作为末端 action-DiT 的 2-expert 对照，不应冒充总参数自动增长 |
| [π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164), Physical Intelligence, 2024 | 多机器人、大规模数据预训练；VLM backbone 与 flow action expert 分工 | backbone、flow action expert、data/embodiment coverage | 否 | 预训练 VLM/flow expert 再微调 | 支持“最后 action expert 是容量/适配插点”的判断；不证明 MIP 或自动结构生长，ClearVLA 仍需保持两遍 ODE 生命周期 |
| [RoboCat](https://arxiv.org/abs/2306.11706), Bousmalis et al., 2023 | 新任务数据到来后少量示范微调并自我生成数据，覆盖任务逐渐扩大 | task/data coverage、共享策略 | 否 | 共享模型 + 新任务微调 | 说明任务覆盖扩展可先于结构扩展；可把新接触模式/长时域阶段作为“有效难度”而非步长窗口触发信号 |

这篇机器人数据 scaling law 还做了一个不能忽略的模型侧对照：在固定 32 个环境-物体对、使用 50% 示范时，视觉编码器从 ViT-S/14→B/14→L/14 的分数约为 0.66→0.81→0.90；而 action diffusion U-Net 的最大特征维度从 512 扩到 2048 并没有提升，large 反而约 0.83（small/base 约 0.88/0.90）。作者给出的解释是当前动作分布可能已被小 U-Net 充分建模，或动作扩容架构本身尚未找到可扩展形式。迁移到 ClearVLA 的含义是：不能预设“最后 action DiT 加宽”必然有效；必须同时做 evidence/视觉表示容量与 terminal action 容量的单轴对照。

该工作的官方源码是 [Fanqi-Lin/Data-Scaling-Laws](https://github.com/Fanqi-Lin/Data-Scaling-Laws)。其 README 把扩容做成可复现实验开关：通过 `task.dataset.dataset_idx` 选择训练环境-物体对（最多 32 对），通过 `task.dataset.use_ratio∈(0,1]` 选择每对示范比例；仓库同时提供 small/base/large U-Net 与 ViT-S/B/L 的训练/评估脚本。这个实现层面的做法值得借鉴：把“数据覆盖桶”和“模型档位”显式参数化，而不是在同一个 run 中隐式改变采样或网络。

按 active `TimeDomainMMDiTBlock` 的线性层粗略计数（忽略 bias、Norm 和 head），单块参数量约为
`(14 + 2·ffn_expansion)·hidden²`；当前 `ffn_expansion=2` 时约为 `18·hidden²`。因此
hidden 512→768 使主干块参数/乘加约变为 2.25×，512→1024 约 4×；depth 3→4
只增加约 33% 的块计算；operator rank 32→64 主要让 contraction bank 线性增大，
不会把所有投影都放大。这个量级关系支持先做 rank/depth/末端 residual，再考虑全局
hidden widening，而不是把所有轴一起放大。

另一个迁移风险是 attention 参数化：当前 `time_domain_mmdit.py` 约 616 行使用
`1/√d_head`，并在多处使用标准 `MultiheadAttention`；μP 的官方实现要求对随宽度变化的
维度采用专门的初始化、学习率和 attention 缩放，并通过 coordinate check 验证激活坐标
不随宽度爆炸/塌缩。因此 μP 只能作为独立的 M/L 参数化 profile，不能直接覆盖默认
Schema30 的 attention 公式。

## 对 ClearVLA 的工程化建议（仅设计，不改源码）

### 可复用源码入口

| 方向 | 源码/项目入口 | 复用边界 |
|---|---|---|
| μP/μTransfer | [microsoft/mup](https://github.com/microsoft/mup) | 可复用 base-shape、初始化/LR 缩放和 coordinate-check；不提供 ClearVLA checkpoint 的函数保持迁移 |
| 机器人数据 scaling | [Fanqi-Lin/Data-Scaling-Laws](https://github.com/Fanqi-Lin/Data-Scaling-Laws) | 可复用 `dataset_idx`/`use_ratio` 式覆盖桶实验和环境-物体对照；其 UMI/Diffusion Policy 数据接口不能直接替换 ClearVLA |
| Octo | [octo-models/octo](https://github.com/octo-models/octo) | 参考共享 backbone + 新 observation/action adapter；不直接移植其 action token ABI |
| OpenVLA | [openvla/openvla](https://github.com/openvla/openvla) | 参考大规模预训练、LoRA/量化和多 embodiment 微调；不作为 Net2Net 迁移实现 |
| RDT | [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) | 参考扩散 Transformer 和统一物理 action space；其 bimanual ABI 与 ClearVLA 不同 |
| π0 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 参考 VLM + flow action expert 的分工；不改变 ClearVLA 两遍 ODE/codec 合同 |
| MoDE | [project page/code](https://mbreuss.github.io/MoDE_Diffusion_Policy/) | 参考噪声条件稀疏 denoiser 和 expert caching；需重新验证 action-token 时间一致性 |

Net2Net、Network Morphism、Progressive Nets、DEN 和 R2R 的论文提供算法/推导，
但没有一个可直接套到当前 MMDiT 的官方迁移脚本；这些路线在 ClearVLA 上必须先做
逐算子 state-dict surgery 与函数等价测试。

### 三档模型族

- **S**：当前 hidden=512、rank=32、默认深度和 ABI。
- **M**：只扩一个主轴（例如 hidden/FFN 宽度），其余不变；用 Net2Wider 或零初始化 residual 做函数保持迁移，再用 μP 迁移优化器超参。
- **L**：在 M 的基础上再扩第二轴（深度或 rank/adapter）；保持 Schema、G/S/W/P 载荷形状、K=4、N=49、24×18 不变。

第一轮实验必须一次只改变一个轴，并保留从 S 直接训练和 S→M migration 两个对照。

### 可扩而不可随意扩的轴

可扩：G/S/W/P/bottom 内部 hidden、FFN 中间宽度、残差块深度、低秩 operator rank、末端 action adapter/专家数。

暂不扩：action_dim、24-step horizon、18D action field、4 个 interval、K=4、N=49、camera 数、3×3 factual microgrid。这些是语义/ABI/计算合同，比例放大不等于能力扩展。

### 触发判据

用独立 episode/trajectory 计数并报告覆盖向量：任务、物体/颜色、embodiment/夹爪、接触/开合事件、长时域阶段、视觉运动/遮挡/光照。仅当新增数据使这些覆盖桶和验证集尾部难例持续增加，且 S 模型在固定 compute 下出现容量受限信号，才建议升到 M/L。容量 gate 接近 1 只能说明收缩饱和，不能单独证明欠容量。

### 验证与回滚

每次 migration 都应检查：旧 checkpoint 与新模型在同一 observation/noise/time 上的 velocity/action 最大绝对误差、均方误差、分位数；旧验证集成功率不回退；新增难度桶是否改善；activation/gradient RMS 与 μP 预期同阶；optimizer state 是否按新形状复制/零扩展。任何 ABI 或旧任务回归都回滚到 S。

### Active mainline source audit（避免把“可配置”误当成“已接线”）

当前 active mainline 的真正 action-DiT 路径是
`clearvla/mainline/v120_core/time_domain_mmdit.py`：
`TimeDomainMMDiTBlock`（约 521 行）→ `TerminalActionController`（约 744 行）→
`EvidenceLatentMMDiTActionDecoder`（约 890 行）。后者在约 995 行按
`latent_cvae_mmdit_depth` 建 `blocks`，约 1023 行建低秩 contraction bank，约
1062–1084 行创建 velocity head 和 terminal controller。

`clearvla/mainline/model/restored_bottom.py:_build_decoder_config` 只把
`BottomConfig.evidence_depth/operator_rank/operator_groups/controller_depth/controller_heads`
映射到 active core；`BottomConfig.ffn_expansion` 没有映射到
`latent_cvae_ffn_expansion`，目前只被兼容 wrapper `clearvla/mainline/model/bottom.py`
读取。因此“把 BottomConfig.ffn_expansion 从 2 改成 3”本身不会扩 active action-DiT，
必须先增加显式的 profile 字段/映射并重新序列化组件身份。这是实现扩容前必须补的
接线测试，而不是现在直接改默认值的授权。

首个末端扩容原型可放在 active decoder 的最后一个 `TimeDomainMMDiTBlock` 与
`TerminalActionController` 之间：新增 residual action block，外包一个零初始化的
输出标量/最后线性层，使迁移初始满足 `F_M(x)=F_S(x)`；再逐步学习该标量。这样不改
`TerminalActionController` 的 owner、velocity/gripper ABI 或 ODE 调用次数。若验证通过，
再考虑全 depth 或 hidden widening。

## 迁移限制

- Net2Net/Network Morphism 的严格等价多在 MLP/CNN 和理想激活条件下成立，MMDiT、Flow Matching、LayerNorm、cross-attention、低秩收缩需要逐算子证明和数值测试。
- μP 解决参数化/超参尺度传递，不保证复制 checkpoint 后的函数等价。
- Progressive/DEN/MoE/动态深度提供结构或条件容量，但会引入路由、负载均衡、旧任务回归和实时性变量。
- Open X、Octo、OpenVLA、RDT、π0 的“更大数据/模型更好”多为跨数据集或跨 embodiment 结果，不是严格单变量 scaling law；ClearVLA 必须做自己的 S/M/L × 有效覆盖桶消融。
