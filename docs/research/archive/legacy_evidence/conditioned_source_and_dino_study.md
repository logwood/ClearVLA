# Conditioned source、初始状态与 DINO 轨迹实验

日期：2026-07-17
范围：只做机制实验和静态核对，不改正式训练/采样路径。

## 结论摘要

1. **不能把 V90 的 arm multiscale source 原样推广到 gripper。** 它确实改善了协方差条件数和有效维数，但在同总能量下显著放大时间差分和加速度；toy OOD 也不优于 AR(0.95)。“更满秩”不是“更符合任务”。
2. **source 的 scale 必须保持单一、不可学习的所有权。** 联合学习 shift+scale 时，scale 从 1.0 缩到 0.286，field loss 从 0.0367 降到 0.0090；这是优化几何退化，不是模型更懂动作。
3. **完整 24 步 condition shift 在本任务中会变成隐藏 action proposal。** 低带宽 DC shift 和零均值时间 shift 都没有收益；只有 full-horizon shift 略降 loss，但 source-target 距离反而几乎翻倍。
4. **初始状态应该保留双重但分权的角色。** 它可以决定 source 的固定边界均值，同时还应作为独立 boundary token 进入 vector-field reader。不能先和意图/视觉压成一根 global condition，也不应让语义网络自由改 source scale。
5. **typed reader 有实际机制收益，但证据仍是 toy 级。** 参数量几乎相同的情况下，boundary/event/timing 三类 token 相比 single fused token，IID MSE 低约 14%，组合 OOD 低约 28%，并且跨 seed 方差更小。
6. **提供的 DINO cache 没有时序塌缩。** top/wrist 的 global temporal effective rank 都约 12.5；保留空间结构后 wrist 为 19.6。原图变化与 DINO patch 变化相关为 0.764/0.551。
7. **两个相机不能预先混成一个摘要。** top/wrist 的慢变量 CKA=0.780，但 temporal-delta CKA 只有 0.155：共享任务阶段，局部变化却明显互补。应保留 camera type、patch position 和时间地址后再由 reader 检索。
8. **wrist 输入存在 27/338=7.99% 的精确重复转移。** 原图和 DINO token 的重复位置完全一致，说明这是输入采样/相机 cadence，不是 DINO 编码塌缩。是否跨数据集普遍存在仍需全数据 probe。

## 环境与可复现产物

- Python: `D:\CondaData\envs\torch312\python.exe`，Python 3.12.13
- PyTorch: 2.13.0+cu130，CUDA 可用，RTX 4060 Laptop 8GB
- CUDA float64 eig/cholesky、BF16 forward/backward 均通过
- 可运行项目 import；环境缺少 `h5py`、`pytest`、`scipy`、`pandas`、`matplotlib`
- `tests.test_temporal_dct`: 15/15 通过

产物：

- `clearvla/tools/probe_conditioned_source_design.py`
- `docs/research/conditioned_source_design_full.json`
- `docs/research/conditioned_source_shift_bandwidth.json`
- `clearvla/tools/probe_dino_trajectory_cache.py`
- `docs/research/dino_trajectory_cache_probe.json`
- `docs/research/dino_trajectory_contact_sheet.png`
- `docs/research/dino_mismatch_contact_sheet.png`

## 1. 现有 Source 合同

`BoundaryConditionedArmSource` 当前满足以下边界：

- 只读 `action_state`，不读 target、DINO、controller 或 task semantics。
- `boundary_multiscale` 把 identity、一次积分、二次积分三个 covariance component 分别 trace-normalize 后混合。
- component shape 和总 scale 分权；innovation 权重大于零保证 full rank。
- `PhysicalActionCodec.sample_noise()` 先在 native arm 空间采样，再一致地编码为 abs/delta manifold。
- gripper 仍在 native 24-step 空间采白噪声，再经 Parseval frame 展开；V90 没改 gripper source。

这是一个清楚的实现合同，但手工 covariance 是否正确仍是经验问题。

## 2. 几何与桥实验

### 2.1 Parseval frame

| 指标 | 结果 |
|---|---:|
| singular min/max | 0.99999998 / 1.00000004 |
| Gram identity max error | 7.28e-8 |
| FP32 `highest` roundtrip RMSE | 8.45e-8 |
| TF32 `high` roundtrip RMSE | 3.87e-4 |

结论：frame 本身正确；默认 `highest` 下没有坐标误差。只有显式切到 TF32 `high` 才出现约 4e-4 的矩阵舍入误差，不能把它误诊为 Parseval 几何错误。

### 2.2 V90 与 AR(0.95) 的同能量对比

V90 默认 `scale=0.8, weights=0.50/0.35/0.15`，与 AR(0.95) 的 source RMS 基本相同。

| Source | RMS | effective dim | condition | delta RMS | acceleration RMS | first std | terminal std |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR(0.95) | 0.804 | 2.16 | 397.4 | 0.315 | 0.448 | 0.312 | 0.956 |
| V90 multiscale | 0.800 | 4.72 | 21.0 | 0.803 | 1.369 | 0.581 | 1.048 |

V90 用更好的数值条件换来了 2.55x delta 和 3.06x acceleration。它不是失败证明，但说明当前 A/B 同时改变了“秩”和“时间粗糙度”，不能只按 pflow 初值判断满秩机制。

在合成 gripper-like 轨迹的小网络实验中：

| Source | IID MSE | compositional OOD MSE |
|---|---:|---:|
| AR(0.95) | 0.0601 | 0.2530 |
| V90-like multiscale, scale matched | 0.0490 | 0.3058 |
| raw DCT covariance, shrink=0.30 | 0.0404 | 0.2477 |
| empirical residual covariance, shrink=0.30 | 0.0511 | 0.2980 |
| white, scale matched | 0.0438 | 0.4016 |

结论只能用于否证：满秩 multiscale 并非天然更稳健；它不能证明 raw-DCT 方案可直接上线，因为现有 DCT 统计来自绝对轨迹，97% 能量在 DC，且不是 `future-current_state` residual covariance。

### 2.3 Trace matching 的边界

所有独立、零相关且 trace-matched 的 source，平均 bridge RMS 都约为 0.903。改变 covariance 方向不会自动缩短独立 coupling 的平均欧氏距离；它改变的是局部条件场的难度和频谱，而不是总距离。

因此 source A/B 至少要同时报告：

- total RMS / trace
- first-step、adjacent delta、acceleration、terminal std
- effective dimension / condition
- IID 与组合 OOD field error
- event/hold 分层误差

只对齐一个总 RMS 不足以隔离变量。

## 3. Condition-aware Source 的可辨识性

三 seed、2000 step 的联合 source/field 小实验：

| Source 参数化 | final field loss | source scale | source-target RMSE |
|---|---:|---:|---:|
| fixed | 0.03673 | 1.000 | 1.150 |
| condition global/DC shift | 0.03741 | 1.000 | 1.283 |
| condition zero-mean temporal shift | 0.03640 | 1.000 | 2.381 |
| condition full 24-step shift | 0.03417 | 1.000 | 2.227 |
| full shift + learned scale | 0.00900 | 0.286 | 1.354 |

解释：

- affine 的大幅降 loss 来自压缩随机 source，不应当算能力收益。
- full-horizon shift 的小幅收益并没有缩短桥；它把条件编码成一条确定、容易被 field 读回的 24-step 轨迹。
- DC shift 和 centered shift 单独都不挣钱，说明 full shift 的自由度组合本身是关键，也正因此接近 action proposal shortcut。
- 这与 CAR-Flow 的结论不冲突：CAR-Flow 明确限制为 shift-only 以消灭 scale-collapse；但本项目的 condition 几乎直接决定动作，完整 horizon shift 的带宽仍然需要单独约束和审计。

当前建议：**不要让 DINO/task/controller 直接生成 source 的完整 24x7 shift。** 若以后重试，先只允许 observed boundary state 决定固定 mean；语义信息继续进入 vector field。任何额外 shift 都必须报告 DC/non-DC RMS、与 target 的距离和反事实依赖。

## 4. Reader 拓扑实验

三个 reader 参数量为 85.7k-86.0k，使用相同 action query、自注意力、FFN 和训练预算：

| Reader | IID MSE | OOD MSE | state shuffle ratio | semantic shuffle ratio |
|---|---:|---:|---:|---:|
| single fused token | 0.02588 | 0.20157 | 89.6x | 117.6x |
| replicated fused token | 0.02560 | 0.21769 | 91.7x | 116.5x |
| typed boundary/event/timing tokens | **0.02222** | **0.14451** | 85.2x | 134.1x |

复制同一个 fused token 没有帮助，说明收益不是 token 数量本身。typed reader 的 OOD 均值和方差都更好，且对 semantic shuffle 更敏感，符合“角色分离后再检索”的机制预期。

限制：这是人工可组合任务，只有三个初始化；可作为接口设计证据，不能代替真实训练 A/B。

## 5. DINO 真实轨迹审查

### 5.1 对齐与数值

- decoded image 与 DINO cache 的 source fingerprint 完全相同。
- 339 帧，top/wrist 双相机；DINO shape=`[339,2,256,768]`，全部 finite。
- DINO token 是从这份 decoded RGB cache 直接编码并丢弃 CLS 后保存，不是另一套图像源。

### 5.2 时间信息

| 指标 | top | wrist |
|---|---:|---:|
| global temporal entropy-rank | 12.53 | 12.55 |
| spatially pooled temporal entropy-rank | 13.29 | 19.56 |
| global top-4 energy | 0.698 | 0.703 |
| raw-image vs DINO patch step Pearson | 0.764 | 0.551 |
| lag-1 global cosine | 0.993 | 0.974 |
| lag-24 global cosine | 0.965 | 0.824 |
| nonlocal nearest cosine (exclude +/-8) | 0.988 | 0.940 |

DINO 没有塌缩，但 global mean 明显存在时间 alias：相隔较远的帧仍可非常相似。它能提供视觉语义，不能独自承担 progress/timestamp；action state、相机类型、patch 地址和显式时间必须保留。

### 5.3 相机分工

- top/wrist global feature linear CKA = 0.780
- top/wrist temporal delta CKA = 0.155
- 时间反转 control delta CKA = 0.122
- framewise delta cosine 均值约 0.002

两相机共享慢任务阶段，但瞬时变化几乎不同向。最合理的 reader 是相机内先保留 patch 结构，再由 typed cross-attention/scene slots 融合；不能先 mean-pool 后相加。现有 `rdt2_control_interface.SceneTaskCompiler` 的 camera embedding 和 per-camera slot 方向是合理的，应保留。

### 5.4 重复帧

wrist 有 27 个精确重复 transition，位置在约 205-262 和 308-333 的周期性采样段；原图与 token 重复索引逐个相同。可能是 wrist 相机低帧率或记录 cadence，不是 DINO 编码问题。

下一次全数据 probe 应报告每 camera/episode 的：duplicate fraction、run/cadence、timestamp gap、动作事件附近可见帧数。若 8% 级别普遍存在，训练时不能把两个相机的同一数组 index 默认解释成同一瞬时时刻。

## 6. 结构决策

建议的职责分配：

1. `source geometry owner`：固定 scale；arm/gripper 分别定义 native covariance；只允许 observed boundary state 决定固定 mean。
2. `boundary evidence owner`：action/state 以独立 typed tokens 进入 reader，不因已经进入 source mean 而删除。
3. `semantic evidence owner`：DINO top、DINO wrist、intent、stage 分别保留 type/address；reader 负责检索和融合。
4. `flow-time owner`：继续通过显式 time embedding/AdaLN 调制，不从 x_t 粗糙度推断阶段。
5. `action owner`：只有 MMDiT/vector field 输出完整 horizon action update；source shifter 不能成为第二个 24-step policy head。

## 7. 下一步最小充分实验

1. 在训练机对全部 episode 读取 `future_action - current_action_state`，按 arm/gripper、hold/open/close、事件相位统计 conditional residual covariance；不要复用绝对轨迹 DCT energy。
2. 用 Ledoit-Wolf/diagonal shrinkage 或固定 full-rank floor 做 covariance family，但把 scale 固定为独立超参；扫描 shape 时保持 trace、首步和 delta 诊断可见。
3. V90 只作为 arm source A/B 跑，不同步改 gripper。比较 AR(0.95)、V90 multiscale、dataset-residual shrinkage 三种。
4. reader A/B 只比较 single fused 与 typed boundary/camera/semantic tokens，保持参数量、source 和随机种子一致。
5. 对 DINO 全数据加 timestamp/duplicate/event-visibility probe；当前单 episode 不能回答 gripper 事件是否可见。

## 8. 文献对应

- [Flow Matching for Generative Modeling, ICLR 2023](https://openreview.net/pdf?id=PqvMRDCJT9t)：FM 回归由指定 probability path 决定的 vector field；source/path 是训练几何的一部分。
- [CAR-Flow, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/8838314582eb031f55aee65a703e76c5-Paper-Conference.pdf)：condition-aware shift 可减轻条件注入负担，但 unrestricted scale 存在零代价 collapse；论文因此采用 shift-only。
- [Perceiver, ICML 2021](https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf)：用 typed/positioned inputs 经 cross-attention 汇入 latent，而不是先拍平成同质摘要。
- [Attention Bottlenecks for Multimodal Fusion, NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html)：通过专门 latent bottleneck 控制跨模态共享，而不是无约束 early fusion。
- [Multimodal Diffusion Transformer, RSS 2024](https://www.roboticsproceedings.org/rss20/p121.pdf)：noisy action tokens 用 self-attention，条件通过 cross-attention 融合，支持 action query 与条件 memory 分权。
- [DINOv2](https://arxiv.org/abs/2304.07193)：DINOv2 是通用静态视觉表征；本地轨迹实验表明仍需任务内时间地址和相机角色。

## 置信度边界

- **高置信**：Parseval 几何、source covariance 数值、scale-collapse、cache 指纹/重复帧来自解析或直接数据检查。
- **中等置信**：typed reader 和 source family 的相对趋势来自三 seed toy 学习实验。
- **低到中等置信**：DINO 对真实任务事件的充分性。当前只有一个 episode 且没有同步 action/qpos，不能直接测 gripper event recall 或动作条件互信息。
