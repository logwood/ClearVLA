# ClearVLA 本地日志目录

这里保存决策所需的原始实验日志，不保存 checkpoint、权重、tensor cache
或可再生的大型审计报告。架构文档只记录结论和来源，不复制这些原始文件。

## 放置层级

- `current/`：当前主线的正式实验、不可覆盖的进行中快照和配套 smoke。
  当前 Schema30 运行仍在远端，尚未镜像到本地时以 handoff 中的 run tag 为准。
- `reference/primary/`：仍承担跨版本判断的历史锚点。旧版本不等于低价值；
  只要它是同一数据合同下的关键前后对照，就保留完整原始证据。
- `diagnostics/`：validation-only、counterfactual、ablation 等判因证据。
  它们可以很重要，但不能冒充独立训练结果。
- `reference/incomplete/`：结构上有参考意义、但没有完成预定训练与最终
  validation 的 run。只用于过程分析，不用于最终性能排名。

## 当前目录表

| 重要性 | Run | 状态与边界 | 主要用途 |
| --- | --- | --- | --- |
| 当前远端 | `schema30_pen_b8_20260902_115644` / `schema30_rdt8_b8_20260902_115726` | Schema30 正式训练进行中；本地尚未封存完整三件套 | Pen 核心闭环与 RDT-8 外层接口；实时状态见 handoff |
| 历史失效 | `current/schema29/integration_4125a3d/` 与 `current/schema29/cachefix_d8a77a1/` | 路径名沿用早期收集位置；Schema29 行为结论因参数 VJP 故障失效，禁止续训 | 只保留 loader/启动历史和 autocast-cache 判因 |
| 主要锚点 | `reference/primary/schema25_r1_b8_20260826_220907/` | console 完成 epoch 8 / step 22768；JSONL 只到 step 22202，且没有原始 `run_context` | Schema25 replay 底座及后续版本的基准 |
| 主要锚点 | `reference/primary/schema25_r2_wg01_p202_grip02_b8_20260828_012356/` | commit `956bc6fb`，8/8 validation epochs 完成 | typed-P2/codec 闭环前的完整 R2 对照 |
| 主要锚点 | `reference/primary/schema26_wg01_p202_grip02_b8_20260829_151657/` | commit `33708a4e`，8/8 epochs 完成 | continuous gripper trajectory 改动对照 |
| 主要锚点 | `reference/primary/schema27_w_typed_norm_b8_20260829_224048/` | commit `35ab87ac`，8/8 epochs 完成 | bounded typed-W normalization 改动对照 |
| 主要锚点 | `reference/primary/schema28_action_world_b8_20260831_013140/` | commit `097330a8`，8/8 epochs 完成 | Schema29 之前最近的完整单任务行为锚点 |
| 重要诊断 | `diagnostics/schema25_r1_r2_a01_val_20260827_193618/` | commit `3219431a`，R1 checkpoint 的完整 validation-only replay | matched P2 value、geometry 与 gripper 判因；不是新训练 |
| 补充参考 | `reference/incomplete/schema25_r2_wg01_p202_grip02_b8_20260828_215113/` | commit `e92c7eee`；6 个 validation epochs 完成，训练进入 epoch 7 / step 18016 | 后期 typed-P2/codec 结构的中途行为；禁止当八轮最终值 |

有 `run_context` 的历史 Pen runs 使用相同的 dataset inventory SHA
`48e3b51b46ed19a1c5c099099531591a91974067ea299e041974c86828245066`
和 action-normalizer SHA
`cb686201cec4df6fe221c6ad71cfe24c00739c7d927d813ee494169b8a89867e`。
因此主要锚点可以做纵向比较；仍需结合每版具体指标语义，不能比较后来才新增的
指标并假装早期版本也测量过它。

## 完整 Pen 锚点的最终 validation 快照

| 版本 | action RMSE | gripper RMSE | decoded event F1 | tail / first |
| --- | ---: | ---: | ---: | ---: |
| Schema25 R1 | 0.08018 | 0.15772 | 0.39392 | 9.011 |
| Schema25 R2 initial | 0.07894 | 0.15295 | 0.38356 | 9.006 |
| Schema26 | 0.07943 | 0.15261 | 0.37172 | 8.289 |
| Schema27 | 0.07820 | 0.15362 | 0.32954 | 8.587 |
| Schema28 | 0.07657 | 0.14733 | 0.37715 | 7.658 |

这张表只负责定位对照证据，不是单指标排行榜。结构健康、远端区间、P2、geometry、
gripper event 与梯度路径仍须回到对应 JSONL 和源码共同解释。

## 收集规则

1. 每个 run 保留原 run tag，并优先保存 `.log`、`metrics.jsonl`、
   `run_context.json` 三件套。
2. 正式训练的中途文件放在带抓取时间的 `formal_snapshots/`，后续抓取不得覆盖。
3. 完成 run 才进入 `reference/primary/` 或当前版本的 `formal_completed/`。
4. 不完整 JSONL 必须通过目录名或 `RUN_NOTE.md` 显式标出边界。
5. 大型派生 audit JSON 不入库；需要时从原始三件套重新生成并输出到终端。
6. 不把 raw logs 纳入架构合同或 Git 提交；文档只引用路径、身份和决策统计。
7. `new_logs/` 除本 README 外整体由 Git 忽略；移动或整理本地证据时保留原
   run tag、文件哈希和完整三件套，不用覆盖来“更新”快照。
