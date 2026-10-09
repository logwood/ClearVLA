# 当前研究运行交接

快照核对：2026-10-09 03:32 UTC。状态会变化，行动前读取实际receipt、日志、
checkpoint和本人进程。本文件不证明整机GPU健康。
当前任务与后续交付见[修复计划](../CURRENT_MAINLINE_REPAIR_PLAN.md)。

## 工作目录与身份

| 名称 | 位置/身份 |
| --- | --- |
| 当前开发 | `/data/senwang/clearvla/checkouts/b-v1-structural-repair-20261008` |
| 分支 | `codex/b-v1-structural-repair-20261008` |
| 文档整理前HEAD | `a206d0481f22eb770f8b307bd7877a43d7f1b056` |
| 主线实验根E | `/data/senwang/clearvla/experiments/causal-identity-ab-20261007` |
| 旧A/B候选 | `/data/senwang/clearvla/checkouts/causal-identity-ab-20261007`，不是当前B-v1开发目录 |
| 固定地址记忆生产 | `/data/senwang/clearvla/checkouts/b-v1-address-memory-training-e4be3be6`，不可热改 |
| 本地候选镜像 | 项目 `.work/b-v1-structural-repair-20261008`；archive，不是Git checkout |
| 本地根源码 | `codex/coordinate-contract-calvin-20260927`，有其他工作修改；不等于远端主线 |

后续提交用实际git HEAD核验，不在文档里手填尚不存在的“最终提交”。

## 完整底座checkpoint

`E/B-short-bs8-1024-r1/checkpoints/best.pt`

- 生产源码：`3a84399926d50831470518e1b7f105131d3f1056`。
- epoch1 / global_step12036。
- SHA-256：`ce5753bf98a5e17254e6bbdbdd93fa80d09003a8b9b47d38cf2c3ca76d12ed25`。
- 标准闭环17/18。高分不排除null捷径、间接推动目标等已知问题。
- 迁移、参数继承、fresh optimizer与执行clock要按具体新配置分别声明。

## 已完成与失败的实验

| 实验 | 当前可作出的声明 | 证据位置 |
| --- | --- | --- |
| A / B-v1 / B-v2短跑 | 各1024BS8与256离线、完整18R8已完成；13/17/12成功 | E内各short-final-audit及closed-loop目录 |
| B-v3 regions短跑 | 完整训练/离线/18R8/资格已完成，12/18；未正式推广 | §34.33.23、29及B-regions-v3资格结果 |
| B-v1地址记忆短跑 | 1024BS8更新到13060，offline CUDA故障；无最终checkpoint或本次panel | `B-v1-address-memory-short-bs8-1024-r1` |
| B-v1续训参照 | 同样完成训练后offline失败，最终权重未保存 | `B-v1-continuation-control-short-bs8-1024-r1` |
| 原两短跑评估接续 | 已失败，不能称等待checkpoint | 各GPU1 evaluation receipt/status |
| 两条主线正式训练 | 尚未启动；仍需候选短跑/行为门槛 | 当前修复计划 |

训练保存保护与增益诊断已在开发源码de1e1046加入。
`training_complete.pt`带待验证标记，不等于best/latest；旧权重不能补救恢复。
保留失败工件，下一有意义候选使用修复后的保存边界。

## 独立SAM结构探索

根目录：`/data/senwang/clearvla/experiments/sam-structure-exploration-20261008`。
用户选择B-v2底座，保持与当前主线分开。

- region-fusion long r1：wrapper失败；最近已归档2680更新，无checkpoint/完整离线。
- slot-feedback long r2：03:32 UTC核验本人PID2921404仍存在，
  cwd为`/data/senwang/clearvla/checkouts/sam-structure-training-edbba48b`；
  stdout/stderr为`Bv2-slot-feedback-long-bs8-r2.log`。
  02:57 UTC工件记录7560/11012更新；这不是03:32进度或完成声明。
- 尚无完整行为资格支持合入。保留既有作业和固定源码。

## 运行与资源约定

标准CALVIN panel：18例/6任务各3、seed0、max_steps360、execute_rows8、
stored_target。拿到合格checkpoint后，闭环放物理GPU1并串行安排。
其余新训练/探针根据实际空闲资源安排，不固定GPU4；用户明确留出的GPU0不抢占。
BS8优先；短跑BS4已授权，需声明等样本曝光及不同更新/clock。
不因文档整理重启失败训练或停止独立实验。

所有新运行锁源码/配置/权重和新输出目录。先查receipt/status/本人进程，
避免重复启动；GPU编号以实际物理映射核对。工具操作遵循senwang-server技能，
独立小sh放`/home/sen.wang/mysh`并先读其AGENTS.md。
原数据/权重/大型结果留远端；保留全NPZ，不下载内部大张量。

## Commands

按任务查现有launcher和解析后配置；不把旧workspace launcher写成唯一入口。
普通训练入口为 `python -m clearvla.mainline.train --config <resolved-config>`，
但模型初始化、时钟、GPU和数据参数应从所选实验的已核验launcher取得。
只读日志汇总：

```bash
python -m clearvla.tools.audit_policy_logs <run-directory> --format json
```

当前重要证据：E下`B-v1-v2-v3-consolidated-decision-evidence-r1.json`、
`B-v1-paired-source-log-decision-r1.json`、
`cross-version-next-task-status-20261008-r1.json`。
研究账本§34.33.27–29保留具体源码、复现命令和范围限定。
