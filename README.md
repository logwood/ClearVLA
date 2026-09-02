# ClearVLA 当前入口

本仓库的活动实现只有
[`clearvla/mainline/`](clearvla/mainline/README.md)。版本号只是记录用标签；
模型语义、检查点身份和可恢复边界由 manifest、源码和运行上下文决定。

## 先读什么

| 目的 | 唯一入口 |
|---|---|
| 当前架构、ABI、运行与放行门槛 | [`docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`](docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md) |
| 当前未关闭问题 | [`docs/research/CURRENT_MAINLINE_ISSUES.md`](docs/research/CURRENT_MAINLINE_ISSUES.md) |
| 当前修复顺序与实验门槛 | [`docs/research/CURRENT_MAINLINE_REPAIR_PLAN.md`](docs/research/CURRENT_MAINLINE_REPAIR_PLAN.md) |
| 当前实现的模块边界与启动方式 | [`clearvla/mainline/README.md`](clearvla/mainline/README.md) |
| Python/uv 开发环境 | [`docs/development/uv_environment.md`](docs/development/uv_environment.md) |
| 原始日志的放置、保留和再审计规则 | [`new_logs/README.md`](new_logs/README.md) |
| 研究文档分层与历史入口 | [`docs/research/README.md`](docs/research/README.md) |

`00_CURRENT_ARCHITECTURE_CONTRACT.md` 是当前真相；历史文档只能用于追溯
ancestry、旧日志或既往修复原因，不能授权把旧机制搬回活动主线。

## 当前身份

```text
capability: object_intent_dynamics_323
manifest:   Schema30
topology:   G1 G2 G3 / W1 W2 / P1 P2 P3
intervals:  4-8 / 8-16 / 16-32 / 32-48
training:   fresh, single-stage end-to-end
```

Schema30 的结构修正以及远端 CUDA VJP、Pen/RDT-8 smoke 和只读 checkpoint
验证已通过；Pen 与 RDT-8 正式运行已经从空目录启动。动态 PID、run tag 和最近
step 只记录在
[`ACTIVE_MAINLINE_HANDOFF.md`](docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md)。
Schema29 及更早 checkpoint 不是 Schema30 的 exact-resume 来源。

## 启动入口

先跑小批量 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/clearvla_mainline_smoke \
bash scripts/smoke_mainline.sh
```

确认资源和 backward/validation 合同后，再启动 batch-8：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/clearvla_mainline \
bash scripts/train_mainline.sh
```

已有 checkpoint 只能走只读验证入口：

```bash
CHECKPOINT=/path/to/clearvla-mainline-checkpoint-v4 \
bash scripts/validate_mainline_checkpoint.sh
```

数据、decoded/DINO cache、T5 条件文件和 batch/worker 可通过脚本中列出的
环境变量覆盖；不要把 checkpoint、cache 或 raw log 复制进架构记忆文档。

## 历史资料

旧版 V-numbered launcher、设计稿和
[`legacy/README.md`](legacy/README.md) 仅用于复现或归因，集中在
[`history_design/`](history_design/README.md)、
[`docs/research/archive/`](docs/research/archive/README.md) 和 `legacy/`。
辅助目录只保留当前交接、RDT 详参以及两份压缩历史索引；完整旧文可由各索引
记录的 Git commit 恢复。
