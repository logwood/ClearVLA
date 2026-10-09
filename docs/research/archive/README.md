# 历史证据入口

历史资料用于解释来源、旧日志或旧修复，不参与普通任务的默认阅读。
当前架构、问题和任务从[研究入口](../README.md)查找。

## 2026-10-09 文档整理前完整快照

提交：`a206d0481f22eb770f8b307bd7877a43d7f1b056`。
本轮覆盖更新的旧全文仍完整保存在Git中，不再复制一套长期加载的大文档。

| 旧路径 | 历史用途 |
| --- | --- |
| `docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md` | 3631行混合合同，包含早期M/C修复和旧默认图 |
| `docs/research/CURRENT_MAINLINE_ISSUES.md` | 747行早期问题及阶段状态 |
| `docs/research/CURRENT_MAINLINE_REPAIR_PLAN.md` | 690行早期计划和阶段约束 |
| `docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md` | 827行历次运行交接 |
| `docs/research/DINOV3_INTEGRATION_STATUS.md` | 235行原始权重、转换、早期资格与部署准备 |
| `README.md`、`clearvla/mainline/README.md` | 整理前入口和早期运行假设 |

只提取所需文件/段落，例如：

```bash
git show a206d0481f22eb770f8b307bd7877a43d7f1b056:docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md
```

本地 `.work` archive 不是Git仓库；此命令在当前远端开发checkout或含该提交的
真实clone中执行。研究账本
[G_SLOT_IDENTITY_BINDING_REVIEW.md](../G_SLOT_IDENTITY_BINDING_REVIEW.md)
仍保留原文和节号，包含近期结构、监督、训练与轨迹证据。

## 更早的资料

| 目录 | 内容 |
| --- | --- |
| [replay/](replay/README.md) | Schema25来源、回放与R1/R2修复 |
| [legacy_evidence/](legacy_evidence/README.md) | 早期实验与复现依据 |

已退休的history_design及早期本地日志索引可从提交`b8163cb`找回。
大型工件仍在实验存储。历史文件的命令、资源状况和阶段指示须按当时身份解释。
