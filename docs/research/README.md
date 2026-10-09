# ClearVLA 文档入口

更新：2026-10-09 UTC。当前研究主线为
`codex/b-v1-structural-repair-20261008`。先确认正在操作哪个checkout；
本地旧分支、候选archive与远端固定实验目录不是同一个运行身份。

## 按任务选择资料

| 要做什么 | 读取哪里 |
| --- | --- |
| 改模型或数据流 | [当前架构合同](00_CURRENT_ARCHITECTURE_CONTRACT.md)，从 Agent quick contract 开始 |
| 理解已有证据、定位问题 | [当前问题表](CURRENT_MAINLINE_ISSUES.md) |
| 继续本轮研究/实现 | [当前修复计划](CURRENT_MAINLINE_REPAIR_PLAN.md) |
| 查实验、路径、checkpoint或接续 | [运行交接](auxiliary/ACTIVE_MAINLINE_HANDOFF.md)，再核对实际状态 |
| 查某个历史结论或复现 | [研究账本](G_SLOT_IDENTITY_BINDING_REVIEW.md)，按章节定位 |
| 模块源码入口 | [mainline README](../../clearvla/mainline/README.md) |
| 仿真接口 | [simulation runbook](../development/SIMULATION.md)，核对具体outlet |
| 早期设计/其他分支 | [归档索引](archive/README.md)或[辅助资料](auxiliary/README.md) |

这些入口按需读取，不是一份每次任务都要读完的清单。
运行事实以匹配源码和序列化工件为准；文档中的候选不能自动成为已采用架构。

## 维护方式

每条决定只在所属文件维护一次。合同不放PID，交接不复制模型设计，
计划不重新堆叠完整探针输出。研究账本保留证据，但不是自动执行命令。
原始日志、checkpoint、NPZ和大张量留在实验目录。

## 本次整理解决的冲突

- 旧合同3631行，快速入口在第1695行；历史修复与当前图混在一起。
- 根README仍要求DINO/decoded cache，且写旧的未来区间和fresh-only训练。
- “当前问题/交接”首屏仍是9月M阶段和当时资源状态。
- 旧“超过22 GiB停止”与实际已完成的22.18 GiB运行、用户显存授权不符。
- 日志审计技能强制加载两份长参考；改为按具体指标与源码边界查阅。
- “本轮不推送/不启动”等历史描述不再作为新任务的永久行为约束。

旧全文保存在清理前提交和[归档索引](archive/README.md)，研究账本原文保留。
本次没有改变训练、网络、实验分数或运行中的作业。

## Agent协作依据

采用[OpenAI关于GPT-6 Astra的官方建议](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)：
说明适用范围，按需提供细节，减少重复指令，用明确交付与完成条件支持持续工作。
这是提示和文档设计建议；实际权限仍由用户、宿主与工具决定。
[AGENTS.md官方说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
解释项目指令的发现与作用范围。
