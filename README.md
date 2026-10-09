# ClearVLA

面向长时域机器人操作的对象中心 Vision-Language-Action 研究系统。
当前研究主线是 `codex/b-v1-structural-repair-20261008`，以完整B-v1继续改进
通用目标区分、语言绑定、空间关系和持续操作。项目仍在研究阶段。

## 从这里开始

| 任务 | 文档 |
| --- | --- |
| 看当前模型 | [架构合同](docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md) |
| 看缺陷、证据和下一步 | [问题表](docs/research/CURRENT_MAINLINE_ISSUES.md)、[修复计划](docs/research/CURRENT_MAINLINE_REPAIR_PLAN.md) |
| 查实验和checkpoint身份 | [运行交接](docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md) |
| 找代码与入口 | [mainline](clearvla/mainline/README.md) |
| 查其他文档或历史 | [文档索引](docs/research/README.md) |

按任务查阅。运行事实以对应源码、解析配置、run context和checkpoint payload为准；
本地旧checkout和候选archive不能代替远端固定实验源码。

## 当前选定CALVIN路径

因果双相机RGB → 在线冻结DINOv3 → G对象事实；
S结合完整指令和历史产生共享目标绑定；
W预测物理动作后果，P1/P2/P3保留和编译事实供底层生成24×7命令。
世界端点为4/8/16/24，部署使用Q5提案、一次W重建和Q5精化。
标准18例闭环每次执行前8行后重规划。

RGB/DINO值缓存关闭。SAM可提供经核验的训练辅助标签，也可启发通用低成本结构；
目前没有以SAM teacher替换在线政策。K不固定对应颜色或物体。
当前图及尚未实现的修复在架构合同明确区分。

## 环境、资产与运行

`clearvla/mainline/`是当前能力命名实现；旧V编号源用于对应历史运行。
环境说明见[uv_environment](docs/development/uv_environment.md)，
仿真说明见[SIMULATION](docs/development/SIMULATION.md)。
已有远端训练环境不因文档更新而重装或替换CUDA依赖。

正式数据路径需要匹配的数据/split、语言bank、normalizer、DINO权重及声明的
训练标签资产。在线DINO配置不要求生成decoded或DINO cache。
具体路径与GPU/launcher从运行交接和该次解析配置取得。

训练入口：

```bash
python -m clearvla.mainline.train --config <resolved-config>
```

该示例不隐含初始化、预算或设备选择；这些必须由实验配置和现有launcher声明。
检查脚本在 `scripts/check_static.*`、`scripts/check_light.*`；
运行与本次改动有关的检查，不因改文档触发全量模型测试。

## 实验保存与其他组件

精确续训、显式迁移、只读验证分开。训练完成待验证快照不等于best checkpoint。
原始日志、NPZ、权重和大探针输出留实验存储，仓库只保存决策所需证据与复现入口。

B-spline、composite action、独立flow solver、residual RL及其他outlet保留各自
[组件文档](clearvla/mainline/README.md)。它们存在于仓库不代表已启用于当前主线。
