# Legacy experiment notes

本目录只保存 V40–V42 时代的复现实验脚本和它们的历史说明，不是当前
训练入口，也不参与 `clearvla/mainline/` 的 import 或 checkpoint 语义。

## 已合并的历史说明

原来的六个 README 已合并为本表；它们描述的脚本和机制只用于 ancestry、
旧日志解释或复现，不应作为新实验设计。原文可从历史提交恢复，例如：

```powershell
git show 64f3808:legacy/README_V42_1_CVAE_PRIOR_PATH_FIX.md
```

| 历史主题 | 保留的核心信息 |
|---|---|
| V40.1 + V37 residual action flow | 在稳定 V40.1 速度路径外添加零初始化 residual decoder；`aflow` 从零开始并应保持渐进。 |
| V40.1 + layered V37 action flow | residual blocks 读取分层 token/layer-pair memory；默认 `0:1,1:3,3:5,5:7`，layer memory 默认 detach。 |
| V41 latent-main action decoder | 用单一 `latent_main_action` decoder 替代 direct-plus-residual 最终路径；事件 token 位于同一主 decoder。 |
| V41.1 horizon-dependent depth | 在同一 latent-main decoder 内按 near/mid/far horizon 限制更新深度；不是旁路或第二个 head。 |
| V42 latent-CVAE action head | 用紧凑 `latent_cvae_action` head 替代 V41 大 decoder；prior 用于部署，posterior 只用于训练辅助。 |
| V42.1 CVAE prior-path fix | 修复训练 posterior 与部署 prior 不一致：主策略 loss 走 prior，posterior 只保留弱重建监督。 |

## 历史脚本

`run_v40_1_*`、`run_v41*` 和 `run_v42*` shell 脚本仍按原名保留。它们依赖
旧版数据/代码布局，启动前必须确认对应源码和日志存在；不要把它们当作
当前默认实验。

## 当前替代入口

- 当前实现与启动方式：[`../clearvla/mainline/README.md`](../clearvla/mainline/README.md)
- 当前架构合同：[`../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`](../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md)
- 当前问题与放行顺序：[`../docs/research/CURRENT_MAINLINE_ISSUES.md`](../docs/research/CURRENT_MAINLINE_ISSUES.md)

不要把 checkpoint、cache、raw log 或完整 probe dump 复制进本目录；需要时
从原始运行目录或 Git 历史重新取得。
