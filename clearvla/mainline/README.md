# ClearVLA mainline 源码入口

本目录包含能力命名的模型、训练与部署实现。具体实验由解析配置和组件选择决定，
不是所有可选模块都在同一个图中。当前B-v1主线的语义见
[架构合同](../../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md)。
实验路径、checkpoint和资源状态见
[运行交接](../../docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md)。

## 按边界找源码

| 边界 | 文件 |
| --- | --- |
| 配置/身份 | `config.py`、`manifest.py`、`interfaces.py`、`checkpoint.py` |
| 组合入口 | `model/policy.py`、`model/top.py`、`model/components.py` |
| 视觉/G | `model/restored_observation.py`、`model/grounding.py`、`v120_core/flow_dino_evidence.py` |
| S与共享目标 | `model/intent.py`、`model/task_execution.py`、`model/target_binding.py` |
| 当前测量/指令起点 | `model/source_measurement.py`、`model/instruction_posterior.py` |
| W | `model/dynamics.py` |
| P1/P2/P3 | `model/v120_p1.py`、`model/compiler.py`、`model/horizon_coordination.py` |
| 底层/动作 | `model/restored_bottom.py`、`model/transition.py`、`model/action_codec.py` |
| 监督/优化 | `training/`、`annotation_goal.py`、`causal_identity.py` |
| 采样/验证/日志 | `runtime/` |
| 主训练循环 | `train.py` |

`v120_core/`是受上述接口约束的数值实现，不表示应使用历史monolith launcher。
`observation.py`、`bottom.py`等原型是否激活要看实际组件选择。

## 稳定入口

```bash
python -m clearvla.mainline.train --config <resolved-config>
python -m clearvla.tools.audit_policy_logs <run-directory> --format json
```

远端从固定源码和既有launcher声明的初始化、时钟、设备启动；
运行输出在统一实验根下，具体目录由交接记录。
验证用checkpoint自身的源码/配置，不在固定实验checkout中热改。

## 恢复与验证

- 新训练、精确续训、显式模型迁移、只读checkpoint验证有不同合同。
- fresh optimizer不等于模型执行时钟归零。
- 开发源码的 `training_complete.pt` 保护验证前训练状态，带待验证标记；
  它不是完成离线后的best/latest。
- 报告覆盖真实完成的训练、离线和闭环阶段；源码检查不代替学到的行为。

## 其他组件

- [B-spline](../action_representations/bspline/README.md)
- [Composite action](../action_representations/composite/README.md)
- [Standalone flow solver](../action_solvers/flow_solver/README.md)
- [Residual RL](../rl/README.md)

这些是独立或可选能力。阅读其文档时核对选定outlet和配置。
