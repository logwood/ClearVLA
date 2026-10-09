# DINOv3 在线集成状态

更新：2026-10-09 UTC。当前B-v1主线已使用在线冻结DINOv3 ViT-B/16；
本页只记录编码器资产与存储边界，不再承担训练任务交接。

## 当前资产与配置

- 原始权重：`/data/senwang/clearvla/third_party/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。
- 原始SHA-256：`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。
- 来源：用户提供的第三方再分发镜像，不是Meta官方下载端点。
- 本地转换目录：`/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m`。
- 选定配置：`dinov3_online_v1`、`hdf5-direct`、`dinov3_local_files_only=true`、
  `dino_cache=""`、`image_frame_lru_capacity=0`。
- 冻结的是DINO编码器；下游可训练的G/S/W/P消费者仍有普通梯度。

旧的decoded-cache字段为配置兼容保留，不表示该在线路径读缓存。
紧凑辅助标签与RGB/DINO值缓存分开计成本。
训练Python与CALVIN renderer可能使用不同环境；使用运行receipt核对，
不沿用早期“服务器没有Python3.12”的说法。

## 历史与下一步

9月的原始下载、转换、RoPE恢复与早期部署资格记录保存在
[清理前文档快照](archive/README.md)。它们不是当前launcher或当前进度。
当前模型见[架构合同](00_CURRENT_ARCHITECTURE_CONTRACT.md)，
训练/评估见[交接](auxiliary/ACTIVE_MAINLINE_HANDOFF.md)。
