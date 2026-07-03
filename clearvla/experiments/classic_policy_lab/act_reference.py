from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .resnet import ResNet18FeatureMap


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _sinusoid_table(length: int, dim: int) -> torch.Tensor:
    positions = np.arange(length)[:, None]
    dims = np.arange(dim)[None, :]
    angle = positions / np.power(10000, 2 * (dims // 2) / dim)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return torch.tensor(angle, dtype=torch.float32).unsqueeze(0)


class ImagePositionEmbeddingSine(nn.Module):
    """The normalized 2D sine encoding used by ACT's DETR backbone."""

    def __init__(self, num_pos_feats: int, temperature: int = 10000, scale: float = 2 * math.pi) -> None:
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)
        self.scale = float(scale)

    def forward(self, tensor: Tensor) -> Tensor:
        not_mask = torch.ones_like(tensor[:, :1])
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).squeeze(1).permute(0, 3, 1, 2)
        return pos


class PosEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float, pre_norm: bool = False) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.pre_norm = bool(pre_norm)

    @staticmethod
    def _with_pos(x: Tensor, pos: Optional[Tensor]) -> Tensor:
        return x if pos is None else x + pos

    def forward(self, src: Tensor, *, pos: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if self.pre_norm:
            normed = self.norm1(src)
            q = k = self._with_pos(normed, pos)
            src = src + self.dropout1(self.attn(q, k, normed, key_padding_mask=key_padding_mask)[0])
            normed = self.norm2(src)
            return src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(normed)))))
        q = k = self._with_pos(src, pos)
        src = self.norm1(src + self.dropout1(self.attn(q, k, src, key_padding_mask=key_padding_mask)[0]))
        return self.norm2(src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src))))))


class PosEncoder(nn.Module):
    def __init__(self, layer: PosEncoderLayer, count: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = _clones(layer, count)
        self.norm = norm

    def forward(self, src: Tensor, *, pos: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            src = layer(src, pos=pos, key_padding_mask=key_padding_mask)
        return self.norm(src) if self.norm is not None else src


class PosDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float, pre_norm: bool = False) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.pre_norm = bool(pre_norm)

    @staticmethod
    def _with_pos(x: Tensor, pos: Optional[Tensor]) -> Tensor:
        return x if pos is None else x + pos

    def forward(self, tgt: Tensor, memory: Tensor, *, memory_pos: Tensor, query_pos: Tensor) -> Tensor:
        if self.pre_norm:
            normed = self.norm1(tgt)
            q = k = self._with_pos(normed, query_pos)
            tgt = tgt + self.dropout1(self.self_attn(q, k, normed)[0])
            normed = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.cross_attn(self._with_pos(normed, query_pos), self._with_pos(memory, memory_pos), memory)[0])
            normed = self.norm3(tgt)
            return tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(normed)))))
        q = k = self._with_pos(tgt, query_pos)
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(q, k, tgt)[0]))
        tgt = self.norm2(tgt + self.dropout2(self.cross_attn(self._with_pos(tgt, query_pos), self._with_pos(memory, memory_pos), memory)[0]))
        return self.norm3(tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(tgt))))))


class PosDecoder(nn.Module):
    def __init__(self, layer: PosDecoderLayer, count: int, norm: nn.Module) -> None:
        super().__init__()
        self.layers = _clones(layer, count)
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, *, memory_pos: Tensor, query_pos: Tensor) -> Tensor:
        outputs = []
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_pos=memory_pos, query_pos=query_pos)
            outputs.append(self.norm(tgt))
        return torch.stack(outputs)


class ACTDETRTransformer(nn.Module):
    def __init__(self, *, dim: int, heads: int, encoder_layers: int, decoder_layers: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = PosEncoderLayer(dim, heads, ffn_dim, dropout)
        decoder_layer = PosDecoderLayer(dim, heads, ffn_dim, dropout)
        self.encoder = PosEncoder(encoder_layer, encoder_layers)
        self.decoder = PosDecoder(decoder_layer, decoder_layers, nn.LayerNorm(dim))
        self._reset()

    def _reset(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, image_src: Tensor, image_pos: Tensor, query_embed: Tensor, latent: Tensor, qpos: Tensor, extra_pos: Tensor) -> Tensor:
        batch, channels, height, width = image_src.shape
        src = image_src.flatten(2).permute(2, 0, 1)
        pos = image_pos.flatten(2).permute(2, 0, 1)
        query_pos = query_embed.unsqueeze(1).repeat(1, batch, 1)
        additional_pos = extra_pos.unsqueeze(1).repeat(1, batch, 1)
        src = torch.cat([torch.stack([latent, qpos], dim=0), src], dim=0)
        pos = torch.cat([additional_pos, pos], dim=0)
        memory = self.encoder(src, pos=pos)
        tgt = torch.zeros_like(query_pos)
        return self.decoder(tgt, memory, memory_pos=pos, query_pos=query_pos).transpose(1, 2)


@dataclass(frozen=True)
class ACTReferenceConfig:
    state_dim: int = 7
    action_dim: int = 7
    camera_names: tuple[str, ...] = ("top", "wrist")
    chunk_len: int = 25
    hidden_dim: int = 512
    ffn_dim: int = 3200
    heads: int = 8
    transformer_encoder_layers: int = 4
    transformer_decoder_layers: int = 7
    style_encoder_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0
    resnet18_weights: Path | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["camera_names"] = list(self.camera_names)
        out["resnet18_weights"] = None if self.resnet18_weights is None else str(self.resnet18_weights)
        return out


class ACTReference(nn.Module):
    """ACT CVAE network faithfully adapted from the official ALOHA release.

    Adaptations are explicit: dimensions and camera names are configurable and
    the local ResNet implementation avoids a hard torchvision runtime dependency.
    The architecture remains ACT: shared ResNet-18 camera backbone, width-wise
    camera concatenation, DETR encoder-decoder, CVAE style encoder, L1 + KL loss.
    """

    def __init__(self, config: ACTReferenceConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.backbone = ResNet18FeatureMap(frozen_batch_norm=True, weights=config.resnet18_weights)
        self.image_proj = nn.Conv2d(512, dim, kernel_size=1)
        self.image_pos = ImagePositionEmbeddingSine(dim // 2)
        self.transformer = ACTDETRTransformer(
            dim=dim,
            heads=config.heads,
            encoder_layers=config.transformer_encoder_layers,
            decoder_layers=config.transformer_decoder_layers,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
        self.query_embed = nn.Embedding(config.chunk_len, dim)
        self.action_head = nn.Linear(dim, config.action_dim)
        self.is_pad_head = nn.Linear(dim, 1)
        self.input_proj_robot_state = nn.Linear(config.state_dim, dim)

        self.cls_embed = nn.Embedding(1, dim)
        self.encoder_action_proj = nn.Linear(config.action_dim, dim)
        self.encoder_joint_proj = nn.Linear(config.state_dim, dim)
        self.latent_proj = nn.Linear(dim, config.latent_dim * 2)
        self.register_buffer("style_pos_table", _sinusoid_table(2 + config.chunk_len, dim), persistent=False)
        style_layer = PosEncoderLayer(dim, config.heads, config.ffn_dim, config.dropout)
        self.style_encoder = PosEncoder(style_layer, config.style_encoder_layers)
        self.latent_out_proj = nn.Linear(config.latent_dim, dim)
        self.additional_pos_embed = nn.Embedding(2, dim)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1), persistent=False)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _latent(self, qpos: Tensor, actions: Tensor | None, is_pad: Tensor | None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch = qpos.shape[0]
        if actions is None:
            latent_sample = torch.zeros((batch, self.config.latent_dim), device=qpos.device, dtype=qpos.dtype)
            return self.latent_out_proj(latent_sample), None, None
        action_embed = self.encoder_action_proj(actions)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1).permute(1, 0, 2)
        if is_pad is None:
            raise ValueError("ACT training requires is_pad")
        prefix_pad = torch.zeros((batch, 2), dtype=torch.bool, device=qpos.device)
        key_padding_mask = torch.cat([prefix_pad, is_pad], dim=1)
        pos = self.style_pos_table[:, :encoder_input.shape[0]].permute(1, 0, 2).repeat(1, batch, 1)
        cls_output = self.style_encoder(encoder_input, pos=pos, key_padding_mask=key_padding_mask)[0]
        latent_info = self.latent_proj(cls_output)
        mu, logvar = latent_info.chunk(2, dim=-1)
        latent_sample = mu + torch.exp(logvar / 2) * torch.randn_like(mu)
        return self.latent_out_proj(latent_sample), mu, logvar

    def forward(self, qpos: Tensor, image: Tensor, actions: Tensor | None = None, is_pad: Tensor | None = None) -> dict[str, Tensor | None]:
        if image.ndim != 5 or image.shape[1] != len(self.config.camera_names):
            raise ValueError(f"ACT image must be [B,Cam,3,H,W], got {tuple(image.shape)}")
        image = (image - self.image_mean) / self.image_std
        latent, mu, logvar = self._latent(qpos, actions, is_pad)
        features = []
        positions = []
        for camera_index in range(image.shape[1]):
            fmap = self.image_proj(self.backbone(image[:, camera_index]))
            features.append(fmap)
            positions.append(self.image_pos(fmap))
        # Official ACT folds the camera axis into image width.
        image_src = torch.cat(features, dim=3)
        image_pos = torch.cat(positions, dim=3)
        qpos_token = self.input_proj_robot_state(qpos)
        hs = self.transformer(image_src, image_pos, self.query_embed.weight, latent, qpos_token, self.additional_pos_embed.weight)
        decoded = hs[-1]
        return {
            "actions": self.action_head(decoded),
            "is_pad_logits": self.is_pad_head(decoded).squeeze(-1),
            "mu": mu,
            "logvar": logvar,
        }

    def compute_loss(self, qpos: Tensor, image: Tensor, actions: Tensor, is_pad: Tensor) -> dict[str, Tensor]:
        output = self(qpos, image, actions=actions, is_pad=is_pad)
        pred = output["actions"]
        mask = (~is_pad).unsqueeze(-1).to(pred.dtype)
        l1 = (torch.abs(pred - actions) * mask).mean()
        mu = output["mu"]
        logvar = output["logvar"]
        assert mu is not None and logvar is not None
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
        return {"loss": l1 + self.config.kl_weight * kl, "l1": l1, "kl": kl}

    @torch.no_grad()
    def predict(self, qpos: Tensor, image: Tensor) -> Tensor:
        return self(qpos, image)["actions"]
