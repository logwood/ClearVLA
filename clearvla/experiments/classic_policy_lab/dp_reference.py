from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .resnet import ResNet18GlobalPool


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: Tensor) -> Tensor:
        half = self.dim // 2
        exponent = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=x.device, dtype=x.dtype) * -exponent)
        embedding = x[:, None].to(dtype=x.dtype) * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """The FiLM residual block from the official Diffusion Policy U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = int(out_channels)
        self.cond_predict_scale = bool(cond_predict_scale)
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            scale, bias = embed.reshape(embed.shape[0], 2, self.out_channels, 1).unbind(dim=1)
            out = scale * out + bias
        else:
            out = out + embed
        return self.blocks[1](out) + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Full ConditionalUnet1D architecture from Diffusion Policy."""

    def __init__(
        self,
        *,
        input_dim: int,
        global_cond_dim: int | None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        all_dims = [input_dim, *down_dims]
        cond_dim = diffusion_step_embed_dim + (0 if global_cond_dim is None else global_cond_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )
        self.up_modules = nn.ModuleList()
        reversed_pairs = list(reversed(in_out[1:]))
        for dim_in, dim_out in reversed_pairs:
            # Match the published DP U-Net: each decoder stage upsamples.
            # With N encoder stages, there are N-1 decoder stages and N-1
            # corresponding upsampling operations, restoring the original horizon.
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self, sample: Tensor, timestep: Tensor | float | int, *, global_cond: Tensor | None = None
    ) -> Tensor:
        x = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timesteps = timestep[None].to(x.device)
        else:
            timesteps = timestep.to(x.device)
        timesteps = timesteps.expand(x.shape[0]).to(dtype=x.dtype)
        feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            feature = torch.cat([feature, global_cond], dim=-1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, feature)
            x = resnet2(x, feature)
            skips.append(x)
            x = downsample(x)
        for module in self.mid_modules:
            x = module(x, feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, feature)
            x = resnet2(x, feature)
            x = upsample(x)
        return self.final_conv(x).transpose(1, 2)


def _betas_for_alpha_bar(num_steps: int, max_beta: float = 0.999) -> Tensor:
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for index in range(num_steps):
        t1 = index / num_steps
        t2 = (index + 1) / num_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler:
    """Self-contained squared-cosine DDPM scheduler matching DP defaults.

    The official repository imports diffusers.DDPMScheduler. This local version
    keeps the same training process and DDPM reverse equation without adding a
    heavyweight runtime dependency. Deterministic evaluation can disable the
    posterior noise explicitly for stable offline comparisons.
    """

    def __init__(
        self,
        *,
        num_train_timesteps: int = 100,
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
    ) -> None:
        if prediction_type not in {"epsilon", "sample"}:
            raise ValueError(prediction_type)
        self.config = SimpleNamespace(
            num_train_timesteps=int(num_train_timesteps),
            clip_sample=bool(clip_sample),
            prediction_type=prediction_type,
        )
        self.betas = _betas_for_alpha_bar(num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self._previous: dict[int, int] = {}

    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        alpha = self.alphas_cumprod.to(original.device)[timesteps].reshape(
            -1, *([1] * (original.ndim - 1))
        )
        return alpha.sqrt() * original + (1 - alpha).sqrt() * noise

    def set_timesteps(self, num_inference_steps: int) -> None:
        if not 1 <= num_inference_steps <= self.config.num_train_timesteps:
            raise ValueError("invalid inference step count")
        values = (
            torch.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps)
            .round()
            .long()
        )
        # Avoid duplicate rounded timesteps for unusual requests.
        unique = []
        for value in values.tolist():
            if not unique or value != unique[-1]:
                unique.append(int(value))
        self.timesteps = torch.tensor(unique, dtype=torch.long)
        self._previous = {
            value: (unique[index + 1] if index + 1 < len(unique) else -1)
            for index, value in enumerate(unique)
        }

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        t = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        previous = self._previous.get(t, t - 1)
        device = sample.device
        alpha_prod_t = self.alphas_cumprod.to(device)[t]
        alpha_prod_prev = (
            torch.tensor(1.0, device=device)
            if previous < 0
            else self.alphas_cumprod.to(device)[previous]
        )
        beta_prod_t = 1 - alpha_prod_t
        if self.config.prediction_type == "epsilon":
            pred_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        else:
            pred_original = model_output
        if self.config.clip_sample:
            pred_original = pred_original.clamp(-1, 1)
        beta_prod_prev = 1 - alpha_prod_prev
        current_alpha = (alpha_prod_t / alpha_prod_prev).clamp(max=1.0)
        current_beta = 1 - current_alpha
        # DDPM posterior mean, generalized to a subsampled timestep schedule.
        pred_original_coeff = alpha_prod_prev.sqrt() * current_beta / beta_prod_t.clamp_min(1e-8)
        current_sample_coeff = current_alpha.sqrt() * beta_prod_prev / beta_prod_t.clamp_min(1e-8)
        previous_sample = pred_original_coeff * pred_original + current_sample_coeff * sample
        if not deterministic and previous >= 0:
            noise = torch.randn(
                sample.shape, dtype=sample.dtype, device=device, generator=generator
            )
            variance = (beta_prod_prev / beta_prod_t.clamp_min(1e-8) * current_beta).clamp_min(0)
            previous_sample = previous_sample + variance.sqrt() * noise
        return previous_sample


class RandomOrCenterCrop(nn.Module):
    def __init__(self, crop_hw: tuple[int, int] | None) -> None:
        super().__init__()
        self.crop_hw = crop_hw

    def forward(self, x: Tensor) -> Tensor:
        if self.crop_hw is None:
            return x
        crop_h, crop_w = self.crop_hw
        height, width = x.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError(f"crop={self.crop_hw} exceeds image={x.shape[-2:]}")
        if self.training:
            y = torch.randint(0, height - crop_h + 1, (x.shape[0],), device=x.device)
            z = torch.randint(0, width - crop_w + 1, (x.shape[0],), device=x.device)
            return torch.stack(
                [
                    frame[..., int(yi) : int(yi) + crop_h, int(zi) : int(zi) + crop_w]
                    for frame, yi, zi in zip(x, y, z)
                ],
                dim=0,
            )
        y0 = (height - crop_h) // 2
        x0 = (width - crop_w) // 2
        return x[..., y0 : y0 + crop_h, x0 : x0 + crop_w]


class DPImageObsEncoder(nn.Module):
    """DP-style multi-image observation encoder with separate camera ResNets."""

    def __init__(
        self,
        *,
        camera_names: tuple[str, ...],
        state_dim: int,
        crop_hw: tuple[int, int] | None = (84, 84),
        group_norm: bool = True,
        resnet18_weights: Path | None = None,
        share_rgb_model: bool = False,
    ) -> None:
        super().__init__()
        self.camera_names = camera_names
        self.state_dim = int(state_dim)
        self.crop = RandomOrCenterCrop(crop_hw)
        self.share_rgb_model = bool(share_rgb_model)
        if share_rgb_model:
            self.shared_encoder = ResNet18GlobalPool(
                group_norm=group_norm, weights=resnet18_weights
            )
            self.camera_encoders = nn.ModuleDict()
        else:
            self.shared_encoder = None
            self.camera_encoders = nn.ModuleDict(
                {
                    name: ResNet18GlobalPool(group_norm=group_norm, weights=resnet18_weights)
                    for name in camera_names
                }
            )
        self.output_dim = 512 * len(camera_names) + state_dim

    def forward(self, images: Tensor, state: Tensor) -> Tensor:
        # images [B,Cam,3,H,W], state [B,D]
        if images.ndim != 5 or images.shape[1] != len(self.camera_names):
            raise ValueError(f"images must be [B,Cam,3,H,W], got {tuple(images.shape)}")
        features = []
        for index, name in enumerate(self.camera_names):
            image = self.crop(images[:, index])
            encoder = (
                self.shared_encoder
                if self.shared_encoder is not None
                else self.camera_encoders[name]
            )
            assert encoder is not None
            features.append(encoder(image))
        features.append(state)
        return torch.cat(features, dim=-1)


@dataclass(frozen=True)
class DPReferenceConfig:
    state_dim: int = 7
    action_dim: int = 7
    camera_names: tuple[str, ...] = ("top", "wrist")
    prediction_horizon: int = 16
    obs_horizon: int = 2
    action_steps: int = 8
    diffusion_train_steps: int = 100
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    crop_hw: tuple[int, int] | None = (84, 84)
    obs_encoder_group_norm: bool = True
    share_rgb_model: bool = False
    resnet18_weights: Path | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["camera_names"] = list(self.camera_names)
        out["down_dims"] = list(self.down_dims)
        out["crop_hw"] = None if self.crop_hw is None else list(self.crop_hw)
        out["resnet18_weights"] = (
            None if self.resnet18_weights is None else str(self.resnet18_weights)
        )
        return out


class DPReference(nn.Module):
    """Image-conditioned U-Net Diffusion Policy reference.

    This reproduces the official CNN policy structure: per-camera ResNet image
    encoders, concatenated low-dimensional state, flattened observation history
    as global condition, ConditionalUnet1D with FiLM, and epsilon prediction.
    """

    def __init__(self, config: DPReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.obs_encoder = DPImageObsEncoder(
            camera_names=config.camera_names,
            state_dim=config.state_dim,
            crop_hw=config.crop_hw,
            group_norm=config.obs_encoder_group_norm,
            resnet18_weights=config.resnet18_weights,
            share_rgb_model=config.share_rgb_model,
        )
        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=self.obs_encoder.output_dim * config.obs_horizon,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_train_steps,
            clip_sample=True,
            prediction_type="epsilon",
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_observation(self, obs_image: Tensor, obs_state: Tensor) -> Tensor:
        batch, obs_steps = obs_image.shape[:2]
        if obs_steps != self.config.obs_horizon:
            raise ValueError(f"obs horizon mismatch: {obs_steps} != {self.config.obs_horizon}")
        features = self.obs_encoder(
            obs_image.reshape(batch * obs_steps, *obs_image.shape[2:]),
            obs_state.reshape(batch * obs_steps, -1),
        )
        return features.reshape(batch, -1)

    def compute_loss(self, obs_image: Tensor, obs_state: Tensor, actions: Tensor) -> Tensor:
        global_cond = self.encode_observation(obs_image, obs_state)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.config.diffusion_train_steps, (actions.shape[0],), device=actions.device
        )
        noisy = self.scheduler.add_noise(actions, noise, timesteps)
        pred = self.model(noisy, timesteps, global_cond=global_cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict_trajectory(
        self,
        obs_image: Tensor,
        obs_state: Tensor,
        *,
        inference_steps: int,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        global_cond = self.encode_observation(obs_image, obs_state)
        trajectory = torch.randn(
            (obs_image.shape[0], self.config.prediction_horizon, self.config.action_dim),
            device=obs_image.device,
            dtype=obs_image.dtype,
            generator=generator,
        )
        self.scheduler.set_timesteps(inference_steps)
        for timestep in self.scheduler.timesteps.to(obs_image.device):
            pred = self.model(trajectory, timestep, global_cond=global_cond)
            trajectory = self.scheduler.step(
                pred, timestep, trajectory, generator=generator, deterministic=deterministic
            )
        return trajectory

    @torch.no_grad()
    def predict_action(
        self,
        obs_image: Tensor,
        obs_state: Tensor,
        *,
        inference_steps: int,
        deterministic: bool = True,
    ) -> Tensor:
        trajectory = self.predict_trajectory(
            obs_image, obs_state, inference_steps=inference_steps, deterministic=deterministic
        )
        start = self.config.obs_horizon - 1
        return trajectory[:, start : start + self.config.action_steps]


class EMAModel:
    """Small EMA helper following DP's training recipe."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.averaged_model = copy.deepcopy(model).eval()
        for parameter in self.averaged_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name, parameter in self.averaged_model.named_parameters():
            parameter.mul_(self.decay).add_(source[name], alpha=1 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, buffer in self.averaged_model.named_buffers():
            if name in source_buffers:
                buffer.copy_(source_buffers[name])
