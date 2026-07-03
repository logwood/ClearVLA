from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.evaluation.metrics import compute_metrics
from .act_reference import ACTReference
from .dp_reference import DPReference
from .normalizer import ArrayNormalizer


def _concat(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("evaluation produced no batches")
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def evaluate_act(
    model: ACTReference,
    loader: DataLoader,
    *,
    device: torch.device,
    action_normalizer: ArrayNormalizer,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    pred_norm, target_norm, prior_norm, past_norm = [], [], [], []
    losses = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        qpos = batch["qpos"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)
        result = model.compute_loss(qpos, image, actions, is_pad)
        pred = model.predict(qpos, image)
        losses.append({key: float(value.detach().cpu()) for key, value in result.items()})
        pred_norm.append(pred.cpu().numpy())
        target_norm.append(actions.cpu().numpy())
        prior_norm.append(action_normalizer.encode(batch["prior_raw"].numpy()))
        past_norm.append(action_normalizer.encode(batch["past_raw"].numpy()))
    metrics = compute_metrics(
        pred_norm=_concat(pred_norm),
        target_norm=_concat(target_norm),
        prior_norm=_concat(prior_norm),
        past_norm=_concat(past_norm),
        normalizer=action_normalizer,  # structural protocol: decode + std
    )
    for key in losses[0]:
        metrics[f"val_{key}"] = float(np.mean([row[key] for row in losses]))
    return metrics


@torch.no_grad()
def evaluate_dp(
    model: DPReference,
    loader: DataLoader,
    *,
    device: torch.device,
    action_normalizer: ArrayNormalizer,
    inference_steps: int,
    max_batches: int = 0,
    deterministic: bool = True,
) -> dict[str, Any]:
    model.eval()
    pred_norm, target_norm, prior_norm, past_norm = [], [], [], []
    denoise_losses = []
    start = model.config.obs_horizon - 1
    end = start + model.config.action_steps
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        obs_image = batch["obs_image"].to(device, non_blocking=True)
        obs_state = batch["obs_state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        denoise_losses.append(float(model.compute_loss(obs_image, obs_state, action).cpu()))
        trajectory = model.predict_trajectory(obs_image, obs_state, inference_steps=inference_steps, deterministic=deterministic)
        pred_norm.append(trajectory[:, start:end].cpu().numpy())
        target_norm.append(action[:, start:end].cpu().numpy())
        prior_norm.append(action_normalizer.encode(batch["prior_raw"].numpy()))
        past_norm.append(action_normalizer.encode(batch["past_raw"].numpy()))
    metrics = compute_metrics(
        pred_norm=_concat(pred_norm),
        target_norm=_concat(target_norm),
        prior_norm=_concat(prior_norm),
        past_norm=_concat(past_norm),
        normalizer=action_normalizer,
    )
    metrics["val_denoise_loss"] = float(np.mean(denoise_losses))
    metrics["inference_steps"] = int(inference_steps)
    metrics["execution_slice"] = [int(start), int(end)]
    return metrics


@torch.no_grad()
def evaluate_rdt_small(
    model,
    vision_encoder,
    language_conditioner,
    loader: DataLoader,
    *,
    device: torch.device,
    action_normalizer: ArrayNormalizer,
    inference_steps: int,
    sampler: str,
    max_batches: int = 0,
    deterministic: bool = True,
    eval_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate RDT-small on raw-unit action metrics with fixed sampling noise."""
    model.eval()
    vision_encoder.eval()
    pred_norm, target_norm, prior_norm, past_norm = [], [], [], []
    denoise_losses = []
    model_dtype = next(model.parameters()).dtype
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        state = batch["state"].to(device, dtype=model_dtype, non_blocking=True)
        images = batch["obs_image"].to(device, non_blocking=True)
        actions = batch["action"].to(device, dtype=model_dtype, non_blocking=True)
        ctrl_freqs = batch["ctrl_freq"].to(device, dtype=model_dtype, non_blocking=True)
        img_tokens = vision_encoder(images).to(device=device, dtype=model_dtype)
        lang_tokens, lang_mask = language_conditioner.batch(
            state.shape[0], device=device, dtype=model_dtype
        )
        denoise_losses.append(float(model.compute_loss(
            state=state,
            actions=actions,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
            img_tokens=img_tokens,
            ctrl_freqs=ctrl_freqs,
        ).detach().cpu()))
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(eval_seed) + batch_index)
        pred = model.predict_action(
            state=state,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
            img_tokens=img_tokens,
            ctrl_freqs=ctrl_freqs,
            inference_steps=inference_steps,
            sampler=sampler,
            deterministic=deterministic,
            generator=generator,
        )
        pred_norm.append(pred.float().cpu().numpy())
        target_norm.append(actions.float().cpu().numpy())
        prior_norm.append(action_normalizer.encode(batch["prior_raw"].numpy()))
        past_norm.append(action_normalizer.encode(batch["past_raw"].numpy()))
    metrics = compute_metrics(
        pred_norm=_concat(pred_norm),
        target_norm=_concat(target_norm),
        prior_norm=_concat(prior_norm),
        past_norm=_concat(past_norm),
        normalizer=action_normalizer,
    )
    metrics["val_denoise_loss"] = float(np.mean(denoise_losses))
    metrics["inference_steps"] = int(inference_steps)
    metrics["sampler"] = str(sampler)
    metrics["eval_seed"] = int(eval_seed)
    return metrics


@torch.no_grad()
def evaluate_rdt2_fm(
    model,
    conditioner,
    loader: DataLoader,
    *,
    device: torch.device,
    action_normalizer: ArrayNormalizer,
    inference_steps: int,
    max_batches: int = 0,
    eval_seed: int = 0,
    instruction: str = "",
    image_ablation: str = "normal",
) -> dict[str, Any]:
    """Evaluate RDT2-FM with fixed flow noise and explicit visual ablations.

    Online visual plugins receive ablated RGB tensors.  Cached DINOv2 plugins
    receive the same ablation request in token space.  ``shuffle-episode`` is
    dataset-aware, so it also works with evaluation batch size one.
    """
    allowed = {"normal", "zero", "mean", "shuffle-batch", "shuffle-episode", "top-only", "wrist-only"}
    if image_ablation not in allowed:
        raise ValueError(f"unsupported image_ablation={image_ablation!r}; choices={sorted(allowed)}")
    model.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()
    pred_norm, target_norm, prior_norm, past_norm = [], [], [], []
    flow_losses = []
    model_dtype = next(model.parameters()).dtype
    camera_names = tuple(getattr(loader.dataset, "camera_names", ("top", "wrist")))
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        state = batch["state"].to(device, dtype=model_dtype, non_blocking=True)
        images = batch["obs_image"].to(device, non_blocking=True)
        actions = batch["action"].to(device, dtype=model_dtype, non_blocking=True)
        sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
        conditioner_ablation = image_ablation
        if image_ablation == "shuffle-episode":
            if not hasattr(loader.dataset, "cross_episode_keys") or not hasattr(loader.dataset, "load_images_for_keys"):
                raise TypeError("shuffle-episode requires an RDT2FMWindowDataset with explicit image-key loading")
            sample_keys = loader.dataset.cross_episode_keys(sample_keys, seed=int(eval_seed) + batch_index)
            images = loader.dataset.load_images_for_keys(sample_keys).to(device, non_blocking=True)
            # The replacement was already performed at the dataset/cache key level.
            conditioner_ablation = "normal"
        instructions = [instruction] * state.shape[0]
        condition = conditioner.encode(
            images,
            instructions,
            sample_keys=sample_keys,
            image_ablation=conditioner_ablation,
            camera_names=camera_names,
        )
        condition = condition.to(device=device, dtype=model_dtype)
        kwargs = {
            "lang_tokens": condition.dense_tokens,
            "lang_kv_cache": condition.kv_cache,
            "lang_attn_mask": condition.attention_mask,
        }
        flow_losses.append(float(model.compute_loss(
            state_tokens=state,
            action_gt=actions,
            **kwargs,
        ).detach().cpu()))
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(eval_seed) + batch_index)
        noisy = torch.randn(
            (state.shape[0], model.pred_horizon, model.action_dim),
            device=device,
            dtype=model_dtype,
            generator=generator,
        )
        pred = model.predict_action(
            state_tokens=state,
            noisy_action=noisy,
            inference_steps=inference_steps,
            **kwargs,
        )
        pred_norm.append(pred.float().cpu().numpy())
        target_norm.append(actions.float().cpu().numpy())
        prior_norm.append(action_normalizer.encode(batch["prior_raw"].numpy()))
        past_norm.append(action_normalizer.encode(batch["past_raw"].numpy()))
    metrics = compute_metrics(
        pred_norm=_concat(pred_norm),
        target_norm=_concat(target_norm),
        prior_norm=_concat(prior_norm),
        past_norm=_concat(past_norm),
        normalizer=action_normalizer,
    )
    metrics["val_flow_mse"] = float(np.mean(flow_losses))
    metrics["inference_steps"] = int(inference_steps)
    metrics["eval_seed"] = int(eval_seed)
    metrics["image_ablation"] = str(image_ablation)
    return metrics

