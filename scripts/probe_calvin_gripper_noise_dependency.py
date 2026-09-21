"""Read-only causal probe for CALVIN binary-gripper endpoint conditioning.

The probe holds one validation observation/cache fixed and changes only the
six legacy continuous-gripper coordinates of the physical action field.  It
is intentionally checkpoint-read-only and never writes into the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import sample_cached_action


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-loader-batches", type=int, default=32)
    parser.add_argument("--require-invariant", action="store_true")
    return parser


def _command_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    prediction = logits.detach().float().argmax(dim=-1)
    positive = prediction == 1
    target_positive = target == 1
    true_positive = (positive & target_positive).float().sum()
    false_positive = (positive & ~target_positive).float().sum()
    false_negative = (~positive & target_positive).float().sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy": float((prediction == target).float().mean().cpu()),
        "predicted_positive_rate": float(positive.float().mean().cpu()),
        "target_positive_rate": float(target_positive.float().mean().cpu()),
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "f1": float(f1.cpu()),
    }


def _disagreement(first: Tensor, second: Tensor) -> float:
    return float(
        (
            first.detach().float().argmax(dim=-1)
            != second.detach().float().argmax(dim=-1)
        )
        .float()
        .mean()
        .cpu()
    )


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.max_loader_batches <= 0:
        raise ValueError("batch and loader limits must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this full-model probe requires an available CUDA device")

    config = load_config(args.config)
    config.validate()
    if str(config.bottom.gripper_output_mode) != "calvin_binary_command":
        raise ValueError("probe requires bottom.gripper_output_mode=calvin_binary_command")
    dtype = resolve_compute_dtype(config)

    bundle = load_mainline_data(config)
    loader = bundle.loader(
        "val",
        batch_size=args.batch_size,
        workers=0,
        device=device,
        shuffle=False,
    )
    selected_raw: dict[str, Tensor] | None = None
    selected_index = -1
    selected_rate = -1.0
    for index, raw in enumerate(loader):
        rate = float((raw["policy_action_raw"][..., -1] >= 0.0).float().mean())
        selected_raw = raw
        selected_index = index
        selected_rate = rate
        if 0.05 < rate < 0.95:
            break
        if index + 1 >= args.max_loader_batches:
            break
    if selected_raw is None:
        raise RuntimeError("validation loader produced no batch")
    batch = to_training_batch(
        selected_raw,
        goal=bundle.goal,
        config=config,
        device=device,
    )

    model = ClearVLAMainlinePolicy(config).to(device)
    model.configure_action_normalizer(bundle.action_normalizer)
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    saved_model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(saved_model, dict):
        raise ValueError("checkpoint has no model state")
    model.load_state_dict(saved_model, strict=True)
    checkpoint_epoch = int(payload.get("epoch", -1))
    checkpoint_step = int(payload.get("global_step", -1))
    del payload, saved_model
    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    target = (batch.action_target.raw_units[..., -1] >= 0.0).long()
    target_field = model.outlet_adapter.encode(
        batch.action_target.normalized,
        batch.online.history.action_state,
    )
    base_noise = model.outlet_adapter.sample_noise(
        batch.online.history.batch,
        device=device,
        dtype=batch.online.history.action_state.dtype,
        generator=generator,
    )
    other_noise = model.outlet_adapter.sample_noise(
        batch.online.history.batch,
        device=device,
        dtype=batch.online.history.action_state.dtype,
        generator=generator,
    )
    tail_start = 2 * int(model.outlet_adapter.arm_dim)

    endpoint_fields = {
        "clean_target_field": target_field.clone(),
        "target_arm_random_gripper": target_field.clone(),
        "target_arm_zero_gripper": target_field.clone(),
        "target_arm_negated_target_gripper": target_field.clone(),
    }
    endpoint_fields["target_arm_random_gripper"][..., tail_start:] = base_noise[
        ..., tail_start:
    ]
    endpoint_fields["target_arm_zero_gripper"][..., tail_start:] = 0.0
    endpoint_fields["target_arm_negated_target_gripper"][..., tail_start:] = -target_field[
        ..., tail_start:
    ]

    autocast_enabled = dtype in {torch.bfloat16, torch.float16}
    endpoint_logits: dict[str, Tensor] = {}
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            cache, _training_state, _static_metrics = model.encode_online(
                batch.online,
                training_mask=False,
                geometry_supervision=False,
                collect_diagnostics=False,
            )
        endpoint_time = torch.ones(
            batch.online.history.batch,
            device=device,
            dtype=torch.float32,
        )
        for name, field in endpoint_fields.items():
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                output = model.velocity(
                    cache,
                    noisy_action_field=field,
                    time=endpoint_time,
                    collect_diagnostics=False,
                )
            logits = output.bottom.gripper_command_logits
            if logits is None:
                raise RuntimeError("CALVIN model did not expose command logits")
            endpoint_logits[name] = logits.detach().float()

        base_result = sample_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=base_noise,
            dtype=dtype,
        )
        gripper_changed_noise = base_noise.clone()
        gripper_changed_noise[..., tail_start:] = other_noise[..., tail_start:]
        gripper_changed_result = sample_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=gripper_changed_noise,
            dtype=dtype,
        )
        arm_changed_noise = base_noise.clone()
        arm_changed_noise[..., :tail_start] = other_noise[..., :tail_start]
        arm_changed_result = sample_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=arm_changed_noise,
            dtype=dtype,
        )

    base_logits = base_result.gripper_command_logits
    gripper_changed_logits = gripper_changed_result.gripper_command_logits
    arm_changed_logits = arm_changed_result.gripper_command_logits
    if base_logits is None or gripper_changed_logits is None or arm_changed_logits is None:
        raise RuntimeError("sampler did not preserve CALVIN command logits")

    endpoint_clean = endpoint_logits["clean_target_field"]
    endpoint_summary: dict[str, Any] = {}
    for name, logits in endpoint_logits.items():
        endpoint_summary[name] = {
            **_command_metrics(logits, target),
            "logit_rms_delta_vs_clean": _rms(logits - endpoint_clean),
            "command_disagreement_vs_clean": _disagreement(logits, endpoint_clean),
        }

    final_tail_identity = {
        "base": float(
            (
                base_result.physical_field[..., tail_start:]
                - base_noise[..., tail_start:]
            )
            .abs()
            .max()
            .cpu()
        ),
        "gripper_changed": float(
            (
                gripper_changed_result.physical_field[..., tail_start:]
                - gripper_changed_noise[..., tail_start:]
            )
            .abs()
            .max()
            .cpu()
        ),
        "arm_changed": float(
            (
                arm_changed_result.physical_field[..., tail_start:]
                - arm_changed_noise[..., tail_start:]
            )
            .abs()
            .max()
            .cpu()
        ),
    }
    endpoint_invariant = all(
        _disagreement(logits, endpoint_clean) == 0.0
        and _rms(logits - endpoint_clean) == 0.0
        for logits in endpoint_logits.values()
    )
    sampled_gripper_invariant = (
        _disagreement(gripper_changed_logits, base_logits) == 0.0
        and _rms(gripper_changed_logits - base_logits) == 0.0
        and _rms(
            gripper_changed_result.action[..., :-1] - base_result.action[..., :-1]
        )
        == 0.0
    )
    report = {
        "schema": "clearvla-calvin-gripper-noise-causal-probe-v1",
        "acceptance": {
            "endpoint_gripper_field_invariant": endpoint_invariant,
            "sampled_initial_gripper_noise_invariant": sampled_gripper_invariant,
            "passed": endpoint_invariant and sampled_gripper_invariant,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "epoch": checkpoint_epoch,
            "global_step": checkpoint_step,
        },
        "batch": {
            "loader_batch_index": selected_index,
            "rows": int(target.numel()),
            "target_positive_rate": selected_rate,
        },
        "physical_field": {
            "dimension": int(model.outlet_adapter.physical_dim),
            "arm_channel_end": tail_start,
            "gripper_channels": int(model.outlet_adapter.physical_dim - tail_start),
        },
        "fixed_endpoint_time_1": endpoint_summary,
        "five_step_single_pass": {
            "base": _command_metrics(base_logits, target),
            "change_only_initial_gripper_noise": {
                **_command_metrics(gripper_changed_logits, target),
                "command_disagreement_vs_base": _disagreement(
                    gripper_changed_logits, base_logits
                ),
                "logit_rms_delta_vs_base": _rms(gripper_changed_logits - base_logits),
                "arm_action_rmse_delta_vs_base": _rms(
                    gripper_changed_result.action[..., :-1]
                    - base_result.action[..., :-1]
                ),
            },
            "change_only_initial_arm_noise": {
                **_command_metrics(arm_changed_logits, target),
                "command_disagreement_vs_base": _disagreement(
                    arm_changed_logits, base_logits
                ),
                "logit_rms_delta_vs_base": _rms(arm_changed_logits - base_logits),
                "arm_action_rmse_delta_vs_base": _rms(
                    arm_changed_result.action[..., :-1] - base_result.action[..., :-1]
                ),
            },
            "gripper_field_final_minus_initial_max_abs": final_tail_identity,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_invariant and not report["acceptance"]["passed"]:
        raise SystemExit("CALVIN command path still depends on the legacy gripper field")


if __name__ == "__main__":
    main()
