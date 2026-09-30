"""Inspect the real-batch future-W gradient route without an optimizer step.

This follows the formal training CLI through data loading, online DINOv3
construction, model encoding and loss composition.  It reports whether the
candidate and matched observed-control W objects remain attached to autograd,
then takes VJPs from the future-dynamics scalar to both W activations and the
three output heads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-autocast-cache", action="store_true")
    return parser


def _autocast(device: torch.device, dtype: torch.dtype, *, cache_enabled: bool):
    enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=bool(cache_enabled),
    )


def _rms(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().square().mean().sqrt())


def _vjp(loss: torch.Tensor, value: torch.Tensor) -> float | None:
    if not loss.requires_grad or not value.requires_grad:
        return None
    gradient = torch.autograd.grad(
        loss,
        value,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return _rms(gradient[0])


def _grad_fn_names(value: torch.Tensor, limit: int = 80) -> list[str]:
    """Return a bounded autograd node walk for a diagnostic report."""

    root = value.grad_fn
    if root is None:
        return []
    queue = [root]
    seen: set[int] = set()
    names: list[str] = []
    while queue and len(names) < limit:
        node = queue.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(type(node).__name__)
        for child, _ in getattr(node, "next_functions", ()):
            if child is not None:
                queue.append(child)
    return names


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    data = replace(config.data, num_workers=int(args.num_workers))
    optimizer_config = replace(config.optimizer, batch_size=int(args.batch_size))
    runtime = replace(config.runtime, compute_dtype=str(args.dtype))
    config = replace(config, data=data, optimizer=optimizer_config, runtime=runtime)
    config.validate()

    device = torch.device(args.device)
    dtype = resolve_compute_dtype(config)
    torch.manual_seed(config.data.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.data.seed)

    bundle = load_mainline_data(config)
    if config.data.visual_feature_mode == "dinov3_online_v1":
        from clearvla.vision.online_pipeline import OnlineVisionPipeline

        bundle = replace(
            bundle,
            visual_encoder=OnlineVisionPipeline.from_config(config, device),
        )

    loader_generator = torch.Generator().manual_seed(config.data.seed + 101)
    flow_generator = torch.Generator(device=device if device.type == "cuda" else "cpu").manual_seed(
        config.data.seed + 102
    )
    condition_generator = torch.Generator(
        device=device if device.type == "cuda" else "cpu"
    ).manual_seed(config.data.seed + 103)
    loader = bundle.loader(
        "train",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        generator=loader_generator,
    )
    raw_batch = next(iter(loader))
    batch = to_training_batch(
        raw_batch,
        goal=bundle.goal,
        visual_encoder=bundle.visual_encoder,
        config=config,
        device=device,
    )

    model = ClearVLAMainlinePolicy(config).to(device).train()
    model.configure_action_normalizer(bundle.action_normalizer)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        total_steps=max(config.optimizer.epochs * max(len(loader), 1), 1),
        minimum_ratio=config.optimizer.min_lr_ratio,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        dtype=dtype,
        train_flow_generator=flow_generator,
        train_condition_generator=condition_generator,
    )

    captured: dict[str, object] = {}
    original_build = model.build_training_targets
    head_forward: list[dict[str, object]] = []
    head_outputs: list[torch.Tensor] = []

    def capture_delta_head(_module, _args, output):
        head_outputs.append(output)
        head_forward.append(
            {
                "requires_grad": bool(output.requires_grad),
                "grad_fn_present": bool(output.grad_fn is not None),
                "rms": _rms(output),
                "data_ptr": int(output.data_ptr()),
                "weight_data_ptr": int(_module.weight.data_ptr()),
                "weight_requires_grad": bool(_module.weight.requires_grad),
            }
        )

    delta_hook = model.world.dynamics.delta_head.register_forward_hook(capture_delta_head)

    def capture_targets(training_state, future, **kwargs):
        result = original_build(training_state, future, **kwargs)
        captured["training_state"] = training_state
        captured["targets"] = result[0]
        return result

    model.build_training_targets = capture_targets  # type: ignore[method-assign]
    try:
        with _autocast(device, dtype, cache_enabled=not args.disable_autocast_cache):
            ledger, metrics = engine._forward(
                batch,
                training=True,
                collect_diagnostics=True,
                generator=flow_generator,
                condition_generator=condition_generator,
            )
    finally:
        delta_hook.remove()

        targets = captured["targets"]
        state = captured["training_state"]
        candidate = state.top.predicted_dynamics
        supervised = targets.supervised_world
        if supervised is None:
            raise RuntimeError("matched-world config did not materialize supervised W")
        future_loss = ledger.terms["future_dynamics"]
        heads = {
            "delta": model.world.dynamics.delta_head.weight,
            "transport": model.world.dynamics.transport_head.weight,
            "covariance": model.world.dynamics.covariance_head.weight,
        }
        all_world_parameters = tuple(model.world.dynamics.named_parameters())
        world_direct_gradients = torch.autograd.grad(
            supervised.dynamics.semantic_delta.sum(),
            tuple(parameter for _, parameter in all_world_parameters),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        head_vjps = {}
        for name, parameter in heads.items():
            gradient = torch.autograd.grad(
                future_loss,
                parameter,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0] if future_loss.requires_grad else None
            direct_gradient = torch.autograd.grad(
                supervised.dynamics.semantic_delta.sum(),
                parameter,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            head_vjps[name] = {
                "requires_grad": bool(parameter.requires_grad),
                "vjp_rms": _rms(gradient),
                "direct_semantic_sum_vjp_rms": _rms(direct_gradient),
            }
        head_output_vjps = []
        for output in head_outputs:
            if not output.requires_grad:
                head_output_vjps.append(None)
                continue
            head_output_vjps.append(
                _rms(
                    torch.autograd.grad(
                        output.sum(),
                        model.world.dynamics.delta_head.weight,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                )
            )

        report = {
            "config": str(args.config),
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "autocast_cache_enabled": not args.disable_autocast_cache,
            "world_supervision_mode": config.top.world_supervision_mode,
            "delta_head_weight_data_ptr": int(model.world.dynamics.delta_head.weight.data_ptr()),
            "future_loss": float(future_loss.detach().float()),
            "future_loss_requires_grad": bool(future_loss.requires_grad),
            "future_loss_grad_fn_present": bool(future_loss.grad_fn is not None),
            "candidate_semantic": {
                "requires_grad": bool(candidate.semantic_delta.requires_grad),
                "grad_fn_present": bool(candidate.semantic_delta.grad_fn is not None),
                "rms": _rms(candidate.semantic_delta),
                "loss_vjp_rms": _vjp(future_loss, candidate.semantic_delta),
            },
            "supervised_semantic": {
                "requires_grad": bool(supervised.dynamics.semantic_delta.requires_grad),
                "grad_fn_present": bool(supervised.dynamics.semantic_delta.grad_fn is not None),
                "rms": _rms(supervised.dynamics.semantic_delta),
                "loss_vjp_rms": _vjp(future_loss, supervised.dynamics.semantic_delta),
                "grad_fn_nodes": _grad_fn_names(supervised.dynamics.semantic_delta),
            },
            "candidate_is_supervised": bool(candidate is supervised.dynamics),
            "supervised_transport": {
                "requires_grad": bool(supervised.dynamics.transport_mean.requires_grad),
                "grad_fn_present": bool(supervised.dynamics.transport_mean.grad_fn is not None),
                "rms": _rms(supervised.dynamics.transport_mean),
                "loss_vjp_rms": _vjp(future_loss, supervised.dynamics.transport_mean),
            },
            "current_support_mean": float(
                targets.current_loss_support.detach().float().mean()
            ),
            "future_interval_valid_mean": None
            if targets.future_interval_valid is None
            else float(targets.future_interval_valid.float().mean()),
            "heads": head_vjps,
            "world_direct_semantic_sum_nonnull": [
                name
                for (name, _), gradient in zip(
                    all_world_parameters, world_direct_gradients, strict=True
                )
                if gradient is not None
            ],
            "engine_probe": {
                key: float(value.detach().float())
                for key, value in metrics.items()
                if key.startswith("gradient_probe_w_")
                or key.startswith("loss_future_prediction_")
            },
            "delta_head_forward_calls": head_forward,
            "delta_head_output_sum_vjp_rms": head_output_vjps,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
