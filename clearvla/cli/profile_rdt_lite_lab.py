from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict

import torch

from clearvla.cli.common import resolve_device
from clearvla.experiments.rdt_lite_lab.model import RDTLiteModel, RDTLiteModelConfig
from clearvla.experiments.rdt_lite_lab.schedule import (
    CosineDiffusionSchedule,
    DiffusionScheduleConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile corrected lightweight RDT-style policy")
    p.add_argument("--objective", choices=["rdt_denoise", "pi_flow"], default="rdt_denoise")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--state-history-len", type=int, default=1)
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--camera-order", nargs="+", default=["top", "wrist"])
    p.add_argument("--patch-grid", nargs=2, type=int, default=[16, 16])
    p.add_argument("--teacher-dim", type=int, default=384)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ffn-hidden", type=int, default=384)
    p.add_argument(
        "--conditioning-mode", choices=["concat", "camera_alternate", "alternate"], default="concat"
    )
    p.add_argument("--include-visual-delta-tokens", action="store_true")
    p.add_argument("--decoder-output-init-std", type=float, default=1e-3)
    p.add_argument("--control-frequency-hz", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sampling-steps", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    return p.parse_args()


def _module_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    cameras = tuple(str(value) for value in args.cameras)
    config = RDTLiteModelConfig(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        chunk_len=args.chunk_len,
        obs_horizon=args.obs_horizon,
        state_history_len=args.state_history_len,
        camera_names=cameras,
        camera_order=tuple(str(value) for value in args.camera_order),
        patch_grid=(int(args.patch_grid[0]), int(args.patch_grid[1])),
        teacher_dim=args.teacher_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        conditioning_mode=args.conditioning_mode,
        include_visual_delta_tokens=args.include_visual_delta_tokens,
        independent_camera_dropout=0.0,
        time_encoding="rdt_discrete" if args.objective == "rdt_denoise" else "pi_continuous",
        decoder_output_init_std=args.decoder_output_init_std,
        control_frequency_hz=args.control_frequency_hz,
    )
    model = RDTLiteModel(config).to(device).eval()
    sampling_steps = int(args.sampling_steps or (5 if args.objective == "rdt_denoise" else 10))
    state_history = torch.randn(
        args.batch_size, args.state_history_len, args.state_dim, device=device
    )
    visual = torch.randn(
        args.batch_size,
        args.obs_horizon,
        len(cameras),
        config.patch_count,
        args.teacher_dim,
        device=device,
    )
    schedule = CosineDiffusionSchedule(DiffusionScheduleConfig())
    breakdown = OrderedDict(
        visual_adaptor=_module_parameters(model.visual_adaptor),
        state_adaptor=_module_parameters(model.state_adaptor),
        action_adaptor=_module_parameters(model.action_adaptor),
        time_embedder=_module_parameters(model.time_embedder),
        frequency_embedder=_module_parameters(model.frequency_embedder),
        blocks=_module_parameters(model.blocks),
        decoder=_module_parameters(model.decoder),
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            model.sample_actions(
                objective=args.objective,
                state_history=state_history,
                visual_tokens=visual,
                steps=sampling_steps,
                diffusion_schedule=schedule,
            )  # type: ignore[arg-type]
        _sync(device)
        durations: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            model.sample_actions(
                objective=args.objective,
                state_history=state_history,
                visual_tokens=visual,
                steps=sampling_steps,
                diffusion_schedule=schedule,
            )  # type: ignore[arg-type]
            _sync(device)
            durations.append(time.perf_counter() - started)
    print(
        json.dumps(
            {
                "schema": "clearvla-rdt-lite-profile-v13.1",
                "device": str(device),
                "objective": args.objective,
                "sampling_steps": sampling_steps,
                "config": config.to_dict(),
                "camera_schedule": model.camera_schedule(),
                "parameter_count": model.parameter_count(),
                "parameter_breakdown": breakdown,
                "latency_ms_mean": 1000.0 * sum(durations) / len(durations),
                "latency_ms_min": 1000.0 * min(durations),
                "latency_ms_max": 1000.0 * max(durations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
