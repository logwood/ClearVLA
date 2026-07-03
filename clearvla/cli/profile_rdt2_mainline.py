from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import (
    RDT2FMReferenceConfig,
    estimate_rdt2_fm_parameter_count,
)
from clearvla.experiments.classic_policy_lab.rdt2_mainline import MainlineRDT2FM, MainlineRDT2FMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile a v29 RDT2 mainline preset with clean future-DINO residual dynamics")
    p.add_argument("--model-size", choices=["small", "medium", "official"], default="medium")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--dense-token-dim", type=int, default=768)
    p.add_argument("--visual-corrector", choices=["none", "query-latent"], default="none")
    p.add_argument("--future-latent-variant", choices=["none", "world-only", "closed-loop"], default="none")
    p.add_argument("--future-latent-grid-size", type=int, default=8)
    p.add_argument("--future-latent-offsets", nargs="+", type=int, default=[8, 16, 24])
    p.add_argument("--future-latent-hidden-size", type=int, default=768)
    p.add_argument("--future-latent-depth", type=int, default=6)
    p.add_argument("--future-latent-heads", type=int, default=8)
    p.add_argument("--future-latent-kv-heads", type=int, default=4)
    p.add_argument("--future-latent-modulation-rank", type=int, default=192)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def config_for(
    name: str,
    *,
    action_dim: int,
    state_dim: int,
    dense_token_dim: int,
    visual_corrector: str,
    future_latent_variant: str,
    future_latent_grid_size: int,
    future_latent_offsets: tuple[int, ...],
    future_latent_hidden_size: int,
    future_latent_depth: int,
    future_latent_heads: int,
    future_latent_kv_heads: int,
    future_latent_modulation_rank: int,
) -> MainlineRDT2FMConfig:
    common = dict(
        action_dim=action_dim,
        state_dim=state_dim,
        lang_adaptor="mlp2x_silu",
        lang_token_dim=dense_token_dim,
        visual_corrector=visual_corrector,
        future_latent_variant=future_latent_variant,
        future_latent_dim=dense_token_dim,
        future_latent_grid_size=future_latent_grid_size,
        future_latent_offsets=future_latent_offsets,
        future_latent_hidden_size=future_latent_hidden_size,
        future_latent_depth=future_latent_depth,
        future_latent_heads=future_latent_heads,
        future_latent_kv_heads=future_latent_kv_heads,
        future_latent_modulation_rank=future_latent_modulation_rank,
    )
    if name == "small":
        return MainlineRDT2FMConfig(
            **common,
            hidden_size=256,
            depth=6,
            num_heads=4,
            num_kv_heads=2,
            history_hidden_size=64,
            fast_exit_layer=2,
            prefix_exit_layer=4,
            visual_start_layer=2,
            modulation_rank=64,
        )
    if name == "medium":
        return MainlineRDT2FMConfig(
            **common,
            hidden_size=512,
            depth=8,
            num_heads=8,
            num_kv_heads=4,
            history_hidden_size=128,
            fast_exit_layer=2,
            prefix_exit_layer=4,
            visual_start_layer=2,
            modulation_rank=128,
        )
    return MainlineRDT2FMConfig(
        **common,
        hidden_size=1024,
        depth=14,
        num_heads=8,
        num_kv_heads=4,
        history_hidden_size=256,
        fast_exit_layer=4,
        prefix_exit_layer=8,
        visual_start_layer=4,
        modulation_rank=256,
    )


def main() -> None:
    args = parse_args()
    cfg = config_for(
        args.model_size,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        dense_token_dim=args.dense_token_dim,
        visual_corrector=args.visual_corrector,
        future_latent_variant=args.future_latent_variant,
        future_latent_grid_size=args.future_latent_grid_size,
        future_latent_offsets=tuple(args.future_latent_offsets),
        future_latent_hidden_size=args.future_latent_hidden_size,
        future_latent_depth=args.future_latent_depth,
        future_latent_heads=args.future_latent_heads,
        future_latent_kv_heads=args.future_latent_kv_heads,
        future_latent_modulation_rank=args.future_latent_modulation_rank,
    )
    model = MainlineRDT2FM(cfg)
    mainline = model.parameter_count()
    reference = estimate_rdt2_fm_parameter_count(
        RDT2FMReferenceConfig(
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            lang_adaptor="mlp2x_silu",
            lang_token_dim=args.dense_token_dim,
        )
    )
    world_parameters = sum(parameter.numel() for parameter in model.future_latent_parameters())
    report = {
        "model": "MainlineRDT2FM",
        "model_size": args.model_size,
        "parameter_count": mainline,
        "reference_parameter_count_same_local_interface": reference,
        "parameter_reduction": 1.0 - mainline / reference,
        "visual_corrector_mode": args.visual_corrector,
        "future_latent_variant": args.future_latent_variant,
        "future_world_parameter_count": world_parameters,
        "future_world_policy_shared_parameters": 0,
        "future_latent_total_tokens": (
            len(args.future_latent_offsets)
            * cfg.future_latent_num_cameras
            * args.future_latent_grid_size**2
            if args.future_latent_variant != "none"
            else 0
        ),
        "config": cfg.__dict__,
        "network_changes": [
            "history-only learned trajectory prior for the policy",
            "prior-relative arm residual flow with bounded continuous gripper",
            "isolated future-DINO residual world model with dedicated full-action encoder",
            "dedicated world-model timestep, task embedding, modulation, and Transformer blocks",
            "per-time/per-camera/per-channel residual normalization",
            "motion-weighted residual flow and explicit corrupted-action dependency objective",
            "optional conservative demonstration-relative consequence consistency through a parameter-detached world model",
            "no future token or world-model hidden state enters policy inference",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
