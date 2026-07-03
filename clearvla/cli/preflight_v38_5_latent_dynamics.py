from __future__ import annotations

"""Preflight structural checks for V38.6.2 action-centered controlled-residual latent-dynamics policy.

This is not a quality benchmark.  It is a fast structural gate that fails before
long training if the rollout objective can bypass the intended path:

    current visual/state/history + action-centered coefficient -> delta -> future/effect

The checks are intentionally synthetic so they do not depend on HDF5/DINO cache
availability.  They verify graph connectivity, no target-future input leakage,
rollout tail binding, and that a tiny synthetic action-conditioned future task
can create counterfactual separation in tens of optimizer steps.
"""

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clearvla.experiments.observed_state_lab.policy_v38 import V38PolicyConfig, V38PolicySystem, TemporalWorldActionDiT


def _small_config() -> V38PolicyConfig:
    return V38PolicyConfig(
        hidden_size=16,
        num_heads=4,
        depth=1,
        action_decoder_depth=1,
        proposal_depth=1,
        dropout=0.0,
        proposal_dropout=0.0,
        action_horizon=4,
        first_execution_steps=1,
        mid_execution_steps=2,
        executed_history_length=3,
        visual_token_dim=32,
        visual_history_length=3,
        num_cameras=2,
        patches_per_camera=4,
        future_anchors=2,
        target_future_count=2,
        future_grid_size=1,
        canvas_registers=2,
        action_basis_tokens=1,
        rollout_tail_start_step=2,
        rollout_tail_full_step=3,
    )


def _random_batch(cfg: V38PolicyConfig, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    b = batch_size
    return {
        "visual": torch.randn(
            b,
            cfg.visual_history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.visual_token_dim,
            device=device,
        ),
        "target_visual": torch.randn(
            b,
            cfg.target_future_count,
            cfg.visual_history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.visual_token_dim,
            device=device,
        ),
        "state_history": torch.randn(b, cfg.executed_history_length, cfg.state_dim, device=device),
        "executed_history": torch.randn(b, cfg.executed_history_length, cfg.action_dim, device=device),
        "state": torch.randn(b, cfg.state_dim, device=device),
        "target_action": torch.randn(b, cfg.action_horizon, cfg.action_dim, device=device),
    }


def _action_conditioned_target_visual(
    model: V38PolicySystem,
    cfg: V38PolicyConfig,
    visual: torch.Tensor,
    target_action: torch.Tensor,
    *,
    scale: float = 0.4,
) -> torch.Tensor:
    """Construct future visual tokens whose residual target is action-dependent.

    The V38.6 target head is frozen linear projection W: D -> H.  We generate a
    desired hidden target y(action) and use residual_d = y @ W so that, for
    orthonormal rows, target_proj(residual_d) approximates y.  This creates a
    tiny synthetic task where real actions should beat hold/shuffled actions.
    """
    with torch.no_grad():
        w = model.planner.rollout_codec.target_proj.weight.detach()  # [H,D]
        action_to_hidden = torch.randn(cfg.action_dim, cfg.hidden_size, device=visual.device) * scale
        action_summary = target_action.mean(dim=1)
        base = action_summary @ action_to_hidden  # [B,H]
        desired = []
        for k in range(cfg.future_anchors):
            by_camera = []
            for c in range(cfg.num_cameras):
                by_camera.append(base + 0.1 * float(k + 1) + 0.05 * float(c))
            desired.append(torch.stack(by_camera, dim=1))
        desired_h = torch.stack(desired, dim=1)  # [B,K,C,H], G=1 in preflight config.
        residual_d = desired_h @ w  # [B,K,C,D]
        target_visual = torch.zeros(
            visual.shape[0],
            cfg.target_future_count,
            cfg.visual_history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.visual_token_dim,
            device=visual.device,
            dtype=visual.dtype,
        )
        for k in range(cfg.target_future_count):
            kk = min(k, cfg.future_anchors - 1)
            for h in range(cfg.visual_history_length):
                target_visual[:, k, h] = visual[:, -1]
                target_visual[:, k, h] = target_visual[:, k, h] + residual_d[:, kk, :, None, :]
        return target_visual


def _forward(
    model: V38PolicySystem,
    batch: dict[str, torch.Tensor],
    *,
    make_counterfactuals: bool = True,
) -> dict[str, torch.Tensor]:
    return model.flow_training_forward(
        batch["visual"],
        batch["state_history"],
        batch["executed_history"],
        batch["state"],
        batch["target_action"],
        target_visual=batch["target_visual"],
        make_counterfactuals=make_counterfactuals,
        proposal_dropout=0.0,
    )


def _residual_target(out: dict[str, torch.Tensor]) -> torch.Tensor:
    # Match the real V38.6 training objective: counterfactual separation is
    # tested on the action-controlled delta after removing the visual base.
    target = out["rollout_effect_target"].detach()
    base = out.get("rollout_base_effect_pred")
    if base is None:
        return target
    return target - base.detach()


def _dyn_losses(out: dict[str, torch.Tensor], margin: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = _residual_target(out)
    real = F.mse_loss(out["rollout_delta_pred"], target)
    hold = F.mse_loss(out["rollout_delta_pred_hold_action"], target)
    shuffle = F.mse_loss(out["rollout_delta_pred_shuffle_action"], target)
    contrast = F.relu(torch.as_tensor(margin, device=real.device, dtype=real.dtype) + real - hold) + F.relu(
        torch.as_tensor(margin, device=real.device, dtype=real.dtype) + real - shuffle
    )
    return real + contrast, {"real": real, "hold": hold, "shuffle": shuffle, "contrast": contrast}


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    cfg = _small_config()
    model = V38PolicySystem(cfg).to(device)
    model.train()

    report: dict[str, Any] = {
        "schema": "clearvla-v38-6-2-action-centered-residual-preflight-v1",
        "seed": args.seed,
        "device": str(device),
        "config": {
            "hidden_size": cfg.hidden_size,
            "depth": cfg.depth,
            "action_horizon": cfg.action_horizon,
            "future_anchors": cfg.future_anchors,
            "future_token_count": cfg.future_token_count,
            "patches_per_camera": cfg.patches_per_camera,
        },
        "checks": {},
    }

    # Static API contract: target future is not accepted by the internal policy forward.
    planner_sig = inspect.signature(TemporalWorldActionDiT.forward)
    system_sig = inspect.signature(V38PolicySystem._policy_forward)
    forbidden_names = {"target_visual", "future_noisy_latent", "future_training_pack", "target_future"}
    planner_forbidden = sorted(set(planner_sig.parameters) & forbidden_names)
    system_forbidden = sorted(set(system_sig.parameters) & forbidden_names)
    report["checks"]["static_forward_signature"] = {
        "ok": not planner_forbidden and not system_forbidden,
        "planner_forbidden_params": planner_forbidden,
        "system_forbidden_params": system_forbidden,
        "planner_params": list(planner_sig.parameters),
        "system_params": list(system_sig.parameters),
    }

    # Graph path and no-leak checks.
    batch = _random_batch(cfg, args.batch_size, device)
    batch["visual"].requires_grad_(True)
    batch["target_visual"].requires_grad_(True)
    batch["target_action"].requires_grad_(True)
    out = _forward(model, batch, make_counterfactuals=True)
    dyn_loss = F.mse_loss(out["rollout_delta_pred"], _residual_target(out))
    dyn_loss.backward(retain_graph=True)
    action_grad_mean = float(batch["target_action"].grad.abs().mean().detach().cpu())
    action_grad_max = float(batch["target_action"].grad.abs().max().detach().cpu())
    visual_grad_mean = float(batch["visual"].grad.abs().mean().detach().cpu())
    visual_grad_max = float(batch["visual"].grad.abs().max().detach().cpu())
    target_visual_grad = batch["target_visual"].grad
    target_visual_grad_max = 0.0 if target_visual_grad is None else float(target_visual_grad.abs().max().detach().cpu())
    real_hold_diff = float(F.mse_loss(out["rollout_delta_pred"], out["rollout_delta_pred_hold_action"]).detach().cpu())
    real_shuffle_diff = float(F.mse_loss(out["rollout_delta_pred"], out["rollout_delta_pred_shuffle_action"]).detach().cpu())
    report["checks"]["graph_dependency"] = {
        "ok": action_grad_max > args.min_action_grad and visual_grad_max > args.min_visual_grad,
        "rollout_loss": float(dyn_loss.detach().cpu()),
        "action_grad_mean": action_grad_mean,
        "action_grad_max": action_grad_max,
        "visual_grad_mean": visual_grad_mean,
        "visual_grad_max": visual_grad_max,
        "real_hold_output_mse": real_hold_diff,
        "real_shuffle_output_mse": real_shuffle_diff,
    }
    report["checks"]["target_future_no_input_leak"] = {
        "ok": target_visual_grad_max == 0.0,
        "target_visual_grad_max": target_visual_grad_max,
        "future_conditioned_action_loss": float(out["future_conditioned_action_loss"].detach().cpu()),
    }
    alpha = out["rollout_alpha"][0, :, 0].detach().cpu().tolist()
    report["checks"]["tail_binding_alpha"] = {
        "ok": alpha[0] == 0.0 and alpha[-1] == 1.0 and all(alpha[i] <= alpha[i + 1] for i in range(len(alpha) - 1)),
        "alpha": alpha,
    }

    # Tiny synthetic overfit: not a real benchmark, just verifies that the
    # architecture and contrast loss can create action-conditioned rollout
    # separation before spending full training time.
    torch.manual_seed(args.seed + 17)
    model = V38PolicySystem(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.micro_lr, weight_decay=0.0)
    micro = _random_batch(cfg, args.batch_size, device)
    micro["target_visual"] = _action_conditioned_target_visual(model, cfg, micro["visual"], micro["target_action"])
    history = []
    for step in range(args.micro_steps + 1):
        out = _forward(model, micro, make_counterfactuals=True)
        loss, rows = _dyn_losses(out, args.margin)
        if step in {0, args.micro_steps // 2, args.micro_steps}:
            history.append(
                {
                    "step": step,
                    "real": float(rows["real"].detach().cpu()),
                    "hold": float(rows["hold"].detach().cpu()),
                    "shuffle": float(rows["shuffle"].detach().cpu()),
                    "delta_hold": float((rows["hold"] - rows["real"]).detach().cpu()),
                    "delta_shuffle": float((rows["shuffle"] - rows["real"]).detach().cpu()),
                    "contrast": float(rows["contrast"].detach().cpu()),
                }
            )
        if step == args.micro_steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = history[-1]
    report["checks"]["synthetic_counterfactual_microfit"] = {
        "ok": final["delta_hold"] > args.min_micro_delta and final["delta_shuffle"] > args.min_micro_delta,
        "steps": args.micro_steps,
        "lr": args.micro_lr,
        "margin": args.margin,
        "history": history,
    }

    ok = all(bool(row.get("ok")) for row in report["checks"].values())
    report["ok"] = ok
    if not all(_finite(v) for check in report["checks"].values() for v in _flatten_floats(check)):
        report["ok"] = False
        report["nonfinite"] = True
    return report


def _flatten_floats(x: Any) -> list[float]:
    vals: list[float] = []
    if isinstance(x, float):
        vals.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            vals.extend(_flatten_floats(v))
    elif isinstance(x, (list, tuple)):
        for v in x:
            vals.extend(_flatten_floats(v))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="V38.6 controlled residual latent dynamics preflight structural checks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--micro-steps", type=int, default=100)
    parser.add_argument("--micro-lr", type=float, default=3e-3)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--min-action-grad", type=float, default=1e-8)
    parser.add_argument("--min-visual-grad", type=float, default=1e-8)
    parser.add_argument("--min-micro-delta", type=float, default=1e-2)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    report = run_preflight(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
