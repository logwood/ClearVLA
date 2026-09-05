"""Bounded real Pen CUDA/BF16 hybrid training and rollout-owner preflight."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from clearvla.mainline.checkpoint import build_checkpoint_identity
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import load_checkpoint_for_validation, save_checkpoint
from clearvla.mainline.runtime.hybrid import differentiable_hybrid_rollout, role_contract
from clearvla.mainline.runtime.identity import dataset_identity, language_identity
from clearvla.mainline.training.engine import MainlineTrainingEngine, validate_finite_training_batch
from clearvla.mainline.training.hybrid import hybrid_rollout_terms
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def emit(path, value):
    line = json.dumps(value, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.hybrid.enabled or config.optimizer.batch_size != 8:
        raise ValueError("this gate requires the explicit hybrid Pen B8 config")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this gate requires exactly one selected CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = args.output_dir / "gate.jsonl"
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)
    bundle = load_mainline_data(config)
    identity = build_checkpoint_identity(
        config,
        repo_root=Path.cwd(),
        dataset=dataset_identity(bundle, config),
        language=language_identity(bundle, config),
    )
    model = ClearVLAMainlinePolicy(config).to(device)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        total_steps=max(config.optimizer.warmup_steps + 1, 1000),
        minimum_ratio=0.1,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        dtype=torch.bfloat16,
        train_flow_generator=torch.Generator(device=device).manual_seed(32001),
        train_condition_generator=torch.Generator(device=device).manual_seed(32002),
    )
    loader = bundle.loader("val", batch_size=8, workers=0, device=device, shuffle=False)
    batch = to_training_batch(next(iter(loader)), goal=bundle.goal, config=config, device=device)
    validate_finite_training_batch(batch)
    emit(
        log,
        {
            "kind": "start",
            "identity": identity.as_dict(),
            "hybrid_role_boundary": dict(role_contract(model).identity),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "batch_size": 8,
            "dtype": "bf16",
            "initialization": "random_no_checkpoint",
            "training_steps": 2,
        },
    )
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        started = time.perf_counter()
        result = engine.train_step(batch, collect_diagnostics=True)
        torch.cuda.synchronize()
        metrics = result.materialize()
        if not all(
            abs(metrics[name]) < 1e-5 for name in ("loss_ledger_gap", "loss_contribution_gap")
        ):
            raise RuntimeError("formal loss ledger did not close")
        if not metrics["loss_contrib_hybrid_rollout_decoded_action"] > 0:
            raise RuntimeError("hybrid rollout has no formal action contribution")
        if metrics["hybrid_solver_total_dynamic_calls"] != 17:
            raise RuntimeError("hybrid rollout forward count differs")
        bad = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        if bad:
            raise RuntimeError("nonfinite parameter gradients: " + str(bad[:8]))
        emit(
            log,
            {
                "kind": "train_step",
                "step": step + 1,
                "seconds": time.perf_counter() - started,
                "metrics": metrics,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            },
        )
        if torch.cuda.max_memory_reserved() > 22 * 2**30:
            raise RuntimeError("hybrid B8 exceeds the 22 GiB memory budget")
    optimizer.zero_grad(set_to_none=True)
    # The ordinary Teacher loss has now opened W's zero-initialized output
    # matrices. This probe isolates rollout loss from the pointwise anchor.
    model.eval()
    model.set_training_step(1200)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        cache, training_state, _ = model.encode_online(batch.online, geometry_supervision=False)
        noise = model.outlet_adapter.sample_noise(
            8,
            device=device,
            dtype=torch.float32,
            generator=torch.Generator(device=device).manual_seed(32003),
        )
        rollout = differentiable_hybrid_rollout(model, cache, config, noise, dtype=torch.bfloat16)
        terms = hybrid_rollout_terms(config, rollout, batch.action_target, batch.online.history)
    rollout.proposal.action.retain_grad()
    terms["hybrid_rollout"].backward()
    prefixes = (
        "observation.",
        "grounding.",
        "intent.",
        "world.",
        "p1.",
        "policy_compiler.",
        "transition.",
        "execution_bottom.decoder.spine.coarse_lifts.",
        "execution_bottom.decoder.spine.detail_lifts.",
        "execution_bottom.decoder.terminal_controller.velocity_head.",
    )
    owners = {}
    for prefix in prefixes:
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        ]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError("missing/nonfinite rollout owner " + prefix)
        norm = sum(g.detach().double().square().sum().item() for g in grads) ** 0.5
        owners[prefix] = norm
        if norm == 0:
            raise RuntimeError("zero rollout owner " + prefix)
    proposal_grad = rollout.proposal.action.grad
    if (
        proposal_grad is None
        or not torch.isfinite(proposal_grad).all()
        or proposal_grad.abs().sum() == 0
    ):
        raise RuntimeError("rollout loss did not reach proposal through rebuilt W")
    model.set_training_step(engine.global_step)
    del rollout, terms, training_state, cache
    optimizer.zero_grad(set_to_none=True)
    # The gate checkpoint is proof only; formal training must initialize anew.
    gate_checkpoint = args.output_dir / "gate_checkpoint.pt"
    save_checkpoint(
        gate_checkpoint,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=0,
        global_step=engine.global_step,
        best_metric=None,
    )
    restored = load_checkpoint_for_validation(
        gate_checkpoint, model=model, config=config, identity=identity
    )
    if restored.global_step != 2:
        raise RuntimeError("checkpoint replay changed training position")
    summary = {
        "passed": True,
        "rollout_only_owner_l2": owners,
        "proposal_action_grad_l2": proposal_grad.double().norm().item(),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "checkpoint_roundtrip_step": restored.global_step,
        "source_digest": identity.source.digest,
        "config_digest": identity.config_digest,
        "manifest": identity.manifest,
        "git_commit": identity.git_commit,
    }
    if summary["peak_reserved_gib"] > 22:
        raise RuntimeError("hybrid gate peak exceeds the 22 GiB budget")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(log, {"kind": "summary", **summary})


if __name__ == "__main__":
    main()
