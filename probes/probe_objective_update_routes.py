"""Matched ordinary backward passes; never update the admitted live model.

Only the scalar objective configuration changes after normal checkpoint
admission. Each pass restores all global and engine RNGs and model buffers.
The ordinary production backward/clipping runs before CPU AdamW proposals are
measured. Differences are conditional interventions, not additive loss shares.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import hashlib
import json
import random
import sys

import numpy as np
import torch
from clearvla.mainline import train
from clearvla.mainline.training.engine import MainlineTrainingEngine


class PassComplete(Exception):
    pass


class ProbeComplete(Exception):
    pass


def norm2(value):
    return float(value.double().square().sum())


def pair_stats(values, reference):
    a2 = b2 = d2 = dot = 0.0
    for a, b in zip(values, reference, strict=True):
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        a2 += norm2(a)
        b2 += norm2(b)
        d2 += norm2(a - b)
        dot += float((a.double() * b.double()).sum())
    return {"l2": a2 ** .5, "reference_l2": b2 ** .5,
            "difference_l2": d2 ** .5,
            "relative_difference": (d2 / max(b2, 1e-300)) ** .5,
            "cosine": dot / max((a2 * b2) ** .5, 1e-300)}


def proposed_update(group, previous, source, gradients):
    bases = [p.detach().cpu().clone() for p in group["params"]]
    copies = [torch.nn.Parameter(x.clone()) for x in bases]
    opt = torch.optim.AdamW(copies, lr=group["lr"], betas=group["betas"],
                           eps=group["eps"], weight_decay=group["weight_decay"],
                           amsgrad=group["amsgrad"])
    for p, g, old_id in zip(copies, gradients, previous["params"], strict=True):
        p.grad = g
        old = source["optimizer"]["state"].get(old_id)
        if old is not None:
            if old["exp_avg"].shape != p.shape or old["exp_avg_sq"].shape != p.shape:
                raise ValueError("source optimizer moment shape differs")
            opt.state[p] = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                            for key, value in old.items()}
    opt.step()
    return [p.detach() - base for p, base in zip(copies, bases, strict=True)]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--drift-checkpoint", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    source = torch.load(args.source_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    drift = torch.load(args.drift_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    original_train_step = MainlineTrainingEngine.train_step

    def inspect(engine, batch, **unused):
        if engine.optimizer.state or engine.global_step != source["global_step"]:
            raise ValueError("requires admitted source clock and fresh LIVE optimizer; CPU proposals retain original moments")
        config = engine.config
        actual_names = {id(p): n for n, p in engine.model.named_parameters()}
        uniform = {"calvin_frame_weight_mode": "uniform", "calvin_frame_motion_gain": 0.,
                   "calvin_frame_event_gain": 0., "calvin_frame_event_radius": 0,
                   "calvin_frame_max_weight": 1.}
        variants = [("formal", {}), ("formal_repeat_early", {}), ("uniform_rows", uniform),
                    ("no_transition", {"gripper_command_transition": 0.}),
                    ("no_endpoint", {}),
                    ("legacy_objectives", dict(uniform, gripper_command_transition=0.)),
                    ("formal_repeat", {})]
        engine.model.train()
        engine.model.set_training_step(engine.global_step)
        buffers = {n: b.detach().clone() for n, b in engine.model.named_buffers()}
        versions = {n: p._version for n, p in engine.model.named_parameters()}
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        generators = {n: getattr(engine, n).get_state().clone() for n in
                      ("train_flow_generator", "train_condition_generator") if getattr(engine, n) is not None}
        original_forward = engine._forward
        original_lifecycle = engine._gradient_lifecycle
        original_step = engine.optimizer.step
        previous_groups = source["optimizer"]["param_groups"]
        for current, previous in zip(engine.optimizer.param_groups, previous_groups, strict=True):
            if current["name"] != previous["name"] or tuple(current["parameter_names"]) != tuple(previous["parameter_names"]):
                raise ValueError("optimizer named parameter inventory differs")
            for field in ("betas", "eps", "weight_decay", "amsgrad"):
                if current[field] != previous[field]:
                    raise ValueError("optimizer hyperparameter differs: " + field)
        report = {"global_step": engine.global_step, "batch_size": batch.online.batch,
                  "sample_ids": batch.audit.sample_index.detach().cpu().tolist(),
                  "drift_global_step": drift["global_step"],
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "scope": "Same ordinary bs8 batch and RNG; objective-only no-update interventions. CPU AdamW proposals use original moments. Not a closed-loop result or independent loss attribution.",
                  "variants": []}
        references = {}
        current_result = {}

        def capture_forward(*a, **kw):
            ledger, metrics = original_forward(*a, **kw)
            # Keep a fully valid admitted model/config. This diagnostic removes
            # one already-computed scalar after the ordinary forward; disabling
            # its selected architecture in production is a different contract.
            if current_result["name"] in {"no_endpoint", "legacy_objectives"}:
                omitted = ledger.contributions["annotated_goal"]
                groups = dict(ledger.groups)
                groups["representation"] = groups["representation"] - omitted
                contributions = dict(ledger.contributions)
                contributions["annotated_goal"] = torch.zeros_like(omitted)
                ledger = replace(ledger, total=sum(groups.values()), groups=groups,
                                 contributions=contributions)
                ledger.validate()
                current_result["diagnostic_omitted_annotated_goal"] = float(omitted.detach())
            current_result["loss_total"] = float(ledger.total.detach())
            current_result["loss_contributions"] = {k: float(v.detach()) for k, v in ledger.contributions.items()}
            current_result["loss_groups"] = {k: float(v.detach()) for k, v in ledger.groups.items()}
            return ledger, metrics

        def capture_lifecycle(*a, **kw):
            current_result["pre_clip"] = {}
            for group in engine.optimizer.param_groups:
                current_result["pre_clip"][group["name"]] = sum(
                    float(p.grad.detach().float().square().sum()) for p in group["params"] if p.grad is not None) ** .5
            result = original_lifecycle(*a, **kw)
            current_result["global_gradient_norm"] = float(result[2])
            return result

        def intercept(*a, **kw):
            current_result["groups"] = []
            for group, previous in zip(engine.optimizer.param_groups, previous_groups, strict=True):
                name = group["name"]
                grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in group["params"]]
                update = proposed_update(group, previous, source, grads)
                observed = [drift["model"][n].float() - source["model"][n].float() for n in (actual_names[id(p)] for p in group["params"])]
                if current_result["name"] == "formal":
                    references[name] = (grads, update)
                ref_grad, ref_update = references[name]
                current_result["groups"].append({"name": name, "lr": group["lr"],
                    "gradient_tensors": sum(g is not None for g in grads),
                    "post_clip_gradient_vs_formal": pair_stats(grads, ref_grad),
                    "continued_adam_update_vs_formal": pair_stats(update, ref_update),
                    "continued_adam_update_vs_observed_drift": pair_stats(update, observed)})
            if engine.optimizer.state or engine.global_step != source["global_step"]:
                raise RuntimeError("live optimizer or clock changed")
            if any(p._version != versions[n] for n, p in engine.model.named_parameters()):
                raise RuntimeError("live model parameters changed")
            raise PassComplete()

        engine._forward = capture_forward
        engine._gradient_lifecycle = capture_lifecycle
        engine.optimizer.step = intercept
        try:
            for name, overrides in variants:
                engine.config = replace(config, objectives=replace(config.objectives, **overrides))
                engine.config.validate()
                random.setstate(random_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                torch.cuda.set_rng_state_all(cuda_states)
                for n, state in generators.items():
                    getattr(engine, n).set_state(state)
                with torch.no_grad():
                    for n, b in engine.model.named_buffers():
                        b.copy_(buffers[n])
                current_result = {"name": name, "objective_overrides": overrides}
                try:
                    original_train_step(engine, batch, collect_diagnostics=False)
                    raise RuntimeError("optimizer interception was bypassed")
                except PassComplete:
                    pass
                report["variants"].append(current_result)
                args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("VARIANT", name, "loss", current_result["loss_total"], "gradient", current_result["global_gradient_norm"], flush=True)
                engine.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            report["parameters_unchanged"] = True
            report["optimizer_unchanged"] = True
            report["state"] = "complete"
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        finally:
            engine.config = config
            engine._forward = original_forward
            engine._gradient_lifecycle = original_lifecycle
            engine.optimizer.step = original_step
        raise ProbeComplete()

    MainlineTrainingEngine.train_step = inspect
    sys.argv = [sys.argv[0], *remaining]
    try:
        train.main()
    except ProbeComplete:
        print("COMPLETE", args.report, flush=True)
    finally:
        MainlineTrainingEngine.train_step = original_train_step


if __name__ == "__main__":
    main()
