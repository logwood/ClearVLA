"""Compare exact CPU AdamW updates from one identical clipped production gradient.

The real training entry loads the source model strictly and builds one ordinary
batch8 forward/backward. Its optimizer step is intercepted before any live
parameter or moment update. Two CPU copies receive the same gradient and LR:
fresh moments versus the original named-group checkpoint moments. Outputs are
diagnostics, never a saved hybrid or a trained checkpoint.
"""
from pathlib import Path
import argparse
import json
import hashlib
import sys

import torch
from clearvla.mainline import train
from clearvla.mainline.training.engine import MainlineTrainingEngine


class ProbeComplete(Exception):
    pass


def norm2(x):
    return float(x.double().square().sum())


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--drift-checkpoint", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    original_train_step = MainlineTrainingEngine.train_step
    source = torch.load(args.source_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    drift = torch.load(args.drift_checkpoint, map_location="cpu", mmap=True, weights_only=False)

    def inspect(engine, batch, **kwargs):
        if engine.optimizer.state:
            raise ValueError("expected original fresh optimizer initialization")
        if engine.global_step != source["global_step"]:
            raise ValueError("training clock differs from source checkpoint")
        model_names = {id(parameter): name for name, parameter in engine.model.named_parameters()}
        versions = {name: value._version for name, value in engine.model.named_parameters()}
        original_step = engine.optimizer.step

        def intercept(*unused, **unused_kwargs):
            report = {
                "global_step": engine.global_step,
                "batch_size": batch.online.batch,
                "sample_ids": batch.audit.sample_index.detach().cpu().tolist(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Same clipped production gradient and current LR; only CPU AdamW moment state differs. No live optimizer step, checkpoint, or runtime state is written.",
                "groups": [],
            }
            old_groups = source["optimizer"]["param_groups"]
            if len(old_groups) != len(engine.optimizer.param_groups):
                raise ValueError("optimizer group inventory differs")
            for current, previous in zip(engine.optimizer.param_groups, old_groups, strict=True):
                if current["name"] != previous["name"] or tuple(current["parameter_names"]) != tuple(previous["parameter_names"]):
                    raise ValueError("optimizer parameter names/order differ")
                for field in ("betas", "eps", "weight_decay", "amsgrad"):
                    if current[field] != previous[field]:
                        raise ValueError("optimizer hyperparameter differs: " + field)
                names = [model_names[id(p)] for p in current["params"]]
                bases = [p.detach().cpu().clone() for p in current["params"]]
                gradients = [None if p.grad is None else p.grad.detach().cpu().clone() for p in current["params"]]
                old_ids = previous["params"]
                deltas = {}
                for variant in ("fresh", "continued"):
                    copies = [torch.nn.Parameter(value.clone()) for value in bases]
                    opt = torch.optim.AdamW(
                        copies, lr=current["lr"], betas=current["betas"], eps=current["eps"],
                        weight_decay=current["weight_decay"], amsgrad=current["amsgrad"],
                    )
                    for copy, gradient, source_id in zip(copies, gradients, old_ids, strict=True):
                        copy.grad = gradient
                        state = source["optimizer"]["state"].get(source_id)
                        if variant == "continued" and state is not None:
                            if state["exp_avg"].shape != copy.shape or state["exp_avg_sq"].shape != copy.shape:
                                raise ValueError("moment tensor shape mismatch")
                            opt.state[copy] = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                                               for key, value in state.items()}
                    opt.step()
                    deltas[variant] = [copy.detach() - base for copy, base in zip(copies, bases, strict=True)]
                    del opt, copies
                observed = [drift["model"][name].float() - source["model"][name].float() for name in names]
                totals = {variant: sum(norm2(delta) for delta in values) for variant, values in deltas.items()}
                drift2 = sum(norm2(delta) for delta in observed)
                dots = {variant: sum(float((delta.double() * target.double()).sum())
                                      for delta, target in zip(values, observed, strict=True))
                        for variant, values in deltas.items()}
                interdot = sum(float((a.double() * b.double()).sum()) for a, b in zip(deltas["fresh"], deltas["continued"], strict=True))
                item = {
                    "name": current["name"], "lr": current["lr"],
                    "parameter_tensors": len(names), "gradient_tensors": sum(g is not None for g in gradients),
                    "actual_512_update_l2": drift2 ** 0.5,
                    "proposed_update_l2": {key: value ** 0.5 for key, value in totals.items()},
                    "fresh_over_continued_l2": (totals["fresh"] / max(totals["continued"], 1e-300)) ** 0.5,
                    "cosine_with_actual_512_update": {key: dots[key] / max((totals[key] * drift2) ** 0.5, 1e-300) for key in totals},
                    "fresh_continued_cosine": interdot / max((totals["fresh"] * totals["continued"]) ** 0.5, 1e-300),
                }
                report["groups"].append(item)
                print(current["name"], item["proposed_update_l2"], "ratio", item["fresh_over_continued_l2"], flush=True)
            if engine.optimizer.state or engine.global_step != source["global_step"]:
                raise RuntimeError("live optimizer state changed")
            if any(p._version != versions[name] for name, p in engine.model.named_parameters()):
                raise RuntimeError("live model was updated")
            report["parameters_unchanged"] = True
            report["optimizer_unchanged"] = True
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            raise ProbeComplete()

        engine.optimizer.step = intercept
        try:
            return original_train_step(engine, batch, collect_diagnostics=False)
        finally:
            engine.optimizer.step = original_step

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
