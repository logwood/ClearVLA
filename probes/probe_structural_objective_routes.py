"""Ordinary backward of existing weighted objectives on one admitted val batch.

The ordinary training forward/backward is used with restored randomness and
buffers for each existing scalar. No optimizer or scheduler update is admitted.
This is a connectivity/direction diagnostic, not training or a convergence test.
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


def summarize(parameters, reference=None):
    norm = weight = dot = previous_norm = difference = 0.
    connected = nonzero = count = 0
    for name, parameter in parameters:
        count += parameter.numel()
        weight += float(parameter.detach().float().square().sum())
        grad = parameter.grad
        if grad is not None:
            connected += 1
            grad = grad.detach().float()
            square = float(grad.square().sum())
            norm += square
            nonzero += square > 0
        if reference is not None:
            previous = reference.get(name)
            if previous is not None:
                previous = previous.to(parameter.device)
                previous_norm += float(previous.square().sum())
                if grad is not None:
                    dot += float((grad * previous).sum())
                    difference += float((grad - previous).square().sum())
                else:
                    difference += float(previous.square().sum())
            elif grad is not None:
                difference += float(grad.square().sum())
    result = {"parameter_tensors": len(parameters), "parameter_elements": count,
              "connected_tensors": connected, "nonzero_gradient_tensors": nonzero,
              "raw_gradient_l2": norm ** .5, "parameter_l2": weight ** .5}
    if reference is not None:
        result.update(cosine_to_total=dot / max((norm * previous_norm) ** .5, 1e-30),
                      relative_to_total_norm=(norm / max(previous_norm, 1e-30)) ** .5,
                      relative_difference_to_total=(difference / max(previous_norm, 1e-30)) ** .5)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.report.exists() or "--validate-checkpoint" not in remaining:
        raise ValueError("requires fresh report and normal read-only checkpoint admission")
    old_eval, old_train = MainlineTrainingEngine.eval_step, MainlineTrainingEngine.train_step

    def inspect(engine, batch, **unused):
        if engine.optimizer.state:
            raise ValueError("read-only validation must not own restored live optimizer moments")
        engine.model.train()
        engine.model.set_training_step(engine.global_step)
        step = engine.global_step
        schedule = engine.schedule.step_index
        versions = {n: p._version for n, p in engine.model.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in engine.model.named_buffers()}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        generators = {n: getattr(engine, n).get_state().clone() for n in ["train_flow_generator", "train_condition_generator"] if getattr(engine, n) is not None}
        parameters = list(engine.model.named_parameters())
        prefixes = {
            "G_grounder": ("grounding.grounder.",),
            "S_language": ("intent.organizer.goal_input.", "intent.organizer.goal_read.", "intent.organizer.goal_self."),
            "S_binding": ("intent.organizer.shared_binder.",),
            "S_correspondence": tuple("intent.organizer.instruction_progress." + s for s in ["query.", "key.", "query_position.", "key_position.", "null_key"]),
            "S_change_content_image": tuple("intent.organizer.instruction_progress.values." + s for s in ["content.", "image.", "joint_"]),
            "S_change_robot": ("intent.organizer.instruction_progress.values.robot.",),
            "S_change_status": ("intent.organizer.instruction_progress.values.status.",),
            "S_change_output": ("intent.organizer.instruction_progress.output.",),
            "S_coarse": ("intent.coarse_action.",),
            "W_dynamics": ("world.dynamics.",),
            "P3_world_feedback": ("policy_compiler.plan_compiler.world_feedback_read.",),
            "robot_response_predictor": ("policy_compiler.plan_compiler.robot_observer.response.",),
            "P3_robot_innovation_read": tuple("policy_compiler.plan_compiler.robot_observer." + s for s in ["error_value.", "plan_query.", "error_output."]),
        }
        owners = {key: [(n, p) for n, p in parameters if n.startswith(start)] for key, start in prefixes.items()}
        if any(not p for p in owners.values()):
            raise ValueError("owner inventory missing: " + str([k for k, p in owners.items() if not p]))
        report = {"global_step": step, "batch_size": batch.online.batch,
                  "sample_ids": batch.audit.sample_index.cpu().tolist(),
                  "episode_ids": batch.audit.episode_index.cpu().tolist(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "scope": __doc__, "owners": {k: [n for n, p in v] for k, v in owners.items()}, "variants": []}
        original_forward, original_lifecycle, original_step = engine._forward, engine._gradient_lifecycle, engine.optimizer.step
        current, reference = {}, {}
        objectives = ["total", "action_flow", "gripper_command", "object_reconstruction", "intent_online",
                      "coarse_action", "future_dynamics", "future_transition", "flow_geometry", "annotated_goal",
                      "robot_response", "total_repeat"]

        def forward(*a, **kw):
            ledger, metrics = original_forward(*a, **kw)
            current["original_contributions"] = {k: float(v.detach()) for k, v in ledger.contributions.items()}
            name = current["name"]
            if name not in ["total", "total_repeat"]:
                selected = [k for k in ledger.contributions if k.startswith("flow_")] if name == "flow_geometry" else [name]
                missing = set(selected) - ledger.contributions.keys()
                if missing:
                    raise ValueError("missing selected objective: " + str(missing))
                total = sum(ledger.contributions[k] for k in selected)
                current["selected_contributions"] = selected
                current["selected_requires_grad"] = total.requires_grad
                if not total.requires_grad:
                    current.update(loss=float(total), skip_reason="selected scalar has no differentiable path")
                    raise PassComplete()
                zero = torch.zeros_like(total)
                ledger = replace(ledger, total=total,
                    groups={"action": total, "representation": zero, "execution": zero},
                    contributions={k: v if k in selected else torch.zeros_like(v) for k, v in ledger.contributions.items()},
                    # Other objectives are already measured above. Keeping
                    # their unused graphs alive through a partial backward
                    # inflates memory relative to ordinary total.backward().
                    terms={})
                ledger.validate()
            current["loss"] = float(ledger.total.detach())
            return ledger, {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        def lifecycle(*a, **kw):
            current["owners"] = {key: summarize(values, reference.get(key)) for key, values in owners.items()}
            if current["name"] == "total":
                for key, values in owners.items():
                    reference[key] = {n: None if p.grad is None else p.grad.detach().float().cpu().clone() for n, p in values}
            return original_lifecycle(*a, **kw)

        def no_step(*a, **kw):
            if engine.optimizer.state or engine.global_step != step or engine.schedule.step_index != schedule:
                raise RuntimeError("live optimizer/clock changed")
            if versions != {n: p._version for n, p in engine.model.named_parameters()}:
                raise RuntimeError("live model parameters changed")
            raise PassComplete()

        engine._forward, engine._gradient_lifecycle, engine.optimizer.step = forward, lifecycle, no_step
        try:
            for name in objectives:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
                for n, state in generators.items():
                    getattr(engine, n).set_state(state)
                with torch.no_grad():
                    for n, b in engine.model.named_buffers():
                        b.copy_(buffers[n])
                current = {"name": name}
                current["memory_before_forward_bytes"] = torch.cuda.memory_allocated()
                try:
                    with torch.enable_grad():
                        old_train(engine, batch, collect_diagnostics=False)
                    raise RuntimeError("optimizer interception was bypassed")
                except PassComplete:
                    pass
                report["variants"].append(current)
                args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("OBJECTIVE_ROUTE", name, current.get("loss"), current.get("skip_reason", "backward complete"), flush=True)
                engine.optimizer.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
            report.update(state="complete", parameters_unchanged=True, optimizer_unchanged=True, clock_unchanged=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        finally:
            engine._forward, engine._gradient_lifecycle, engine.optimizer.step = original_forward, original_lifecycle, original_step
        raise ProbeComplete()

    MainlineTrainingEngine.eval_step = inspect
    sys.argv = [sys.argv[0], *remaining]
    try:
        train.main()
    except ProbeComplete:
        print("COMPLETE", args.report, flush=True)
    finally:
        MainlineTrainingEngine.eval_step = old_eval


if __name__ == "__main__":
    main()
