"""Separate arm-field and W-cache exposure at the existing binary command head.

Use the actual checkpoint-validation entry, identical samples and observed-row
support. Extra endpoint calls never integrate velocity, rebuild observation,
change physical time, update weights, or provide oracle labels to deployment.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import sys

import torch
import torch.nn.functional as F
from clearvla.mainline import train
from clearvla.mainline.training import engine as engine_module
from clearvla.mainline.training.engine import MainlineTrainingEngine, _autocast
from clearvla.mainline.endpoint_supervision import clean_endpoint_field, endpoint_condition


class ProbeComplete(Exception):
    pass


def counts(logits, target, valid, event, boundary):
    pred = logits.argmax(-1).bool()
    truth = target.bool()
    predicted_previous = torch.cat((boundary[:, None], pred[:, :-1]), dim=1)
    true_previous = torch.cat((boundary[:, None], truth[:, :-1]), dim=1)
    predicted_event = pred.long() - predicted_previous.long()
    true_event = truth.long() - true_previous.long()
    ce = F.cross_entropy(logits.float().flatten(0, 1), target.long().flatten(), reduction="none").reshape_as(valid)
    result = {}
    for name, mask in [("all", valid), ("first8", valid & (torch.arange(valid.shape[1], device=valid.device)[None] < 8)),
                       ("complete_samples", valid & valid.all(-1, keepdim=True)), ("event", valid & event)]:
        result[name] = {"rows": int(mask.sum()), "correct": int(((pred == truth) & mask).sum()),
                        "positive": int((pred & mask).sum()), "target_positive": int((truth & mask).sum()),
                        "true_positive": int((pred & truth & mask).sum()), "ce_sum": float(ce[mask].sum()),
                        "events_predicted": int(((predicted_event != 0) & mask).sum()),
                        "events_target": int(((true_event != 0) & mask).sum()),
                        "events_matched_type_and_row": int(((predicted_event == true_event) & (true_event != 0) & mask).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-batches", type=int, default=16)
    args, remaining = parser.parse_known_args()
    if "--validate-checkpoint" not in remaining:
        raise ValueError("requires ordinary read-only checkpoint validation")
    if args.report.exists() or args.probe_batches < 1:
        raise ValueError("fresh output and positive probe batch count required")
    original_eval = MainlineTrainingEngine.eval_step
    original_flow = engine_module.sample_flow_matching
    original_sample = train.sample_refined_cached_action_with_cache
    pending = {}
    report = {"scope": "Matched endpoint head evaluation, same observed rows; coarse/refined W cache crossed with clean labelled/generated arm field. No optimization or new closed loop.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "batches": []}

    def flow(*a, **kw):
        result = original_flow(*a, **kw)
        pending["flow"] = result
        return result

    def eval_step(engine, batch, **kw):
        pending["engine"] = engine
        pending["batch"] = batch
        return original_eval(engine, batch, **kw)

    @torch.no_grad()
    def sample(model, cache, config, **kw):
        result, refined = original_sample(model, cache, config, **kw)
        engine, batch = pending["engine"], pending["batch"]
        if model is not engine.model or cache.history is not batch.online.history:
            # Conditioning may produce a new immutable history wrapper, so
            # validate the actual admitted numeric current state as well.
            if model is not engine.model or not torch.equal(cache.history.state, batch.online.history.state):
                raise ValueError("validation sample lost its matched batch")
        if config.data.data_profile != "calvin_relative_7d_v1" or not model.outlet_adapter.is_binary_command:
            raise ValueError("focused diagnostic requires the CALVIN binary outlet")
        versions = {n: p._version for n, p in model.named_parameters()}
        labelled = clean_endpoint_field(replace(pending["flow"], source_physical_noise=result.initial_physical_noise), arm_dim=model.outlet_adapter.arm_dim)
        # Retain the SAME neutralized binary source-noise lanes in both fields.
        generated = torch.cat((result.physical_field[..., :2 * model.outlet_adapter.arm_dim],
                               labelled[..., 2 * model.outlet_adapter.arm_dim:]), dim=-1)
        valid = batch.action_target.row_valid
        if valid is None:
            valid = torch.ones_like(batch.action_target.raw_units[..., -1], dtype=torch.bool)
        truth = batch.action_target.raw_units[..., -1] > 0
        boundary = batch.action_target.gripper_transition_boundary_raw_units[..., -1] > 0
        previous = torch.cat((boundary[:, None], truth[:, :-1]), dim=1)
        event = truth != previous
        item = {"index": len(report["batches"]), "sample_ids": batch.audit.sample_index.cpu().tolist(),
                "episode_ids": batch.audit.episode_index.cpu().tolist(), "global_step": engine.global_step,
                "conditions": {}, "generated_vs_label_arm_field_rms": float(
                    ((generated[..., :2 * model.outlet_adapter.arm_dim] - labelled[..., :2 * model.outlet_adapter.arm_dim]).float().square().mean(-1))[valid].mean().sqrt())}
        outputs = {}
        for cache_name, value_cache in [("coarse", cache), ("refined", refined)]:
            for field_name, field in [("labelled", labelled), ("generated", generated)]:
                time, context = endpoint_condition(field, context_enabled=config.runtime.deployment_flow_schedule is not None)
                with _autocast(engine.device, engine.dtype):
                    output = model.velocity(value_cache, noisy_action_field=field, time=time,
                                            flow_step_context=context, require_execution_supervision=False,
                                            collect_diagnostics=False)
                logits = output.bottom.gripper_command_logits.float()
                name = cache_name + "_" + field_name
                outputs[name] = logits
                item["conditions"][name] = counts(logits, truth, valid, event, boundary)
                item["conditions"][name]["per_sample_first8_open_probability"] = logits.softmax(-1)[:, :8, 1].cpu().tolist()
        item["repeat_vs_original_sample"] = {
            "logit_rms": float((outputs["refined_generated"] - result.gripper_command_logits.float()).square().mean().sqrt()),
            "observed_command_differences": int(((outputs["refined_generated"].argmax(-1) != result.gripper_command_logits.argmax(-1)) & valid).sum())}
        item["paired_command_changes"] = {
            a + "__" + b: int(((outputs[a].argmax(-1) != outputs[b].argmax(-1)) & valid).sum())
            for a, b in [("coarse_labelled", "coarse_generated"), ("refined_labelled", "refined_generated"),
                         ("coarse_labelled", "refined_labelled"), ("coarse_generated", "refined_generated")]}
        if any(p._version != versions[n] for n, p in model.named_parameters()):
            raise RuntimeError("endpoint diagnostic mutated model parameters")
        report["batches"].append(item)
        report["parameters_unchanged"] = True
        report["state"] = "complete" if len(report["batches"]) >= args.probe_batches else "running"
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("ENDPOINT_DISTRIBUTION", item["index"], item["paired_command_changes"], item["repeat_vs_original_sample"], flush=True)
        if report["state"] == "complete":
            raise ProbeComplete()
        return result, refined

    MainlineTrainingEngine.eval_step = eval_step
    engine_module.sample_flow_matching = flow
    train.sample_refined_cached_action_with_cache = sample
    sys.argv = [sys.argv[0], *remaining]
    try:
        train.main()
    except ProbeComplete:
        print("COMPLETE", args.report, flush=True)
    finally:
        MainlineTrainingEngine.eval_step = original_eval
        engine_module.sample_flow_matching = original_flow
        train.sample_refined_cached_action_with_cache = original_sample


if __name__ == "__main__":
    main()
