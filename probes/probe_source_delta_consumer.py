"""Capture eight real protected-carrier sources and the actual bottom read.

Evaluation only. Reuses the complete proposal/W/refined sampler. A local
return-frame observer reads actual production tensors without another softmax
or separate source contraction. The optional baseline-source clamp changes only
the named raw P2 term, then recomputes downstream consumers normally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from probes import probe_target_binding_measurement_v2 as base
from probes.source_delta_ledger import analyze, rms, stats
from clearvla.mainline.v120_core.role_delta_attnres import RoleDeltaAttnRes


NAMES = (
    "f_lattice", "f_target", "e_semantic_w", "e_semantic_s_target",
    "e_semantic_w_s", "e_geometry", "i_semantic", "i_geometry",
)


def observe_return(fn, args, kwargs, code=None, owner=None):
    """Read one exact function's return locals, preserving nested observers."""
    code = fn.__func__.__code__ if code is None else code
    owner = fn.__self__ if owner is None else owner
    captured = []
    previous = sys.getprofile()

    def observer(frame, event, arg):
        if event == "return" and frame.f_code is code and frame.f_locals.get("self") is owner:
            captured.append(dict(frame.f_locals))
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observer)
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    if len(captured) != 1:
        raise RuntimeError(f"expected one real producer return, got {len(captured)}: {code.co_name}")
    return result, captured[0]


def cpu(value):
    return value.detach().float().cpu().numpy().copy()


class Capture:
    def __init__(self, model, current, call_indices):
        self.model, self.current = model, current
        self.call_indices = set(call_indices)
        self.saved = []
        self.reset()
        compiler = model.policy_compiler
        reader = compiler.effect_reader
        old_terminal = reader.temporal_terminal

        def terminal(*args, **kwargs):
            reads = []
            handle = reader.semantic_value.register_forward_hook(
                lambda _m, _i, out: reads.append(out)
            )
            try:
                result, local = observe_return(
                    old_terminal, args, kwargs,
                    code=type(reader).temporal_terminal.__code__, owner=reader,
                )
            finally:
                handle.remove()
            effect = result[0]
            if not reads or reads[0].shape != effect.semantic.shape:
                raise RuntimeError("cannot align the real W semantic terminal")
            terms = {
                "e_semantic_w": reads[0],
                "e_semantic_s_target": local["target_value_effect"][..., 0, :].to(reads[0].dtype),
                "e_semantic_w_s": local.get("semantic_delta", torch.zeros_like(reads[0])),
            }
            event_key = (
                str(current.get("_sampling_stage", "")),
                int(current.get("_sampling_integration_index", -1)),
                int(current.get("_sampling_step", -1)),
            )
            if current.get("_sampling_label"):
                if self.clamp_source is not None:
                    reference = self.clamp_reference.get(event_key)
                    if reference is None:
                        raise RuntimeError(f"missing baseline clamp node: {event_key}")
                    terms[self.clamp_source] = reference[self.clamp_source].to(reads[0])
                    effect = replace(effect, semantic=(
                        terms["e_semantic_w"] + terms["e_semantic_s_target"]
                        + terms["e_semantic_w_s"]
                    ))
                    result = effect, result[1]
                    self.clamp_hits.add(event_key)
                self.term_events[event_key] = terms
            self.raw_semantics[int(effect.semantic.data_ptr())] = terms
            return result

        self.patch(reader, "temporal_terminal", terminal)
        old_compile = compiler.compile

        def compile_wrapper(*args, **kwargs):
            result, local = observe_return(
                old_compile, args, kwargs,
                code=type(compiler).compile.__code__, owner=compiler,
            )
            raw = local["raw_effect"]
            terms = self.raw_semantics.get(int(raw.semantic.data_ptr()))
            if terms is None:
                raise RuntimeError("raw P2 semantic producer identity did not match compile")
            scale = local["effect_contract"]
            # These are all scaled by the ORIGINAL joint semantic+geometry
            # contract. Retain the small additive/cast rounding separately.
            scaled = {n: v * scale.to(dtype=v.dtype) for n, v in terms.items()}
            consequence = local["consequence"]
            self.compiled[int(consequence.protected_consequence.data_ptr())] = {
                "semantic_terms": scaled,
                "p2_scale": scale,
                "semantic_actual": local["effect"].semantic,
                "semantic_raw": raw.semantic,
                "replay": {
                    "candidate_world": local["candidate_world"],
                    "context": local["context"],
                    "query": local["p1_action_query"],
                    "raw_terms": terms,
                },
            }
            return result

        self.patch(compiler, "compile", compile_wrapper)
        decoder = model.execution_bottom.decoder
        detail = decoder.protected_detail_basis_attnres
        old_detail = detail.forward

        def detail_wrapper(*args, **kwargs):
            ordinal = self.detail_ordinal
            self.detail_ordinal += 1
            result, local = observe_return(
                old_detail, args, kwargs,
                code=RoleDeltaAttnRes.forward.__code__, owner=detail,
            )
            if self.pending is not None and ordinal == self.protected_ordinal:
                self.actual_reader = {
                    k: local[k] for k in ("source_probability", "value_scale", "raw_delta_values", "routed_values")
                }
                self.actual_reader["output"] = result[0]
            return result

        self.patch(detail, "forward", detail_wrapper)
        old_bottom = decoder._read_policy_delta_bank

        def bottom_wrapper(action_query, bank, **kwargs):
            self.pending = bank
            self.detail_ordinal = 0
            self.protected_ordinal = int(bank.protected_policy_precision is not None)
            self.actual_reader = None
            result = old_bottom(action_query, bank, **kwargs)
            stage = str(current.get("_sampling_stage", ""))
            call = int(current.get("_sampling_step", -1))
            if current.get("_sampling_label") and call in self.call_indices:
                if self.actual_reader is None:
                    raise RuntimeError("protected reader did not execute")
                ptr = int(bank.protected_detail.data_ptr())
                sources = current.get("_protected_ledger_by_ptr", {}).get(ptr)
                semantic = self.compiled.get(ptr)
                if not sources or not semantic:
                    raise RuntimeError("missing exact carrier producer identity")
                values = dict(sources)
                values.update(semantic["semantic_terms"])
                record = {
                    "stage": stage,
                    "integration_index": int(current.get("_sampling_integration_index", -1)),
                    "integration_call_index": call,
                    "time": float(current["_sampling_time"]),
                    "arrays": {
                        "source_names": np.array(NAMES),
                        "source_values": np.stack([cpu(values[n]) for n in NAMES]),
                        "beta": cpu(self.actual_reader["source_probability"]),
                        "value_scale": cpu(self.actual_reader["value_scale"]),
                        "reader_values": cpu(self.actual_reader["raw_delta_values"]),
                        "reader_output": cpu(self.actual_reader["output"]),
                        "p2_effect_scale": cpu(semantic["p2_scale"]),
                        "p2_semantic_actual": cpu(semantic["semantic_actual"]),
                        "action_query": cpu(action_query),
                        "optional_bridge": cpu(result[0]),
                    },
                }
                record["dtypes"] = {n: str(values[n].dtype) for n in NAMES}
                record["dtypes"]["reader_values"] = str(self.actual_reader["raw_delta_values"].dtype)
                record["dtypes"]["reader_output"] = str(self.actual_reader["output"].dtype)
                record["replay"] = semantic["replay"]
                self.events.append(record)
            self.pending = None
            return result

        self.patch(decoder, "_read_policy_delta_bank", bottom_wrapper)

    def reset(self):
        self.raw_semantics, self.compiled = {}, {}
        self.events = []
        self.pending = None
        self.detail_ordinal = 0
        self.protected_ordinal = 0
        self.actual_reader = None
        self.term_events = {}
        self.clamp_source = None
        self.clamp_reference = {}
        self.clamp_hits = set()

    def patch(self, owner, name, fn):
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, fn)

    def close(self):
        for owner, name, fn in reversed(self.saved):
            setattr(owner, name, fn)


def key(event):
    return event["stage"], event["integration_index"], event["integration_call_index"]


def export(event, common, path):
    metadata = dict(common)
    metadata.update({k: v for k, v in event.items() if k not in {"arrays", "replay"}})
    payload = dict(event["arrays"])
    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)
    return payload


def cross_world_read(capture, left, right):
    """2x2 local P2 replay; W always travels with its own action identity.

    Columns change P1 query and S read context together. Rows change the
    candidate-world forecast with its validated action condition. This is a
    local forward counterfactual, never a relabeled candidate cache or rollout.
    """
    reader = capture.model.policy_compiler.effect_reader
    contexts = [left["replay"], right["replay"]]
    terms = {(0, 0): contexts[0]["raw_terms"], (1, 1): contexts[1]["raw_terms"]}
    for world_index, read_index in ((0, 1), (1, 0)):
        world, read = contexts[world_index], contexts[read_index]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            effect, _ = reader.forward_candidate(
                read["query"], world["candidate_world"], read["context"].intent.policy_dock(),
                action_condition=world["context"].action_condition,
                collect_diagnostics=True,
            )
        terms[world_index, read_index] = capture.raw_semantics[int(effect.semantic.data_ptr())]
    result = {}
    for name in ("e_semantic_w", "e_semantic_s_target", "e_semantic_w_s"):
        y00, y01, y10, y11 = [cpu(terms[k][name]).astype(np.float64) for k in ((0, 0), (0, 1), (1, 0), (1, 1))]
        delta = y11 - y00
        value = ((y10 - y00) + (y11 - y01)) * 0.5
        read = ((y01 - y00) + (y11 - y10)) * 0.5
        result[name] = {
            "raw_producer_delta": stats(delta),
            "candidate_world_factor": stats(value, delta),
            "p1_query_and_s_read_factor": stats(read, delta),
            "fixed_baseline_world_read_change": stats(y01 - y00, delta),
            "closure_error": stats(delta - value - read),
        }
    return {
        "units": "P2 semantic hidden value before its joint semantic/geometry RMS contract",
        "interpretation": "symmetric local 2x2 factor decomposition; no task-success claim",
        "world_action_identity_preserved": True,
        "sources": result,
    }


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def natural_instruction_comparisons(records, output_dir):
    """Compare ordinary baseline commands; never infer physical target identity."""
    anchors, results = {}, []
    for record in records:
        with np.load(output_dir / record["native_actions_file"], allow_pickle=False) as saved:
            action = saved["baseline"].copy()
        binding = np.asarray(record["binding"]["baseline"], dtype=np.float64)
        layout = record["layout"]
        if layout not in anchors:
            anchors[layout] = record["instruction"], action, binding
            continue
        instruction0, action0, binding0 = anchors[layout]
        results.append({
            "layout": layout,
            "instructions": [instruction0, record["instruction"]],
            "binding_mass_difference": stats(binding - binding0),
            "native_arm_24_difference": stats(action[..., :6] - action0[..., :6]),
            "native_arm_first8_difference": stats(action[..., :8, :6] - action0[..., :8, :6]),
            "native_gripper_difference": stats(action[..., 6:] - action0[..., 6:]),
            "interpretation": "natural instruction sensitivity, not target correctness or causal attribution",
        })
    return results


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite results: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    policy = base.ClearVLACheckpointPolicy(
        args.checkpoint, device=torch.device(args.device), t5_condition=args.t5_condition, seed=0
    )
    model = policy.bundle.model
    current = {}
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    identity = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha(args.checkpoint),
        "source_commit": commit,
        "probe_sha256": file_sha(__file__),
        "noise_seed": 12345,
        "limitations": "synthetic supported-K binding swap; per-camera semantic object identity is not established",
    }
    base_saved = base.install_wrappers(model, current)
    original_read = next(fn for _o, n, fn in base_saved if n == "_read_policy_delta_bank")
    deep_saved = base.install_deep_wrappers(model, current, original_decoder_read=original_read)
    sampling_saved = base.install_sampling_trace(model, current)
    binder = base.TaskConditionedTargetBinder.forward
    capture = Capture(model, current, args.calls)
    records = []
    try:
        for layout in args.layouts:
            history = base.history_with_world(args.observation_dir / f"{layout}.npz")
            for index, instruction in enumerate(args.instructions):
                common = dict(identity, layout=layout, instruction=instruction,
                              observation_sha256=file_sha(args.observation_dir / f"{layout}.npz"))
                run_kwargs = dict(policy=policy, history=history, instruction=instruction,
                                  current=current, model=model, mode="full", min_coordinate_separation=0.15)
                capture.reset()
                baseline = base._run_one(**run_kwargs, binder_forward=binder, swap=False, swap_pair=None)
                baseline_events = capture.events
                baseline_terms = dict(capture.term_events)
                repeat_stats = []
                for _ in range(args.repeats - 1):
                    capture.reset()
                    repeated = base._run_one(**run_kwargs, binder_forward=binder, swap=False, swap_pair=baseline["pair_info"])
                    repeat_stats.append({
                        "action": base._compare_path_actions(baseline, repeated),
                        "reader_output_rmse_by_event": [
                            {"key": key(a), "rmse": rms(a["arrays"]["reader_output"] - b["arrays"]["reader_output"])}
                            for a, b in zip(baseline_events, capture.events, strict=True)
                        ],
                    })
                capture.reset()
                swapped = base._run_one(
                    **run_kwargs, binder_forward=base._swap_wrapper(binder, current, baseline["pair_info"]),
                    swap=True, swap_pair=baseline["pair_info"],
                )
                paired = []
                for event_index, (left, right) in enumerate(zip(baseline_events, capture.events, strict=True)):
                    if key(left) != key(right):
                        raise RuntimeError("sampling event scopes differ")
                    prefix = f"{layout}_{index:02d}_event{event_index:02d}"
                    p0 = export(left, dict(common, state="baseline"), args.output_dir / (prefix + "_baseline.npz"))
                    p1 = export(right, dict(common, state="binding_swap"), args.output_dir / (prefix + "_swapped.npz"))
                    pair = analyze(p0, p1)
                    pair["input_files"] = [prefix + "_baseline.npz", prefix + "_swapped.npz"]
                    pair["p2_semantic_split_residual"] = {
                        label: rms(p["p2_semantic_actual"].astype(np.float64) - p["source_values"][2:5].astype(np.float64).sum(0))
                        for label, p in (("baseline", p0), ("swapped", p1))
                    }
                    if args.cross_world:
                        pair["world_read_factorial"] = cross_world_read(capture, left, right)
                    paired.append(pair)
                native_path = args.output_dir / f"{layout}_{index:02d}_native_actions.npz"
                clamp_result = None
                if args.clamp_source is not None:
                    capture.reset()
                    capture.clamp_source = args.clamp_source
                    capture.clamp_reference = baseline_terms
                    clamped = base._run_one(
                        **run_kwargs, binder_forward=base._swap_wrapper(binder, current, baseline["pair_info"]),
                        swap=True, swap_pair=baseline["pair_info"],
                    )
                    if capture.clamp_hits != set(baseline_terms):
                        raise RuntimeError("baseline clamp did not cover the same complete sampler nodes")
                    clamp_result = {
                        "source": args.clamp_source,
                        "scope": "raw P2 source clamped to baseline at all aligned proposal/refined nodes; downstream recomputed",
                        "matched_node_count": len(capture.clamp_hits),
                        "action_vs_baseline": {}, "action_vs_swapped": {},
                    }
                    for n in baseline["nodes"]:
                        if n.startswith("final_native_action"):
                            clamp_result["action_vs_baseline"][n] = base.compare(clamped["nodes"][n], baseline["nodes"][n])
                            clamp_result["action_vs_swapped"][n] = base.compare(clamped["nodes"][n], swapped["nodes"][n])
                    arm_delta = swapped["nodes"]["final_native_action_arm"] - baseline["nodes"]["final_native_action_arm"]
                    clamp_result["remaining_arm_delta"] = stats(
                        clamped["nodes"]["final_native_action_arm"] - baseline["nodes"]["final_native_action_arm"], arm_delta)
                    np.savez_compressed(args.output_dir / f"{layout}_{index:02d}_clamped_native_actions.npz",
                        clamped=clamped["nodes"]["final_native_action"],
                        metadata_json=np.array(json.dumps(dict(common,clamp_source=args.clamp_source),sort_keys=True)))
                np.savez_compressed(native_path,
                    baseline=baseline["nodes"]["final_native_action"],
                    swapped=swapped["nodes"]["final_native_action"],
                    metadata_json=np.array(json.dumps(common, sort_keys=True)))
                records.append({
                    "layout": layout, "instruction": instruction, "pair_info": baseline["pair_info"],
                    "repeats": repeat_stats,
                    "binding": {
                        label: np.asarray(result["nodes"]["s_binding_mass"]).tolist()
                        for label, result in (("baseline", baseline), ("swapped", swapped))
                    },
                    "action_differences": {
                        n: base.compare(swapped["nodes"][n], baseline["nodes"][n])
                        for n in baseline["nodes"] if n.startswith("final_native_action")
                    },
                    "events": paired,
                    "native_actions_file": native_path.name,
                    "baseline_source_clamp": clamp_result,
                })
                print(json.dumps({"instruction": instruction, "captured_events": len(paired)}), flush=True)
    finally:
        capture.close()
        for o, n, fn in reversed(sampling_saved):
            setattr(o, n, fn)
        for o, n, fn in reversed(deep_saved):
            setattr(o, n, fn)
        base.restore_wrappers(base_saved)
    result = {
        "identity": identity, "records": records,
        "natural_instruction_comparisons": natural_instruction_comparisons(records, args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--t5-condition", type=Path, required=True)
    p.add_argument("--observation-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layouts", nargs="+", default=["standard"])
    p.add_argument("--instructions", nargs="+", default=list(base.DEFAULT_INSTRUCTIONS))
    p.add_argument("--calls", nargs="+", type=int, default=[0, 5])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cross-world", action="store_true")
    p.add_argument("--clamp-source", choices=("e_semantic_w", "e_semantic_s_target", "e_semantic_w_s"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
