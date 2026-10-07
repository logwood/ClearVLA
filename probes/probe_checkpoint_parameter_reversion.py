"""Localize candidate drift with reversible, explicitly hybrid parameter probes.

No hybrid is saved as a checkpoint or deployed. Same current observation,
instruction reference, original sampling noise and production sampler are used.
Parameter-group reversions are interventions in a co-adapted model, not
independent additive contribution percentages.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import torch

from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory


def array(x):
    return x.detach().float().cpu().numpy().copy()


def rms(x):
    return float(np.sqrt(np.square(np.asarray(x, dtype=np.float64)).mean()))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "reference-checkpoint", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--case-ids", type=int, nargs="+", default=[4, 8, 13])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    if sha(args.reference_checkpoint) != args.reference_sha256:
        raise ValueError("reference checkpoint byte identity mismatch")
    policy = ClearVLACheckpointPolicy(
        args.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"),
        seed=0,
    )
    model = policy.bundle.model
    new = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    old = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    for name in ("dimensions", "observation", "top", "bottom"):
        if new["config"][name] != old["config"][name]:
            raise ValueError("model configuration differs at " + name)
    if new["component_selection"] != old["component_selection"]:
        raise ValueError("component selection mismatch")
    # Data provenance changed in this repair; the physical feature charts did not.
    if new["data_state"]["normalizer_metadata"] != old["data_state"]["normalizer_metadata"]:
        raise ValueError("normalizer metadata differs")
    current = model.state_dict()
    if set(current) != set(old["model"]) or set(current) != set(new["model"]):
        raise ValueError("state dictionary inventory differs")
    for key, value in current.items():
        if value.shape != old["model"][key].shape or value.shape != new["model"][key].shape:
            raise ValueError("state shape differs: " + key)
    names = set(dict(model.named_parameters()))
    groups = {
        "conditioning": lambda k: k.startswith("conditioning."),
        "vision_and_query": lambda k: k.startswith(("observation.", "bridge.")),
        "G": lambda k: k.startswith("grounding."),
        "S_language": lambda k: k.startswith("intent.organizer.goal_"),
        "S_other": lambda k: k.startswith("intent.organizer.") and not k.startswith("intent.organizer.goal_"),
        "coarse": lambda k: k.startswith("intent.coarse_action."),
        "W_and_transition": lambda k: k.startswith(("world.", "transition.")),
        "P1": lambda k: k.startswith("p1."),
        "P2": lambda k: k.startswith(("policy_compiler.effect_reader.", "policy_compiler.consequence.")),
        "P3": lambda k: k.startswith("policy_compiler.plan_compiler."),
        "bottom": lambda k: k.startswith("execution_bottom."),
    }
    selected = {name: sorted(k for k in current if predicate(k)) for name, predicate in groups.items()}
    if not all(selected.values()):
        raise ValueError("empty parameter intervention")
    write(args.output / "identity.json", {
        "candidate_sha256": policy.bundle.checkpoint_sha256,
        "reference_sha256": args.reference_sha256,
        "script_sha256": sha(__file__),
        "production_source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "state_keys": len(current),
        "groups": {key: {"keys": len(keys), "parameters": sum(current[k].numel() for k in keys if k in names)}
                   for key, keys in selected.items()},
        "scope": "zero-update reversible parameter interventions on candidate source; no hybrid checkpoint is saved; no oracle enters policy",
    })
    saved_methods = []
    nodes = {}
    sample_holder = []

    def tap(owner, name, key, getter):
        original = getattr(owner, name)
        saved_methods.append((owner, name, original))
        def wrapped(*positional, **kwargs):
            value = original(*positional, **kwargs)
            nodes[key] = array(getter(value))
            return value
        setattr(owner, name, wrapped)

    tap(model.intent.coarse_action, "forward", "coarse", lambda out: out.action_prediction)
    tap(model.p1, "build_static", "P1.protected_detail", lambda out: out[0].protected_detail)
    tap(model.policy_compiler.plan_compiler, "forward", "P3.temporal", lambda out: out[0].temporal)
    import clearvla.simulation.clearvla_policy as frontend
    original_sample = frontend.sample_action
    def capture(*positional, **kwargs):
        value = original_sample(*positional, **kwargs)
        sample_holder[:] = [value]
        return value
    frontend.sample_action = capture

    def restore(keys, source):
        with torch.no_grad():
            for key in keys:
                current[key].copy_(source["model"][key])

    def native(sample):
        result = policy.bundle.action_normalizer.decode(array(sample.action)[0])
        result[:, -1] = array(sample.gripper_command)[0]
        return result

    rows = []
    try:
        for row in json.loads(args.plan.read_text()):
            if row["case_id"] not in args.case_ids:
                continue
            restore(current, new)
            case = Path(row["case"])
            with np.load(case / "trajectory.npz", allow_pickle=False) as data:
                observation = calvin_policy_observation(
                    {"rgb_obs": {key: data[key][0] for key in ("rgb_static", "rgb_gripper")},
                     "robot_obs": data["robot_obs"][0]},
                    np.zeros(7, dtype=np.float32),
                )
                recorded = data["raw_chunks"][0, :8].copy()
            history = CausalHistory(executed_world=True)
            history.reset(observation, reset_action=np.zeros(7, dtype=np.float32))
            policy.reset()
            nodes.clear()
            _, online = policy.act_with_input(history.snapshot(), row["instruction"])
            sample = sample_holder[0]
            baseline = native(sample)
            base_nodes = {key: value.copy() for key, value in nodes.items()}
            entry = {"case_id": row["case_id"], "state": 0,
                     "baseline_first8": baseline[:8].tolist(),
                     "recorded_arm_rmse": rms(baseline[:8, :6] - recorded[:, :6]), "variants": []}
            for variant in ["repeat", "all_reference_state"] + list(selected):
                restore(current, new)
                if variant == "all_reference_state":
                    restore(current, old)
                elif variant != "repeat":
                    restore(selected[variant], old)
                nodes.clear()
                with torch.no_grad():
                    result = sample_action(model, online, policy.bundle.config,
                                           initial_physical_noise=sample.initial_physical_noise)
                action = native(result)
                item = {"variant": variant, "native_first8": action[:8].tolist(),
                        "mean_shift": (action[:8] - baseline[:8]).mean(0).tolist(),
                        "first8_arm_rmse": rms(action[:8, :6] - baseline[:8, :6]),
                        "gripper_mismatches": int(np.count_nonzero(action[:8, 6] != baseline[:8, 6])),
                        "node_rmse": {key: rms(value - base_nodes[key]) for key, value in nodes.items()}}
                entry["variants"].append(item)
                write(args.output / "results.json", rows + [entry])
                print(row["case_id"], variant, item["mean_shift"][:3], flush=True)
            rows.append(entry)
        if len(rows) != len(args.case_ids):
            raise ValueError("missing requested cases")
        write(args.output / "complete.json", {"complete": True, "cases": args.case_ids,
              "scope": "No checkpoint update; original full-state reversion is a positive control; do not promote hybrids."})
    finally:
        restore(current, new)
        frontend.sample_action = original_sample
        for owner, name, original in saved_methods:
            setattr(owner, name, original)


if __name__ == "__main__":
    main()
