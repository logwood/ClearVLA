"""Trace current observed gripper width through real stage-boundary histories.

Width-only sensor interventions keep RGB, past commands and prior states
factual. They are explicitly not physical counterfactual rollouts. Current
state and its timestamp-zero duplicate are updated together using the real
state feature encoder/normalizer. No contact/object truth enters the model.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json

import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action


def array(x):
    return x.detach().float().cpu().numpy().copy()


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    for name in ["checkpoint", "plan", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"), seed=0)
    model = policy.bundle.model
    import clearvla.simulation.clearvla_policy as frontend
    original_sample = frontend.sample_action
    original_encode = model.encode_online
    captures = {}

    def encode(*a, **kw):
        result = original_encode(*a, **kw)
        captures["cache"] = result[0]
        captures["coarse"] = result[1].top.coarse_action
        return result

    def sampled(*a, **kw):
        result = original_sample(*a, **kw)
        captures["sample"] = result
        return result

    model.encode_online = encode
    frontend.sample_action = sampled
    report = {"checkpoint_sha256": policy.bundle.checkpoint_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Width-only current sensor interventions; no physical rollout and no privileged target/contact inputs. Full downstream recomputation, shared initial noise, exact current-state duplicates.", "records": []}

    def summarize(result):
        native = policy.bundle.action_normalizer.decode(array(result.action)[0])
        native[:, -1] = array(result.gripper_command)[0]
        cache = captures["cache"]
        coarse = policy.bundle.action_normalizer.decode(array(captures["coarse"].action_prediction)[0])
        return {"native": native.tolist(), "first8_mean": native[:8].mean(0).tolist(),
                "coarse_first8_mean": coarse[:8].mean(0).tolist(),
                "opening_innovation_feature": float(cache.robot_feedback.innovation[0, -1]),
                "binding": array(cache.top.intent.target_binding.mass)[0].tolist(),
                "null": float(cache.top.intent.target_binding.null_mass[0]),
                "first8_p_open": array(result.gripper_command_logits.softmax(-1))[0, :8, 1].tolist()}

    try:
        plan = json.loads(args.plan.read_text())
        for cid, selected in [(1, [32, 40, 48]), (3, [32, 40, 48]), (7, [32, 40, 48]),
                              (10, [120, 136]), (17, [120, 136])]:
            row = next(r for r in plan if r["case_id"] == cid)
            case = Path(row["case"])
            with np.load(case / "trajectory.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(max(selected) + 1):
                previous = np.zeros(7, np.float32) if t == 0 else data["executed"][t - 1]
                obs = calvin_policy_observation({"rgb_obs": {k: data[k][t] for k in ["rgb_static", "rgb_gripper"]},
                                                "robot_obs": data["robot_obs"][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                if t not in selected:
                    if t == 0:
                        policy.act_with_input(history.snapshot(), row["instruction"])
                    elif t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                factual, online = policy.act_with_input(history.snapshot(), row["instruction"])
                seed = captures["sample"].initial_physical_noise
                raw = data["robot_obs"][t, :7].copy()
                item = {"case_id": cid, "state": t, "observed_opening_m": float(raw[6]),
                        "recorded_arm_rmse": rms(factual[:8, :6] - data["raw_chunks"][t // 8, :8, :6]),
                        "recorded_gripper_mismatches": int(np.sum(factual[:8, 6] != data["raw_chunks"][t // 8, :8, 6])),
                        "baseline": summarize(captures["sample"]), "variants": []}
                for name, opening in [("repeat", float(raw[6])), ("empty_width", 0.0),
                                      ("held_object_width", .0412), ("open_width", .08)]:
                    altered_raw = raw.copy()
                    altered_raw[6] = opening
                    state = policy._normal(altered_raw[None], action=False)
                    timing = online.history.timing
                    current_rows = timing.state_offsets == 0
                    assert int(current_rows.sum()) == 1
                    state_history = torch.where(current_rows[..., None], state[:, None], online.history.state_history)
                    changed = replace(online, history=replace(online.history, state=state, state_history=state_history))
                    if name == "repeat":
                        torch.testing.assert_close(state, online.history.state, atol=0, rtol=0)
                        torch.testing.assert_close(state_history, online.history.state_history, atol=0, rtol=0)
                    with torch.no_grad():
                        result = sample_action(model, changed, policy.bundle.config, initial_physical_noise=seed)
                    value = summarize(result)
                    value.update(name=name, opening_m=opening,
                        first8_arm_delta_rms=rms(np.asarray(value["native"])[:8, :6] - factual[:8, :6]))
                    item["variants"].append(value)
                report["records"].append(item)
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("OPENING", cid, t, flush=True)
        report["state"] = "complete"
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        frontend.sample_action = original_sample
        model.encode_online = original_encode


if __name__ == "__main__":
    main()
