"""Trace factual outcome producers and their admitted policy consumers.

Read-only replay of saved observations and confirmed commands. Zero controls
remove declared internal values without changing RGB, history, target binding,
or random action seed. They are interface interventions, not physical rollouts
or independent contribution percentages. Simulator truth is not a policy input.
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


def array(value):
    return value.detach().float().cpu().numpy().copy()


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    for name in ["checkpoint", "plan", "output"]:
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"), seed=0)
    model = policy.bundle.model
    import clearvla.simulation.clearvla_policy as frontend
    organizer = model.intent.organizer
    compiler = model.policy_compiler.plan_compiler
    teacher = model.training_targets.teacher
    world = model.world.dynamics
    observer = compiler.robot_observer
    originals = dict(sample=frontend.sample_action, encode=model.encode_online,
        organize=organizer.forward, measure=teacher.measure_observations,
        predict=world.predict_executed_endpoint, observe=observer.observe,
        robot_read=observer.read, world_read=compiler.world_feedback_read.forward)
    capture, mode = {}, {"name": "baseline"}

    def sampled(*a, **kw):
        result = originals["sample"](*a, **kw)
        capture["sample"] = result
        return result

    def encode(*a, **kw):
        result = originals["encode"](*a, **kw)
        capture["cache"] = result[0]
        capture["coarse"] = result[1].top.coarse_action.action_prediction
        return result

    def organize(*a, **kw):
        result, metrics = originals["organize"](*a, **kw)
        capture["unmodified_intent"] = result
        if mode["name"] in ["state_change_zero", "all_short_feedback_zero"]:
            result = replace(result, state_change_evidence=torch.zeros_like(result.state_change_evidence))
        return result, metrics

    def measure(*a, **kw):
        result = originals["measure"](*a, **kw)
        capture["measured"] = result
        return result

    def predict(*a, **kw):
        result = originals["predict"](*a, **kw)
        capture["predicted"] = result
        return result

    def observe(step, current_state):
        result, loss = originals["observe"](step, current_state)
        capture["robot_observed_delta"] = current_state - step.previous_state
        capture["robot_innovation"] = result.innovation
        return result, loss

    def robot_read(*a, **kw):
        result = originals["robot_read"](*a, **kw)
        return torch.zeros_like(result) if mode["name"] in ["explicit_feedback_zero", "all_short_feedback_zero"] else result

    def world_read(*a, **kw):
        result = originals["world_read"](*a, **kw)
        return torch.zeros_like(result) if mode["name"] in ["explicit_feedback_zero", "all_short_feedback_zero"] else result

    def summarize(result):
        native = policy.bundle.action_normalizer.decode(array(result.action)[0])
        native[:, -1] = array(result.gripper_command)[0]
        coarse = policy.bundle.action_normalizer.decode(array(capture["coarse"])[0])
        intent = capture["cache"].top.intent
        return {"first8_native": native[:8].tolist(), "first8_mean": native[:8].mean(0).tolist(),
                "coarse_first8": coarse[:8].tolist(), "coarse_first8_mean": coarse[:8].mean(0).tolist(),
                "binding": array(intent.target_binding.mass)[0].tolist(),
                "binding_null": array(intent.target_binding.null_mass)[0].tolist(),
                "state_change_rms": rms(array(intent.state_change_evidence))}

    versions = {n: p._version for n, p in model.named_parameters()}
    frontend.sample_action, model.encode_online = sampled, encode
    organizer.forward, teacher.measure_observations = organize, measure
    world.predict_executed_endpoint, observer.observe = predict, observe
    observer.read, compiler.world_feedback_read.forward = robot_read, world_read
    report = {"checkpoint_sha256": policy.bundle.checkpoint_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": __doc__, "records": []}
    try:
        plan = json.loads(args.plan.read_text())
        for cid, states in [(1, [40]), (3, [40]), (7, [40]), (10, [120, 136]), (17, [120, 136])]:
            row = next(r for r in plan if r["case_id"] == cid)
            with np.load(Path(row["case"]) / "trajectory.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in ["rgb_static", "rgb_gripper", "robot_obs", "executed", "raw_chunks"]}
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(max(states) + 1):
                previous = np.zeros(7, np.float32) if t == 0 else data["executed"][t - 1]
                obs = calvin_policy_observation({"rgb_obs": {k: data[k][t] for k in ["rgb_static", "rgb_gripper"]}, "robot_obs": data["robot_obs"][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                mode["name"] = "baseline"
                if t not in states:
                    if t == 0:
                        policy.act_with_input(history.snapshot(), row["instruction"])
                    elif t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                action, online = policy.act_with_input(history.snapshot(), row["instruction"])
                seed = capture["sample"].initial_physical_noise
                baseline = summarize(capture["sample"])
                measured, predicted = capture["measured"], capture["predicted"]
                feedback = capture["cache"].world_feedback.feedback
                actual_sem = measured.successor_per_support[:, 0] - measured.current_reference[:, 0]
                actual_image = measured.transport_per_support[:, 0]
                torch.testing.assert_close(actual_sem - predicted.semantic.float(), feedback.semantic, atol=1e-6, rtol=0.)
                torch.testing.assert_close(actual_image - predicted.image.float(), feedback.image, atol=1e-6, rtol=0.)
                original_intent = capture["unmodified_intent"]
                replaced_intent = replace(original_intent, state_change_evidence=-original_intent.state_change_evidence)
                old_dock, new_dock = original_intent.action_dock(), replaced_intent.action_dock()
                # All coarse inputs are the very same objects. This proves
                # lack of a dedicated route, not lack of all RGB/history clues.
                identical_dock = all(getattr(old_dock, k) is getattr(new_dock, k) for k in old_dock.__dataclass_fields__)
                assert identical_dock
                item = {"case_id": cid, "state": t, "baseline": baseline,
                    "recorded_arm_rmse": rms(action[:8, :6] - data["raw_chunks"][t // 8, :8, :6]),
                    "recorded_gripper_mismatches": int(np.sum(action[:8, 6] != data["raw_chunks"][t // 8, :8, 6])),
                    "reversing_state_change_keeps_every_coarse_input_identical": identical_dock,
                    "producers": {"W_measured_semantic_rms": rms(array(actual_sem)),
                        "W_predicted_semantic_rms": rms(array(predicted.semantic)),
                        "W_semantic_innovation_rms": rms(array(feedback.semantic)),
                        "W_measured_image_per_K_camera": array(actual_image)[0].tolist(),
                        "W_predicted_image_per_K_camera": array(predicted.image)[0].tolist(),
                        "W_image_innovation_per_K_camera": array(feedback.image)[0].tolist(),
                        "robot_observed_delta": array(capture["robot_observed_delta"])[0].tolist(),
                        "robot_innovation": array(capture["robot_innovation"])[0].tolist()}, "variants": []}
                for name in ["repeat", "state_change_zero", "explicit_feedback_zero", "all_short_feedback_zero"]:
                    mode["name"] = name
                    with torch.no_grad():
                        result = sample_action(model, online, policy.bundle.config, initial_physical_noise=seed)
                    value = summarize(result)
                    value.update(name=name,
                        first8_arm_delta_rms=rms(np.asarray(value["first8_native"])[:, :6] - action[:8, :6]),
                        coarse_arm_delta_rms=rms(np.asarray(value["coarse_first8"])[:, :6] - np.asarray(baseline["coarse_first8"])[:, :6]))
                    np.testing.assert_allclose(value["binding"], baseline["binding"], atol=0., rtol=0.)
                    item["variants"].append(value)
                report["records"].append(item)
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("OUTCOME_ROUTE", cid, t, flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        report.update(state="complete", parameters_unchanged=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        frontend.sample_action, model.encode_online = originals["sample"], originals["encode"]
        organizer.forward, teacher.measure_observations = originals["organize"], originals["measure"]
        world.predict_executed_endpoint, observer.observe = originals["predict"], originals["observe"]
        observer.read, compiler.world_feedback_read.forward = originals["robot_read"], originals["world_read"]


if __name__ == "__main__":
    main()
