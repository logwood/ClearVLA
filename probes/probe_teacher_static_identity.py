"""Same-chart controls of the production Teacher's observation measurement.

Use actual past G facts captured during factual full-checkpoint replay. Present
their exact raw current chart as a hypothetical static successor, at offset 0
and the four declared W endpoints, with original versus zero transport prior. These
are isolated measurement checks, not policy conditions, training or rollouts.
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


def array(value):
    return value.detach().float().cpu().numpy().copy()


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float) ** 2)))


def summary(measured):
    sem = array(measured.successor_per_support - measured.current_reference)
    image = array(measured.transport_per_support)
    return dict(semantic_delta_rms=rms(sem), image_delta_rms=rms(image),
        image_delta_per_K_camera=image[0, 0].tolist(), null=array(measured.null_probability)[0, 0, :, 0].tolist(),
        real_mass=array(measured.association_real_mass_per_support)[0, 0, :, 0].tolist())


def main():
    ap = argparse.ArgumentParser()
    for name in ["checkpoint", "plan", "output"]:
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"), seed=0)
    model = policy.bundle.model
    teacher = model.training_targets.teacher
    original = teacher.measure_observations
    captured = {}

    def measure(*a, **kw):
        result = original(*a, **kw)
        captured.update(source=kw, result=result)
        return result

    teacher.measure_observations = measure
    versions = {n: p._version for n, p in model.named_parameters()}
    report = {"checkpoint_sha256": policy.bundle.checkpoint_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "current_reference_mode": teacher.current_reference_mode, "scope": __doc__, "records": []}
    try:
        plan = json.loads(args.plan.read_text())
        for cid, state in [(1, 40), (3, 40), (7, 40), (17, 136)]:
            row = next(r for r in plan if r["case_id"] == cid)
            with np.load(Path(row["case"]) / "trajectory.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in ["rgb_static", "rgb_gripper", "robot_obs", "executed", "raw_chunks"]}
            policy.reset(); history = CausalHistory(executed_world=True)
            for t in range(state + 1):
                previous = np.zeros(7, np.float32) if t == 0 else data["executed"][t - 1]
                obs = calvin_policy_observation({"rgb_obs": {k: data[k][t] for k in ["rgb_static", "rgb_gripper"]}, "robot_obs": data["robot_obs"][t]}, previous)
                if t == 0: history.reset(obs, reset_action=previous)
                else: history.append(previous, obs)
                if t not in [0, state]:
                    if t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                action, online = policy.act_with_input(history.snapshot(), row["instruction"])
                if t != state: continue
                source, measured = captured["source"], captured["result"]
                facts = source["facts"]
                same = facts.dense_chart.dino_content[:, None].detach()
                assert same.shape == source["observations"].shape
                with torch.no_grad():
                    actual_source_chart = model.observation.observation_supports(online.history.executed_world_window.dino_history[:, -1:])
                source_chart_error = float((actual_source_chart.float() - same.float()).abs().max())
                torch.testing.assert_close(actual_source_chart.float(), same.float(), atol=0., rtol=0.)
                item = dict(case_id=cid, state=t, source_offset=-4,
                    recorded_arm_rmse=rms(action[:8, :6] - data["raw_chunks"][t // 8, :8, :6]),
                    actual=summary(measured), source_motion_rms=rms(array(facts.camera_transport_prior)),
                    exact_past_DINO_source_chart_error=source_chart_error, variants=[])
                with torch.no_grad():
                    saved_mode = teacher.current_reference_mode
                    try:
                        for reference_mode in [saved_mode, "raw_chart_v1"]:
                            # Change only an isolated measurement call. Restore
                            # the admitted selection before any policy replay.
                            teacher.current_reference_mode = reference_mode
                            # The four declared W endpoints must not acquire
                            # motion merely because the search prior widens.
                            for offset in [0, 4, 8, 16, 24]:
                                for motion in ["original", "zero"]:
                                    conditioned = facts if motion == "original" else replace(facts, camera_transport_prior=torch.zeros_like(facts.camera_transport_prior))
                                    static = original(facts=conditioned, observations=same,
                                        relative_offsets=torch.full_like(source["relative_offsets"], offset), observed=source["observed"])
                                    value = summary(static)
                                    # Keep this exact posterior while replacing
                                    # only the current reference by the raw-chart
                                    # expectation at the same G-owned address.
                                    address = torch.where(facts.dense_chart.cell_observed[:, None, ..., 0], facts.object_to_chart.float(), 0.).flatten(2)
                                    address = address / address.sum(-1, keepdim=True).clamp_min(1e-6)
                                    raw_reference = torch.einsum("bkn,bnd->bkd", address, facts.dense_chart.dino_content.float().flatten(1, -2))[:, None]
                                    real = 1 - static.null_probability
                                    delta = static.successor_per_support - static.current_reference
                                    ref_shift = real * (raw_reference - static.current_reference)
                                    value.update(reference_mode=reference_mode, offset=offset, source_motion=motion,
                                        semantic_delta_same_posterior_raw_reference_rms=rms(array(delta - ref_shift)),
                                        semantic_reference_value_shift_rms=rms(array(ref_shift)),
                                        semantic_actual_minus_static_rms=rms(array((measured.successor_per_support-measured.current_reference) - delta)),
                                        image_actual_minus_static_rms=rms(array(measured.transport_per_support-static.transport_per_support)))
                                    item["variants"].append(value)
                    finally:
                        teacher.current_reference_mode = saved_mode
                report["records"].append(item)
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("TEACHER_STATIC", cid, t, flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        report.update(state="complete", parameters_unchanged=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        teacher.measure_observations = original


if __name__ == "__main__":
    main()
