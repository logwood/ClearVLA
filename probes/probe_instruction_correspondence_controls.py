"""Inference-only controls at the S observed-correspondence boundary.

No physical ground truth enters a condition. Ranking-only preserves each row's
real mass, null, and entire probability multiset. Fixed-metric uses the existing
DINO cosine/sqrt(D) convention but preserves the learned null; it changes
concentration and is explicitly not a trained or accepted repair. Robot-zero
removes only S's observed robot-displacement value, not its P3 sibling/history.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
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
    module = model.intent.organizer.instruction_progress
    import clearvla.simulation.clearvla_policy as frontend
    old_sample, old_encode = frontend.sample_action, model.encode_online
    old_forward, old_kernel = module.forward, module._kernel
    captures, mode = {}, {"name": "baseline"}

    def encode(*a, **kw):
        result = old_encode(*a, **kw)
        captures["coarse"] = result[1].top.coarse_action.action_prediction
        captures["binding"] = result[0].top.intent.target_binding
        return result

    def sampled(*a, **kw):
        result = old_sample(*a, **kw)
        captures["sample"] = result
        return result

    def progress(*a, **kw):
        captures["current_dino"] = kw["current_dino"]
        result = old_forward(*a, **kw)
        captures["progress"] = result
        return result

    def kernel(query, content, observed, grid):
        original = old_kernel(query, content, observed, grid)
        if mode["name"] not in ("ranking_only", "fixed_metric_same_null"):
            return original
        with torch.autocast(device_type=query.device.type, enabled=False):
            now = F.normalize(torch.where(observed[..., None], captures["current_dino"].float(), 0.), dim=-1)
            destination = F.normalize(torch.where(observed[..., None], content.float(), 0.), dim=-1)
            cosine = now @ destination.transpose(-2, -1)
            legal = observed[:, :, None, :].expand_as(cosine)
            real, null = original[..., :-1], original[..., -1:]
            if mode["name"] == "ranking_only":
                order = cosine.masked_fill(~legal, -torch.inf).argsort(-1, descending=True)
                replacement = torch.zeros_like(real).scatter(-1, order, real.sort(-1, descending=True).values)
                captures["histogram_error"] = max(captures.get("histogram_error", 0.),
                    float((replacement.sort(-1).values - real.sort(-1).values).abs().max()))
            else:
                logits = (cosine * math.sqrt(now.shape[-1])).masked_fill(~legal, -torch.inf)
                logits = torch.where(legal.any(-1, keepdim=True), logits, 0.)
                conditional = torch.where(legal, logits.softmax(-1), 0.)
                replacement = conditional * real.sum(-1, keepdim=True)
            torch.testing.assert_close(replacement.sum(-1), real.sum(-1), atol=2e-6, rtol=0.)
            return torch.cat((replacement, null), -1)

    def values_hook(m, a, out):
        if mode["name"] == "S_robot_progress_zero":
            out = (*out[:3], torch.zeros_like(out[3]), out[4])
        return out

    def summary(result):
        native = policy.bundle.action_normalizer.decode(array(result.action)[0])
        native[:, -1] = array(result.gripper_command)[0]
        coarse = policy.bundle.action_normalizer.decode(array(captures["coarse"])[0])
        output, evidence = captures["progress"]
        return {"native": native.tolist(), "first8_mean": native[:8].mean(0).tolist(),
                "coarse_first8_mean": coarse[:8].mean(0).tolist(),
                "first8_p_open": array(result.gripper_command_logits.softmax(-1))[0, :8, 1].tolist(),
                "binding": array(captures["binding"].mass)[0].tolist(),
                "binding_null": array(captures["binding"].null_mass)[0].tolist(),
                "progress_rms": rms(array(output)),
                "content_delta_rms": rms(array(evidence.content_delta)),
                "image_delta_rms": rms(array(evidence.image_delta)),
                "current_reference_probability_delta_rms": rms(array(evidence.posterior.current_probability - evidence.posterior.reference_probability)),
                "histogram_error": captures.get("histogram_error", 0.)}

    versions = {n: p._version for n, p in model.named_parameters()}
    frontend.sample_action, model.encode_online = sampled, encode
    module.forward, module._kernel = progress, kernel
    handle = module.values.register_forward_hook(values_hook)
    report = {"checkpoint_sha256": policy.bundle.checkpoint_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": __doc__, "records": []}
    try:
        plan = json.loads(args.plan.read_text())
        for cid, selected in [(1, [0, 40]), (3, [40]), (7, [40]), (10, [136]), (17, [120, 136])]:
            row = next(r for r in plan if r["case_id"] == cid)
            case = Path(row["case"])
            with np.load(case / "trajectory.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in ["rgb_static", "rgb_gripper", "robot_obs", "executed", "raw_chunks"]}
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(max(selected) + 1):
                previous = np.zeros(7, np.float32) if t == 0 else data["executed"][t - 1]
                obs = calvin_policy_observation({"rgb_obs": {k: data[k][t] for k in ["rgb_static", "rgb_gripper"]}, "robot_obs": data["robot_obs"][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                mode["name"] = "baseline"
                if t not in selected:
                    if t == 0:
                        policy.act_with_input(history.snapshot(), row["instruction"])
                    elif t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                action, online = policy.act_with_input(history.snapshot(), row["instruction"])
                seed = captures["sample"].initial_physical_noise
                row_result = {"case_id": cid, "state": t,
                    "recorded_arm_rmse": rms(action[:8, :6] - data["raw_chunks"][t//8, :8, :6]),
                    "recorded_gripper_mismatches": int(np.sum(action[:8, 6] != data["raw_chunks"][t//8, :8, 6])),
                    "baseline": summary(captures["sample"]), "variants": []}
                for name in ["repeat", "ranking_only", "fixed_metric_same_null", "S_robot_progress_zero"]:
                    mode["name"] = name
                    captures["histogram_error"] = 0.
                    with torch.no_grad():
                        result = sample_action(model, online, policy.bundle.config, initial_physical_noise=seed)
                    value = summary(result)
                    value.update(name=name, first8_arm_delta_rms=rms(np.asarray(value["native"])[:8, :6] - action[:8, :6]))
                    np.testing.assert_allclose(value["binding"], row_result["baseline"]["binding"], atol=0., rtol=0.)
                    row_result["variants"].append(value)
                report["records"].append(row_result)
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("CORRESPONDENCE_CONTROL", cid, t, flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        report.update(state="complete", parameters_unchanged=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        handle.remove()
        frontend.sample_action, model.encode_online = old_sample, old_encode
        module.forward, module._kernel = old_forward, old_kernel


if __name__ == "__main__":
    main()
