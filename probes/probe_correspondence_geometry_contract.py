"""Known feature-chart motion and source-logit decomposition, without updates.

Feature permutations are algebraic chart tests, not rendered physical rollouts.
The exact inverse permutation is diagnostic truth only. No changed features or
oracle correspondence are sent to the policy or used as a training objective.
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


def rms(t):
    return float(t.detach().float().square().mean().sqrt())


def arr(t):
    return t.detach().float().cpu().tolist()


def analyze(reader, content, grid):
    content = content.detach().float()
    cameras, patches, channels = content.shape[1:]
    side = math.isqrt(patches)
    assert side * side == patches
    observed = torch.ones(content.shape[:-1], device=content.device, dtype=torch.bool)

    def projected(x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            qc = reader.query(x)
            qp = reader.query_position(grid)[None, None].expand_as(qc)
            kc = reader.key(x)
            kp = reader.key_position(grid)[None, None].expand_as(kc)
        return qc, qp, kc, kp

    qc, qp, kc, kp = projected(content)
    q, k = qc + qp, kc + kp
    def logits(q, k):
        return torch.einsum("bcqh,bcnh->bcqn", q, k) * reader.scale
    full = logits(q, k)
    prior = logits(q.mean(-2, keepdim=True), k)
    interaction = full - prior
    key_center = lambda x: x - x.mean(-1, keepdim=True)
    result = {"self_logit_sources": {}, "shifts": []}
    for name, term in [("full", full), ("query_mean_destination_prior", prior),
                       ("query_dependent_interaction", interaction),
                       ("content_content", logits(qc, kc)), ("content_position", logits(qc, kp)),
                       ("position_content", logits(qp, kc)), ("position_position", logits(qp, kp))]:
        result["self_logit_sources"][name] = [rms(key_center(term)[0, c]) for c in range(cameras)]
    result["query_mean_explained_logit_variance_fraction"] = [
        float(key_center(prior)[0, c].square().mean() / key_center(full)[0, c].square().mean().clamp_min(1e-30))
        for c in range(cameras)]
    base_index = torch.arange(patches, device=content.device).reshape(side, side)
    pixel = torch.stack(torch.meshgrid(torch.arange(side, device=content.device), torch.arange(side, device=content.device), indexing="ij"), -1).reshape(-1, 2)
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (2, 0), (0, 1), (0, -1), (0, 2)]:
        permutation = torch.roll(base_index, shifts=(dy, dx), dims=(0, 1)).flatten()
        shifted = content[:, :, permutation]
        query_c, query_p, _, _ = projected(shifted)
        query = query_c + query_p
        with torch.autocast(device_type=content.device.type, enabled=False):
            now = reader._kernel(query, shifted, observed, grid)
            start = reader._kernel(query, content, observed, grid)
        reference_index = permutation
        # Ignore wrapped query cells when comparing literal geometric shifts.
        expected = grid - grid[reference_index]
        valid = ((pixel[:, 1] - dx >= 0) & (pixel[:, 1] - dx < side)
                 & (pixel[:, 0] - dy >= 0) & (pixel[:, 0] - dy < side))
        pair = {"dx_cells": dx, "dy_cells": dy, "queries_without_wrap": int(valid.sum()),
                "expected_xy": arr(expected[valid].mean(0)), "variants": {}}
        variants = {"learned": (now, start)}
        # Remove only the query-independent destination prior, keeping the
        # original null and total real mass. No temperature/concentration match.
        def remove_prior(probability, target):
            qq = query - query.mean(-2, keepdim=True)
            _, _, kk, pp = projected(target)
            score = logits(qq, kk + pp)
            return torch.cat((score.softmax(-1) * probability[..., :-1].sum(-1, keepdim=True), probability[..., -1:]), -1)
        variants["query_mean_removed"] = (remove_prior(now, shifted), remove_prior(start, content))
        def fixed(probability, target):
            cosine = F.normalize(shifted, dim=-1) @ F.normalize(target, dim=-1).transpose(-2, -1)
            real = (cosine * math.sqrt(channels)).softmax(-1) * probability[..., :-1].sum(-1, keepdim=True)
            return torch.cat((real, probability[..., -1:]), -1)
        variants["fixed_cosine_same_null"] = (fixed(now, shifted), fixed(start, content))
        for name, (a, b) in variants.items():
            literal_delta = (a[..., :-1] - b[..., :-1]) @ grid
            conditional_delta = ((a[..., :-1] / a[..., :-1].sum(-1, keepdim=True))
                                 - (b[..., :-1] / b[..., :-1].sum(-1, keepdim=True))) @ grid
            matched = b[..., :-1].gather(-1, reference_index[None, None, :, None].expand(1, cameras, -1, 1))[..., 0]
            rank = 1 + (b[..., :-1] > matched[..., None]).sum(-1)
            top1 = b[..., :-1].argmax(-1) == reference_index[None, None]
            rows = []
            for c in range(cameras):
                estimate = literal_delta[0, c, valid]
                oracle = expected[valid]
                item = {"mean_xy": arr(estimate.mean(0)), "conditional_mean_xy": arr(conditional_delta[0, c, valid].mean(0)),
                        "xy_rmse": rms(estimate - oracle), "correct_location_probability_mean": float(matched[0, c, valid].mean()),
                        "correct_location_top1_fraction": float(top1[0, c, valid].float().mean()),
                        "correct_location_rank_mean": float(rank[0, c, valid].float().mean())}
                if dx or dy:
                    direction = oracle.mean(0)
                    gain = (estimate * direction).sum(-1) / direction.square().sum()
                    item.update(displacement_gain_mean=float(gain.mean()), wrong_sign_fraction=float((gain < 0).float().mean()))
                rows.append(item)
            pair["variants"][name] = rows
        result["shifts"].append(pair)
    return result


def main():
    ap = argparse.ArgumentParser()
    for key in ["checkpoint", "plan", "output"]:
        ap.add_argument("--" + key, required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"), seed=0)
    model = policy.bundle.model
    reader = model.intent.organizer.instruction_progress
    capture = {}
    handle = reader.register_forward_hook(lambda m, a, out: capture.__setitem__("evidence", out[1]))
    versions = {n: p._version for n, p in model.named_parameters()}
    report = {"checkpoint_sha256": policy.bundle.checkpoint_sha256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": __doc__, "records": []}
    try:
        plan = json.loads(args.plan.read_text())
        for cid, state in [(1, 40), (3, 40), (7, 40), (17, 136)]:
            row = next(r for r in plan if r["case_id"] == cid)
            with np.load(Path(row["case"]) / "trajectory.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in ["rgb_static", "rgb_gripper", "robot_obs", "executed", "raw_chunks"]}
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(state + 1):
                previous = np.zeros(7, np.float32) if t == 0 else data["executed"][t - 1]
                obs = calvin_policy_observation({"rgb_obs": {k: data[k][t] for k in ["rgb_static", "rgb_gripper"]}, "robot_obs": data["robot_obs"][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                if t not in [0, state]:
                    if t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                action, _ = policy.act_with_input(history.snapshot(), row["instruction"])
                if t != state:
                    continue
                posterior = capture["evidence"].posterior
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                    result = analyze(reader, posterior.current, posterior.coordinates)
                report["records"].append({"case_id": cid, "state": t,
                    "recorded_arm_rmse": float(np.sqrt(np.mean((action[:8, :6] - data["raw_chunks"][t//8, :8, :6]) ** 2))), **result})
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("GEOMETRY_CONTRACT", cid, t, flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        report.update(state="complete", parameters_unchanged=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
