"""Audit the physical identity read by S's actual current/reference posterior.

Simulator masks are report-only. Area-pooled mask fractions and bilinear
declared-grid footprints are both reported; neither proves a DINO token has
pure physical-object content. All model inputs/history remain factual.
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


def cpu(x):
    return x.detach().float().cpu().numpy().copy()


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def partition(masks, coordinates, side):
    fractions, footprints = [], []
    for mask in masks:
        raw = torch.as_tensor(mask, dtype=torch.float32, device=coordinates.device)[None]
        fractions.append(F.adaptive_avg_pool2d(raw, (side, side))[0].flatten(-2))
        grid = coordinates.reshape(1, side, side, 2)
        footprints.append(F.grid_sample(raw, grid, align_corners=True, padding_mode="zeros")[0].flatten(-2))
    return {"area": torch.stack(fractions), "declared_grid": torch.stack(footprints)}


def summarize(capture, current_masks, reference_masks):
    output, evidence = capture["progress"]
    p = evidence.posterior
    p.validate(strict=True)
    side = math.isqrt(p.current.shape[-2])
    assert side * side == p.current.shape[-2]
    current = partition(current_masks, p.coordinates, side)
    reference = partition(reference_masks, p.coordinates, side)
    result = {"binding": cpu(evidence.binding.mass)[0].tolist(),
              "binding_null": cpu(evidence.binding.null_mass)[0].tolist(),
              "view_weight": cpu(evidence.view_weight)[0].tolist(),
              "current_null": cpu(p.current_null)[0].tolist(),
              "reference_null": cpu(p.reference_null)[0].tolist(),
              "progress_rms": rms(cpu(output)),
              "robot_delta": cpu(evidence.robot_delta)[0].tolist(),
              "image_delta": cpu(evidence.image_delta)[0].tolist(),
              "laws": {}}
    binding = evidence.binding.mass[0].float()
    view = evidence.view_weight[0].float()
    for name, law, masks in [("G_source", p.source_probability, current),
                             ("current_read", p.current_probability, current),
                             ("reference_read", p.reference_probability, reference)]:
        law = law[0].float()
        real = law.sum(-1)
        parts = {}
        for method, region in masks.items():
            mass = torch.einsum("kcn,con->kco", law, region)
            block = mass.sum(-1)
            read = (mass * binding[:, None, None] * view[..., None]).sum((0, 1))
            parts[method] = {"object_mass_K_C_O": cpu(mass).tolist(),
                             "nonblock_real_mass_K_C": cpu(real - block).tolist(),
                             "target_law_object_mass": cpu(read).tolist(),
                             "target_law_nonblock_real_mass": float(((real - block) * binding[:, None] * view).sum())}
        result["laws"][name] = {"real_mass_K_C": cpu(real).tolist(), "partitions": parts,
            "K_pairwise_probability_rms": rms(cpu(law[:, None] - law[None, :]))}
    names = ["content", "image", "joint", "robot", "status"]
    values = capture["values"]
    weight = capture["module"].output.weight.detach().float()
    width = values[0].shape[-1]
    pieces = [F.linear(v.detach().float(), weight[:, i*width:(i+1)*width]) for i, v in enumerate(values)]
    result["progress_linear_sources"] = {name: {"input_rms": rms(cpu(value)), "projected_rms": rms(cpu(piece))}
                                         for name, value, piece in zip(names, values, pieces)}
    result["linear_source_sum_vs_actual_raw_rms"] = rms(cpu(sum(pieces) - capture["raw"].float()))
    result["actual_raw_rms"] = rms(cpu(capture["raw"]))
    if "kernels" in capture:
        result["kernel_audit"] = []
        for index, kernel in enumerate(capture["kernels"]):
            law = kernel[0].float()
            real, null = law[..., :-1], law[..., -1]
            destination = current["area"] if index == 0 else reference["area"]
            source = current["area"]
            region = torch.einsum("cqn,con->cqo", real, destination)
            count = source.sum(-1).clamp_min(1e-12)
            transition = torch.einsum("caq,cqo->cao", source, region) / count[..., None]
            mean_null = torch.einsum("caq,cq->ca", source, null) / count
            entropy = -(law * law.clamp_min(1e-30).log()).sum(-1) / math.log(law.shape[-1])
            diagonal = real.diagonal(dim1=-2, dim2=-1)
            rank = (real > diagonal[..., None]).float().sum(-1) + 1
            source_features = F.normalize(p.current[0].detach().float(), dim=-1)
            destination_features = F.normalize((p.current if index == 0 else p.reference)[0].detach().float(), dim=-1)
            cosine = source_features @ destination_features.transpose(-2, -1)
            order = cosine.argsort(dim=-1, descending=True)
            sorted_mass = real.sort(dim=-1, descending=True).values
            # Preserve each row's exact real probability multiset and null.
            # This is a ranking-only diagnostic, not a trained replacement.
            reranked = torch.zeros_like(real).scatter(-1, order, sorted_mass)
            reranked_region = torch.einsum("cqn,con->cqo", reranked, destination)
            reranked_transition = torch.einsum("caq,cqo->cao", source, reranked_region) / count[..., None]
            result["kernel_audit"].append({"destination": "current" if index == 0 else "reference",
                "object_area_C_O": cpu(source.sum(-1)).tolist(),
                "object_to_object_mass_C_O_O": cpu(transition).tolist(),
                "object_to_null_C_O": cpu(mean_null).tolist(),
                "same_histogram_frozen_cosine_object_to_object_mass": cpu(reranked_transition).tolist(),
                "mean_entropy_C": cpu(entropy.mean(-1)).tolist(),
                "max_probability_mean_C": cpu(real.max(-1).values.mean(-1)).tolist(),
                "diagonal_probability_mean_C": cpu(diagonal.mean(-1)).tolist(),
                "diagonal_rank_mean_C": cpu(rank.mean(-1)).tolist(),
                "diagonal_top1_fraction_C": cpu((rank == 1).float().mean(-1)).tolist(),
                "same_histogram_max_abs_error": float((reranked.sort(-1).values - real.sort(-1).values).abs().max())})
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ["checkpoint", "plan", "masks", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kernel-audit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"), seed=0)
    model = policy.bundle.model
    module = model.intent.organizer.instruction_progress
    capture = {"module": module}
    original_kernel = module._kernel
    if args.kernel_audit:
        capture["kernels"] = []
        def record_kernel(*a, **kw):
            value = original_kernel(*a, **kw)
            capture["kernels"].append(value)
            return value
        module._kernel = record_kernel
    handles = [module.register_forward_hook(lambda m, a, out: capture.__setitem__("progress", out)),
               module.values.register_forward_hook(lambda m, a, out: capture.__setitem__("values", out)),
               module.output.register_forward_hook(lambda m, a, out: capture.__setitem__("raw", out))]
    versions = {n: p._version for n, p in model.named_parameters()}
    report = {"checkpoint_sha256": policy.bundle.checkpoint_sha256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": __doc__, "records": []}
    windows = [(1, [24, 40]), (3, [40]), (5, [24]), (7, [40]), (11, [24]),
               (10, [136, 352]), (17, [120, 128, 136, 144, 184]), (18, [32])]
    if args.kernel_audit:
        windows = [(1, [40]), (3, [40]), (7, [40]), (17, [120, 136])]
    try:
        plan = json.loads(args.plan.read_text())
        for cid, selected in windows:
            row = next(r for r in plan if r["case_id"] == cid)
            case = Path(row["case"])
            with np.load(case / "trajectory.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in ["rgb_static", "rgb_gripper", "robot_obs", "executed", "raw_chunks"]}
            def load_masks(t):
                with np.load(args.masks / case.name / f"state_{t:03d}.npz", allow_pickle=False) as z:
                    return [z[k + "_masks"].copy() for k in ["top", "wrist"]], z["object_names"].tolist(), json.loads(str(z["metadata_json"]))
            reference_masks, names, reference_meta = load_masks(0)
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
                if args.kernel_audit:
                    capture["kernels"].clear()
                action, _ = policy.act_with_input(history.snapshot(), row["instruction"])
                if args.kernel_audit:
                    assert len(capture["kernels"]) == 2
                masks, current_names, current_meta = load_masks(t)
                assert names == current_names
                record = {"case_id": cid, "state": t, "target": row["target"], "objects": names,
                          "recorded_arm_rmse": rms(action[:8, :6] - data["raw_chunks"][t//8, :8, :6]),
                          "recorded_gripper_mismatches": int(np.sum(action[:8, 6] != data["raw_chunks"][t//8, :8, 6])),
                          "mask_reference_metadata": reference_meta, "mask_current_metadata": current_meta,
                          **summarize(capture, masks, reference_masks)}
                report["records"].append(record)
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("CORRESPONDENCE", cid, t, flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        report.update(state="complete", parameters_unchanged=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        module._kernel = original_kernel
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
