"""Factual G reconstruction VJP, including assignment and value branches.

No optimizer updates. Masks partition reports only. Captured local facts are
held fixed, so this qualifies the full grounder, not its upstream producer or
the complete multitask/Adam training update. FP32 rerun fidelity is reported.
"""
from pathlib import Path
from dataclasses import fields, is_dataclass, replace
import argparse
import copy
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from probes.probe_source_delta_consumer import observe_return


def clone_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float() if x.is_floating_point() else x.detach().clone()
    if is_dataclass(x):
        return replace(x, **{f.name: clone_tree(getattr(x, f.name)) for f in fields(x)})
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone_tree(v) for v in x)
    return x


def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def vector_grad(loss, parameters, retain_graph=True):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, values))


def dot(a, b):
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b))


def norm(a):
    return dot(a, a).sqrt()


def report_weights(owner, masks):
    return [F.adaptive_avg_pool2d(torch.as_tensor(m, device=owner.device, dtype=torch.float32)[None], owner.shape[-2:])[0] for m in masks]


def owner_metrics(owner, observed, weights):
    joint = (owner * observed[:, None]).sum((0, 3, 4)) / observed.sum()
    pk = joint.sum(1, keepdim=True)
    pc = joint.sum(0, keepdim=True)
    mi = (joint * (joint.clamp_min(1e-30).log() - (pk * pc).clamp_min(1e-30).log())).sum()
    means = (owner * observed[:, None]).sum((-1, -2)) / observed.sum((-1, -2))[:, None]
    # Indices fixed by the preceding four-window evidence, not optimized here.
    metrics = {"camera_owner_mi_nats": mi, "top_K1_wrist_K3_mean_owner": (means[0, 0, 0] + means[0, 2, 1]) / 2}
    for c, w in enumerate(weights):
        active = [j for j in range(len(w)) if float(w[j].sum()) > 1e-6]
        region = [(owner[0, :, c] * w[j]).sum((-1, -2)) / w[j].sum() for j in active]
        if len(region) > 1:
            metrics[["top", "wrist"][c] + "_physical_region_owner_squared_separation"] = torch.stack([(a-b).square().sum() for i, a in enumerate(region) for b in region[i+1:]]).mean()
    return metrics


def rank_budget(local, weights, names):
    owner, value = local["reconstruction_owner"].detach(), local["content"].detach()[0]
    target, pos = local["target_content"].detach()[0], local["decoded_position"].detach()[0]
    observed = local["observed"].detach()[0, ..., 0]
    residual_target = target-pos
    camera_means = (residual_target * observed[..., None]).sum((1, 2)) / observed.sum((1, 2))[:, None]
    regions = []
    for c, weights_c in enumerate(weights):
        for j, name in enumerate(names):
            w = weights_c[j] * observed[c]
            if float(w.sum()) <= 1e-6:
                continue
            mu = (residual_target[c] * w[..., None]).sum((0, 1)) / w.sum()
            cost = (value-mu).square().mean(-1)
            best, runner = cost.argsort()[:2].tolist()
            cam_cost = (value-camera_means[c]).square().mean(-1)
            margin = cost[runner]-cost[best]
            camera_margin = cam_cost[runner]-cam_cost[best]
            object_correction = -2*((value[runner]-value[best])*(mu-camera_means[c])).mean()
            if not torch.allclose(margin, camera_margin+object_correction, atol=1e-6, rtol=1e-4):
                raise AssertionError("per-K ranking decomposition failed")
            regions.append({"camera": ["top", "wrist"][c], "object": name, "best_K": best+1, "runner_K": runner+1,
                "margin": float(margin), "camera_mean_margin": float(camera_margin), "object_specific_correction": float(object_correction),
                "object_from_camera_mean_rms": float((mu-camera_means[c]).square().mean().sqrt())})
    shape = local["candidate_shape"]
    read_by_camera = local["read"].detach().reshape(1, owner.shape[1], *shape).sum(tuple(range(3, 3+len(shape)-1)))
    return {"regions": regions, "between_camera_residual_mean_rms": float((camera_means[0]-camera_means[1]).square().mean().sqrt()),
        "global_K_read_camera_mass": read_by_camera[0].cpu().tolist()}


def audit(grounder, captured, masks, names):
    module = copy.deepcopy(grounder).float().eval().requires_grad_(True)
    # Copy the class method rather than the production instance's observer.
    module.forward = type(module).forward.__get__(module, type(module))
    local_facts = clone_tree(captured["local_facts"])
    with torch.autocast(device_type="cuda", enabled=False):
        output, local = observe_return(module.forward, (local_facts,), {"collect_diagnostics": False})
    parameters = tuple(module.parameters())
    loss = local["reconstruction_error"]
    terminals = tuple(local[k] for k in ("reconstruction_owner", "content", "decoded_position"))
    adjoints = torch.autograd.grad(loss, terminals, retain_graph=True)
    branches = {name: (terminal*adjoint.detach()).sum() for name, terminal, adjoint in zip(("assignment", "content", "position"), terminals, adjoints)}
    gradients = {"full": vector_grad(loss, parameters)}
    gradients.update({name: vector_grad(scalar, parameters) for name, scalar in branches.items()})
    summed = tuple(sum(parts) for parts in zip(*(gradients[k] for k in branches)))
    closure = norm(tuple(a-b for a,b in zip(gradients["full"], summed))) / norm(gradients["full"]).clamp_min(1e-30)
    if float(closure) > 1e-4:
        raise AssertionError(f"grounder reconstruction VJP branches fail closure: {float(closure)}")
    owner = local["reconstruction_owner"]
    observed = local["observed"][..., 0]
    # Test the tempting one-line view-value replacement as an isolated
    # objective diagnostic. It is never used by the factual policy or trainer.
    camera_content = output[0].camera_content
    if camera_content is None:
        raise ValueError("requires declared per-camera observation values")
    slot_residual = local["content"] - local["aggregated_content"]
    alternatives = {}
    for name, value in (("view_values", camera_content), ("view_values_plus_slot_residual", camera_content+slot_residual[:, :, None])):
        prediction = torch.einsum("bkcyx,bkcd->bcyxd", owner, value) + owner.sum(1)[..., None]*local["decoded_position"]
        alternative = ((prediction-local["target_content"]).square().mean(-1)*observed).sum()/observed.sum()
        alternatives[name] = float(alternative.detach())
        gradients[name] = vector_grad(alternative, parameters)
    weights = report_weights(owner, masks)
    metrics = owner_metrics(owner, observed, weights)
    rates = {}
    for name, metric in metrics.items():
        mg = vector_grad(metric, parameters)
        rates[name] = {"value": float(metric.detach()), "per_unit_parameter_descent": {branch: float(-dot(mg, grad)/norm(grad).clamp_min(1e-30)) for branch, grad in gradients.items()}}
    # A finite ray check uses an isolated module and restores every parameter.
    # This is unpreconditioned raw reconstruction descent, not a training step.
    originals = tuple(p.detach().clone() for p in parameters)
    radius = float(norm(originals))*1e-5
    gnorm = float(norm(gradients["full"]))
    ray = {}
    for sign in (-1, 1):
        with torch.no_grad():
            for p, base, grad in zip(parameters, originals, gradients["full"]):
                p.copy_(base-sign*radius/gnorm*grad)
            with torch.autocast(device_type="cuda", enabled=False):
                _, probe = observe_return(module.forward, (local_facts,), {"collect_diagnostics": False})
            ray[str(sign)] = {"loss": float(probe["reconstruction_error"]), **{k: float(v) for k,v in owner_metrics(probe["reconstruction_owner"], observed, weights).items()}}
    with torch.no_grad():
        for p, base in zip(parameters, originals):
            p.copy_(base)
    result = {"source_loss": float(captured["reconstruction_error"]), "fp32_loss": float(loss.detach()),
        "fp32_vs_source_owner_rms": float((owner.detach()-captured["reconstruction_owner"].float()).square().mean().sqrt()),
        "fp32_vs_source_content_rms": float((local["content"].detach()-captured["content"].float()).square().mean().sqrt()),
        "branch_gradient_l2": {k: float(norm(v)) for k,v in gradients.items()}, "alternative_objective_loss": alternatives, "gradient_sum_relative_error": float(closure),
        "assignment_content_gradient_cosine": float(dot(gradients["assignment"], gradients["content"])/(norm(gradients["assignment"])*norm(gradients["content"])).clamp_min(1e-30)),
        "metrics": rates, "finite_ray": {"parameter_l2_radius": radius, "relative_radius": 1e-5, "sign_plus_is_descent": True, "values": ray},
        "rank_budget": rank_budget(local, weights, names), "scope": "full grounder parameters; factual upstream local facts fixed; raw reconstruction only; not multitask/Adam update"}
    return result


def main():
    ap=argparse.ArgumentParser()
    for name in ("checkpoint", "plan", "masks", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    a=ap.parse_args(); a.output.mkdir(exist_ok=False); torch.set_num_threads(4)
    policy=ClearVLACheckpointPolicy(a.checkpoint, device=torch.device("cuda:0"),
        t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
        dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"), seed=0)
    model=policy.bundle.model; grounder=model.grounding.grounder; old=grounder.forward; sources=[]
    def capture(*args, **kw):
        out, local=observe_return(old, args, kw)
        sources.append({k: local[k] for k in ("local_facts", "reconstruction_owner", "content", "reconstruction_error")})
        return out
    grounder.forward=capture; rows=[]
    try:
        plan=json.loads(a.plan.read_text())
        for cid,state in ((1,24),(5,24),(11,24),(17,136)):
            item=next(r for r in plan if r["case_id"]==cid); case=Path(item["case"])
            with np.load(case/"trajectory.npz",allow_pickle=False) as z:
                data={k:z[k] for k in ("rgb_static","rgb_gripper","robot_obs","executed","raw_chunks")}
            policy.reset(); history=CausalHistory(executed_world=True)
            for t in range(state+1):
                command=np.zeros(7,np.float32) if t==0 else data["executed"][t-1]
                observation=calvin_policy_observation({"rgb_obs":{k:data[k][t] for k in ("rgb_static","rgb_gripper")},"robot_obs":data["robot_obs"][t]},command)
                if t==0: history.reset(observation,reset_action=command)
                else: history.append(command,observation)
                if t not in (0,state):
                    if t%8==0: model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                    continue
                sources.clear(); action,_=policy.act_with_input(history.snapshot(),item["instruction"])
                if t==state:
                    with np.load(a.masks/case.name/f"state_{t:03d}.npz",allow_pickle=False) as z:
                        masks=[z[k+"_masks"].copy() for k in ("top","wrist")]; names=z["object_names"].tolist()
                    # Remove observer before copying the module (bound closures).
                    grounder.forward=old
                    with torch.inference_mode(False),torch.enable_grad(): result=audit(grounder,sources[0],masks,names)
                    grounder.forward=capture
                    rows.append({"case_id":cid,"state":t,"recorded_arm_rmse":float(np.sqrt(np.mean((action[:8,:6]-data["raw_chunks"][t//8,:8,:6])**2))),**result})
                    dump(a.output/"results.json",rows); print(cid,t,"complete",flush=True)
        dump(a.output/"complete.json",{"checkpoint_sha256":policy.bundle.checkpoint_sha256,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"windows":len(rows),
            "policy_parameters_unchanged":True,"scope":"four factual windows; isolated grounder copies; physical masks for reports only; no optimizer/production/loss changes"})
    finally:
        grounder.forward=old


if __name__=="__main__": main()
