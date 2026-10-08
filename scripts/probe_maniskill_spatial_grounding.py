"""Frozen ManiSkill spatial audit; actor masks are evaluator-only labels.

Run from the checkpoint's ORIGINAL source closure. The script adds hooks but
never edits model weights/configuration or sends segmentation/poses to policy.
Every query uses the real RGB/state/history path and seed-zero Q5 refinement.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch

import clearvla.mainline.model.grounding as grounding_module
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from clearvla.vision.entity_chart import pushforward_log_to_current_image

from probe_maniskill_failure import Capture, array, h5_history, load_policy
from record_maniskill_npz import atomic_json


def raster_mass(coordinates, mass, side=336):
    """Exact bilinear push of [C,N] mass at [C,N,2] align-corners points."""
    xy = coordinates.float().clamp(-1, 1)
    c = int(xy.shape[0])
    xy = xy.reshape(c, -1, 2)
    values = mass.float().reshape(c, -1)
    if xy.shape[:2] != values.shape or not torch.isfinite(values).all():
        raise ValueError("spatial mass and coordinates differ or are nonfinite")
    x, y = (xy[..., 0] + 1) * ((side-1)/2), (xy[..., 1] + 1) * ((side-1)/2)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x-x0, y-y0
    image = values.new_zeros(c, side*side)
    for ox, oy, w in ((0,0,(1-dx)*(1-dy)), (1,0,dx*(1-dy)),
                      (0,1,(1-dx)*dy), (1,1,dx*dy)):
        index = (y0.long()+oy).clamp_max(side-1)*side + (x0.long()+ox).clamp_max(side-1)
        image.scatter_add_(1, index, values*w)
    torch.testing.assert_close(image.sum(-1), values.sum(-1), rtol=2e-5, atol=2e-6)
    return image.reshape(c, side, side)


def simulator_labels(env):
    """Read same-frame renderer IDs without changing policy observation mode."""
    n = env._env.unwrapped
    ids = [int(n.cubeA.per_scene_id[0]), int(n.cubeB.per_scene_id[0])]
    masks = []
    for camera in ("base_camera", "hand_camera"):
        seg = n._sensors[camera].get_obs(
            rgb=False, depth=False, position=False, segmentation=True
        )["segmentation"][0, ..., 0].cpu().numpy()
        masks.append(np.stack([seg == actor for actor in ids]))
    return np.stack(masks), np.stack([
        n.cubeA.pose.raw_pose[0, :3].cpu().numpy(),
        n.cubeB.pose.raw_pose[0, :3].cpu().numpy(),
    ])


def region_statistics(density, masks):
    """Conditional-on-camera mass; dilation acknowledges sparse source samples."""
    mass = np.asarray(density, dtype=np.float64)
    if mass.ndim == 3:
        mass = mass[None]
    camera_mass = mass.sum((-2,-1))
    conditional = mass / np.maximum(camera_mass[..., None, None], 1e-30)
    result = {"camera_mass": camera_mass.tolist(), "regions": {}}
    for object_index, object_name in enumerate(("red", "green")):
        for radius in (0, 12, 24):
            region = np.stack([
                cv2.dilate(x[object_index].astype(np.uint8),
                           np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
                for x in masks
            ])
            fraction = region.mean((-2,-1))
            read = (conditional * region[None]).sum((-2,-1))
            result["regions"][f"{object_name}_r{radius}"] = {
                "area_fraction": fraction.tolist(),
                "conditional_mass": read.tolist(),
                "enrichment_over_uniform": (read / np.maximum(fraction[None], 1e-30)).tolist(),
            }
    return result


class SpatialCapture(Capture):
    def infer_spatial(self, history):
        model = self.policy.bundle.model
        self.cache = self.state = None
        self.g_source = None
        self.p1_densities = []
        original_encode = model.encode_online
        original_push = grounding_module.pushforward_log_to_current_image
        reader = model.p1.factual_reader
        original_micro = reader._posterior_microgrid_expectation

        def encode(*args, **kwargs):
            result = original_encode(*args, **kwargs)
            self.cache, self.state, _ = result
            return result

        def push(log_mass, supported, spatial, **kwargs):
            if self.g_source is None:
                self.g_source = (log_mass, supported, spatial)
            return original_push(log_mass, supported, spatial, **kwargs)

        def micro(route, fine_logits, candidate_valid, coordinates, radius,
                  rgb, detail, center_rgb, center_detail, *tail, **kwargs):
            result = original_micro(route, fine_logits, candidate_valid, coordinates,
                                    radius, rgb, detail, center_rgb, center_detail,
                                    *tail, **kwargs)
            # Follow the actual nine-cell law, including per-cell boundary masking.
            density = torch.zeros(2, 336, 336, device=route.device)
            center_density = None
            for oy in (-1., 0., 1.):
                for ox in (-1., 0., 1.):
                    xy = coordinates.float() + radius.float()[..., None, None] * coordinates.new_tensor([ox, oy])
                    valid = candidate_valid & (xy.abs() <= 1).all(-1)
                    observed = valid[:, None, None]
                    logits = fine_logits.float().masked_fill(~observed, torch.finfo(torch.float32).min)
                    logits = logits.masked_fill(~observed.any(-1, keepdim=True), 0)
                    weights = logits.softmax(-1) * observed
                    joint = (route.float()[..., None] * weights).sum((0,1,2))
                    d = raster_mass(xy[0], joint)
                    density += d / 9
                    if ox == 0 and oy == 0:
                        center_density = d
            self.p1_densities.append((density, center_density, route.shape[1]*route.shape[2]))
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(model, "encode_online", encode))
            stack.enter_context(patch.object(grounding_module, "pushforward_log_to_current_image", push))
            stack.enter_context(patch.object(reader, "_posterior_microgrid_expectation", micro))
            action, online, tensors = super().infer(history)
        if self.g_source is None or not self.p1_densities:
            raise RuntimeError("expected current-image G and full-posterior P1 path")
        logs, supported, spatial = self.g_source
        image = pushforward_log_to_current_image(logs, supported, spatial, rows=336, columns=336)
        tensors["g_density"] = array(image.normalized((2,3,4))[0][0])
        reconstructed8 = pushforward_log_to_current_image(
            logs, supported, spatial, rows=8, columns=8
        ).normalized((2,3,4))[0]
        torch.testing.assert_close(reconstructed8, self.state.top.facts.object_to_chart, rtol=1e-5, atol=1e-6)
        total_queries = sum(x[2] for x in self.p1_densities)
        tensors["p1_micro_density"] = array(sum(x[0] for x in self.p1_densities)/total_queries)
        tensors["p1_center_density"] = array(sum(x[1] for x in self.p1_densities)/total_queries)
        uniform_source = supported.float()
        uniform_source /= uniform_source.sum((2,3,4,5), keepdim=True).clamp_min(1)
        # Equal local hypotheses, before the dense K binder chooses among them.
        pre = pushforward_log_to_current_image(
            uniform_source.clamp_min(1e-30).log(), supported, spatial, rows=336, columns=336
        ).normalized((2,3,4))[0]
        tensors["prebinder_density"] = array(pre[0, :1])
        f = self.state.top.facts
        tensors["reconstruction_error_map"] = array(
            (f.reconstructed_dino.float()-f.dense_chart.dino_content.float()).square().mean(-1)[0])
        tensors["raw_dino_current"] = array(online.observation.dino_history[0, -1])
        tensors["g_dino_chart"] = array(f.dense_chart.dino_content[0])
        tensors["reconstructed_dino"] = array(f.reconstructed_dino[0])
        tensors["coarse_action"] = array(self.state.top.coarse_action.action_prediction)
        tensors["initial_noise"] = array(self.sample.initial_physical_noise)
        return action, online, tensors


def save_case(args, capture, label, history, masks, poses, metadata):
    capture.policy.reset()
    action, online, tensors = capture.infer_spatial(history)
    tensors.update(masks=masks, object_positions=poses, robot_state=history.state,
                   rgb=np.stack([history.rgb_history[c][-1] for c in ("top","wrist")]))
    np.savez_compressed(args.output / (label+".npz"), **tensors)
    cv2.imwrite(str(args.output/(label+".png")), cv2.cvtColor(
        np.concatenate(list(tensors["rgb"]), axis=1), cv2.COLOR_RGB2BGR))
    row = dict(label=label, **metadata, first8_mean_xyz=action[:8,:3].mean(0).tolist(),
               first8_gripper=action[:8,-1].tolist(), positions=poses.tolist(),
               tcp_xyz=history.state[:3].tolist(), visible_pixels=masks.sum((-2,-1)).tolist(),
               binding=tensors["binding_probability"].tolist())
    for name in ("prebinder_density","g_density","p1_center_density","p1_micro_density"):
        row[name] = region_statistics(tensors[name], masks)
    # Attribution composition only: S's K mass times G's joint image distribution.
    bound = (tensors["g_density"]*tensors["binding_probability"][0,:-1,None,None,None]).sum(0)
    row["binding_composed_density"] = region_statistics(bound, masks)
    print(json.dumps({k:row[k] for k in ("label","first8_mean_xyz","visible_pixels")}),flush=True)
    return row


def run(args):
    policy=load_policy(args)
    capture=SpatialCapture(policy)
    env=ManiSkillStackCubeEnv()
    rows=[]
    try:
        if args.mode == "placements":
            import sapien
            for moved in ("red", "green"):
                for x,y in ((-.10,-.18),(.10,-.18),(-.10,.18),(.10,.18),(0.,-.08),(0.,.08)):
                    reset=env.reset(seed=5000015)
                    n=env._env.unwrapped
                    actor=n.cubeA if moved=="red" else n.cubeB
                    pose=actor.pose.raw_pose[0].cpu().numpy().copy()
                    pose[:3]=[x,y,.02]
                    actor.set_pose(sapien.Pose(p=pose[:3],q=pose[3:]))
                    obs=env._observation(n.get_obs())
                    np.testing.assert_allclose(obs.state,reset.observation.state,atol=1e-6,rtol=0)
                    history=CausalHistory();history.reset(obs)
                    masks,poses=simulator_labels(env)
                    label=f"{moved}_x{x:+.2f}_y{y:+.2f}"
                    row=save_case(args,capture,label,history.snapshot(),masks,poses,dict(moved=moved,xy=[x,y]))
                    rows.append(row)
                    atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows))
        elif args.mode == "resets":
            for seed in range(5000000,5000018):
                reset=env.reset(seed=seed);history=CausalHistory();history.reset(reset.observation)
                masks,poses=simulator_labels(env)
                rows.append(save_case(args,capture,f"seed_{seed}",history.snapshot(),masks,poses,dict(seed=seed)))
                atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows))
        elif args.mode == "experts":
            splits=json.loads((args.data/"prepared/splits.json").read_text())["splits"]
            for split in ("train","val"):
                for name in splits[split][:3]:
                    with h5py.File(args.data/"experts"/(name+".hdf5")) as h:
                        actions=h["action"][:]
                        close=np.flatnonzero((actions[:,-1]<0)&(h["action_state"][:,-1]>0))
                        opening=np.flatnonzero((actions[:,-1]>0)&(h["action_state"][:,-1]<0))
                        c=int(close[0]);o=int(opening[opening>c][0])
                        stages={0:"reset",16:"approach",max(0,c-8):"preclose",c:"close",
                                c+8:"lift",(c+o)//2:"transport",o:"release"}
                        reset=env.reset(seed=int(h.attrs["seed"]))
                        for t in range(max(stages)+1):
                            if t in stages:
                                masks,poses=simulator_labels(env)
                                np.testing.assert_allclose(reset.observation.state,h["state"][t],atol=.001,rtol=0)
                                # Teacher-forced history stays EXACTLY the dataset's,
                                # while replay segmentation supplies audit-only labels.
                                hist=h5_history(h,t)
                                row=save_case(args,capture,f"{split}_{name}_{stages[t]}",
                                              hist,masks,poses,dict(split=split,episode=name,step=t,stage=stages[t]))
                                row["arm_rmse_first8"]=float(np.sqrt(np.mean(
                                    (capture.tensors["action"][:8,:6]-actions[t:t+8,:6])**2)))
                                rows.append(row)
                                atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows))
                            if t<max(stages):reset=env.step(actions[t])
        else:
            raise ValueError(args.mode)
    finally:
        env.close()
    atomic_json(args.output/"provenance.json",policy.deployment_health())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("placements","resets","experts"))
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--dino",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    with torch.no_grad():run(args)


if __name__=="__main__":
    main()

