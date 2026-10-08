"""Exchange the two physical cube poses, preserving occupied geometry.

This rerenders real RGB with red/green identities exchanged, fixed robot and
seed-zero noise. Segmentation is an audit label only. Frozen single-observation
queries are not closed-loop success results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import sapien
from probe_maniskill_failure import load_policy
from probe_maniskill_spatial_grounding import SpatialCapture,save_case,simulator_labels
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from clearvla.simulation.history import CausalHistory
from record_maniskill_npz import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","checkpoint","dino","output","baseline"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);rows=[]
    policy=load_policy(args);capture=SpatialCapture(policy);env=ManiSkillStackCubeEnv()
    try:
        with torch.no_grad():
            for seed in range(5000000,5000018):
                reset=env.reset(seed=seed);n=env._env.unwrapped
                with np.load(args.baseline/f"seed_{seed}.npz") as z:
                    baseline={k:z[k].copy() for k in ("action","object_positions","rgb","robot_state","initial_noise","binding_probability")}
                np.testing.assert_array_equal(np.stack(list(reset.observation.rgb.values())),baseline["rgb"])
                np.testing.assert_array_equal(reset.observation.state,baseline["robot_state"])
                pose_a=n.cubeA.pose.raw_pose[0].cpu().numpy().copy()
                pose_b=n.cubeB.pose.raw_pose[0].cpu().numpy().copy()
                n.cubeA.set_pose(sapien.Pose(p=pose_b[:3],q=pose_b[3:]))
                n.cubeB.set_pose(sapien.Pose(p=pose_a[:3],q=pose_a[3:]))
                obs=env._observation(n.get_obs())
                np.testing.assert_array_equal(obs.state,reset.observation.state)
                history=CausalHistory();history.reset(obs)
                masks,poses=simulator_labels(env)
                row=save_case(args,capture,f"seed_{seed}_swap",history.snapshot(),masks,poses,dict(seed=seed))
                np.testing.assert_array_equal(capture.sample.initial_physical_noise.cpu().numpy(),baseline["initial_noise"])
                original=baseline["action"][:8,:2].mean(0)
                swapped=capture.tensors["action"][:8,:2].mean(0)
                displacement=poses[0,:2]-baseline["object_positions"][0,:2]
                row.update(original_mean8_xy=original.tolist(),swapped_mean8_xy=swapped.tolist(),
                    red_displacement_xy=displacement.tolist(),
                    action_delta_xy=(swapped-original).tolist(),
                    response_projection=float(np.dot(swapped-original,displacement)/max(np.dot(displacement,displacement),1e-12)),
                    action_change_norm=float(np.linalg.norm(swapped-original)),
                    binding_max_change=float(np.max(np.abs(capture.tensors["binding_probability"]-baseline["binding_probability"]))))
                rows.append(row);atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows))
    finally:env.close()
    atomic_json(args.output/"provenance.json",policy.deployment_health())

if __name__=="__main__":main()

