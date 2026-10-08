"""Spatial measurements along all 18 archived autonomous failures.

Replay the complete acknowledged action stream and verify native robot/object
states against the full NPZ. Inspect approach, closing plan (or step48 if never
closed), step160 and step392. Preserve episode-start reference and advance the
actual policy RNG once per replan; compare queried actions with saved chunks.
This is deterministic replay and attribution, not a new benchmark policy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from clearvla.simulation.admission import STACKCUBE_INSTRUCTION
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from probe_maniskill_failure import load_policy
from probe_maniskill_spatial_grounding import SpatialCapture,simulator_labels,region_statistics
from record_maniskill_npz import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","checkpoint","dino","output","rollouts"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);rows=[];episodes=[]
    policy=load_policy(args);capture=SpatialCapture(policy);env=ManiSkillStackCubeEnv()
    try:
        with torch.no_grad():
            for seed in range(5000000,5000018):
                folder=next(args.rollouts.glob(f"episode_*_seed_{seed}"))
                with np.load(folder/"trajectory.npz") as z:
                    actions=z["executed"].copy();state=z["robot_obs_trajectory"].copy()
                    positions=z["object_positions"].copy();chunks=z["raw_chunks"].copy()
                closes=np.flatnonzero((actions[:,-1]<0)&(state[:-1,-1]>0))
                close=int(closes[0]) if len(closes) else None
                closing_plan=(close//8)*8 if close is not None else 48
                selected={16:"approach",closing_plan:"closing_plan" if close is not None else "no_close_48",
                          160:"middle_failure",392:"late_failure"}
                current=env.reset(seed=seed);history=CausalHistory();history.reset(current.observation);policy.reset()
                state_error=0.;object_error=0.;action_error=0.
                for t in range(401):
                    state_error=max(state_error,float(np.max(np.abs(current.observation.state-state[t,:7]))))
                    actual=np.array([current.evaluation.metrics["telemetry_cube_a_pose"][:3],
                                     current.evaluation.metrics["telemetry_cube_b_pose"][:3]])
                    object_error=max(object_error,float(np.max(np.abs(actual-positions[t,:2]))))
                    if t==0:
                        initial=policy.act(history.snapshot(),STACKCUBE_INSTRUCTION)
                        action_error=float(np.max(np.abs(initial-chunks[0])))
                    elif t in selected:
                        masks,poses=simulator_labels(env)
                        action,online,tensors=capture.infer_spatial(history.snapshot())
                        tensors.update(masks=masks,object_positions=poses,robot_state=current.observation.state,
                                       rgb=np.stack(list(current.observation.rgb.values())))
                        label=f"seed_{seed}_t{t:03d}_{selected[t]}"
                        np.savez_compressed(args.output/(label+".npz"),**tensors)
                        cv2.imwrite(str(args.output/(label+".png")),cv2.cvtColor(
                            np.concatenate(list(current.observation.rgb.values()),axis=1),cv2.COLOR_RGB2BGR))
                        difference=float(np.max(np.abs(action-chunks[t//8])))
                        row={"label":label,"seed":seed,"step":t,"stage":selected[t],"first_close":close,
                             "source_episode":str(folder),"action_max_difference_from_archive":difference,
                             "robot_max_difference_so_far":state_error,"object_max_difference_so_far":object_error,
                             "visible_pixels":masks.sum((-2,-1)).tolist(),"binding":tensors["binding_probability"].tolist(),
                             "first8_mean_xyz":action[:8,:3].mean(0).tolist(),
                             "first8_gripper":action[:8,-1].tolist()}
                        action_error=max(action_error,difference)
                        for key in ("prebinder_density","g_density","p1_center_density","p1_micro_density"):
                            row[key]=region_statistics(tensors[key],masks)
                        rows.append(row);print(json.dumps({k:row[k] for k in ("label","action_max_difference_from_archive","robot_max_difference_so_far","object_max_difference_so_far")}),flush=True)
                        atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows,episodes=episodes))
                    elif t%8==0 and t<400:
                        policy.bundle.model.outlet_adapter.sample_noise(
                            1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                    if t<400:
                        current=env.step(actions[t]);history.append(actions[t],current.observation)
                episodes.append(dict(seed=seed,steps=400,robot_max_error=state_error,object_max_error=object_error,
                                     action_max_error=action_error,source=str(folder)))
                atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows,episodes=episodes))
    finally:env.close()
    atomic_json(args.output/"provenance.json",policy.deployment_health())

if __name__=="__main__":main()

