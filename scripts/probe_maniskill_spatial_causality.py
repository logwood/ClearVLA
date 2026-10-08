"""Matched rendered-scene and named-cache interventions; diagnostics only.

Hybrid cameras/caches are deliberately inconsistent causal interventions, never
a deployed policy or benchmark improvement. Complete 2^3 factorials preserve
all top (G/S/W) fields together, P1 together, and CT source together. Real robot,
history, instruction and initial action noise remain identical.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from clearvla.mainline.runtime.sampling import sample_refined_cached_action
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from probe_maniskill_failure import Capture, array, load_policy, native
from record_maniskill_npz import atomic_json


class CacheCapture(Capture):
    def infer_cache(self, history):
        model=self.policy.bundle.model
        encode=model.encode_online
        def wrapped(*args,**kwargs):
            result=encode(*args,**kwargs);self.cache=result[0]
            return result
        with patch.object(model,"encode_online",wrapped):
            action,online,tensors=self.infer(history)
        return action,online,tensors,self.cache


def physical_history(env,moved,xy):
    import sapien
    reset=env.reset(seed=5000015)
    n=env._env.unwrapped
    actor=n.cubeA if moved=="red" else n.cubeB
    pose=actor.pose.raw_pose[0].cpu().numpy().copy()
    pose[:3]=[*xy,.02]
    actor.set_pose(sapien.Pose(p=pose[:3],q=pose[3:]))
    obs=env._observation(n.get_obs())
    np.testing.assert_allclose(obs.state,reset.observation.state,atol=1e-6,rtol=0)
    h=CausalHistory();h.reset(obs)
    return h.snapshot()


def infer(cap,h):
    cap.policy.reset()
    a,o,t,c=cap.infer_cache(h)
    return a,o,t,c,cap.sample.initial_physical_noise.clone()


def run(args):
    policy=load_policy(args);cap=CacheCapture(policy);env=ManiSkillStackCubeEnv()
    rows=[]
    try:
        pairs=[("x_yneg",(-.10,-.18),(.10,-.18)),
               ("x_ypos",(-.10,.18),(.10,.18)),("y_xzero",(0.,-.08),(0.,.08))]
        for moved in ("red","green"):
            for pair,xy_a,xy_b in pairs:
                label=f"{moved}_{pair}"
                ha=physical_history(env,moved,xy_a)
                hb=physical_history(env,moved,xy_b)
                a,oa,ta,ca,noise=infer(cap,ha)
                b,ob,tb,cb,noise_b=infer(cap,hb)
                torch.testing.assert_close(noise,noise_b,atol=0,rtol=0)
                np.testing.assert_array_equal(ha.state,hb.state)
                result={"a":a,"b":b,"noise":array(noise)}
                for bits in range(8):
                    # Keep top and its exact current-state/reference owners
                    # together. Their values are matched, but tensor identity
                    # is also part of the cache admission contract.
                    base=cb if bits&1 else ca
                    cache=replace(base,
                        factual_dock=cb.factual_dock if bits&2 else ca.factual_dock,
                        transition_source=cb.transition_source if bits&4 else ca.transition_source)
                    object.__setattr__(cache,"_seed_context",base.seed_context)
                    sample=sample_refined_cached_action(policy.bundle.model,cache,policy.bundle.config,
                                                       initial_physical_noise=noise)
                    action=native(sample,policy)
                    result[f"factorial_{bits:03b}"]=action
                np.testing.assert_allclose(result["factorial_000"],a,atol=.002,rtol=0)
                np.testing.assert_allclose(result["factorial_111"],b,atol=.002,rtol=0)
                for camera in ("top","wrist"):
                    rgb=dict(ha.rgb_history);rgb[camera]=hb.rgb_history[camera]
                    h=replace(ha,rgb_history=rgb)
                    result[f"donor_{camera}"]=infer(cap,h)[0]
                for key in ("g_content","g_geometry","g_camera_coordinates","binding_probability",
                            "p1_detail","p2_geometry","p2_semantic","p3_state_change"):
                    result["a_"+key]=ta[key];result["b_"+key]=tb[key]
                np.savez_compressed(args.output/(label+".npz"),**result)
                cv2.imwrite(str(args.output/(label+".png")),cv2.cvtColor(np.concatenate(
                    [np.concatenate([h.rgb_history[c][-1] for c in ("top","wrist")],axis=1)
                     for h in (ha,hb)],axis=0),cv2.COLOR_RGB2BGR))
                row={"label":label,"moved":moved,"a_xy":xy_a,"b_xy":xy_b,
                     "mean8":{k:v[:8,:3].mean(0).tolist() for k,v in result.items()
                              if k in ("a","b","donor_top","donor_wrist") or k.startswith("factorial_")},
                     "identity_a_max_error":float(np.max(np.abs(result["factorial_000"]-a))),
                     "identity_b_max_error":float(np.max(np.abs(result["factorial_111"]-b)))}
                rows.append(row);print(json.dumps(row),flush=True)
                atomic_json(args.output/"summary.json",dict(note=__doc__,rows=rows))
                del ca,cb
    finally:env.close()
    atomic_json(args.output/"provenance.json",policy.deployment_health())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","checkpoint","dino","output"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    with torch.no_grad():run(args)
if __name__=="__main__":main()

