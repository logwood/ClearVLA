"""Diagnostic linear readouts of frozen reset representations.

Fit only on 59 training episodes, choose ridge strength on seven validation
episodes, report seven untouched test episodes. Privileged object positions are
labels for this diagnostic; the policy and its deployed inputs remain unchanged.
A readable representation is not evidence that the trained policy uses it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from probe_maniskill_failure import Capture,array,h5_history,load_policy
from record_maniskill_npz import atomic_json


def ridge_readouts(features,target,splits):
    train=splits=="train";val=splits=="val";test=splits=="test"
    rows={}
    for name,x in features.items():
        x=x.astype(np.float64)
        # Per-feature centering; one global train-only RMS avoids amplifying
        # near-constant coordinates/noise. Kernel scaling controls dimension.
        mean=x[train].mean(0);x=x-mean
        scale=np.sqrt(np.mean(x[train]**2))
        x=x/max(scale,1e-12)
        k=x@x[train].T/max(x.shape[1],1)
        ky=k[train]
        ym=target[train].mean(0);y=target[train]-ym
        choices=[]
        for alpha in (1e-5,1e-4,1e-3,.01,.1,1.,10.,100.):
            coef=np.linalg.solve(ky+alpha*np.eye(len(ky)),y)
            pred=k@coef+ym
            error=np.linalg.norm((pred-target).reshape(-1,2,2),axis=-1)
            choices.append((float(np.sqrt(np.mean((pred[val]-target[val])**2))),alpha,pred,error))
        best=min(choices,key=lambda z:z[0]);_,alpha,pred,error=best
        rows[name]={"alpha_selected_on_val":alpha,"features":x.shape[1],
            "train_mean_object_xy_error_cm":(100*error[train].mean(0)).tolist(),
            "val_mean_object_xy_error_cm":(100*error[val].mean(0)).tolist(),
            "test_mean_object_xy_error_cm":(100*error[test].mean(0)).tolist(),
            "test_per_episode_object_xy_error_cm":(100*error[test]).tolist(),
            "predictions":pred.tolist(),
            "val_component_rmse_by_alpha":{str(z[1]):z[0] for z in choices}}
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","checkpoint","dino","output","contracts"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);policy=load_policy(args);capture=Capture(policy)
    contracts={x["episode"]:x for x in json.loads(args.contracts.read_text())["episodes"]}
    split_names=json.loads((args.data/"prepared/splits.json").read_text())["splits"]
    features={k:[] for k in ("state_only","dino_global","dino_spatial","g_content","g_geometry","g_all","p1_detail")}
    targets=[];splits=[];names=[]
    try:
        with torch.no_grad():
            for split in ("train","val","test"):
                for name in split_names[split]:
                    with h5py.File(args.data/"experts"/(name+".hdf5")) as h:
                        history=h5_history(h,0)
                    policy.reset()
                    action,online,tensors=capture.infer(history)
                    raw=array(online.observation.dino_history[0,-1])
                    data={"state_only":history.state,
                          "dino_global":raw.mean(1),
                          "dino_spatial":raw,
                          "g_content":tensors["g_content"],
                          "g_geometry":np.concatenate([tensors["g_geometry"].ravel(),tensors["g_camera_coordinates"].ravel()]),
                          "g_all":np.concatenate([tensors[k].ravel() for k in
                              ("g_content","g_semantic","g_geometry","g_camera_coordinates","g_camera_support")]),
                          "p1_detail":tensors["p1_detail"]}
                    for k,v in data.items():features[k].append(v.reshape(-1))
                    c=contracts[name]
                    target=np.array(c["red_xyz"][:2]+c["green_xyz"][:2],np.float32)
                    targets.append(target);splits.append(split);names.append(name)
                    print(json.dumps({"episode":name,"split":split,"n":len(names)}),flush=True)
    finally:
        pass
    arrays={k:np.stack(v) for k,v in features.items()}
    target=np.stack(targets);split_array=np.array(splits)
    np.savez_compressed(args.output/"frozen_features.npz",**arrays,targets=target,splits=split_array,names=np.array(names))
    result=ridge_readouts(arrays,target,split_array)
    train=split_array=="train";test=split_array=="test"
    mean=target[train].mean(0)
    result["constant_train_mean"]={"test_mean_object_xy_error_cm":
        (100*np.linalg.norm((target[test]-mean).reshape(-1,2,2),axis=-1).mean(0)).tolist()}
    atomic_json(args.output/"readouts.json",dict(note=__doc__,target_order=["red_x","red_y","green_x","green_y"],rows=result,episodes=names,splits=splits))
    atomic_json(args.output/"provenance.json",policy.deployment_health())

if __name__=="__main__":main()

