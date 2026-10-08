"""RGB-only planar calibration diagnostic, fitted on training episodes.

Pure-color masks are inherited from the existing trajectory color audit.
The wooden table must be excluded by channel dominance. Low-visibility cases
remain explicitly unlocalized; this is not a deployed controller.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import h5py
import numpy as np


def color_masks(rgb):
    h,s,v=cv2.split(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
    r,g,b=rgb.astype(np.float32).transpose(2,0,1)
    return np.stack([
        ((h<4)|(h>176))&(r>2*g)&(r>2*b)&(s>110)&(v>45),
        (h>40)&(h<82)&(g>1.8*r)&(g>1.8*b)&(s>110)&(v>45)])


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","features","output"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    z=np.load(args.features);points=[];counts=[]
    for name in z["names"]:
        with h5py.File(args.data/"experts"/(name+".hdf5")) as h:
            masks=color_masks(h["observations/images/cam_high"][0])
        pair=[];numbers=[]
        for m in masks:
            yy,xx=np.where(m)
            pair.append([xx.mean() if len(xx) else np.nan,yy.mean() if len(yy) else np.nan])
            numbers.append(len(xx))
        points.append(pair);counts.append(numbers)
    xy=np.array(points);counts=np.array(counts);target=z["targets"].reshape(-1,2,2)
    train=z["splits"]=="train"
    valid=np.isfinite(xy).all(-1)&(counts>=20)
    src=xy[train].reshape(-1,2);dst=target[train].reshape(-1,2)
    admitted=valid[train].reshape(-1)
    cv2.setRNGSeed(0)
    H,inliers=cv2.findHomography(src[admitted],dst[admitted],method=cv2.RANSAC,ransacReprojThreshold=.01)
    pred=cv2.perspectiveTransform(xy.reshape(-1,1,2),H).reshape(-1,2,2)
    err=np.linalg.norm(pred-target,axis=-1)*100
    safe_error=np.where(valid,err,np.nan)
    report={"note":__doc__,"training_inliers":int(inliers.sum()),"training_available":int(admitted.sum()),
            "homography":H.tolist(),"threshold_pixels":20,"episodes":z["names"].tolist(),
            "splits":z["splits"].tolist(),"centroids":xy.tolist(),"counts":counts.tolist(),
            "localization_available":valid.tolist(),"predictions":np.where(valid[...,None],pred,np.nan).tolist(),
            "per_episode_xy_error_cm":safe_error.tolist(),"summary":{}}
    for split in ("train","val","test"):
        select=z["splits"]==split
        report["summary"][split]={"episodes":int(select.sum()),
            "available_per_object":valid[select].sum(0).tolist(),
            "mean_xy_error_cm":np.nanmean(safe_error[select],axis=0).tolist(),
            "max_xy_error_cm":np.nanmax(safe_error[select],axis=0).tolist()}
    args.output.write_text(json.dumps(finite_json(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["summary"],indent=2))

if __name__=="__main__":main()

