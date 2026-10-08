"""Offline structural audit of saved G posteriors; never runs or edits a policy.

Distinguish within-slot averaging, across-slot collapse, and shared-K mixing.
Actor masks select an evaluator permutation, not an online object identity.
The label-blind mode is the barycenter of the most massive 25x25 window.
"""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np


def box_sum(a, radius=12):
    a = np.asarray(a, dtype=np.float64)
    p = np.pad(a, ((radius, radius), (radius, radius)))
    s = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    n = 2 * radius + 1
    return s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]


def center(a):
    mass = a.sum()
    if mass <= 0:
        raise ValueError('empty spatial distribution')
    return np.array([(a.sum(0) * np.arange(a.shape[1])).sum(),
                     (a.sum(1) * np.arange(a.shape[0])).sum()]) / mass


def mode_summary(p):
    window = box_sum(p)
    y, x = np.unravel_index(window.argmax(), window.shape)
    cut = np.zeros_like(p)
    cut[max(0,y-12):y+13,max(0,x-12):x+13] = p[max(0,y-12):y+13,max(0,x-12):x+13]
    local = center(cut)
    return local, float(cut.sum())


def examine(path, arm, panel):
    with np.load(path, allow_pickle=False) as z:
        d = z['g_density'].astype(np.float64)
        masks = z['masks'].transpose(1,0,2,3)
        binding = z['binding_probability'][0].astype(np.float64)
        actual = z['g_camera_coordinates'][0].astype(np.float64) if 'g_camera_coordinates' in z else None
        content = z['g_content'][0].astype(np.float64) if 'g_content' in z else None
    k, c, h, w = d.shape
    cm = d.sum((-2,-1))
    p = d / cm[...,None,None]
    centers = np.array([[center(v) for v in obj] for obj in p])
    modes = np.empty_like(centers)
    peak_mass = np.empty((k,c))
    spread = np.empty((k,c))
    for ki in range(k):
        for ci in range(c):
            modes[ki,ci], peak_mass[ki,ci] = mode_summary(p[ki,ci])
            cx,cy=centers[ki,ci]
            spread[ki,ci] = np.sqrt((p[ki,ci].sum(0)*(np.arange(w)-cx)**2).sum()
                                     +(p[ki,ci].sum(1)*(np.arange(h)-cy)**2).sum())
    visible = masks.sum((-2,-1)) >= 20
    targets = np.array([[center(m) if m.any() else [0.,0.] for m in obj] for obj in masks])
    regions = np.array([[box_sum(m)>0 for m in obj] for obj in masks])
    mass = np.einsum('kcyx,ocyx->koc',d,regions.astype(float))
    nv = np.maximum(visible.sum(-1),1)
    dist = ((centers[:,None]-targets[None])**2).sum(-1)
    cost = np.where(visible[None],-np.log(np.maximum(mass*nv[None,:,None],1e-30))
                    + dist / ((h-1)/2)**2,0).sum(-1)/nv
    pair = min(permutations(range(k),2), key=lambda s:cost[s[0],0]+cost[s[1],1])
    common = dict(arm=arm,panel=panel,case=path.stem,path=str(path))
    rows=[]
    for oi, name in enumerate(('red','green')):
        ki=pair[oi]
        for ci in range(c):
            if not visible[oi,ci]: continue
            mean_error=float(np.linalg.norm(centers[ki,ci]-targets[oi,ci]))
            mode_error=float(np.linalg.norm(modes[ki,ci]-targets[oi,ci]))
            rows.append(dict(**common,object=name,camera=('global','wrist')[ci],slot=ki,
                mean=centers[ki,ci].tolist(),mode=modes[ki,ci].tolist(),target=targets[oi,ci].tolist(),
                mean_error_px=mean_error,mode_error_px=mode_error,
                mean_mode_distance_px=float(np.linalg.norm(centers[ki,ci]-modes[ki,ci])),
                spread_rms_px=float(spread[ki,ci]),peak_window_mass=float(peak_mass[ki,ci]),
                target_region_mass=float(mass[ki,oi,ci]/cm[ki,ci]),
                mean_far_mode_near=mean_error>24 and mode_error<=24,
                both_far=mean_error>24 and mode_error>24,
                binding_mass=float(binding[ki]),other_object_binding_mass=float(binding[pair[1-oi]])))
    bound=(d*binding[:-1,None,None,None]).sum(0)
    bound_rows=[]
    for ci in range(c):
        if not visible[0,ci]:continue
        b=bound[ci]/bound[ci].sum()
        bc=center(b)
        bound_rows.append(dict(**common,camera=('global','wrist')[ci],
            red_mean_error_px=float(np.linalg.norm(bc-targets[0,ci])),
            matched_red_mean_error_px=float(np.linalg.norm(centers[pair[0],ci]-targets[0,ci])),
            binding_composed_mean=bc.tolist(),red_target=targets[0,ci].tolist(),
            red_region_mass=float((b*regions[0,ci]).sum()),
            green_region_mass=float((b*regions[1,ci]).sum())))
    overlaps=[]; separations=[]
    for ci in range(c):
        for a,b in permutations(range(k),2):
            if a>=b:continue
            overlaps.append(float(np.sqrt(p[a,ci]*p[b,ci]).sum()))
            separations.append(float(np.linalg.norm(centers[a,ci]-centers[b,ci])))
    real=binding[:-1]/binding[:-1].sum()
    scene=(1-binding[:-1]);scene/=scene.sum()
    conditional_region=mass/cm[:,None]
    cross_view=[]
    if visible.all():
        for ki in range(k):
            # Diagnostic only: do two observed views confidently associate
            # the SAME K with different cubes? Exclude weak/background reads.
            object_mass=conditional_region[ki]
            total=object_mass.sum(0)
            red_fraction=object_mass[0]/np.maximum(total,1e-30)
            admitted=bool((total>=.2).all())
            conflicting=admitted and bool((red_fraction[0]>=.7 and red_fraction[1]<=.3)
                                         or (red_fraction[1]>=.7 and red_fraction[0]<=.3))
            cross_view.append(dict(slot=ki,admitted=admitted,conflicting=conflicting,
                                   red_fraction=red_fraction.tolist(),region_mass=object_mass.tolist(),
                                   matched_object=('red' if pair[0]==ki else 'green' if pair[1]==ki else None)))
    diag=dict(**common,slot_centers=centers.tolist(),slot_mode=modes.tolist(),
        matched_slots=list(pair),binding=binding.tolist(),
        target_scene_weight_total_variation=float(np.abs(real-scene).sum()/2),
        cross_view_identity=cross_view,
        binding_conditional_real=real.tolist(),binding_effective_real_slots=float(np.exp(-(real*np.log(np.maximum(real,1e-30))).sum())),
        mean_slot_overlap=float(np.mean(overlaps)),max_slot_overlap=float(max(overlaps)),
        mean_slot_center_separation_px=float(np.mean(separations)),
        every_pair_overlap_above_095=bool(min(overlaps)>.95),
        max_real_binding=float(real.max()))
    if actual is not None:
        diag['exported_coordinate_moment_max_error_px']=float(np.max(np.abs((actual+1)*(h-1)/2-centers)))
    if content is not None:
        u=content/np.linalg.norm(content,axis=-1,keepdims=True)
        diag['content_pair_cosine']=float((u@u.T)[~np.eye(k,dtype=bool)].mean())
    return rows,bound_rows,diag


def summarize(rows, keys):
    result={'count':len(rows)}
    for key in keys:
        a=np.array([r[key] for r in rows],float)
        result[key]={'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())}
    for key in ('mean_far_mode_near','both_far','every_pair_overlap_above_095'):
        if rows and key in rows[0]: result[key+'_count']=sum(r[key] for r in rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    views=[]; bounds=[]; cases=[]
    for arm in ('control','candidate'):
        for panel,folder in [('heldout',args.root/(arm+'-heldout')),('placements',args.root/(arm+'-placements')),
                             ('failed_states',args.root/'report/failed-state-probe'/arm)]:
            for path in sorted(folder.glob('*.npz')):
                a,b,c=examine(path,arm,panel);views.extend(a);bounds.extend(b);cases.append(c)
            print(arm,panel,'done',flush=True)
    aggregate={}
    for arm in ('control','candidate'):
        aggregate[arm]={}
        for panel in ('heldout','placements','failed_states'):
            vr=[x for x in views if x['arm']==arm and x['panel']==panel]
            br=[x for x in bounds if x['arm']==arm and x['panel']==panel]
            cr=[x for x in cases if x['arm']==arm and x['panel']==panel]
            aggregate[arm][panel]={
                'matched_views':summarize(vr,['mean_error_px','mode_error_px','spread_rms_px','peak_window_mass','target_region_mass','mean_mode_distance_px']),
                'binding_composition':summarize(br,['red_mean_error_px','matched_red_mean_error_px','red_region_mass','green_region_mass']),
                'cases':summarize(cr,['mean_slot_overlap','max_slot_overlap','mean_slot_center_separation_px','binding_effective_real_slots','max_real_binding']),
            }
    payload=dict(note=__doc__,aggregate=aggregate,views=views,binding_composition=bounds,cases=cases)
    (args.output/'collapse-audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('Saved',args.output/'collapse-audit.json',flush=True)


if __name__=='__main__':main()
