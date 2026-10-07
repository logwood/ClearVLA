"""Audit causal short-RGB composition to instruction start against rigid labels.

Only past/current RGB enters the estimator. Rigid/body arrays score outputs
afterward. Compare strict intermediate support and final cycle/photo support;
neither is claimed calibrated visibility or a production replacement. The
only persistent estimator state is two coordinate maps and support maps.
"""
from pathlib import Path
import argparse, hashlib, json
import cv2
import numpy as np
from probe_observed_rgb_motion_candidate import score, estimate


def warp(field, xy):
    return cv2.remap(np.asarray(field, np.float32), xy[..., 0], xy[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def inside(xy, shape):
    h, w = shape
    return (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1)


def dump(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def main():
    q = argparse.ArgumentParser()
    for key in ('plan','labels','output'): q.add_argument('--'+key, type=Path, required=True)
    q.add_argument('--stride', type=int, default=4)
    a = q.parse_args(); a.output.mkdir(exist_ok=False); cv2.setNumThreads(1)
    if a.stride < 1: raise ValueError('positive observation stride required')
    complete = json.loads((a.labels/'complete.json').read_text())
    if not complete.get('rigid_simulation_oracle_audit_only') or complete['pair_mode'] != 'current_start':
        raise ValueError('separate current-to-start oracle audit labels required')
    index = {(r['case'], r['step']):r for r in json.loads((a.labels/'results.json').read_text())}
    if len(index) != complete['windows']: raise ValueError('non-unique audit windows')
    report = dict(identity=dict(scope=__doc__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(), label_identity=complete,
        opencv_version=cv2.__version__, stride=a.stride, policy_changed=False), records=[], complete=False)
    for row in json.loads(a.plan.read_text()):
        case = Path(row['case']); wanted = sorted(set(row['steps']))
        if any(t % a.stride for t in wanted): raise ValueError('query not on declared chain clock')
        with np.load(case/'trajectory.npz', allow_pickle=False) as z:
            rgbs = {k:z[k] for k in ('rgb_static','rgb_gripper')}
        pending = {t:dict(case=case.name, case_id=row.get('case_id'), step=t, cameras=[]) for t in wanted}
        for camera, key in enumerate(('rgb_static','rgb_gripper')):
            rgb = rgbs[key]; h,w = rgb.shape[1:3]; yy,xx = np.mgrid[:h,:w].astype(np.float32)
            grid = np.stack((xx,yy),-1)
            to_start = grid.copy(); from_start = grid.copy()
            strict_to = np.ones((h,w),np.float32); strict_from = np.ones((h,w),np.float32)
            in_bounds_to = np.ones((h,w),np.float32); in_bounds_from = np.ones((h,w),np.float32)
            for t in range(0,max(wanted)+1,a.stride):
                if t:
                    # No learned/raw descriptor caches or simulator state.
                    reverse = estimate(rgb[t],rgb[t-a.stride],'dis_medium')
                    forward = estimate(rgb[t-a.stride],rgb[t],'dis_medium')
                    old_to, old_from = to_start, from_start
                    to_start = warp(old_to,reverse['xy'])
                    from_start = warp(forward['xy'],old_from)
                    strict_to = reverse['accepted'].astype(np.float32)*(warp(strict_to,reverse['xy'])>1-1e-6)
                    strict_from = strict_from*(warp(forward['accepted'].astype(np.float32),old_from)>1-1e-6)
                    in_bounds_to = inside(reverse['xy'],(h,w))*(warp(in_bounds_to,reverse['xy'])>1-1e-6)
                    in_bounds_from = in_bounds_from*(warp(inside(forward['xy'],(h,w)).astype(np.float32),old_from)>1-1e-6)
                if t not in pending: continue
                # Verify the estimator state BEFORE reading the scoring labels.
                if not np.isfinite(to_start).all() or not np.isfinite(from_start).all(): raise ValueError('nonfinite chain')
                cycle = np.linalg.norm(warp(from_start,to_start)-grid,axis=-1)
                # Float photometry makes the final gate explicit; intermediate
                # gates retain the candidate's declared OpenCV runtime behavior.
                photo = np.abs(rgb[t].astype(np.float32)-warp(rgb[0],to_start)).mean(-1)/255.
                bounds = inside(to_start,(h,w)) & (in_bounds_to>0) & (warp(in_bounds_from,to_start)>1-1e-6)
                endpoint = bounds & (cycle<.75) & (photo<.08)
                strict = endpoint & (strict_to>0) & (warp(strict_from,to_start)>1-1e-6)
                entry = index[(case.name,t)]; path = Path(entry['production_labels'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['production_labels_sha256']: raise ValueError('label hash changed')
                with np.load(path,allow_pickle=False) as z: labels={k:z[k] for k in z.files}
                if list(labels['source_frames']) != [t,0]: raise ValueError('audit reference clock differs')
                variants = {}
                for mode, accepted in [('strict_chain',strict),('endpoint_gate',endpoint)]:
                    variants[mode] = score(dict(xy=to_start,accepted=accepted),labels,camera,(h,w))
                if t==0 and (not np.array_equal(to_start,grid) or not strict.all()): raise ValueError('initial same-image chain is not identity')
                pending[t]['cameras'].append(dict(camera=camera,variants=variants,
                    strict_full_image_fraction=float(strict.mean()),endpoint_full_image_fraction=float(endpoint.mean()),
                    cycle_finite=True,label_sha256=entry['production_labels_sha256']))
        for t in wanted:
            report['records'].append(pending[t]);dump(a.output/'results.json',report)
            print('RGB_CHAIN',case.name,t,flush=True)
    report['complete']=True;dump(a.output/'results.json',report)


if __name__=='__main__':main()
