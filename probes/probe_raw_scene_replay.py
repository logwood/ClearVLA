"""Independent raw CALVIN replay admission; simulator state is scoring only.

Scene metadata chooses the renderer asset, never an identity-label producer.
Do not export oracle masks until native RGB has passed exact comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def make_environment(raw_split, scene):
    import hydra
    import calvin_env
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(Path(raw_split)/'.hydra/merged_config.yaml')
    cfg.scene=OmegaConf.load(Path(calvin_env.__file__).parents[1]/'conf/scene'/(scene+'.yaml'))
    for key in list(cfg.cameras):
        if key not in ('static','gripper'):del cfg.cameras[key]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize('.')
    # The library get_env(scene=...) discards OmegaConf.merge's returned
    # object. Explicit replacement here is local to this read-only scorer.
    return hydra.utils.instantiate(cfg.env,show_gui=False,use_vr=False,use_scene_info=True)


def scene_for(path):
    path=Path(path);index=int(path.stem.split('_')[-1])
    scenes=np.load(path.parent/'scene_info.npy',allow_pickle=True).item()
    found=[name for name,(lo,hi) in scenes.items() if int(lo)<=index<=int(hi)]
    if len(found)!=1:raise ValueError('ambiguous raw scene provenance')
    return found[0]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--export-exact-masks',action='store_true')
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    rows=json.loads(args.plan.read_text())['records'];files={}
    for row in rows:
        for path,sha in zip(row['raw_files'],row['sha256']):files[path]=sha
    if args.limit:files=dict(list(files.items())[:args.limit])
    results=dict(complete=False,records=[],scope=__doc__,plan_sha256=digest(args.plan),
                 script_sha256=digest(__file__),production_changed=False)
    current=None;env=None
    try:
        for filename,expected in files.items():
            path=Path(filename)
            if digest(path)!=expected:raise ValueError('raw data bytes changed')
            scene=scene_for(path)
            if scene!=current:
                if env is not None:env.close()
                env=make_environment(path.parent,scene);current=scene
            with np.load(path,allow_pickle=False) as z:
                data={k:z[k] for k in ('robot_obs','scene_obs','rgb_static','rgb_gripper','depth_static','depth_gripper')}
            controls=[];exact=False;mask_arrays={}
            for method in ('pose_only','library_reset_one_substep'):
                if method=='pose_only':
                    env.scene.reset(data['scene_obs']);env.robot.reset(data['robot_obs'])
                else:env.reset(robot_obs=data['robot_obs'],scene_obs=data['scene_obs'])
                cameras=[];images=[]
                for camera,key in zip(env.cameras,('static','gripper')):
                    rgb,depth=camera.render();images.append(rgb)
                    diff=np.abs(rgb.astype(float)-data['rgb_'+key].astype(float))
                    cameras.append(dict(camera=key,rgb_exact=bool(np.array_equal(rgb,data['rgb_'+key])),
                        rgb_mae=float(diff.mean()),rgb_max=float(diff.max()),
                        unequal_pixels=int(np.any(diff!=0,axis=-1).sum()),
                        depth_mae=float(np.abs(depth-data['depth_'+key]).mean()),
                        depth_max=float(np.abs(depth-data['depth_'+key]).max())))
                controls.append(dict(reset=method,cameras=cameras))
                if all(c['rgb_exact'] for c in cameras):
                    exact=True
                    if args.export_exact_masks:
                        import pybullet as pb
                        from probe_rgbd_correspondence import matrix
                        for camera,key,rgb in zip(env.cameras,('static','gripper'),images):
                            frame=pb.getCameraImage(width=camera.width,height=camera.height,
                                viewMatrix=matrix(camera,'view').T.flatten().tolist(),
                                projectionMatrix=matrix(camera,'projection').T.flatten().tolist(),
                                physicsClientId=camera.cid,flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
                            np.testing.assert_array_equal(np.asarray(frame[2])[...,:3],rgb)
                            label=np.asarray(frame[4]).reshape(camera.height,camera.width)
                            mask_arrays['audit_body_'+key]=np.where(label<0,-1,label&((1<<24)-1))
                            mask_arrays['audit_link_'+key]=np.where(label<0,-2,(label>>24)-1)
                    break
            record=dict(path=str(path),raw_sha256=expected,scene=scene,exact_rgb=exact,controls=controls)
            if mask_arrays:
                target=args.output/(path.stem+'-oracle.npz');np.savez_compressed(target,**mask_arrays)
                record.update(oracle_path=str(target),oracle_sha256=digest(target),oracle_role='scoring only')
            results['records'].append(record)
            (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(path.name,scene,exact,[(r['reset'],[c['rgb_mae'] for c in r['cameras']]) for r in controls],flush=True)
        results.update(complete=True,all_exact=all(r['exact_rgb'] for r in results['records']))
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
