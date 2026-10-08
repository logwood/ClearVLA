"""Replay declared panel observations exactly and export metric sensor depth.

No RGB/features are cached and no simulator identity is exported. This is an
audit artifact; production continues reading its original raw sensor files.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
from clearvla.benchmarks.calvin_eval import _environment,_official_task_assets


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int)
    a=p.parse_args();a.output.mkdir(exist_ok=False);plan=json.loads(a.plan.read_text())
    out=dict(complete=False,records=[],script_sha256=sha(__file__),plan_sha256=sha(a.plan),scope=__doc__)
    if a.index is None:
        for i in range(len(plan)):
            dest=a.output/('worker_%02d'%i)
            with (a.output/('worker_%02d.log'%i)).open('x') as log:
                subprocess.run([sys.executable,'-B','-u',__file__,'--plan',str(a.plan),'--output',str(dest),'--index',str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
            child=json.loads((dest/'results.json').read_text());assert child['complete']
            out['records'].extend(child['records']);dump(a.output/'results.json',out)
    else:
        row=plan[a.index];case=Path(row['case']);result=json.loads((case/'result.json').read_text())
        with np.load(case/'trajectory.npz',allow_pickle=False) as z:
            data={k:z[k] for k in ('executed','rgb_static','rgb_gripper')}
        official,*_=_official_task_assets();env=_environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'),show_gui=False)
        robot,scene=official.get_env_state_for_initial_condition(result['initial_state']);env.reset(robot_obs=robot,scene_obs=scene)
        wanted=sorted(set(row['steps'])|{max(0,s-4) for s in row['steps']})
        for step in range(max(wanted)+1):
            if step:env.step(data['executed'][step-1].copy())
            if step not in wanted:continue
            fields={}
            for cam,key in zip(env.cameras,('static','gripper')):
                rgb,depth=cam.render();np.testing.assert_array_equal(rgb,data['rgb_'+key][step])
                fields['depth_'+key]=depth
            target=a.output/('depth_%03d.npz'%step);np.savez_compressed(target,**fields)
            out['records'].append(dict(case=str(case),case_id=row['case_id'],step=step,depth_path=str(target),depth_sha256=sha(target),rgb_exact=True))
            dump(a.output/'results.json',out)
    out['complete']=True;dump(a.output/'results.json',out)


if __name__=='__main__':main()
