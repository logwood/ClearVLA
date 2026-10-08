"""Qualify a different-motion witness on raw sensor-only candidate regions."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from sensor_motion_groups import propose_motion_negatives


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ('plan','regions','calibration','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    rows=json.loads(a.plan.read_text())['records'];regions=json.loads(a.regions.read_text())
    assert regions['complete'];lookup={r['path']:r for r in regions['records']}
    calibration=json.loads(a.calibration.read_text());records=[]
    report=dict(complete=False,records=records,plan_sha256=sha(a.plan),regions_sha256=sha(a.regions),
        script_sha256=sha(__file__),oracle_used=False,production_changed=False)
    for row in rows:
        before,now=row['raw_files'];r=lookup[now]
        assert [sha(f) for f in row['raw_files']]==row['sha256']
        assert sha(r['labels_path'])==r['labels_sha256']
        with np.load(before,allow_pickle=False) as b,np.load(now,allow_pickle=False) as c,np.load(r['labels_path'],allow_pickle=False) as z:
            # Explicit sensor allowlist inside each producer call.
            cameras=[propose_motion_negatives(dict(rgb=c['rgb_'+key],depth=c['depth_'+key]),
                dict(rgb=b['rgb_'+key],depth=b['depth_'+key]),projection,z['partition_'+key],z['groups_'+key])
                for key,projection in zip(('static','gripper'),calibration['projection'])]
        records.append(dict(path=now,previous=before,cameras=cameras))
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(Path(now).name,[x['accepted_pairs'] for x in cameras],flush=True)
    report['complete']=True;(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
