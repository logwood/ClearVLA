"""Audit motion-confirmed negatives on complete, exactly replayed panels."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from probe_mask_surface_agreement import proposal_partition
from sensor_motion_groups import propose_motion_negatives


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score_pairs(part,groups,pairs,body,link,objects):
    rows=[];keys=np.unique(np.stack((body[body>=0],link[body>=0]),-1),axis=0)
    counts=[np.array([int(((part==g)&(body==b)&(link==l)).sum()) for b,l in keys]) for g in range(len(groups))]
    is_block=np.array([b in objects for b,l in keys])
    for i,j in pairs:
        a,b=counts[i],counts[j]
        rows.append(dict(groups=[i,j],all_pairs=int((part==i).sum())*int((part==j).sum()),
            identified_pairs=int(a.sum())*int(b.sum()),wrong_same_rigid=int(a@b),
            block_pairs=int(a[is_block].sum())*int(b[is_block].sum()),wrong_same_block=int(a[is_block]@b[is_block])))
    return dict(pairs=rows,totals={key:sum(r[key] for r in rows) for key in ('all_pairs','identified_pairs','wrong_same_rigid','block_pairs','wrong_same_block')})


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','depths','sam','groups','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    plan=json.loads(a.plan.read_text());cal=json.loads(a.calibration.read_text())
    sam=json.loads(a.sam.read_text());geo=json.loads(a.groups.read_text());dep=json.loads(a.depths.read_text())
    assert sam['complete'] and geo['complete'] and dep['complete']
    sams={(r['case_id'],r['step']):r for r in sam['records']};geos={(r['case_id'],r['step']):r for r in geo['records']}
    depths={(r['case_id'],r['step']):r for r in dep['records']}
    report=dict(complete=False,records=[],scope=__doc__,production_changed=False,script_sha256=sha(__file__))
    for row in plan:
        case=Path(row['case'])
        with np.load(case/'trajectory.npz',allow_pickle=False) as z:
            rgb={key:z['rgb_'+key] for key in ('static','gripper')}
        for step in row['steps']:
            k=(row['case_id'],step);s=sams[k];g=geos[k];previous=max(0,step-4)
            assert sha(s['proposal_path'])==s['proposal_sha256'] and sha(g['groups_path'])==g['groups_sha256']
            d0=depths[(k[0],previous)];d1=depths[k]
            assert sha(d0['depth_path'])==d0['depth_sha256'] and sha(d1['depth_path'])==d1['depth_sha256']
            cameras=[]
            with np.load(s['proposal_path'],allow_pickle=False) as masks,np.load(g['groups_path'],allow_pickle=False) as groups,np.load(d0['depth_path'],allow_pickle=False) as old,np.load(d1['depth_path'],allow_pickle=False) as now:
                for ci,key in enumerate(('static','gripper')):
                    part,pairs=proposal_partition(masks['masks_'+('top' if ci==0 else 'wrist')],groups['group_'+key],interior_radius=1)
                    proposal=propose_motion_negatives(dict(rgb=rgb[key][step],depth=now['depth_'+key]),
                        dict(rgb=rgb[key][previous],depth=old['depth_'+key]),cal['projection'][ci],part,pairs)
                    # Only after producer returns, score with independent labels.
                    bodies={o['body'] for o in g['audit']['movable_coverage']}
                    score=score_pairs(part,pairs,proposal['accepted_pairs'],groups['audit_body_'+key],groups['audit_link_'+key],bodies)
                    cameras.append(dict(proposal=proposal,audit=score))
            report['records'].append(dict(case_id=k[0],step=step,cameras=cameras))
            (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
