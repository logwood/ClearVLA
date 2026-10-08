"""Audit candidate negatives on the real 32x32 sampler and a declared K chart.

No model update or label admission. For each region's mean sampled K read, the
TV between its interpolation stencils bounds the TV obtainable from ANY K
probabilities on that chart (data processing). This is a spatial-read bound,
not a feature-information or whole-network bound. Oracle maps only score the
already selected points. Existing sensor-positive labels are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np

from clearvla.mainline.data.identity_correspondence import IdentityLabelProducer
from probe_mask_surface_agreement import proposal_partition, negative_clique_size, audit


def stencils(y, x, shape, side):
    h, w = shape
    u = x.astype(np.float64)*(side-1)/(w-1)
    v = y.astype(np.float64)*(side-1)/(h-1)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    x1 = np.minimum(x0+1,side-1); y1 = np.minimum(y0+1,side-1)
    dx = u-x0; dy = v-y0
    weights = np.zeros((len(y),side*side),np.float64)
    row = np.arange(len(y))
    for xx,yy,ww in ((x0,y0,(1-dx)*(1-dy)),(x1,y0,dx*(1-dy)),
                     (x0,y1,(1-dx)*dy),(x1,y1,dx*dy)):
        np.add.at(weights,(row,yy*side+xx),ww)
    np.testing.assert_allclose(weights.sum(1),1.,rtol=0.,atol=1e-12)
    return weights


def describe(values):
    a=np.asarray(values,float)
    return dict(count=int(a.size),minimum=float(a.min()),median=float(np.median(a)),
                maximum=float(a.max()),mean=float(a.mean())) if a.size else dict(count=0)


def score(partition, groups, side):
    y,x=IdentityLabelProducer._sample(partition.shape)
    selected=partition[y,x]
    weights=stencils(y,x,partition.shape,side)
    h,w=partition.shape
    ay=np.rint(np.linspace(0,h-1,side)).astype(int)
    ax=np.rint(np.linspace(0,w-1,side)).astype(int)
    atom_region=partition[ay[:,None],ax[None,:]].ravel()
    alive=np.unique(selected[selected>=0]); means={}
    regions=[]
    for group in alive:
        mean=weights[selected==group].mean(0);means[int(group)]=mean
        regions.append(dict(group=int(group),points=int((selected==group).sum()),
            native_pixels=int((partition==group).sum()),
            own_region_canonical_atoms=int((atom_region==group).sum()),
            mean_read_stencil_mass_on_own_region=float(mean[atom_region==group].sum())))
    pairs=[]
    for a,b in itertools.combinations(alive,2):
        if np.all(groups[a]!=groups[b]):
            tv=.5*np.abs(means[int(a)]-means[int(b)]).sum()
            if not -1e-12<=tv<=1+1e-12:raise ValueError('stencil TV outside probability bound')
            pairs.append(dict(groups=[int(a),int(b)],max_possible_mean_K_TV=float(tv)))
    # Preserve the exact actual producer sampler; never resample to an object.
    sampled_map=np.full(partition.shape,-1,np.int64)
    sampled_map[y,x]=selected
    return dict(regions=regions,negative_pairs=pairs,
        native_groups=len(groups),sampled_groups=len(alive),supported_sample_points=int((selected>=0).sum()),
        sampled_negative_clique=negative_clique_size(groups[alive])), sampled_map


def main():
    p=argparse.ArgumentParser()
    for key in ('sam','groups','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--canonical-side',type=int,required=True)
    p.add_argument('--interior-radius',type=int,choices=(0,1),required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    sam=json.loads(args.sam.read_text());geo=json.loads(args.groups.read_text())
    if not sam['complete'] or not geo['complete']:raise ValueError('incomplete proposal sources')
    index={(r['case_id'],r['step']):r for r in geo['records']}
    result=dict(complete=False,records=[],canonical_side=args.canonical_side,
        interior_radius=args.interior_radius,scope=__doc__,production_changed=False,
        inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.sam,args.groups,Path(__file__))})
    for row in sam['records']:
        g=index[(row['case_id'],row['step'])]
        for path,expected in ((row['proposal_path'],row['proposal_sha256']),(g['groups_path'],g['groups_sha256'])):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('proposal bytes changed')
        cams=[]
        with np.load(row['proposal_path'],allow_pickle=False) as s,np.load(g['groups_path'],allow_pickle=False) as z:
            for sk,gk,ck in (('masks_top','group_static','static'),('masks_wrist','group_gripper','gripper')):
                part,groups=proposal_partition(s[sk],z[gk],interior_radius=args.interior_radius)
                value,sampled=score(part,groups,args.canonical_side)
                # Oracle accessed only after constructing the sampled proposal.
                body=z['audit_body_'+ck];link=z['audit_link_'+ck]
                keys=np.unique(np.stack((body[body>=0],link[body>=0]),-1),axis=0)
                truth=np.stack([(body==i)&(link==j) for i,j in keys]) if len(keys) else np.empty((0,*body.shape),bool)
                value['rigid_link_scoring']=audit(sampled,groups,truth)
                value['oracle_scope']='scoring only; audit field names retain block for generic entity masks'
                cams.append(value)
        result['records'].append(dict(case=row['case_id'],step=row['step'],cameras=cams))
    result['complete']=True
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    summary=dict(windows=len(result['records']),canonical_side=args.canonical_side,cameras=[])
    for c in range(2):
        rows=[r['cameras'][c] for r in result['records']]
        regions=[a for r in rows for a in r['regions']];pairs=[a for r in rows for a in r['negative_pairs']]
        summary['cameras'].append(dict(
            native_groups=sum(r['native_groups'] for r in rows),sampled_groups=sum(r['sampled_groups'] for r in rows),
            supported_sample_points=sum(r['supported_sample_points'] for r in rows),
            regions_without_own_canonical_atom=sum(r['own_region_canonical_atoms']==0 for r in regions),
            mean_stencil_mass_on_own_region=describe([r['mean_read_stencil_mass_on_own_region'] for r in regions]),
            max_possible_mean_K_TV=describe([a['max_possible_mean_K_TV'] for a in pairs]),
            sampled_negative_clique=describe([r['sampled_negative_clique'] for r in rows]),
            wrong_negative_rigid_pairs=sum(r['rigid_link_scoring']['wrong_negative_same_block_pairs'] for r in rows)))
    (args.output/'decision-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
