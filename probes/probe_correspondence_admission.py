"""Check proposed correspondence against physical masks; masks are audit only."""
from pathlib import Path
import hashlib, json, sys
import torch
import torch.nn.functional as F


def audit(grounder,captured,native,masks,names):
    x=native[0].detach().float()
    result=[]
    grids=torch.stack(torch.meshgrid(torch.linspace(-1,1,16,device=x.device),
        torch.linspace(-1,1,16,device=x.device),indexing='ij'),-1)[...,[1,0]][None]
    label=[]
    for m in masks:
        point=F.grid_sample(torch.as_tensor(m,device=x.device,dtype=torch.float32)[None],
            grids,align_corners=True)[0].flatten(1).T
        confidence,idx=point.max(-1)
        label.append(torch.where(confidence>.9,idx,torch.full_like(idx,-1)))
    for mode in ('raw','view_centered'):
        f=x if mode=='raw' else x-x.mean(1,keepdim=True)
        f=F.normalize(f,dim=-1)
        affinity=f[0]@f[1].T
        forward=affinity.argmax(-1);backward=affinity.argmax(-2)
        mutual=backward[forward]==torch.arange(256,device=x.device)
        margin=affinity.topk(2,-1).values.diff(dim=-1).neg().flatten()
        rows=[]
        for source,target,score in ((0,1,affinity),(1,0,affinity.T)):
            target_idx=score.argmax(-1);reverse=score.argmax(-2)
            reciprocal=reverse[target_idx]==torch.arange(256,device=x.device)
            for j,obj in enumerate(names):
                src=label[source]==j
                for rule,keep in [('all_nearest',src),('mutual_nearest',src & reciprocal)]:
                    target_label=label[target][target_idx[keep]]
                    rows.append({'source':source,'target':target,'object':obj,'rule':rule,
                        'source_cells':int(src.sum()),'accepted_cells':int(keep.sum()),
                        'correct_cells':int((target_label==j).sum()),
                        'wrong_object_cells':int(((target_label>=0)&(target_label!=j)).sum()),
                        'background_cells':int((target_label<0).sum())})
        result.append({'mode':mode,'cross_view_mutual_cells':int(mutual.sum()),'regions':rows})
    return {'scope':'frozen observed full-RGB endpoint DINO descriptors; physical masks audit only',
            'cross_view':result}


if __name__=='__main__':
    import probe_grounding_identity_provenance as driver
    driver.audit=audit
    driver.upstream_support=lambda *a,**k: []
    driver.base_support=lambda *a,**k: {}
    driver.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    p=output/'complete.json';done=json.loads(p.read_text())
    done['dependency_driver_sha256']=done.pop('script_sha256')
    done['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p.write_text(json.dumps(done,indent=2)+'\n')
