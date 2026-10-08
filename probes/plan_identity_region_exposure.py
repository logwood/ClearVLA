"""Resolve the unchanged formal samplers before generating auxiliary labels.

This reads episode metadata and raw hashes, never model predictions or oracle
labels. Prefetch-only rows are declared separately and do not become updates.
"""
from pathlib import Path
from itertools import islice
import argparse,hashlib,json
import numpy as np
import torch
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data
from clearvla.mainline.train import BOUNDARY_VALIDATION_MAX_BATCHES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train-batches',type=int,required=True)
    p.add_argument('--val-batches',type=int,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=load_config(a.config)
    if cfg.optimizer.batch_size!=8 or cfg.top.identity_supervision_mode!='rgbd_temporal_conditional_v2':
        raise ValueError('plan from the unchanged v2 BS8 sampler before adding annotations')
    bundle=load_mainline_data(cfg)
    settings=[('train',a.train_batches),('val',a.val_batches)]+[(s,BOUNDARY_VALIDATION_MAX_BATCHES) for s in ('val_prefix','val_tail') if s in bundle.datasets]
    plans={};exposures=[];hashes={}
    for split,limit in settings:
        kwargs=dict(batch_size=8,workers=0,device=torch.device('cpu'),shuffle=split=='train')
        if split=='train':kwargs['generator']=torch.Generator().manual_seed(cfg.data.seed+101)
        elif split=='val':kwargs['task_panel_max_batches']=a.val_batches if bundle.is_multitask else None
        else:kwargs['coverage_panel_max_batches']=BOUNDARY_VALIDATION_MAX_BATCHES
        loader=bundle.loader(split,**kwargs);sampler=loader.batch_sampler
        if hasattr(sampler,'set_epoch'):sampler.set_epoch(1)
        base=bundle.datasets[split].base;offsets=np.asarray(base.config.visual_history_offsets[-2:])
        budget=limit+(cfg.data.num_workers+2 if split=='train' else 0)
        for batch_index,indices in enumerate(islice(iter(sampler),budget),1):
            keys=[]
            for index in indices:
                ref=base.refs[int(index)];episode=base.episodes[ref.episode_idx]
                context=int(episode.context_start)
                handle=base.image_store._h5(episode.path);source_split=str(handle.attrs['source_split'])
                assert context==int(handle.attrs['context_start'])
                frames=ref.center+base.config.image_offset+offsets
                if base.config.causal_reset_padding:frames=np.maximum(frames,0)
                frames=(frames+context).astype(int).tolist();key=(source_split,*frames)
                keys.append('/'.join(map(str,key)))
                if key in plans:continue
                raw=[Path(cfg.data.calvin_raw_source)/source_split/('episode_%07d.npz'%f) for f in frames]
                for f in raw:
                    if str(f) not in hashes:hashes[str(f)]=sha(f)
                plans[key]=dict(episode=ref.episode_idx,episode_path=str(episode.path),
                    raw_source_split=source_split,source_split=source_split,context_start=context,
                    source_frames=frames,raw_files=list(map(str,raw)),sha256=[hashes[str(f)] for f in raw])
            exposures.append(dict(split=split,batch=batch_index,trained_or_scored=batch_index<=limit,
                                  sample_indices=list(map(int,indices)),source_keys=keys))
        print(split,'declared_batches',sum(x['split']==split for x in exposures),'unique_pairs',len(plans),flush=True)
    report=dict(complete=True,records=list(plans.values()),exposure=exposures,config_sha256=sha(a.config),
        base_config=str(a.config),batch_size=8,epoch=1,train_batches=a.train_batches,val_batches=a.val_batches,
        unique_raw_files=len(hashes),script_sha256=sha(__file__),scope=__doc__)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
