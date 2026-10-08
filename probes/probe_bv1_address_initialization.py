"""Read-only full-model migration check; dataset identity borrowed, not inventoried."""
from pathlib import Path
import argparse,dataclasses,gc,json,torch
from clearvla.mainline.config import load_config
from clearvla.mainline.checkpoint import build_checkpoint_identity,checkpoint_identity_from_mapping
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import load_checkpoint_for_initialization
from clearvla.mainline.runtime.address_memory_migration import KEY

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--repo",type=Path,required=True);a.add_argument("--checkpoint",type=Path,required=True)
    a.add_argument("--config",type=Path,action="append",required=True);a.add_argument("--output",type=Path,required=True)
    r=a.parse_args();torch.set_num_threads(4);torch.manual_seed(0)
    if r.output.exists():raise FileExistsError(r.output)
    payload=torch.load(r.checkpoint,map_location="cpu",mmap=True,weights_only=False)
    saved=checkpoint_identity_from_mapping(payload["identity"]);reports=[]
    for cp in r.config:
        cfg=load_config(cp)
        identity=build_checkpoint_identity(cfg,repo_root=r.repo,dataset=saved.dataset,language=saved.language)
        model=ClearVLAMainlinePolicy(cfg)
        init=load_checkpoint_for_initialization(r.checkpoint,model=model,config=cfg,identity=identity,model_contract_migration="b_v1_address_memory_v1")
        current=model.state_dict()
        assert init.global_step==12036
        assert all(torch.equal(value,current[name]) for name,value in payload["model"].items())
        added=set(current)-set(payload["model"])
        assert added==({KEY} if cfg.top.entity_address_memory_mode!="none" else set())
        if added:assert torch.count_nonzero(current[KEY]).item()==0
        reports.append(dict(config=str(cp),mode=cfg.top.entity_address_memory_mode,global_step=init.global_step,
            inherited_tensors_exact=len(payload["model"]),added=sorted(added),
            new_parameters=sum(current[k].numel() for k in added)))
        del current,model;gc.collect()
    r.output.write_text(json.dumps(dict(complete=True,records=reports,
        scope="full real model migration/state parity; loader must independently verify actual dataset at launch; no optimizer or behavior claim"),indent=2)+"\n")
    print("INITIALIZATION_COMPLETE",reports,flush=True)
if __name__=="__main__":main()
