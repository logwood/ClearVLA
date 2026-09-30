"""Independent-process parity probe against a supplied source root."""
import sys, json, hashlib
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root),str(root/'tests')]
import torch
from dataclasses import replace
from test_mainline_task_execution_integration import config, _batch, _model_engine
from clearvla.mainline.global_task import COMPILED_TASK_GLOBAL
from clearvla.mainline.runtime.sampling import sample_action

def digest(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
report={}
for amp in (False,True):
    torch.manual_seed(34671)
    c=config(amp)
    c=replace(c,top=replace(c.top,object_view_mode='per_camera_values_v1'),bottom=replace(c.bottom,global_condition_mode=COMPILED_TASK_GLOBAL))
    m,e=_model_engine(c)
    b=_batch()
    params=hashlib.sha256()
    for name,t in m.state_dict().items(): params.update((name+str(tuple(t.shape))+digest(t)).encode())
    m.eval()
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        cache,_,_=m.encode_online(b.online)
        sampled=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(48675))
    report['bf16' if amp else 'fp32']={'state_dict':params.hexdigest(),'relation':digest(cache.top.intent.task_relation.values), 'carrier':digest(cache.top.intent.public_interval_carrier),'global':digest(cache.top.intent.compiled_global_task.tokens),'action':digest(sampled.action),'gripper':digest(sampled.gripper_command)}
    del m,e,b,cache,sampled
    import gc;gc.collect()
Path(sys.argv[2]).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report))
