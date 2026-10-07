"""Join actual sampler draws with audited physical geometry on shared age supports.

Only exact audited annotation ages are admitted. This does not interpolate
unmeasured object truth and is not an estimate certified for every training row.
No RGB loading, DINO encoder, optimizer, checkpoint or model input is changed.
"""
import argparse,collections,hashlib,json
from pathlib import Path
import torch
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--geometry",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--comparison-batches",type=int,action="append",default=[])
    a=ap.parse_args()
    if any(n<=0 for n in a.comparison_batches):raise ValueError("comparison batches must be positive")
    if a.output.exists():raise FileExistsError(a.output)
    config=load_config(a.config);bundle=load_mainline_data(config)
    records=json.loads(a.geometry.read_text())
    geometry={(r["episode"],int(r["age"])):r for r in records if r["split"]=="train"}
    refs=bundle.datasets["train"].base.refs
    def row(index):
        ref=refs[int(index)];episode=bundle.episodes[ref.episode_idx]
        age=int(ref.center)-(int(episode.source_start)-int(episode.context_start))
        return geometry.get((episode.episode_id,age))
    def summary(indices):
        accepted=[r for i in indices if (r:=row(i)) is not None]
        n=len(accepted); by_task=collections.Counter(r["task"] for r in accepted)
        return {"total_draws":len(indices),"exact_audited_age_draws":n,
          "unique_exact_windows":len({(r["episode"],r["age"]) for r in accepted}),
          "target_nearest_fraction":sum(r["target_nearest"] for r in accepted)/max(n,1),
          "near_proxy_fraction":sum(r["near_proxy"] for r in accepted)/max(n,1),
          "early_age_lt16_fraction":sum(r["age"]<16 for r in accepted)/max(n,1),
          "early_nonnearest_fraction":sum(r["age"]<16 and not r["target_nearest"] for r in accepted)/max(n,1),
          "far_nonnearest_fraction":sum(r["target_xy_distance"]>.15 and not r["target_nearest"] for r in accepted)/max(n,1),
          "by_task":dict(by_task),"by_age":dict(sorted(collections.Counter(r["age"] for r in accepted).items()))}
    loader=bundle.loader("train",batch_size=config.optimizer.batch_size,workers=0,device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(config.data.seed+101))
    sampler=loader.batch_sampler
    if hasattr(sampler,"set_epoch"):sampler.set_epoch(1)
    draws=[int(i) for batch in sampler for i in batch]
    configured_batches=int(config.runtime.max_train_batches)
    configured_batches=min(len(loader),configured_batches) if configured_batches>0 else len(loader)
    report={"config":str(a.config),"geometry":str(a.geometry),"batch_size":config.optimizer.batch_size,
      "batches_in_epoch":len(loader),"uniform_on_same_audited_support":summary(list(range(len(refs)))),
      "full_sampler_iterator_on_same_support":summary(draws),
      "configured_run_batches":configured_batches,
      "configured_run_on_same_support":summary(draws[:configured_batches*config.optimizer.batch_size]),
      "comparison_caps_on_same_support":{str(n):summary(draws[:n*config.optimizer.batch_size]) for n in a.comparison_batches},
      "identities":{"config_sha256":hashlib.sha256(a.config.read_bytes()).hexdigest(),
        "geometry_sha256":hashlib.sha256(a.geometry.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
      "scope":"source geometry and actual sampler joined only at exact common ages; unmeasured ages excluded"}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
