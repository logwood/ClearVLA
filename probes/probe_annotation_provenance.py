"""Exercise production loading/resolution with an audit-only provenance repair.

No HDF5, production loader or checkpoint is changed. Raw annotation identity is
admitted only after the existing strict overlay verifies split/task/text/frame
bounds and real-prefix length.
"""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import argparse,json,hashlib
from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.annotation_endpoint import resolve_annotation_endpoint
from clearvla.benchmarks.calvin_raw import CalvinRawReader,virtualize_calvin_cached_prefix

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);m=json.loads(a.manifest.read_text())
 names=[x for split in ['train','val','test'] for x in m['splits'][split]]
 episodes,skipped=load_episodes(Path(m['cached_prefix_root']),'*.hdf5',cameras=('top','wrist'),min_length=1,action_key='action',action_state_key='action_state',state_key='state',camera_key_overrides={'top':'observations/images/cam_high','wrist':'observations/images/cam_right_wrist'},episode_names=names)
 assert not skipped,(len(skipped),skipped[:3]);print('loaded',len(episodes),flush=True)
 reader=CalvinRawReader(Path(m['raw_source']),val_fraction=m['train_validation_fraction'],split_seed=m['split_seed'])
 overlaid,report=virtualize_calvin_cached_prefix(episodes,reader,expected_splits=m['splits'],raw_episode_map=m['raw_episode_map'])
 raw={ep.episode_id:ep for split in ['train','val','test'] for ep in reader.episodes(split)};split_by={ep:split for split in ['train','val','test'] for ep in m['splits'][split]};counts={split:{'before':Counter(),'after':Counter()} for split in ['train','val','test']};rows=[]
 for episode in overlaid:
  source=raw[m['raw_episode_map'][episode.episode_id]];before=resolve_annotation_endpoint(episode)
  if episode.source_annotation_index is not None:assert episode.source_annotation_index==source.annotation.index
  candidate=replace(episode,source_annotation_index=int(source.annotation.index));after=resolve_annotation_endpoint(candidate)
  assert after.status=='annotated-end-observation' and after.index==source.annotation.end-source.context_start
  assert after.index<candidate.cached_frame_count
  split=split_by[episode.episode_id];counts[split]['before'][before.status]+=1;counts[split]['after'][after.status]+=1
  rows.append({'episode':episode.episode_id,'raw_episode':source.episode_id,'split':split,'verified_annotation_index':int(source.annotation.index),'last_real_observation_index':int(after.index),'cached_frame_count':int(candidate.cached_frame_count)})
 summary={'episodes':len(episodes),'counts':counts,'overlay_checks_passed':True,'labels_are_success_flags':False,
  'missing_field':'source_annotation_index','source':'strictly matched raw.annotation.index',
  'unchanged':'RGB/state/action arrays, source bounds, window selection, splits and checkpoint',
  'scope':'audit-only dataclass replacement through actual resolver; production overlay not edited',
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'verified_sources.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
