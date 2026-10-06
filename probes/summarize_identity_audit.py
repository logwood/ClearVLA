"""Reduce admitted mask probes without interpreting mask overlap as identity truth."""
from pathlib import Path
import argparse,json
import numpy as np

def spread(v):
    if not v:return None
    x=np.asarray(v,float);return {'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}

def main():
    q=argparse.ArgumentParser();q.add_argument('--input',type=Path,nargs='+',required=True);q.add_argument('--output',type=Path,required=True)
    a=q.parse_args();states=[];natural=[]
    for root in a.input:
        states+=json.loads((root/'all_identity_summaries.json').read_text())
        natural+=json.loads((root/'all_natural_interventions.json').read_text())
    assert len({(x['case_id'],x['state']) for x in states})==len(states)
    rows=[]
    for s in states:
        binding=np.asarray(s['binding']);row={'case_id':s['case_id'],'state':s['state'],'success':s['success'],'target':s['target'],
            'binding':binding.tolist(),'binding_top_k':int(binding.argmax()+1),'null':s['null'],'cameras':[]}
        for camera,prior in zip(s['coverage']['cameras'],s['prior_coverage']['cameras']):
            p=np.asarray(camera['k_object_probability']);idx=camera['object_names'].index(s['target']);v=p[:,idx]
            row['cameras'].append({'camera':camera['camera'],'target_pixels':camera['visible_pixels'][idx],
                'target_pixel_fraction':camera['image_area_fraction'][idx],
                'pre_G_target_mass':prior['k_object_probability'][0][idx],
                'K_target_mass':v.tolist(),'K_any_object_mass':p.sum(-1).tolist(),
                'K_other_pixel_mass':camera['k_other_pixel_probability'],
                'K_largest_visible_object':[camera['object_names'][i] for i in p.argmax(-1)],
                'best_target_k':int(v.argmax()+1),'best_target_k_binding':float(binding[v.argmax()]),
                'S_weighted_target_mask_mass':float(binding@v),
                'S_weighted_each_object_mask_mass':(binding@p).tolist()})
        rows.append(row)
    natural_rows=[]
    for n in natural:
        records=n['records'];repeat=next(x for x in records if x['repeat']);base_direction=n['baseline_instruction'].split()[-1]
        for r in records:
            kind='repeat' if r['repeat'] else ('color' if r['instruction'].split()[-1]==base_direction else 'direction')
            natural_rows.append({'case_id':n['case_id'],'state':n['state'],'kind':kind,'instruction':r['instruction'],
                'native_arm_rmse':r['arm_first8_difference']['rmse'],'native_arm_relative_rmse':r['arm_first8_difference']['relative_rmse'],
                'binding_max_abs':r['binding_difference']['max_abs'],'binding_top_k':int(np.argmax(r['binding'])+1),
                'gripper_mismatches':r['gripper_first8_mismatches'],'repeat_arm_rmse':repeat['arm_first8_difference']['rmse'],
                'node_rmse':{k:v['rmse'] for k,v in r['nodes'].items()},'G_max_abs':max(r['goal_free_G_max_abs'].values())})
    groups={}
    for name in ['top','wrist']:
        c=[c for r in rows for c in r['cameras'] if c['camera']==name and c['target_pixels']>0]
        groups[name]={'visible_target_windows':len(c),'max_K_target_mask_mass':spread([max(v['K_target_mass']) for v in c]),
            'pre_G_target_mask_mass':spread([v['pre_G_target_mass'] for v in c]),
            'K_any_object_mask_mass':spread([v for x in c for v in x['K_any_object_mass']])}
    summary={'states':len(rows),'cases':len(set(r['case_id'] for r in rows)),'natural_windows':len(natural),
        'source_center_closure_max':max(s['coverage']['source_center_max_abs_error'] for s in states),'coverage':groups,
        'language':{kind:{'arm_rmse':spread([r['native_arm_rmse'] for r in natural_rows if r['kind']==kind]),
            'binding_max_abs':spread([r['binding_max_abs'] for r in natural_rows if r['kind']==kind]),
            'gripper_mismatches_total':sum(r['gripper_mismatches'] for r in natural_rows if r['kind']==kind)} for kind in ['repeat','color','direction']},
        'notes':['Mask coverage is visible spatial support, not semantic information fraction.',
                 'Largest visible object is reported without certification: arbitrarily small coverage is not identity.',
                 'Sampling interventions use recorded causal history and shared initial physical noise, not original rollout RNG.']}
    a.output.mkdir(exist_ok=True,parents=True)
    for name,value in [('summary',summary),('identity_rows',rows),('natural_rows',natural_rows)]:
        (a.output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
