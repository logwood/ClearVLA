"""Audit recorded body contacts, support changes and native execution by phase."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def spans(a):
    d=np.diff(np.r_[False,a,False].astype(int))
    return [[int(s),int(e-1)] for s,e in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]

def main():
    q=argparse.ArgumentParser();q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=q.parse_args();a.output.mkdir(exist_ok=True,parents=True);rows=[]
    for case in sorted(a.root.glob('case_*')):
        if not case.is_dir():continue
        r=json.loads((case/'result.json').read_text());tel=json.loads((case/'environment_info.json').read_text());info=tel['info']
        with np.load(case/'trajectory.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        cid=int(case.name.split('_')[1]);target='block_'+r['task'].split('_')[1];sign=-1 if r['task'].endswith('left') else 1
        objs=info[0]['scene_info']['movable_objects'];robot_uid=info[0]['robot_info']['uid'];names=list(objs)
        pos={n:np.array([f['scene_info']['movable_objects'][n]['current_pos'] for f in info]) for n in names}
        contact_graph={};contact_samples=[]
        for n in names:
            others=set()
            for f in info:
                others.update((int(c[2]),int(c[4])) for c in f['scene_info']['movable_objects'][n]['contacts'] if c[9]>0)
            contact_graph[n]={str(key):spans([any((int(c[2]),int(c[4]))==key and c[9]>0 for c in f['scene_info']['movable_objects'][n]['contacts']) for f in info]) for key in sorted(others)}
        start_support=set((int(c[2]),int(c[4])) for c in info[0]['scene_info']['movable_objects'][target]['contacts'] if c[2]!=robot_uid)
        supports=[set((int(c[2]),int(c[4])) for c in f['scene_info']['movable_objects'][target]['contacts'] if c[2]!=robot_uid) for f in info]
        # CALVIN push_object requires displacement > 0.1 m AND initial support retained.
        support_ok=np.array([bool(start_support) and bool(s) and start_support<=s for s in supports])
        progress=sign*(pos[target][:,0]-pos[target][0,0]);moving=np.abs(np.diff(pos[target][:,0]))>0.0005
        windows=[]
        for s,e in spans(moving):
            e+=1;robot_contacts=[];other_contacts=[]
            for t in range(s,e+1):
                for c in info[t]['scene_info']['movable_objects'][target]['contacts']:
                    if c[9]<=0:continue
                    item={'state':t,'other_uid':int(c[2]),'other_link':int(c[4]),'force':float(c[9])}
                    (robot_contacts if c[2]==robot_uid else other_contacts).append(item)
            windows.append({'states':[s,e],'target_delta':(pos[target][e]-pos[target][s]).tolist(),
                'robot_contact_count':len(robot_contacts),'other_contact_uids':sorted(set(c['other_uid'] for c in other_contacts)),
                'robot_contact_max_force':max([c['force'] for c in robot_contacts],default=0),
                'object_contact_samples':[c for c in other_contacts if c['other_uid'] in [o['uid'] for o in objs.values()]]})
        specific={2:(72,104),5:(32,104),9:(240,288),10:(72,152),11:(32,104),14:(328,360),15:(328,352),17:(120,152)}
        if cid in specific:
            lo,hi=specific[cid]
            for t in range(lo,min(hi,len(info)-1)+1):
                cs=[c for c in info[t]['scene_info']['movable_objects'][target]['contacts'] if c[2]==robot_uid and c[9]>0]
                if cs:
                    contact_samples.append({'state':t,'tcp':d['robot_obs'][t,:3].tolist(),'target':pos[target][t].tolist(),
                        'native_action_causing_transition':d['executed'][t-1].tolist(),'gripper_width':info[t]['robot_info']['gripper_opening_width'],
                        'target_contacts':[list(c) for c in cs]})
        row={'case_id':cid,'case':case.name,'success':r['success'],'target':target,'object_uid_map':{n:o['uid'] for n,o in objs.items()},
             'initial_support':sorted(start_support),'support_lost_spans':spans(~support_ok),
             'max_signed_progress':float(progress.max()),'final_signed_progress':float(progress[-1]),
             'target_z_min':float(pos[target][:,2].min()),'target_z_initial':float(pos[target][0,2]),
             'contact_graph':contact_graph,'target_motion_windows':windows,'target_robot_contact_samples':contact_samples}
        rows.append(row)
        if cid in specific:
            lo,hi=specific[cid];hi=min(hi,len(info)-1);states=np.arange(lo,hi+1);act=d['executed'][lo:hi]
            fig,ax=plt.subplots(3,2,figsize=(12,9),layout='constrained');fig.suptitle(f"Case {cid:02d}: {r['instruction']} ({'success' if r['success'] else 'failure'})")
            for n,color in [('block_red','red'),('block_blue','blue'),('block_pink','magenta')]:
                ax[0,0].plot(states,pos[n][states,0],color=color,label=n)
            ax[0,0].plot(states,d['robot_obs'][states,0],color='black',label='TCP');ax[0,0].set_ylabel('world X (m)');ax[0,0].legend(fontsize=8)
            ax[0,1].plot(states,d['robot_obs'][states,2],label='TCP Z');ax[0,1].plot(states,pos[target][states,2],label='target Z');ax[0,1].set_ylabel('world Z (m)');ax[0,1].legend()
            for j,l in enumerate(['x','y','z']):ax[1,0].step(states[:-1],act[:,j],where='post',label=l)
            ax[1,0].set_ylabel('submitted native xyz');ax[1,0].legend()
            ax[1,1].step(states[:-1],act[:,6],where='post',label='gripper command (+ open)')
            ax[1,1].plot(states,np.array([f['robot_info']['gripper_opening_width'] for f in info])[states]*10,label='opening width x 10');ax[1,1].legend(fontsize=8)
            ctrl=np.array([f['target_pos'] for f in tel['controller_targets']]);gap=np.linalg.norm(ctrl-d['robot_obs'][:,:3],axis=1)
            ax[2,0].plot(states,100*gap[states]);ax[2,0].set_ylabel('controller target / TCP gap (cm)')
            uids={robot_uid:'robot',**{o['uid']:n for n,o in objs.items()}}
            for uid,name in uids.items():
                if uid==objs[target]['uid']:continue
                force=np.array([sum(c[9] for c in f['scene_info']['movable_objects'][target]['contacts'] if c[2]==uid and c[9]>0) for f in info])
                ax[2,1].plot(states,force[states],label=name)
            ax[2,1].set_ylabel('sampled target contact force (N)');ax[2,1].legend(fontsize=8)
            for axes in ax.flat:
                axes.set_xlabel('recorded state');axes.grid(alpha=.2)
                for t in range((lo//8+1)*8,hi,8):axes.axvline(t,color='gray',alpha=.15)
            fig.savefig(a.output/f'case_{cid:02d}_contact_mechanism.png',dpi=140);plt.close(fig)
            points=[lo,min(lo+8,hi),min(lo+16,hi),min(lo+24,hi),hi]
            fig,ax=plt.subplots(2,len(points),figsize=(13,5),layout='constrained')
            for j,t in enumerate(points):
                for k,key in enumerate(['rgb_static','rgb_gripper']):ax[k,j].imshow(d[key][t]);ax[k,j].set_title(f'{key} state {t}');ax[k,j].axis('off')
            fig.savefig(a.output/f'case_{cid:02d}_images.png',dpi=150);plt.close(fig)
    (a.output/'contact_mechanisms.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        if not row['success']:
            print(json.dumps({k:row[k] for k in ['case_id','target','initial_support','support_lost_spans','max_signed_progress','target_z_min','target_motion_windows']},separators=(',',':')))
if __name__=='__main__':main()
