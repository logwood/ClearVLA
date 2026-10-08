"""Compare the predeclared repair arms using renderer labels and complete episodes."""
from __future__ import annotations
import argparse
import html
from itertools import permutations
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):return json.loads(path.read_text())
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def grounding(path):
    with np.load(path,allow_pickle=False) as z:
        density=z['g_density'].astype(np.float64)
        masks=z['masks'].transpose(1,0,2,3)
        binding=z['binding_probability'][0,:-1].astype(np.float64)
    slots,cameras,height,width=density.shape
    visible=masks.sum((-2,-1))>=20
    grid=np.stack(np.meshgrid(np.linspace(-1,1,width),np.linspace(-1,1,height)),axis=-1)
    camera_mass=density.sum((-2,-1))
    conditional=density/np.maximum(camera_mass[:,:,None,None],1e-30)
    center=np.einsum('kcyx,yxd->kcd',conditional,grid)
    target=np.einsum('ocyx,yxd->ocd',masks.astype(float),grid)/np.maximum(masks.sum((-2,-1))[...,None],1)
    region=np.stack([[cv2.dilate(m.astype(np.uint8),np.ones((25,25),np.uint8))>0 for m in views] for views in masks])
    mass=np.einsum('kcyx,ocyx->koc',density,region.astype(float))
    nviews=np.maximum(visible.sum(-1),1)
    nll=np.where(visible[None],-np.log(np.maximum(mass*nviews[None,:,None],1e-30)),0).sum(-1)/nviews
    distance=((center[:,None]-target[None])**2).sum(-1)
    mean_distance=np.where(visible[None],distance,0).sum(-1)/nviews
    cost=nll+mean_distance
    choice=min(permutations(range(slots),2),key=lambda pair:cost[pair[0],0]+cost[pair[1],1])
    rows=[]
    for obj,name in enumerate(('red','green')):
        k=choice[obj]
        for camera in range(cameras):
            if not visible[obj,camera]:continue
            rows.append(dict(object=name,camera=camera,slot=k,
                conditional_region_mass=float(mass[k,obj,camera]/max(camera_mass[k,camera],1e-30)),
                joint_region_mass=float(mass[k,obj,camera]),
                centroid_error_px=float(np.sqrt(distance[k,obj,camera])*(height-1)/2),
                camera_mass=float(camera_mass[k,camera])))
    return dict(views=rows,
        binding_composed_red_region_mass=float(np.einsum('k,kc->',binding,mass[:,0])),
        binding_composed_green_region_mass=float(np.einsum('k,kc->',binding,mass[:,1])),
        binding_to_matched_red=float(binding[choice[0]]),
        binding_to_matched_green=float(binding[choice[1]]),
        binding_null_mass=float(1-binding.sum()))


def paired_episode_interval(control, candidate, key):
    """Bootstrap episodes, keeping the repeated stage/view measurements together."""
    names=sorted({x['episode'] for x in control})
    differences=[]
    for name in names:
        a=[x[key] for x in control if x['episode']==name]
        b=[x[key] for x in candidate if x['episode']==name]
        assert len(a)==len(b) and len(a)>0
        differences.append(float(np.mean(b)-np.mean(a)))
    values=np.asarray(differences)
    rng=np.random.default_rng(20261008)
    samples=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(-1)
    return dict(candidate_minus_control=float(values.mean()),
        episode_bootstrap_95_percentile=np.quantile(samples,[.025,.975]).tolist(),
        per_episode=dict(zip(names,differences)),
        note='Descriptive paired episode bootstrap; one training seed, seven episodes per split.')


def episodes(folder):
    rows=[]
    for d in sorted(folder.glob('episode_*')):
        if not (d/'result.json').exists():continue
        result=load(d/'result.json')
        with np.load(d/'trajectory.npz',allow_pickle=False) as z:
            command=z['executed'];state=z['robot_obs_trajectory'];objects=z['object_positions']
            assert command.shape==(400,7) and state.shape==(401,15)
            assert objects.shape==(401,3,3)
            assert all(np.isfinite(x).all() for x in (command,state,objects))
            assert result['video_frames']==401 and result['steps']==400
            red=objects[:,0];green=objects[:,1];tcp=state[:,:3]
            close=np.flatnonzero((command[:,-1]<0)&(state[:-1,-1]>0))
            first=int(close[0]) if len(close) else None
            height=red[:,2]-red[0,2]
            metrics=[json.loads(x)['metrics'] for x in (d/'telemetry.jsonl').read_text().splitlines()]
            assert len(metrics)==401
            grasp=np.array([x['is_cubeA_grasped'] for x in metrics],bool)
            rows.append(dict(seed=result['seed'],folder=d.name,success=bool(result['success']),
                success_at_end=bool(result['success_at_end']),first_success_step=result['first_success_step'],
                first_close=first,first_close_xy_cm=float(np.linalg.norm(tcp[first,:2]-red[first,:2])*100) if first is not None else None,
                min_xy_cm=float(np.linalg.norm(tcp[:,:2]-red[:,:2],axis=-1).min()*100),
                max_red_lift_cm=float(height.max()*100),grasped=bool(grasp.any()),
                grasp_frames=int(grasp.sum()),
                ik_fallback_steps=sum(bool(x.get('telemetry_ik_fallback',False)) for x in metrics[1:]),
                ik_diagnostic_steps=sum('telemetry_ik_fallback' in x for x in metrics[1:]),
                grasped_lift_above_2cm=bool(((height>.02)&grasp).any()),
                initial_red_xyz=red[0].tolist(),initial_green_xyz=green[0].tolist(),
                first8_mean_xyz=command[:8,:3].mean(0).tolist(),steps=400,observations=401))
    assert len(rows)==18
    return rows


def episode_summary(rows):
    closed=[x['first_close_xy_cm'] for x in rows if x['first_close_xy_cm'] is not None]
    return dict(episodes=len(rows),successes=sum(x['success'] for x in rows),
        successes_at_end=sum(x['success_at_end'] for x in rows),
        ik_fallback_steps=sum(x['ik_fallback_steps'] for x in rows),
        ik_diagnostic_steps=sum(x['ik_diagnostic_steps'] for x in rows),
        closes=len(closed),grasp_episodes=sum(x['grasped'] for x in rows),
        grasped_lifts=sum(x['grasped_lift_above_2cm'] for x in rows),
        median_first_close_xy_cm=float(np.median(closed)) if closed else None,
        median_min_xy_cm=float(np.median([x['min_xy_cm'] for x in rows])),
        median_max_red_lift_cm=float(np.median([x['max_red_lift_cm'] for x in rows])))


def grounding_figure(control,candidate,destination,*,rgb_source=None,title='Fixed observation'):
    """Circles versus crosses make the image-coordinate error directly visible."""
    with np.load(rgb_source or control,allow_pickle=False) as z:rgb=z['rgb']
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for col,(arm,path) in enumerate((('Control',control),('Repair candidate',candidate))):
        measured=grounding(path)
        with np.load(path,allow_pickle=False) as z:density=z['g_density'].astype(float);masks=z['masks']
        for camera in range(2):
            ax=axes[camera,col];ax.imshow(rgb[camera]);errors=[]
            for obj,(name,color) in enumerate((('red','#ff3030'),('green','#2bff52'))):
                selected=[v for v in measured['views'] if v['object']==name and v['camera']==camera]
                if not selected:continue
                value=selected[0];mass=density[value['slot'],camera];total=mass.sum()
                if total<=1e-30:raise ValueError('Cannot display an underflowed conditional centroid')
                yy,xx=np.indices(mass.shape)
                predicted=np.array([(xx*mass).sum(),(yy*mass).sum()])/total
                y,x=np.nonzero(masks[camera,obj]);actual=np.array([x.mean(),y.mean()])
                ax.plot([actual[0],predicted[0]],[actual[1],predicted[1]],color=color,linewidth=1.5)
                ax.scatter(*actual,s=135,marker='o',facecolors='none',edgecolors='black',linewidth=4)
                ax.scatter(*actual,s=135,marker='o',facecolors='none',edgecolors=color,linewidth=2)
                ax.scatter(*predicted,s=130,marker='x',color='black',linewidth=5)
                ax.scatter(*predicted,s=130,marker='x',color=color,linewidth=2.5)
                errors.append(name+' '+str(round(value['centroid_error_px']))+' px')
            ax.set_title(arm+' · '+('global camera' if camera==0 else 'wrist camera')+'\n'+', '.join(errors),fontsize=12)
            ax.axis('off')
    fig.suptitle(title+'\nCircle = cube centre; cross = G read centroid',fontsize=15)
    fig.text(.5,.015,'Same RGB in both columns. Slots are matched by the evaluator; these are image reads, not 3D pose outputs.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.94));fig.savefig(destination,dpi=140);plt.close(fig)


def trajectory_figure(folders,records,destination):
    """Show the entire 400-action execution, including the failed late stages."""
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex='row')
    xy_limits=[];action_limit=.1
    for col,(arm,folder) in enumerate(folders):
        record=records[arm]
        with np.load(folder/record['folder']/'trajectory.npz',allow_pickle=False) as z:
            states=z['robot_obs_trajectory'];objects=z['object_positions'];actions=z['executed']
        telemetry=[json.loads(x)['metrics'] for x in (folder/record['folder']/'telemetry.jsonl').read_text().splitlines()]
        action_limit=max(action_limit,float(np.abs(actions[:,:3]).max())*1.08)
        grasp=np.array([x['is_cubeA_grasped'] for x in telemetry],bool)
        ax=axes[0,col]
        xy=states[:,:2]*100
        ax.plot(xy[:,0],xy[:,1],color='#2774bd',label='TCP path')
        ax.scatter(*xy[0],marker='o',color='#2774bd',label='TCP reset')
        ax.scatter(*xy[-1],marker='x',color='#2774bd',label='TCP end')
        for obj,color,name in ((0,'#d94c46','Red cube'),(1,'#279752','Green cube')):
            path=objects[:,obj,:2]*100
            ax.plot(path[:,0],path[:,1],color=color,alpha=.7)
            ax.scatter(*path[0],marker='s',s=65,color=color,label=name+' reset')
        first=record['first_close']
        if first is not None:
            ax.plot([xy[first,0],objects[first,0,0]*100],[xy[first,1],objects[first,0,1]*100],'--',color='#702a9c',label='First close miss')
            ax.scatter(*xy[first],color='#702a9c',s=35)
        xy_limits.append(np.concatenate([xy,objects[:,:2,:2].reshape(-1,2)*100]))
        ax.set(title=arm+' | '+('stack success' if record['success'] else 'no stack'),xlabel='World X (cm)',ylabel='World Y (cm)',aspect='equal')
        if col==0:ax.legend(fontsize=7,loc='best')
        ax=axes[1,col]
        ax.plot(np.arange(401),(objects[:,0,2]-objects[0,0,2])*100,color='#d94c46',label='Red cube lift')
        ax.fill_between(np.arange(401),0,1,where=grasp,transform=ax.get_xaxis_transform(),color='#2daa75',alpha=.2,label='Native grasp true')
        ax.axhline(2,color='#777',linestyle=':',label='2 cm lift')
        if first is not None:ax.axvline(first,color='#702a9c',linestyle='--',label='First close')
        ax.set(xlabel='Executed action index',ylabel='Lift above reset (cm)')
        if col==0:ax.legend(fontsize=7,loc='best')
        ax=axes[2,col]
        for d,color,name in ((0,'#2879bb','X'),(1,'#dd8c24','Y'),(2,'#6a3d9a','Z')):
            ax.plot(np.arange(400),actions[:,d],color=color,label=name,linewidth=.7)
        ax.set(xlabel='Executed action index',ylabel='Executed native XYZ command',ylim=(-1.05,1.05))
        if col==0:ax.legend(fontsize=7,ncol=3)
    all_xy=np.concatenate(xy_limits)
    lo=all_xy.min(0)-3;hi=all_xy.max(0)+3
    for ax in axes[0]:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    for ax in axes[2]:ax.set_ylim(-action_limit,action_limit)
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.suptitle('Seed '+str(records['control']['seed'])+' — complete matched executions',fontsize=16)
    fig.tight_layout();fig.savefig(destination,dpi=135);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/data/senwang/data/clearvla_sim/stackcube_bc_20260910_v2'))
    args=p.parse_args();r=args.root;out=r/'report';out.mkdir(exist_ok=True)
    target_file=out/'teacher_action_targets.npz'
    if target_file.exists():
        with np.load(target_file,allow_pickle=False) as z:teacher_targets={k:z[k] for k in z.files}
    else:
        teacher_targets={}
        for case in load(r/'control-heldout/summary.json')['rows']:
            with h5py.File(args.data/'experts'/(case['episode']+'.hdf5')) as source:
                teacher_targets[case['label']]=source['action'][case['step']:case['step']+8]
        np.savez_compressed(target_file,**teacher_targets)
    summary=dict(note='One predeclared 256-update pilot per arm. Grounding is evaluated with exact renderer masks, not training color masks. Slot assignment is an evaluator permutation, not a deployed identity guarantee. Partial validation covers 128 windows; 14 held-out episodes receive seven-stage spatial probes.',training={},grounding={},placements={},closed_loop={})
    trajectories={};ground_rows={};ground_cases={}
    source_rows={}
    for arm in ('control','candidate'):
        rows=[json.loads(x) for x in (r/(arm+'-pilot/metrics.jsonl')).read_text().splitlines()]
        epoch=[x for x in rows if x.get('kind')=='epoch'][-1]
        assert epoch['step']==6648
        train_keys=['loss_total','loss_group_action','loss_action_flow','loss_contrib_spatial_grounding','loss_spatial_grounding','loss_spatial_region_nll','loss_spatial_centroid_error_sq','loss_ledger_gap','gradient_epoch_preclip_l2_mean','runtime_seconds_per_batch']
        val_keys=['validation_action_rmse_normalized','validation_first_rmse_normalized','validation_first8_rmse_normalized','validation_tail_rmse_normalized',
            'validation_action_rmse_source_native','validation_arm_rmse_source_native','validation_gripper_rmse_source_native','validation_first_rmse_source_native',
            'validation_gripper_command_accuracy','validation_decoded_gripper_close_precision','validation_decoded_gripper_close_recall',
            'validation_decoded_gripper_close_f1','validation_decoded_gripper_events_predicted','validation_decoded_gripper_events_target',
            'validation_decoded_gripper_event_ratio','loss_spatial_grounding']
        summary['training'][arm]={'step':epoch['step'],'train':{k:epoch['train'].get(k) for k in train_keys},'validation':{k:epoch['validation'].get(k) for k in val_keys}}
        spikes=[x['gradient_global_preclip_l2'] for x in rows if x.get('kind')=='gradient_spike']
        logged=[v for x in rows if x.get('kind')=='train' for v in x['metrics'].values() if isinstance(v,(float,int))]
        summary['training'][arm]['health']=dict(gradient_spikes_above_5=len(spikes),max_spike_before_clipping=max(spikes) if spikes else None,
            nonfinite_logged_train_scalars=sum(not np.isfinite(v) for v in logged),completed_updates=256,
            logged_learning_rate_range=[min(x['metrics']['learning_rate'] for x in rows if x.get('kind')=='train'),
                max(x['metrics']['learning_rate'] for x in rows if x.get('kind')=='train')])
        rows=[];cases=[]
        source_rows[arm]=load(r/(arm+'-heldout/summary.json'))['rows']
        assert len(source_rows[arm])==98
        for case in source_rows[arm]:
            value=grounding(r/(arm+'-heldout')/(case['label']+'.npz'))
            with np.load(r/(arm+'-heldout')/(case['label']+'.npz'),allow_pickle=False) as z:prediction=z['action'][:8]
            target=teacher_targets[case['label']]
            assert prediction.shape==target.shape==(8,7)
            error=prediction-target
            np.testing.assert_allclose(np.sqrt(np.mean(error[:,:6]**2)),case['arm_rmse_first8'],rtol=1e-5,atol=1e-7)
            value.update(translation_rmse_first8=float(np.sqrt(np.mean(error[:,:3]**2))),
                rotation_rmse_first8=float(np.sqrt(np.mean(error[:,3:6]**2))),
                gripper_sign_accuracy_first8=float(np.mean((prediction[:,-1]>0)==(target[:,-1]>0))))
            for view in value['views']:
                rows.append(dict(label=case['label'],split=case['split'],episode=case['episode'],stage=case['stage'],**view))
            cases.append(dict(label=case['label'],split=case['split'],episode=case['episode'],
                stage=case['stage'],arm_rmse_first8=case['arm_rmse_first8'],
                **{k:v for k,v in value.items() if k!='views'}))
        ground_rows[arm]=rows
        ground_cases[arm]=cases
        summary['grounding'][arm]={}
        for split in ('val','test'):
            selected=[x for x in rows if x['split']==split]
            summary['grounding'][arm][split]=dict(views=len(selected),episodes=len(set(x['episode'] for x in selected)),
                mean_conditional_region_mass=float(np.mean([x['conditional_region_mass'] for x in selected])),
                mean_joint_region_mass=float(np.mean([x['joint_region_mass'] for x in selected])),
                mean_centroid_error_px=float(np.mean([x['centroid_error_px'] for x in selected])))
        summary['placements'][arm]=[]
        for x in load(r/(arm+'-placements/summary.json'))['rows']:
            row={k:x[k] for k in ('label','xy','moved','first8_mean_xyz','tcp_xyz')}
            toward=np.array(x['positions'][0][:2])-np.array(x['tcp_xyz'][:2])
            row['first8_mean_xy_projection_toward_red']=float(np.dot(x['first8_mean_xyz'][:2],toward)/max(np.linalg.norm(toward),1e-12))
            summary['placements'][arm].append(row)
        trajectories[arm]=episodes(r/(arm+'-eval18'))
        summary['closed_loop'][arm]=episode_summary(trajectories[arm])
    # Frozen expert observations, evaluator labels and sampling noise must agree.
    for a,b in zip(source_rows['control'],source_rows['candidate']):
        assert all(a[k]==b[k] for k in ('label','split','episode','stage','step'))
        with np.load(r/'control-heldout'/(a['label']+'.npz'),allow_pickle=False) as za, np.load(r/'candidate-heldout'/(b['label']+'.npz'),allow_pickle=False) as zb:
            for key in ('rgb','masks','object_positions','robot_state','initial_noise'):
                np.testing.assert_array_equal(za[key],zb[key],err_msg=a['label']+'/'+key)
    summary['heldout_pairing']={'cases':98,'same_rgb_state_masks_poses_sampling_noise':True}
    summary['grounding_paired_intervals']={}
    summary['teacher_forced_by_stage']={}
    for split in ('val','test'):
        a=[x for x in ground_rows['control'] if x['split']==split]
        b=[x for x in ground_rows['candidate'] if x['split']==split]
        assert [(x['label'],x['object'],x['camera']) for x in a]==[(x['label'],x['object'],x['camera']) for x in b]
        summary['grounding_paired_intervals'][split]={key:paired_episode_interval(a,b,key) for key in ('conditional_region_mass','centroid_error_px')}
        summary['teacher_forced_by_stage'][split]={}
        for arm in ('control','candidate'):
            summary['teacher_forced_by_stage'][split][arm]={}
            for stage in ('reset','approach','preclose','close','lift','transport','release'):
                c=[x for x in ground_cases[arm] if x['split']==split and x['stage']==stage]
                assert len(c)==7
                keys=('arm_rmse_first8','translation_rmse_first8','rotation_rmse_first8','gripper_sign_accuracy_first8',
                    'binding_composed_red_region_mass','binding_composed_green_region_mass','binding_to_matched_red','binding_to_matched_green','binding_null_mass')
                summary['teacher_forced_by_stage'][split][arm][stage]={key:float(np.mean([x[key] for x in c])) for key in keys}
    # The paired episode layouts must be identical across arms and baseline.
    reference=r/'reference-baseline-eval18'
    if not reference.exists():
        reference.mkdir()
        for d in sorted(args.baseline.glob('episode_*')):
            dest=reference/d.name;dest.mkdir()
            for f in d.iterdir():
                if f.is_file() and f.suffix in ('.npz','.mp4','.json','.jsonl'):shutil.copy2(f,dest/f.name)
    trajectories['original']=episodes(reference)
    summary['closed_loop']['original']=episode_summary(trajectories['original'])
    for i in range(18):
        for arm in ('control','candidate'):
            assert trajectories[arm][i]['seed']==trajectories['original'][i]['seed']
            for key in ('initial_red_xyz','initial_green_xyz'):
                np.testing.assert_array_equal(trajectories[arm][i][key],trajectories['original'][i][key])
    summary['per_episode']=trajectories
    binding_rows=[]
    binding_path=out/'binding-counterfactual-r2/summary.json'
    if binding_path.exists():
        evidence=load(binding_path);assert len(evidence['rows'])==42
        summary['binding_counterfactual']={'note':evidence['note'],'admission':evidence['admission'],'arms':{}}
        for arm in ('control','candidate'):
            selected=[x for x in evidence['rows'] if x['arm']==arm]
            by_mode={mode:[dict(episode=x['episode'],error=x['scores'][mode]['translation_rmse']) for x in selected] for mode in ('none','red','green')}
            summary['binding_counterfactual']['arms'][arm]=dict(
                noop_action_delta_rmse_max=max(x['scores']['none']['action_delta_rmse'] for x in selected),
                translation_rmse={mode:float(np.mean([x['error'] for x in values])) for mode,values in by_mode.items()},
                red_minus_noop=paired_episode_interval(by_mode['none'],by_mode['red'],'error'),
                green_minus_noop=paired_episode_interval(by_mode['none'],by_mode['green'],'error'))
            for stage in ('reset','preclose','lift'):
                chosen=[x for x in selected if x['stage']==stage]
                binding_rows.append((arm,stage,*[f"{np.mean([x['scores'][m]['translation_rmse'] for x in chosen]):.5f}" for m in ('none','red','green')]))
    (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'grounding_per_view.json').write_text(json.dumps(ground_rows,indent=2,allow_nan=False)+'\n')
    (out/'grounding_per_query.json').write_text(json.dumps(ground_cases,indent=2,allow_nan=False)+'\n')
    archive=out/'all_54_full_trajectories_npz.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for arm,folder in [('original',reference),('control',r/'control-eval18'),('candidate',r/'candidate-eval18')]:
            for f in sorted(folder.rglob('trajectory.npz')):z.write(f,arm+'/'+f.parent.name+'/'+f.name)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    colors={'control':'#62748a','candidate':'#218d75'}
    for i,arm in enumerate(('control','candidate')):
        x=np.arange(2)+(i-.5)*.3
        ax[0].bar(x,[100*summary['grounding'][arm][s]['mean_conditional_region_mass'] for s in ('val','test')],.3,color=colors[arm],label=arm)
        ax[1].bar(x,[summary['grounding'][arm][s]['mean_centroid_error_px'] for s in ('val','test')],.3,color=colors[arm])
        ax[2].bar(np.arange(3)+(i-.5)*.3,[summary['closed_loop'][arm][k] for k in ('grasp_episodes','grasped_lifts','successes')],.3,color=colors[arm])
    ax[0].set(title='Cube-region read mass',ylabel='Mean camera-conditional mass (%)',xticks=[0,1],xticklabels=['Validation','Test'])
    ax[1].set(title='Cube centroid error',ylabel='Mean distance (pixels)',xticks=[0,1],xticklabels=['Validation','Test'])
    ax[2].set(title='Full closed-loop outcomes',ylabel='Episodes out of 18',xticks=[0,1,2],xticklabels=['Grasp','Grasp + lift','Stack'])
    for a in ax:a.grid(axis='y',alpha=.2)
    ax[0].legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    ground_table=[]
    for arm in ('control','candidate'):
        for split in ('val','test'):
            v=summary['grounding'][arm][split]
            ground_table.append((arm,split,v['episodes'],v['views'],f"{100*v['mean_conditional_region_mass']:.2f}%",f"{v['mean_centroid_error_px']:.2f}"))
    behavior=[]
    for arm in ('original','control','candidate'):
        v=summary['closed_loop'][arm]
        behavior.append((arm,str(v['successes'])+' / '+str(v['successes_at_end']),v['grasp_episodes'],v['grasped_lifts'],v['closes'],v['median_first_close_xy_cm']))
    stage_rows=[]
    for split in ('val','test'):
        for stage in ('reset','approach','preclose','close','lift','transport','release'):
            a=summary['teacher_forced_by_stage'][split]['control'][stage]
            b=summary['teacher_forced_by_stage'][split]['candidate'][stage]
            stage_rows.append((split,stage,f"{a['translation_rmse_first8']:.4f} / {b['translation_rmse_first8']:.4f}",
                f"{a['rotation_rmse_first8']:.4f} / {b['rotation_rmse_first8']:.4f}",
                f"{a['gripper_sign_accuracy_first8']:.3f} / {b['gripper_sign_accuracy_first8']:.3f}",
                f"{a['binding_to_matched_red']:.3f} / {b['binding_to_matched_red']:.3f}",
                f"{a['binding_composed_red_region_mass']:.3f} / {b['binding_composed_red_region_mass']:.3f}"))
    placement_rows=[]
    for a,b in zip(summary['placements']['control'],summary['placements']['candidate']):
        assert a['label']==b['label']
        placement_rows.append((a['label'],f"{a['first8_mean_xyz'][0]:+.4f}",f"{b['first8_mean_xyz'][0]:+.4f}",
            f"{a['first8_mean_xy_projection_toward_red']:+.4f}",f"{b['first8_mean_xy_projection_toward_red']:+.4f}"))
    grounding_images=[]
    for case in source_rows['control']:
        if case['stage']!='reset':continue
        filename='grounding_'+case['label']+'.png'
        grounding_figure(r/'control-heldout'/(case['label']+'.npz'),r/'candidate-heldout'/(case['label']+'.npz'),out/filename,
            title=case['split']+' '+case['episode']+' · reset')
        grounding_images.append('<details><summary>'+case['split']+' '+case['episode']+'</summary><img loading="lazy" src="'+filename+'"></details>')
    cases=[]
    for i in range(18):
        figure='seed_'+str(trajectories['control'][i]['seed'])+'_full.png'
        trajectory_figure([('original',reference),('control',r/'control-eval18'),('candidate',r/'candidate-eval18')],
            {arm:trajectories[arm][i] for arm in ('original','control','candidate')},out/figure)
        links=[]
        for arm,folder in [('original','reference-baseline-eval18'),('control','control-eval18'),('candidate','candidate-eval18')]:
            e=trajectories[arm][i];prefix='../'+folder+'/'+e['folder']+'/'
            links.append(f"<b>{arm}</b>: success={e['success']}, success at end={e['success_at_end']}, grasp={e['grasped']}, close miss={e['first_close_xy_cm']} cm · <a href='{prefix}trajectory.npz'>full NPZ</a> · <a href='{prefix}video.mp4'>full video</a>")
        cases.append('<details><summary>Seed '+str(trajectories['control'][i]['seed'])+'</summary><p>'+'</p><p>'.join(links)+'</p><img loading="lazy" src="'+figure+'"><p>Top: complete TCP and cube XY paths; purple line marks the miss at the first close command. Middle: red-cube lift, with green shading only while ManiSkill reports a grasp. Bottom: all 400 applied XYZ commands.</p></details>')
    mechanism=all(summary['grounding']['candidate'][s]['mean_conditional_region_mass']>summary['grounding']['control'][s]['mean_conditional_region_mass'] and summary['grounding']['candidate'][s]['mean_centroid_error_px']<summary['grounding']['control'][s]['mean_centroid_error_px'] for s in ('val','test'))
    behavior_gain=summary['closed_loop']['candidate']['successes']>summary['closed_loop']['control']['successes']
    verdict=('Grounding improves on both held-out splits. ' if mechanism else 'The pilot does not qualify a consistent grounding improvement. ')+('The common panel also gains stack successes; fresh seeds are still required before promotion.' if behavior_gain else 'The common panel does not gain stack successes; the checkpoint is not a qualified policy repair.')
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>ManiSkill spatial repair comparison</title><style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#17212b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #cdd5de;text-align:left}}th{{background:#edf2f7}}details{{padding:12px;border:1px solid #ccd4dc;margin:10px 0}}a{{color:#1468a0}}</style></head><body>
<h1>Does spatial supervision repair the policy?</h1><p><b>{verdict}</b></p>
<p>Both arms start from epoch 8 / step 6392, retain the same named Adam moments, use the same 256 updates and minibatch seed, and finish at step 6648. Candidate adds only a 0.01-weight current-region objective. Online graph, inputs, action chart and Q5/replan-8 execution are unchanged.</p>
<img src="comparison.png"><h2>Grounding against renderer labels</h2>{table(('Arm','Split','Episodes','Visible object/views','Region mass','Centroid error (px)'),ground_table)}
<p>All seven validation and seven test episodes are probed at seven stages. G slots are matched with a single evaluator permutation across cameras. Regions expand exact visible actor masks by 12 pixels. These measure read localization, not calibrated 3D poses or guaranteed semantic identity; camera-conditional and joint masses remain separate in the JSON.</p>
<details><summary>See the difference: all 14 held-out reset layouts</summary>{''.join(grounding_images)}</details>
<details><summary>Does localization reach target binding and actions? All seven expert stages</summary>
{table(('Split','Stage','XYZ RMSE (C / R)','Rotation RMSE (C / R)','Gripper accuracy (C / R)','S mass on red slot (C / R)','S × G red-region mass (C / R)'),stage_rows)}
<p>RMSE uses the first eight native six-channel arm commands against expert commands, with the same observed expert history and sampling noise. It is not a closed-loop score. C / R means control / repair candidate. S mass uses the evaluator-matched red slot; it does not assert stable physical identity.</p></details>
<details><summary>Controlled object placements: every signed action response</summary>
{table(('Moved cube and XY metres','Control X command','Candidate X command','Control toward-red projection','Candidate toward-red projection'),placement_rows)}
<p>All values average the first eight native commands. Positive XY projection points toward the red cube; negative points away. This tests the immediate response to physical placement, not success or multi-step reachability.</p></details>
<details><summary>Evaluator-only target-binding interventions</summary>
{table(('Arm','Expert stage','Unchanged XYZ RMSE','Red-slot XYZ RMSE','Green-slot XYZ RMSE'),binding_rows)}
<p>On seven validation demonstrations, each existing shared binding is concentrated on the renderer-matched red or green slot while preserving its original real/null mass. G is numerically unchanged and every case has a no-op repeat. The slot remains a learned representation, not an oracle physical state. These fixed-observation probes are not deployment results and do not change either closed-loop policy. <a href="binding-counterfactual-r2/summary.json">Every probe and numerical admission</a>.</p></details>
<h2>Physical outcomes</h2>{table(('Arm','Stacks any / at end (of 18)','Grasps / 18','Grasp + 2cm lift / 18','Episodes with close','Median first-close XY miss (cm)'),behavior)}
<p>The close-error median includes only episodes that issue a close; compare its denominator and per-seed records. All episodes retain 400 actions and 401 observations. A better loss or heatmap alone cannot qualify the repair.</p>
<p><a href="comparison.json">Complete metrics, paired episode uncertainty and results</a> · <a href="grounding_per_view.json">Every grounding view</a> · <a href="grounding_per_query.json">Every binding/action query</a> · <a href="all_54_full_trajectories_npz.zip">All 54 full trajectory NPZ files</a> · <a href="../preflight-admission.json">Preflight and restored-output parity</a> · <a href="../label_audit.json">Training-label audit</a></p>
<h2>Every paired trajectory</h2>{''.join(cases)}
<p>Source and objective specification: <code>configs/mainline/maniskill_spatial_repair_experiment_20261008.json</code>. This is a bounded single-seed training pilot, not a statistically established deployment result.</p></body></html>'''
    (out/'index.html').write_text(body,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('training','grounding','closed_loop')},indent=2))
    print(verdict)


if __name__=='__main__':main()
