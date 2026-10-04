"""C7 native terminal, actual engine, ordinary CE, no runtime holding rule."""
from dataclasses import replace
import gc
import pytest
import torch
from test_causal_stage2_integration import config as previous_config, batch, _model_engine
from test_mainline_global_task import abi_for
from clearvla.mainline.config import load_config, config_from_mapping
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.runtime.deployment import validate_deployment_abi
from clearvla.mainline.runtime.checkpoints import save_checkpoint,load_checkpoint_exact
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.mainline.command_sequence import CONDITIONAL_COMMANDS


def config(amp=False):
    c=previous_config(amp)
    return replace(c,bottom=replace(c.bottom,command_sequence_mode=CONDITIONAL_COMMANDS))


@pytest.fixture(autouse=True)
def release():
    gc.collect();yield;gc.collect()


@pytest.mark.parametrize('amp',[False,True])
def test_zero_start_and_real_two_updates_preserve_arm_ownership(amp):
    c=config(amp);b=batch(c);dtype=torch.bfloat16 if amp else torch.float32
    torch.manual_seed(1720);m,e=_model_engine(c)
    torch.manual_seed(1720);old,_=_model_engine(previous_config(amp))
    for n,p in old.state_dict().items():torch.testing.assert_close(p,m.state_dict()[n],rtol=0,atol=0)
    assert e.dtype==dtype
    m.eval();old.eval()
    with torch.no_grad():
        a=sample_action(m,b.online,c,dtype=dtype,generator=torch.Generator().manual_seed(1730))
        z=sample_action(old,b.online,previous_config(amp),dtype=dtype,generator=torch.Generator().manual_seed(1730))
    torch.testing.assert_close(a.action,z.action,rtol=0,atol=0)
    torch.testing.assert_close(a.physical_field,z.physical_field,rtol=0,atol=0)
    del old,z,a;gc.collect()
    selected={n:p for n,p in m.named_parameters() if '.command_sequence.' in n}
    assert len(selected)==1
    owned=[p for g in e.optimizer.param_groups for p in g['params']]
    assert all(p.requires_grad and sum(p is o for o in owned)==1 for p in selected.values())
    for _ in range(2):
        out=e.train_step(b)
        assert torch.isfinite(out.loss)
    for n,p in selected.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,n
        assert p.detach().abs().max()>0,n
    # A changed temporal head cannot directly modify arm in ONE fixed velocity
    # call. Across proposal/W/refined it can indirectly affect arm via gripper
    # candidate conditioning, so do not make an incorrect whole-sampler claim.
    m.eval()
    with torch.no_grad():
        cache,_,_=m.encode_online(b.online);noise=torch.randn(1,24,18);time=torch.full((1,),.6)
        before=m.velocity(cache,noisy_action_field=noise,time=time)
        weight=next(iter(selected.values()));weight.add_(torch.randn_like(weight)*.3)
        after=m.velocity(cache,noisy_action_field=noise,time=time)
    torch.testing.assert_close(before.bottom.physical_velocity,after.bottom.physical_velocity,rtol=0,atol=0)
    assert not torch.equal(before.bottom.gripper_command_logits,after.bottom.gripper_command_logits)


def test_whole_sampler_boundary_is_once_per_observation_not_unexecuted_proposal():
    c=config();m,_=_model_engine(c);m.eval();b=batch(c)
    count=[];seen=[];steps=[]
    original=m.outlet_adapter.prepare_binary_command_boundary
    def prepare(step):
        value=original(step);count.append(value);steps.append(step);return value
    m.outlet_adapter.prepare_binary_command_boundary=prepare
    ctrl=m.execution_bottom.decoder.terminal_controller
    hook=ctrl.command_sequence.register_forward_pre_hook(lambda module,args:seen.append(args[2]))
    try:
        with torch.no_grad():
            out=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1732))
            out2=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1732))
        assert len(count)==2 and len(seen)==24
        assert count[0] is not count[1]
        assert all(x is count[0].probability for x in seen[:12])
        assert all(x is count[1].probability for x in seen[12:])
        torch.testing.assert_close(out.action,out2.action,rtol=0,atol=0)
        assert all(s is b.online.history.executed_robot_step for s in steps)
    finally:m.outlet_adapter.prepare_binary_command_boundary=original;hook.remove()


def test_future_labels_do_not_change_online_chain_or_logits():
    c=config();m,_=_model_engine(c);m.eval();b=batch(c)
    with torch.no_grad():
        cache,state,_=m.encode_online(b.online)
        noise=torch.randn(1,24,18);time=torch.full((1,),.8)
        before=m.velocity(cache,noisy_action_field=noise,time=time)
        future=replace(b.future,action_sequence=-b.future.action_sequence)
        m.build_training_targets(state,future)
        after=m.velocity(cache,noisy_action_field=noise,time=time)
    torch.testing.assert_close(before.bottom.gripper_command_logits,after.bottom.gripper_command_logits,rtol=0,atol=0)
    torch.testing.assert_close(before.bottom.physical_velocity,after.bottom.physical_velocity,rtol=0,atol=0)


@pytest.mark.parametrize('kind',['missing','foreign','mutated','chart'])
def test_chain_source_rejection_before_decoder(kind):
    c=config();m,_=_model_engine(c);m.eval();b=batch(c)
    with torch.no_grad():cache,_,_=m.encode_online(b.online)
    if kind=='missing':cache=replace(cache,command_boundary=None)
    elif kind=='foreign':cache=replace(cache,command_boundary=replace(cache.command_boundary,source=replace(cache.command_boundary.source,command=cache.command_boundary.source.command.clone())))
    elif kind=='mutated':cache.command_boundary.probability.add_(.1)
    else:cache=replace(cache,command_boundary=replace(cache.command_boundary,normalizer_fingerprint='other'))
    calls=[];h=m.execution_bottom.decoder.register_forward_pre_hook(lambda *a:calls.append(1))
    try:
        with pytest.raises(ValueError,match='command chain'):
            m.velocity(cache,noisy_action_field=torch.zeros(1,24,18),time=torch.zeros(1))
        assert not calls
    finally:h.remove()


def test_modes_preset_and_objectives():
    a=load_config('configs/mainline/dinov3_causal_chain_stage2_calvin.json')
    b=load_config('configs/mainline/dinov3_causal_chain_stage3_calvin.json')
    assert replace(b,bottom=replace(b.bottom,command_sequence_mode=a.bottom.command_sequence_mode),data=replace(b.data,output_dir=a.data.output_dir))==a
    assert b.objectives==a.objectives and b.runtime==a.runtime
    assert config_from_mapping(b.as_dict())==b
    assert 'command_sequence_mode' not in a.as_dict()['bottom']
    for c in [replace(b,top=replace(b.top,robot_feedback_mode='none')),replace(b,bottom=replace(b.bottom,endpoint_supervision_mode='interior_heads_v1')),replace(b,bottom=replace(b.bottom,gripper_output_mode='continuous')),replace(b,bottom=replace(b.bottom,command_sequence_mode='implicit_hold'))]:
        with pytest.raises(ValueError):c.validate()


@pytest.mark.parametrize('mutation',['missing','altered','unselected'])
def test_abi_identity(tmp_path,mutation):
    c=previous_config() if mutation=='unselected' else config()
    _,abi,_=abi_for(c,tmp_path);validate_deployment_abi(abi)
    if mutation=='missing':abi.pop('binary_command_sequence')
    elif mutation=='altered':abi['binary_command_sequence']['boundary']='future_target'
    else:abi['binary_command_sequence']={'schema':'fake'}
    with pytest.raises(ValueError,match='command sequence'):validate_deployment_abi(abi)


def test_new_checkpoint_exact_restore_and_deployment(tmp_path):
    c=config();torch.manual_seed(1735);m,e=_model_engine(c);b=batch(c)
    identity,abi,data=abi_for(c,tmp_path);m.configure_action_normalizer(data.action_normalizer)
    e.train_step(b);m.eval();path=tmp_path/'candidate.pt'
    save_checkpoint(path,model=m,optimizer=e.optimizer,schedule=e.schedule,config=c,identity=identity,epoch=0,global_step=1,best_metric=None,data_state={'action_normalizer':data.action_normalizer.to_dict(),'state_normalizer':data.state_normalizer.to_dict(),'deployment_abi':abi})
    with torch.no_grad():expected=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1736)).action
    other,oe=_model_engine(c)
    load_checkpoint_exact(path,model=other,optimizer=oe.optimizer,schedule=oe.schedule,config=c,identity=identity)
    for n,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[n],rtol=0,atol=0)
    with pytest.raises(ValueError):load_checkpoint_exact(path,model=other,optimizer=oe.optimizer,schedule=oe.schedule,config=previous_config(),identity=identity)
    del m,e,other,oe;gc.collect()
    loaded=load_deployment_checkpoint(path,device=torch.device('cpu'))
    with torch.no_grad():actual=sample_action(loaded.model,b.online,c,generator=torch.Generator().manual_seed(1736)).action
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
