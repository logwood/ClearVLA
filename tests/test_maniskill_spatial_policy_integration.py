"""Native-input source/cache/update/adapter gates for posterior W/P2."""
from dataclasses import replace
import torch
import test_dinov3_spatial_effect_integration as base
from clearvla.mainline.p2_geometry import POSTERIOR_VIEW_TRANSPORT

def posterior_fixture_factory(original):
    def fixture(tmp_path, amp=False):
        cfg,bundle=original(tmp_path,amp)
        cfg=replace(cfg,top=replace(cfg.top,p2_geometry_mode=POSTERIOR_VIEW_TRANSPORT))
        cfg.validate()
        return cfg,bundle
    return fixture

def test_native_posterior_update_and_rebuild(tmp_path,monkeypatch):
    original=base.fixture
    monkeypatch.setattr(base,"fixture",posterior_fixture_factory(original))
    cfg,bundle=base.fixture(tmp_path,True)
    batch=base.batch_for(cfg,bundle)
    model,engine=base.baseline.engine(cfg,bundle)
    tracked={n:p for n,p in model.named_parameters() if "spatial_posterior_context" in n}
    assert len(tracked)==4
    before={n:p.detach().clone() for n,p in tracked.items()}
    engine.train_step(batch)
    for n,p in tracked.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,n
        assert not torch.equal(p,before[n]),n
    model.eval()
    with torch.no_grad():
        cache,_,_=model.encode_online(batch.online)
        assert cache.top.predicted_dynamics.camera_position_probability is cache.top.belief.camera_position_probability
        result=base.sample_action(model,batch.online,cfg,generator=torch.Generator().manual_seed(57))
    assert result.action.shape==(1,24,7) and torch.isfinite(result.action).all()
    assert cache.top.predicted_dynamics.camera_position_probability.shape==(1,cfg.top.object_slots,2,16,16)

def test_native_posterior_fresh_checkpoint_adapter_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setattr(base,"fixture",posterior_fixture_factory(base.fixture))
    base.test_native_fresh_checkpoint_exact_adapter_parity(tmp_path,monkeypatch)

