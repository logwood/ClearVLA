from dataclasses import replace
from pathlib import Path
import hashlib
import json
import pytest
import torch
from test_mainline_checkpoint import _reduced_joint_calvin_config, _dataset
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import (
    CALVIN_ENDPOINT_TRAJECTORY_REPAIR_V1_MIGRATION as MIGRATION,
    CALVIN_ENDPOINT_TRAJECTORY_REPAIR_V1_SOURCE_PATHS as PATHS,
    save_checkpoint, load_checkpoint_for_initialization,
)
from clearvla.mainline.training.optimizer import build_optimizer, WarmupCosineSchedule


def test_endpoint_trajectory_initialization_preserves_all_weights_and_rejects_drift(tmp_path):
    root = Path(__file__).resolve().parents[1]
    c = _reduced_joint_calvin_config()
    target_config = replace(c, objectives=replace(c.objectives,
        gripper_command_transition=.05, calvin_frame_weight_mode="motion_event_v1",
        calvin_frame_motion_gain=.75, calvin_frame_event_gain=1.5,
        calvin_frame_event_radius=1, calvin_frame_max_weight=3.0))
    condition = tmp_path / "language.pt"
    condition.write_bytes(b"fixed external language")
    language = ArtifactIdentity.from_file("t5_goal", condition)
    def identity(config):
        return build_checkpoint_identity(config, repo_root=root,
            dataset=_dataset(), language=language, commit="9"*40)
    saved_id = identity(c)
    current_id = identity(target_config)
    rows = tuple((n, hashlib.sha256(("prior:"+n).encode()).hexdigest() if n in PATHS else h)
        for n,h in saved_id.source.files)
    assert PATHS <= {n for n,h in rows}
    saved_id = replace(saved_id, source=replace(saved_id.source, files=rows,
        digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()))
    source = ClearVLAMainlinePolicy(c)
    opt,_ = build_optimizer(source,c)
    schedule=WarmupCosineSchedule(opt,warmup_steps=2,total_steps=4,minimum_ratio=.1)
    path=tmp_path/"source.pt"
    save_checkpoint(path,model=source,optimizer=opt,schedule=schedule,
        config=c,identity=saved_id,epoch=0,global_step=0,best_metric=None)
    target=ClearVLAMainlinePolicy(target_config)
    state=load_checkpoint_for_initialization(path,model=target,config=target_config,
        identity=current_id,model_contract_migration=MIGRATION)
    assert state.model_contract_migration==MIGRATION
    assert set(state.changed_source_files)==PATHS
    assert source.state_dict().keys()==target.state_dict().keys()
    for name,value in source.state_dict().items():
        assert torch.equal(value,target.state_dict()[name]),name
    bad=replace(target_config,objectives=replace(target_config.objectives,annotated_goal=2.0))
    with pytest.raises(ValueError,match="outside its six"):
        load_checkpoint_for_initialization(path,model=target,config=bad,
            identity=identity(bad),model_contract_migration=MIGRATION)
    changed=list(current_id.source.files)
    changed=[(n,hashlib.sha256(b"unrelated").hexdigest() if n.endswith("/model/grounding.py") else h) for n,h in changed]
    other=replace(current_id,source=replace(current_id.source,files=tuple(changed),
        digest=hashlib.sha256(json.dumps(changed,sort_keys=True,separators=(",",":")).encode()).hexdigest()))
    with pytest.raises(ValueError,match="allow-list"):
        load_checkpoint_for_initialization(path,model=target,config=target_config,
            identity=other,model_contract_migration=MIGRATION)
