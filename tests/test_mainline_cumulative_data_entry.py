"""Production CALVIN loading entry, not just isolated model fixtures.

HDF5, NPY caches, instruction bank, split manifest, normalizer fitting, worker
collation and to_training_batch all run for real. Pixels and embeddings are
synthetic source-labelled transport; no pretrained encoder/robot claim.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from clearvla.data.hdf5_episode import (
    RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING, load_episodes,
)
from clearvla.data.split import EPISODE_SPLIT_MANIFEST_SCHEMA
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import save_episode_tokens
from clearvla.mainline.config import load_config
from clearvla.mainline.data.language import source_instruction_inventory_sha256
from clearvla.mainline.data.loading import (
    load_mainline_data, load_mainline_data_for_smoke, to_training_batch,
)
from clearvla.tools.build_t5_instruction_cache import build_t5_instruction_cache_payload
from clearvla.vision.preprocessing import PreprocessConfig

ROOT = Path(__file__).resolve().parents[1]
PRESETS = (
    "structural_rebuild_annotated_goal_calvin",
    "task_grounded_execution_cumulative_calvin",
)

def source_fixture(tmp_path: Path, preset: str = PRESETS[-1], *, endpoint_case: str = "complete"):
    """Three separate source episodes; annotation starts at local row 24."""
    root = tmp_path / "episodes"
    root.mkdir()
    instructions = (
        "push the red block to the left",
        "push the blue block to the right",
        "push the pink block to the left",
    )
    for j, text in enumerate(instructions):
        t = np.arange(105, dtype=np.float32)
        state = np.stack((
            .02 + .001*t + j*.1, -.13 + .0005*t, .5-.0004*t,
            .1+.001*t, -.2+.002*t, .3-.001*t, .04+.00001*t,
        ), axis=-1).astype(np.float32)
        action = np.stack(
            [np.sin(.09*t+c)*.1 for c in range(6)] +
            [np.where(t < 40, 1., -1.)], axis=-1,
        ).astype(np.float32)
        state[81:] = state[80]
        action[80:, :6] = 0.
        action[80:, 6] = action[79, 6]
        boundary = np.concatenate((np.array([[0,0,0,0,0,0,1]], np.float32), action[:-1]))
        path = root / f"episode_{j}.hdf5"
        with h5py.File(path, "w") as f:
            f["action"] = action
            f["state"] = state
            f["action_state"] = boundary
            for c, name in enumerate(("cam_high", "cam_right_wrist")):
                rgb = np.zeros((105, 8, 8, 3), np.uint8)
                rgb[..., 0] = j+10
                rgb[..., 1] = np.minimum(np.arange(105), 80)[:,None,None]
                rgb[..., 2] = c
                f[f"observations/images/{name}"] = rgb
            f.attrs.update({
                "instruction": text,
                "task": ("push_red_block_left", "push_blue_block_right", "push_pink_block_left")[j],
                "valid_center_start": 24,
                "valid_center_end": 56,
                "terminal_state_index": 80,
                "terminal_padding_mode": RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
                "source_annotation_index": j,
                "context_start": 1000+j*1000,
                "source_start": 1024+j*1000,
                "source_end": 1080+j*1000,
                "source_trajectory_id": f"separate-raw-{j}",
            })
    if endpoint_case != "complete":
        with h5py.File(root / "episode_0.hdf5", "r+") as f:
            if endpoint_case == "missing":
                del f.attrs["source_annotation_index"]
            elif endpoint_case == "censored":
                f.attrs["source_end"] = 1082
            else:
                raise ValueError(endpoint_case)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "schema": EPISODE_SPLIT_MANIFEST_SCHEMA,
        "splits": {k: [f"episode_{j}"] for j,k in enumerate(("train","val","test"))},
    }), encoding="utf-8")
    episodes, skipped = load_episodes(
        root, "*.hdf5", cameras=("top","wrist"), min_length=1,
        action_key="action", state_key="state", action_state_key="action_state",
        camera_key_overrides={"top":"observations/images/cam_high","wrist":"observations/images/cam_right_wrist"},
    )
    assert not skipped
    cache = tmp_path / "dino"
    for j,e in enumerate(episodes):
        rows = np.minimum(np.arange(e.length), 80)
        token = (j*1000 + rows[:,None,None,None]*10 +
                 np.arange(2)[None,:,None,None]).astype(np.float32)
        token = np.broadcast_to(token, (e.length,2,64,16)).copy()
        save_episode_tokens(
            cache_dir=cache, episode=e, camera_names=("top","wrist"),
            preprocessing=PreprocessConfig(resize_hw=(336,336)),
            dinov2_model="synthetic-source-labelled-dino", tokens=token,
        )
    texts = tuple(sorted(instructions))
    tokens = (torch.arange(len(texts)*4*16, dtype=torch.float32).reshape(len(texts),4,16)+1)
    payload = build_t5_instruction_cache_payload(
        instructions=texts, tokens=tokens,
        attention_mask=torch.ones(len(texts),4,dtype=torch.bool),
        model_source="synthetic-source-labelled-t5",
        source_episode_count=len(episodes),
        source_instruction_inventory_sha256=source_instruction_inventory_sha256(instructions),
    )
    lang = tmp_path / "t5.pt"
    torch.save(payload, lang)
    cfg = load_config(ROOT / f"configs/mainline/{preset}.json")
    cfg = replace(
        cfg,
        data=replace(cfg.data, raw_hdf5_root=str(root),
            dino_cache=str(cache), t5_condition=str(lang), split_manifest=str(manifest),
            calvin_raw_source="", decoded_cache=str(tmp_path/"unused-decoded"),
            image_store_mode="hdf5-direct", num_workers=0,
            dinov2_model="synthetic-source-labelled-dino", visual_cache_read_backend="mmap"),
        dimensions=replace(cfg.dimensions,hidden_size=32,num_heads=4,
            visual_token_dim=16,patches_per_camera=64,goal_token_dim=16,goal_max_tokens=4),
        bottom=replace(cfg.bottom,controller_heads=4),
    )
    cfg.validate()
    return cfg, texts, tokens

@pytest.mark.parametrize("preset", PRESETS)
def test_cumulative_formal_entry_retains_real_endpoint_labels_all_splits(tmp_path, preset):
    cfg, _, _ = source_fixture(tmp_path, preset)
    bundle = load_mainline_data(cfg)
    assert set(bundle.datasets) == {"train","val","test"}
    for split, dataset in bundle.datasets.items():
        assert len(dataset) == 56
        assert dataset.base.config.window_boundary_contract == "observed_tail_v1"
        assert dataset.base.config.annotation_goal_mode == "annotated_endpoint_relation_v1"
        rows = [dataset[0], dataset[len(dataset)-1]]
        assert [int(r["center_index"]) for r in rows] == [24,79]
        assert [int(r["instruction_reference_age"]) for r in rows] == [0,55]
        assert int(dataset.base[0]["annotation_endpoint_key"][0,1]) == 80
        assert all(bool(r["annotation_endpoint_declared"]) for r in rows)
        # Last real action is a valid next command; synthetic suffix is not.
        assert int(rows[-1]["future_action_observed"].sum()) == 1
        batch = to_training_batch(default_collate(rows), goal=bundle.goal, config=cfg, device=torch.device("cpu"))
        batch.validate(cfg)
        assert batch.online.history.state.shape == (2,10)
        assert batch.action_target.normalized.shape[-1] == 7

def test_worker_language_mapping_uses_episode_identity_after_row_permutation(tmp_path):
    cfg, texts, tokens = source_fixture(tmp_path)
    bundle = load_mainline_data(cfg)
    selected = [bundle.datasets[k][0] for k in ("test","train","val")]
    batch = to_training_batch(default_collate(selected), goal=bundle.goal, config=cfg, device=torch.device("cpu"))
    for row, sample in enumerate(selected):
        text = bundle.episodes[int(sample["episode_idx"])].instruction
        assert torch.equal(batch.online.goal.tokens[row], tokens[texts.index(text)])

@pytest.mark.parametrize("preset", PRESETS)
def test_cumulative_loader_only_smoke_uses_same_supervision_contract(tmp_path, preset):
    cfg, _, _ = source_fixture(tmp_path,preset)
    bundle = load_mainline_data_for_smoke(cfg, split="train", episode_limit=1)
    assert set(bundle.datasets) == {"train"}
    for dataset in bundle.datasets.values():
        assert bool(dataset[0]["annotation_endpoint_declared"])

def test_missing_endpoint_provenance_keeps_bc_but_does_not_fabricate_goal(tmp_path):
    cfg, _, _ = source_fixture(tmp_path, endpoint_case="missing")
    bundle = load_mainline_data(cfg)
    row = bundle.datasets["train"][0]
    assert not bool(row["annotation_endpoint_declared"])
    assert bool(row["future_action_observed"].any())
    assert int(row["annotation_endpoint_offset_steps"]) == 0

def test_invalid_endpoint_contract_is_still_rejected(tmp_path):
    cfg, _, _ = source_fixture(tmp_path)
    broken = replace(cfg,data=replace(cfg.data,window_boundary_contract="strict_complete_v1"))
    with pytest.raises(ValueError):
        load_mainline_data(broken)

def test_censored_endpoint_does_not_accept_storage_padding_as_truth(tmp_path):
    cfg, _, _ = source_fixture(tmp_path, endpoint_case="censored")
    bundle = load_mainline_data(cfg)
    row = bundle.datasets["train"][0]
    assert not bool(row["annotation_endpoint_declared"])
    assert bool(row["future_action_observed"].any())

class _SourceLabelledEncoder:
    """Only the expensive feature encoder is replaced; pixels identify sources."""
    def encode(self, rgb_history, preprocessing):
        from clearvla.vision.preprocessing import apply_preprocess
        images = np.stack([
            np.stack([apply_preprocess(frame,preprocessing) for frame in rgb_history[camera]])
            for camera in ("top","wrist")
        ], axis=1)
        marker = images[:,:,0,0].astype(np.float32)
        values = (marker[...,0]-10)*1000 + marker[...,1]*10 + marker[...,2]
        tokens = torch.from_numpy(values[:,:,None,None].copy()).expand(-1,-1,64,16).clone()
        return tokens, images

def _policy(bundle, cfg, monkeypatch):
    from types import SimpleNamespace
    from clearvla.mainline.data.language import load_t5_condition_bank
    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
    bank = load_t5_condition_bank(cfg.data.t5_condition,max_tokens=4,expected_width=16)
    # Input producer is real; sampling is replaced and is NOT a model test.
    def no_sampling(model, online, config, **kwargs):
        return SimpleNamespace(
            action=torch.zeros(1,24,7), gripper_command=torch.ones(1,24),
        )
    monkeypatch.setattr("clearvla.simulation.clearvla_policy.sample_action",no_sampling)
    policy = object.__new__(ClearVLACheckpointPolicy)
    policy.bundle = SimpleNamespace(config=cfg, action_normalizer=bundle.action_normalizer,
        state_normalizer=bundle.state_normalizer, language=bank,model=None)
    policy.device = torch.device("cpu")
    policy.encoder = _SourceLabelledEncoder()
    policy.preprocessing = PreprocessConfig(resize_hw=(336,336))
    policy._seed = 0
    policy._generator = torch.Generator().manual_seed(0)
    policy.reset()
    return policy

def _observation(episode, index):
    from clearvla.simulation.contracts import PolicyObservation
    with h5py.File(episode.path,"r") as f:
        rgb = {name: np.asarray(f[key][index]) for name,key in episode.camera_keys.items()}
    return PolicyObservation(rgb=rgb,state=episode.states_raw[index].copy(),
        action_state=episode.action_states_raw[index].copy())

def _assert_same_tree(a,b):
    from dataclasses import fields, is_dataclass
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif is_dataclass(a):
        assert type(a) is type(b)
        for f in fields(a):
            _assert_same_tree(getattr(a,f.name),getattr(b,f.name))
    else:
        assert a == b

def _paired_online(bundle,cfg,age,monkeypatch):
    from clearvla.simulation.history import CausalHistory
    e = bundle.episodes[bundle.splits["train"][0]]
    start = 24
    warm = CausalHistory(executed_world=True)
    warm.reset(_observation(e,0))
    for t in range(start):
        warm.append(e.actions_raw[t],_observation(e,t+1))
    cold = CausalHistory(executed_world=True)
    cold.reset(_observation(e,start))
    wp,cp = _policy(bundle,cfg,monkeypatch),_policy(bundle,cfg,monkeypatch)
    _,wo = wp.act_with_input(warm.snapshot(),e.instruction)
    _,co = cp.act_with_input(cold.snapshot(),e.instruction)
    for t in range(start,start+age):
        obs = _observation(e,t+1)
        warm.append(e.actions_raw[t],obs)
        cold.append(e.actions_raw[t],obs)
    if age:
        _,wo = wp.act_with_input(warm.snapshot(),e.instruction)
        _,co = cp.act_with_input(cold.snapshot(),e.instruction)
    return wo,co

@pytest.mark.parametrize("age",[0,1,4,8,12,24])
def test_recorded_input_matches_online_with_same_real_history(tmp_path,monkeypatch,age):
    cfg,_,_ = source_fixture(tmp_path)
    bundle = load_mainline_data(cfg)
    sample = bundle.datasets["train"][age]
    batch = to_training_batch(default_collate([sample]),goal=bundle.goal,config=cfg,device=torch.device("cpu"))
    warm,cold = _paired_online(bundle,cfg,age,monkeypatch)
    _assert_same_tree(batch.online,warm)
    # Same current scene, proprioception, command boundary and instruction.
    _assert_same_tree(warm.goal,cold.goal)
    _assert_same_tree(warm.instruction_reference,cold.instruction_reference)
    assert torch.equal(warm.history.state,cold.history.state)
    assert torch.equal(warm.history.action_state,cold.history.action_state)
    assert torch.equal(warm.observation.dino_history[:,-1],cold.observation.dino_history[:,-1])
    assert warm.history.timing.action_executed.all()
    if age < 24:
        assert not cold.history.timing.action_executed.all()
    else:
        _assert_same_tree(warm,cold)

def test_action_dropout_does_not_represent_a_cold_environment_reset(tmp_path,monkeypatch):
    from clearvla.mainline.model.components import ConditioningStage
    from clearvla.mainline.model.proposal import HistoryActionProposal
    cfg,_,_ = source_fixture(tmp_path)
    bundle = load_mainline_data(cfg)
    warm,cold = _paired_online(bundle,cfg,0,monkeypatch)
    cfg = replace(cfg,top=replace(cfg.top,action_history_condition_dropout=1.,goal_condition_dropout=0.))
    stage = ConditioningStage(HistoryActionProposal(action_dim=7,hidden=32,heads=4,horizon=24,history_length=8))
    dropped,_,_,_ = stage.prepare(warm,config=cfg,training=True,training_mask=True,
                                 condition_generator=torch.Generator().manual_seed(1))
    assert not dropped.history.timing.action_executed.any()
    assert dropped.history.timing.state_observed.all()
    assert int(cold.history.timing.state_observed.sum()) == 1
    assert torch.equal(dropped.observation.dino_history,warm.observation.dino_history)
    assert not torch.equal(dropped.observation.dino_history,cold.observation.dino_history)

def test_support_census_counts_actual_dataset_without_mutation(tmp_path):
    from scripts.audit_calvin_history_support import context_report
    cfg,_,_ = source_fixture(tmp_path)
    bundle = load_mainline_data(cfg)
    original = tuple(bundle.datasets["train"].base.refs)
    report = context_report(bundle)
    assert report["splits"]["train"]["context_counts"] == [
        {"observed_state_rows":3,"executed_action_rows":8,"windows":56}]
    assert report["reset_context"]["history_state_observed"] == [False,False,True]
    assert report["reset_context"]["history_action_executed"] == [False]*8
    assert report["splits"]["train"]["selected_modes"]["annotation_goal_mode"] == "annotated_endpoint_relation_v1"
    first = report["splits"]["train"]["instruction_starts"][0]
    assert first["previous_step_source_available"] and first["four_step_source_available"]
    assert tuple(bundle.datasets["train"].base.refs) == original


@pytest.mark.parametrize("amp", [False, True])
def test_disk_loaded_cumulative_batch_completes_engine_step_and_sampling(tmp_path, amp):
    """Production loading -> batch -> actual engine; synthetic files, small graph."""
    import gc

    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.sampling import sample_action
    from clearvla.mainline.training.engine import MainlineTrainingEngine
    from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer

    gc.collect()
    torch.manual_seed(65012)
    cfg, _, _ = source_fixture(tmp_path)
    cfg = replace(cfg, runtime=replace(cfg.runtime, compute_dtype="bf16" if amp else "fp32"))
    bundle = load_mainline_data(cfg)
    batch = to_training_batch(
        default_collate([bundle.datasets["train"][0]]), goal=bundle.goal,
        config=cfg, device=torch.device("cpu"),
    )
    model = ClearVLAMainlinePolicy(cfg)
    model.configure_action_normalizer(bundle.action_normalizer)
    optimizer, _ = build_optimizer(model, cfg)
    # Use the configured 500-step LR warmup and 200/1000 execution staging.
    # A single observed update does not stand in for mature-controller behavior.
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=cfg.optimizer.warmup_steps, total_steps=22024,
        minimum_ratio=cfg.optimizer.min_lr_ratio,
    )
    engine = MainlineTrainingEngine(
        model=model, config=cfg, optimizer=optimizer, schedule=schedule,
        device=torch.device("cpu"), dtype=torch.bfloat16 if amp else torch.float32,
        train_flow_generator=torch.Generator().manual_seed(65013),
        train_condition_generator=torch.Generator().manual_seed(65014),
    )
    tracked = {n: p for n,p in model.named_parameters() if p.requires_grad and
               any(s in n for s in ("task_relation_encoder", "annotation_goal"))}
    assert tracked
    before = {n: p.detach().clone() for n,p in tracked.items()}
    engine.train_step(batch)
    assert engine.global_step == 1 and schedule.step_index == 1
    assert any(not torch.equal(before[n], p) for n,p in tracked.items())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(65015))
        b = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(65015))
    assert a.action.shape == (1,24,7) and torch.isfinite(a.action).all()
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
    del engine, model, optimizer, schedule, bundle, batch, tracked, before, a, b
    gc.collect()
