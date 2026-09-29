"""Formal train/validate CLI using artificial source markers, NOT pretrained.

Only the external frozen encoder factory is replaced. Dataset, preflight,
optimizer update, validation, identity and checkpoint writers are real.
"""
from __future__ import annotations

import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from test_dinov3_production_integration import fixture, pipeline
from clearvla.mainline.train import main
from clearvla.vision.online_pipeline import OnlineVisionPipeline


def test_formal_online_training_and_validation_cli(tmp_path, monkeypatch):
    config, bundle = fixture(tmp_path)
    del bundle
    gc.collect()
    output = tmp_path / 'formal-run'
    config = replace(
        config,
        data=replace(config.data, output_dir=str(output), num_workers=0),
        optimizer=replace(config.optimizer, batch_size=1, epochs=1),
        runtime=replace(config.runtime, max_train_batches=1, max_val_batches=1,
                        eval_sampling_diagnostic_batches=0,
                        eval_proposal_ablation_batches=0,
                        eval_execution_ablation_batches=0),
    )
    path = tmp_path / 'formal.json'
    path.write_text(json.dumps(config.as_dict()))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, 'argv', ['clearvla.mainline.train', '--config', str(path), '--device', 'cpu'])
    monkeypatch.setattr(OnlineVisionPipeline, 'from_config', classmethod(lambda cls, cfg, device: pipeline()))
    main()
    context = json.loads((output / 'run_context.json').read_text())
    assert context['config']['data']['visual_feature_mode'] == 'dinov3_online_v1'
    assert context['visual_encoder']['feature_mode'] == 'dinov3_online_v1'
    assert context['visual_encoder']['disk_feature_cache'] is False
    latest = output / 'checkpoints/latest.pt'
    assert latest.is_file() and (output / 'checkpoints/best.pt').is_file()
    # Locally created fixture checkpoint, not an untrusted user pickle.
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    assert saved['global_step'] == 1
    assert saved['data_state']['deployment_abi']['observation']['dinov3']['identity']['encoder']['frozen']
    del saved
    gc.collect()
