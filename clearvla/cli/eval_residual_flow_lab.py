from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clearvla.cli.common import load_and_normalize_episodes, resolve_device
from clearvla.data.normalizer import ZScoreNormalizer
from clearvla.experiments.residual_flow_lab.evaluation import evaluate_residual_flow_model, visual_dependency_report
from clearvla.experiments.residual_flow_lab.model import ResidualFlowLabModel, ResidualFlowLabModelConfig
from clearvla.experiments.vision_usage_lab.dataset import LabEventScoreConfig, LabVisualMode, VisionUsageLabDataset, compute_lab_event_scores
from clearvla.experiments.vision_usage_lab.latent_cache import VisionLatentCacheStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate residual-flow checkpoints under visual counterfactuals")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--latent-cache-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--integration-steps", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _load_payload(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema")
    if schema == "clearvla-residual-flow-lab-export-v1":
        return payload, dict(payload["summary"]["context"])
    if schema == "clearvla-residual-flow-lab-v1":
        return payload, dict(payload["context"])
    raise ValueError(f"unsupported residual-flow checkpoint schema={schema!r}")


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    payload, context = _load_payload(args.checkpoint)
    model_config = ResidualFlowLabModelConfig.from_dict(dict(context["model"]))
    source_args = dict(context["args"])
    cameras = tuple(model_config.camera_names)
    episodes, skipped, train_ids, val_ids, test_ids, _ = load_and_normalize_episodes(
        data_root=args.data_root,
        pattern=args.glob,
        cameras=cameras,
        action_key=str(source_args.get("action_key", "action")),
        camera_key_overrides={
            "top": str(source_args.get("top_key", "observations/images/cam_high")),
            "wrist": str(source_args.get("wrist_key", "observations/images/cam_right_wrist")),
        },
        min_length=max(model_config.past_len, model_config.obs_horizon - 1) + model_config.chunk_len,
        train_frac=float(source_args.get("train_frac", 0.8)),
        val_frac=float(source_args.get("val_frac", 0.1)),
        seed=int(source_args.get("seed", 0)),
    )
    if train_ids != list(context["train_ids"]) or val_ids != list(context["val_ids"]) or test_ids != list(context["test_ids"]):
        raise ValueError("episode split differs from training context")
    normalizer = ZScoreNormalizer.from_dict(dict(context["normalizer"]))
    for episode in episodes:
        episode.actions_norm = normalizer.encode(episode.actions_raw)
    ids = val_ids if args.split == "val" else test_ids
    store = VisionLatentCacheStore(args.latent_cache_dir, camera_names=cameras)
    latent_meta = store.validate_consistent(episodes)
    if latent_meta.patch_grid != model_config.patch_grid or latent_meta.token_dim != model_config.teacher_dim:
        raise ValueError("latent cache shape is incompatible with checkpoint")
    visual_pool = ids if len(ids) > 1 else list(range(len(episodes)))
    dataset_kwargs = dict(
        episodes=episodes,
        episode_ids=ids,
        visual_pool_episode_ids=visual_pool,
        latent_store=store,
        chunk_len=model_config.chunk_len,
        past_len=model_config.past_len,
        obs_horizon=model_config.obs_horizon,
        future_visual_horizons=(1,),
        include_future_visual_delta=False,
        stride=int(source_args.get("stride", 1)),
        prior=str(source_args.get("prior", "blend")),
        prior_beta=float(source_args.get("prior_beta", 0.5)),
        velocity_mode=str(source_args.get("velocity_mode", "ema")),
        ema_decay=float(source_args.get("ema_decay", 0.75)),
        visual_shift=int(source_args.get("visual_shift", 8)),
        negative_visual_min_shift=int(source_args.get("negative_visual_min_shift", 8)),
    )
    loaders: dict[str, DataLoader] = {}
    for mode in LabVisualMode:
        dataset = VisionUsageLabDataset(visual_mode=mode, **dataset_kwargs)
        dataset.attach_event_scores(compute_lab_event_scores(dataset, LabEventScoreConfig(event_quantile=float(source_args.get("event_quantile", 0.70)))))
        loaders[mode.value] = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = ResidualFlowLabModel(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metrics = {
        mode: evaluate_residual_flow_model(model, loader, device=device, normalizer=normalizer, integration_steps=args.integration_steps)
        for mode, loader in loaders.items()
    }
    report = {
        "schema": "clearvla-residual-flow-lab-eval-v1",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "skipped": skipped,
        "metrics": metrics,
        "dependency": visual_dependency_report(metrics),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
