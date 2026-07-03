from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clearvla.cli.common import resolve_device
from clearvla.data.hdf5_episode import load_episodes
from clearvla.experiments.vision_usage_lab.latent_cache import build_all_vision_latent_caches
from clearvla.experiments.vision_usage_lab.teacher import PatchTeacherConfig, build_patch_teacher
from clearvla.vision.online_store import OnlineVisualStore
from clearvla.vision.preprocessing import PreprocessConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict frame-local frozen patch-token caches for vision-usage experiments")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--decoded-image-cache-dir", type=Path, default=None)
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")
    p.add_argument("--teacher-backend", choices=["dinov2_vits14", "tiny_patch"], default="dinov2_vits14")
    p.add_argument("--teacher-image-size", type=int, nargs=2, metavar=("H", "W"), default=(224, 224))
    p.add_argument("--teacher-patch-size", type=int, default=None)
    p.add_argument("--teacher-token-dim", type=int, default=None)
    p.add_argument("--teacher-source", choices=["github", "local"], default="github")
    p.add_argument("--teacher-hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--teacher-local-repo", default=None)
    p.add_argument("--teacher-model-name", default=None)
    p.add_argument("--tiny-seed", type=int, default=17)
    p.add_argument("--batch-frames", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    cameras = tuple(str(x) for x in args.cameras)
    teacher_cfg = PatchTeacherConfig(
        backend=args.teacher_backend,
        image_hw=(int(args.teacher_image_size[0]), int(args.teacher_image_size[1])),
        patch_size=args.teacher_patch_size,
        token_dim=args.teacher_token_dim,
        tiny_seed=args.tiny_seed,
        torch_hub_source=args.teacher_source,
        torch_hub_repo=args.teacher_hub_repo,
        local_repo_dir=args.teacher_local_repo,
        model_name=args.teacher_model_name,
    )
    teacher_cfg.validate()
    episodes, skipped = load_episodes(
        args.data_root,
        args.glob,
        cameras=cameras,
        min_length=1,
        action_key=args.action_key,
        camera_key_overrides={"top": args.top_key, "wrist": args.wrist_key},
    )
    store = OnlineVisualStore(
        camera_names=cameras,
        preprocessing=PreprocessConfig(),
        decoded_cache_dir=args.decoded_image_cache_dir,
    )
    if args.decoded_image_cache_dir is not None:
        for episode in episodes:
            store.validate_episode(episode)
    teacher = build_patch_teacher(teacher_cfg, device=device)
    metas = build_all_vision_latent_caches(
        episodes,
        cache_dir=args.cache_dir,
        camera_names=cameras,
        teacher=teacher,
        teacher_config=teacher_cfg,
        visual_store=store,
        device=device,
        batch_frames=args.batch_frames,
        rebuild=args.rebuild,
    )
    print(json.dumps({
        "episodes": len(episodes),
        "skipped": skipped,
        "cache_dir": str(args.cache_dir),
        "teacher": teacher_cfg.to_dict(),
        "patch_grid": list(teacher_cfg.patch_grid()),
        "token_dim": teacher_cfg.resolved_token_dim(),
        "metas": [meta.to_dict() for meta in metas],
    }, indent=2))


if __name__ == "__main__":
    main()
