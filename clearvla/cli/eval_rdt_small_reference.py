from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clearvla.experiments.classic_policy_lab.cli_common import add_data_args, load_data, make_loader, resolve_device, serializable
from clearvla.experiments.classic_policy_lab.dataset import RDTSmallDatasetConfig, RDTSmallWindowDataset
from clearvla.experiments.classic_policy_lab.evaluation import evaluate_rdt_small
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt_small_reference import (
    DebugPatchVisionEncoder,
    EmptyLanguageConditioner,
    RDTSmallReference,
    RDTSmallReferenceConfig,
    SiglipPatchVisionEncoder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ClearVLA RDT-170M / RDT-small checkpoint")
    add_data_args(p, default_resize=(384, 384))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--inference-steps", type=int, default=None)
    p.add_argument("--sampler", choices=["dpm_solver", "ddpm_debug"], default="dpm_solver")
    p.add_argument("--vision-encoder", choices=["siglip", "patch-debug"], default=None)
    p.add_argument("--siglip-model", default=None)
    p.add_argument("--siglip-local-files-only", action="store_true")
    p.add_argument("--empty-lang-embed", type=Path, default=None)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--stochastic-sampling", action="store_true")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _model_config(data: dict) -> RDTSmallReferenceConfig:
    values = dict(data)
    values.pop("img_cond_len", None)
    values["state_indices"] = tuple(values["state_indices"])
    return RDTSmallReferenceConfig(**values)


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if dtype == torch.bfloat16 and device.type != "cuda":
        raise RuntimeError("--dtype bf16 is intended for CUDA evaluation")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    context = payload["context"]
    action_normalizer = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_normalizer = ArrayNormalizer.from_dict(payload["state_normalizer"])
    model_config = _model_config(context["model"])
    data_config = RDTSmallDatasetConfig(**context["data"])
    episodes, train_ids, val_ids, test_ids, _, _, store, skipped = load_data(
        args,
        min_length=data_config.prediction_horizon + data_config.image_history,
        normalizer_mode=action_normalizer.mode,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        splits=context["splits"],
    )
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    cameras = tuple(str(value) for value in context["args"]["cameras"])
    ds = RDTSmallWindowDataset(episodes, ids, image_store=store, camera_names=cameras, state_normalizer=state_normalizer, action_normalizer=action_normalizer, config=data_config)
    loader = make_loader(ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    model = RDTSmallReference(model_config).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    vision_mode = args.vision_encoder or context["vision"]["mode"]
    patch_grid = int(context["vision"]["patch_grid"])
    if vision_mode == "siglip":
        siglip_name = args.siglip_model or context["vision"]["model"]
        vision_encoder = SiglipPatchVisionEncoder(model_name_or_path=siglip_name, image_history=model_config.image_history, max_cameras=model_config.max_cameras, local_files_only=args.siglip_local_files_only).to(device=device, dtype=dtype)
    else:
        vision_encoder = DebugPatchVisionEncoder(token_dim=model_config.img_token_dim, patch_grid=patch_grid, image_history=model_config.image_history, max_cameras=model_config.max_cameras).to(device)
    empty_path = args.empty_lang_embed
    if empty_path is None:
        empty_path = Path(__file__).resolve().parents[1] / "assets" / "rdt_empty_lang_embed.pt"
    language_conditioner = EmptyLanguageConditioner(token_dim=model_config.lang_token_dim, embedding_path=empty_path)
    inference_steps = int(args.inference_steps or model_config.inference_steps)
    metrics = evaluate_rdt_small(model, vision_encoder, language_conditioner, loader, device=device, action_normalizer=action_normalizer, inference_steps=inference_steps, sampler=args.sampler, max_batches=args.max_val_batches, deterministic=not args.stochastic_sampling, eval_seed=args.eval_seed)
    report = {"schema": "clearvla-rdt-small-reference-eval-v1", "checkpoint": str(args.checkpoint), "split": args.split, "metrics": metrics, "skipped": skipped}
    text = json.dumps(serializable(report), indent=2); print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True); args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
