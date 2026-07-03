from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    resolve_device,
    serializable,
)
from clearvla.experiments.classic_policy_lab.dataset import RDT2FMDatasetConfig, RDT2FMWindowDataset
from clearvla.experiments.classic_policy_lab.evaluation import evaluate_rdt2_fm
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import (
    CachedDinoV2DenseConditioner,
    DebugDenseConditioner,
    DebugKVConditioner,
    DinoV2DenseConditioner,
    NullKVConditioner,
    RDT2VQKVConditioner,
)
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import DinoV2TokenStore
from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import RDT2FMReference, RDT2FMReferenceConfig


IMAGE_ABLATIONS = ["normal", "zero", "mean", "shuffle-batch", "shuffle-episode", "top-only", "wrist-only"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ClearVLA RDT2-FM action-expert checkpoint")
    add_data_args(p, default_resize=(384, 384))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--inference-steps", type=int, default=None)
    p.add_argument("--condition-mode", choices=["none", "debug-kv", "debug-dense", "dinov2", "dinov2-cache", "rdt2-vq"], default=None)
    p.add_argument("--instruction", default=None)
    p.add_argument("--dinov2-model", default=None)
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--rdt2-vq-model", default=None)
    p.add_argument("--rdt2-vq-processor", default=None)
    p.add_argument("--rdt2-vq-local-files-only", action="store_true")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--image-ablation", choices=IMAGE_ABLATIONS, default="normal")
    p.add_argument("--compare-image-ablations", nargs="+", choices=IMAGE_ABLATIONS, default=None,
                   help="Optionally evaluate normal images and several counterfactual image conditions in one reproducible report")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _build_conditioner(
    mode: str,
    context: dict,
    args: argparse.Namespace,
    *,
    model_config: RDT2FMReferenceConfig,
    episodes,
    cameras: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
):
    saved_args = context["args"]
    if mode == "none":
        return NullKVConditioner(depth=model_config.depth, num_kv_heads=model_config.num_kv_heads, head_dim=model_config.head_dim).to(device)
    if mode == "debug-kv":
        return DebugKVConditioner(depth=model_config.depth, num_kv_heads=model_config.num_kv_heads, head_dim=model_config.head_dim, tokens=int(saved_args.get("debug_cond_tokens", 8))).to(device)
    if mode == "debug-dense":
        return DebugDenseConditioner(token_dim=int(saved_args.get("debug_dense_token_dim", 32)), tokens_per_camera=max(1, int(saved_args.get("debug_cond_tokens", 8)) // max(len(cameras), 1))).to(device)
    if mode == "dinov2":
        name = args.dinov2_model or saved_args.get("dinov2_model", "facebook/dinov2-large")
        return DinoV2DenseConditioner(name, local_files_only=args.dinov2_local_files_only).to(device)
    if mode == "dinov2-cache":
        path = args.dinov2_token_cache_dir or saved_args.get("dinov2_token_cache_dir") or context.get("conditioning", {}).get("dinov2_token_cache_dir")
        if path is None:
            raise ValueError("dinov2-cache evaluation requires --dinov2-token-cache-dir or a saved training cache path")
        name = args.dinov2_model or saved_args.get("dinov2_model", "facebook/dinov2-large")
        store = DinoV2TokenStore(
            Path(path),
            episodes=episodes,
            camera_names=cameras,
            preprocessing=preprocessing_from_args(args),
            dinov2_model=name,
        )
        return CachedDinoV2DenseConditioner(store).to(device)
    name = args.rdt2_vq_model or saved_args.get("rdt2_vq_model", "robotics-diffusion-transformer/RDT2-VQ")
    processor = args.rdt2_vq_processor or saved_args.get("rdt2_vq_processor", "Qwen/Qwen2.5-VL-7B-Instruct")
    selected_layers = context["conditioning"].get("selected_layers", list(range(model_config.depth)))
    return RDT2VQKVConditioner(name, selected_layers=selected_layers, processor_name_or_path=processor, dtype=dtype, local_files_only=args.rdt2_vq_local_files_only).to(device)


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
    model_config = RDT2FMReferenceConfig(**context["model"])
    data_config = RDT2FMDatasetConfig(**context["data"])
    episodes, train_ids, val_ids, test_ids, _, _, store, skipped = load_data(
        args,
        min_length=data_config.prediction_horizon + 1,
        normalizer_mode=action_normalizer.mode,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        splits=context["splits"],
    )
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    cameras = tuple(str(value) for value in context["args"]["cameras"])
    if tuple(str(value) for value in args.cameras) != cameras:
        raise ValueError(f"evaluation cameras must match checkpoint cameras: requested={tuple(args.cameras)}, checkpoint={cameras}")
    dataset = RDT2FMWindowDataset(episodes, ids, image_store=store, camera_names=cameras, state_normalizer=state_normalizer, action_normalizer=action_normalizer, config=data_config)
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    model = RDT2FMReference(model_config, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    mode = args.condition_mode or context["conditioning"]["mode"]
    conditioner = _build_conditioner(mode, context, args, model_config=model_config, episodes=episodes, cameras=cameras, device=device, dtype=dtype)
    instruction = context["conditioning"].get("instruction", "") if args.instruction is None else args.instruction
    steps = int(args.inference_steps or model_config.num_inference_timesteps)
    def run_one(image_ablation: str):
        return evaluate_rdt2_fm(
            model,
            conditioner,
            loader,
            device=device,
            action_normalizer=action_normalizer,
            inference_steps=steps,
            max_batches=args.max_val_batches,
            eval_seed=args.eval_seed,
            instruction=instruction,
            image_ablation=image_ablation,
        )

    metrics = run_one(args.image_ablation)
    report = {
        "schema": "clearvla-rdt2-fm-reference-eval-v2",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "condition_mode": mode,
        "image_ablation": args.image_ablation,
        "metrics": metrics,
        "skipped": skipped,
    }
    if args.compare_image_ablations:
        requested = ["normal", *args.compare_image_ablations]
        modes = []
        for value in requested:
            if value not in modes:
                modes.append(value)
        ablation_metrics = {value: (metrics if value == args.image_ablation else run_one(value)) for value in modes}
        normal = ablation_metrics["normal"]
        keys = ("full_mse", "arm_first_rmse", "first4_rmse", "first8_rmse")
        degradation = {}
        for value, rows in ablation_metrics.items():
            if value == "normal":
                continue
            degradation[value] = {
                key: {
                    "absolute": float(rows[key] - normal[key]),
                    "relative": float(rows[key] / normal[key] - 1.0) if normal[key] != 0 else None,
                }
                for key in keys
            }
        report["ablation_metrics"] = ablation_metrics
        report["degradation_vs_normal"] = degradation
    text = json.dumps(serializable(report), indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
