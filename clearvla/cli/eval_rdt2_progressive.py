from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

import torch

from clearvla.cli.eval_rdt2_fm_reference import _build_conditioner
from clearvla.experiments.classic_policy_lab.cli_common import add_data_args, load_data, make_loader, resolve_device, serializable
from clearvla.experiments.classic_policy_lab.dataset import RDT2FMDatasetConfig, RDT2FMWindowDataset
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_progressive import ProgressiveRDT2FM, ProgressiveRDT2FMConfig
from clearvla.experiments.classic_policy_lab.rdt2_progressive_runtime import evaluate_progressive_rdt2_fm

IMAGE_ABLATIONS = ["normal", "zero", "mean", "shuffle-batch", "shuffle-episode", "top-only", "wrist-only"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a v19 progressive RDT2-FM checkpoint")
    add_data_args(p, default_resize=(224, 224))
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
    p.add_argument("--image-ablation", choices=IMAGE_ABLATIONS, default="normal")
    p.add_argument("--compare-image-ablations", nargs="+", choices=IMAGE_ABLATIONS, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/eval_rdt2_progressive.py")
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if dtype == torch.bfloat16 and device.type != "cuda":
        raise RuntimeError("--dtype bf16 is intended for CUDA evaluation")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    context = payload["context"]
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    model_config = ProgressiveRDT2FMConfig(**context["model"])
    data_config = RDT2FMDatasetConfig(**context["data"])
    episodes, train_ids, val_ids, test_ids, _, _, store, skipped = load_data(
        args, min_length=data_config.prediction_horizon + 1, normalizer_mode=action_norm.mode,
        action_normalizer=action_norm, state_normalizer=state_norm, splits=context["splits"],
    )
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    cameras = tuple(str(value) for value in context["args"]["cameras"])
    if tuple(args.cameras) != cameras:
        raise ValueError(f"evaluation cameras must match checkpoint cameras: {tuple(args.cameras)} != {cameras}")
    ds = RDT2FMWindowDataset(episodes, ids, image_store=store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    loader = make_loader(ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    model = ProgressiveRDT2FM(model_config, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    mode = args.condition_mode or context["conditioning"]["mode"]
    conditioner = _build_conditioner(mode, context, args, model_config=model_config, episodes=episodes, cameras=cameras, device=device, dtype=dtype)
    instruction = context["conditioning"].get("instruction", "") if args.instruction is None else args.instruction
    steps = int(args.inference_steps or model_config.num_inference_timesteps)
    def run(mode_name: str):
        return evaluate_progressive_rdt2_fm(model, conditioner, loader, device=device, action_normalizer=action_norm, inference_steps=steps, max_batches=args.max_val_batches, instruction=instruction, image_ablation=mode_name)
    metrics = run(args.image_ablation)
    report = {"schema": "clearvla-rdt2-progressive-eval-v1", "checkpoint": str(args.checkpoint), "split": args.split, "condition_mode": mode, "image_ablation": args.image_ablation, "metrics": metrics, "skipped": skipped}
    if args.compare_image_ablations:
        modes = []
        for value in ["normal", *args.compare_image_ablations]:
            if value not in modes:
                modes.append(value)
        rows = {value: (metrics if value == args.image_ablation else run(value)) for value in modes}
        normal = rows["normal"]
        keys = ("full_mse", "arm_first_rmse", "fast_exit_arm_first_rmse", "prefix_exit_first4_rmse")
        report["ablation_metrics"] = rows
        report["degradation_vs_normal"] = {value: {key: {"absolute": float(data[key] - normal[key]), "relative": float(data[key] / normal[key] - 1.0) if normal[key] else None} for key in keys} for value, data in rows.items() if value != "normal"}
    text = json.dumps(serializable(report), indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
