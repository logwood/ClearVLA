from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

import torch

from clearvla.experiments.classic_policy_lab.cli_common import add_data_args, load_data, make_loader, preprocessing_from_args, resolve_device, serializable
from clearvla.experiments.classic_policy_lab.dataset import RDT2FMDatasetConfig, RDT2FMWindowDataset
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import CachedDinoV2DenseConditioner, DebugDenseConditioner, DinoV2DenseConditioner
from clearvla.experiments.classic_policy_lab.rdt2_control_interface import ControlInterfaceRDT2FMConfig, RDT2ControlInterface
from clearvla.experiments.classic_policy_lab.rdt2_control_interface_runtime import evaluate_control_interface_rdt2_fm
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import DinoV2TokenStore


IMAGE_ABLATIONS = ["normal", "zero", "mean", "shuffle-batch", "shuffle-episode", "top-only", "wrist-only"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a v21 RDT2-FM control-interface checkpoint")
    add_data_args(p, default_resize=(224, 224))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--inference-steps", type=int, default=None)
    p.add_argument("--condition-mode", choices=["debug-dense", "dinov2", "dinov2-cache"], default=None)
    p.add_argument("--instruction", default=None)
    p.add_argument("--debug-cond-tokens", type=int, default=None)
    p.add_argument("--debug-dense-token-dim", type=int, default=None)
    p.add_argument("--dinov2-model", default=None)
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--image-ablation", choices=IMAGE_ABLATIONS, default="normal")
    p.add_argument("--compare-image-ablations", nargs="+", choices=IMAGE_ABLATIONS, default=None)
    p.add_argument("--collect-diagnostics", action="store_true")
    p.add_argument("--diagnostic-batches", type=int, default=8)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _build_conditioner(args: argparse.Namespace, context: dict, *, episodes, cameras: tuple[str, ...], device: torch.device):
    saved = context["args"]
    mode = args.condition_mode or context["conditioning"]["mode"]
    if mode == "debug-dense":
        token_dim = int(args.debug_dense_token_dim or saved.get("debug_dense_token_dim", 32))
        token_count = int(args.debug_cond_tokens or saved.get("debug_cond_tokens", 8))
        return mode, DebugDenseConditioner(token_dim=token_dim, tokens_per_camera=max(1, token_count // max(len(cameras), 1))).to(device)
    name = args.dinov2_model or saved.get("dinov2_model", "facebook/dinov2-base")
    if mode == "dinov2":
        return mode, DinoV2DenseConditioner(name, local_files_only=args.dinov2_local_files_only).to(device)
    cache = args.dinov2_token_cache_dir or saved.get("dinov2_token_cache_dir") or context["conditioning"].get("dinov2_token_cache_dir")
    if cache is None:
        raise ValueError("dinov2-cache evaluation requires --dinov2-token-cache-dir or a saved training cache path")
    store = DinoV2TokenStore(Path(cache), episodes=episodes, camera_names=cameras, preprocessing=preprocessing_from_args(args), dinov2_model=name)
    return mode, CachedDinoV2DenseConditioner(store).to(device)


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/eval_rdt2_control_interface.py")
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
    model_config = ControlInterfaceRDT2FMConfig(**context["model"])
    data_config = RDT2FMDatasetConfig(**context["data"])
    episodes, train_ids, val_ids, test_ids, _, _, store, skipped = load_data(
        args,
        min_length=data_config.prediction_horizon + 1,
        normalizer_mode=action_norm.mode,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        splits=context["splits"],
    )
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    cameras = tuple(str(value) for value in context["args"]["cameras"])
    if tuple(str(value) for value in args.cameras) != cameras:
        raise ValueError(f"evaluation cameras must match checkpoint cameras: {tuple(args.cameras)} != {cameras}")
    dataset = RDT2FMWindowDataset(episodes, ids, image_store=store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    model = RDT2ControlInterface(model_config, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    mode, conditioner = _build_conditioner(args, context, episodes=episodes, cameras=cameras, device=device)
    if int(conditioner.token_dim) != model_config.dense_token_dim:
        raise ValueError(f"condition token width changed: checkpoint={model_config.dense_token_dim}, evaluator={conditioner.token_dim}")
    instruction = context["conditioning"].get("instruction", "") if args.instruction is None else args.instruction
    steps = int(args.inference_steps or model_config.num_inference_timesteps)

    def run_one(ablation: str):
        return evaluate_control_interface_rdt2_fm(
            model,
            conditioner,
            loader,
            device=device,
            action_normalizer=action_norm,
            inference_steps=steps,
            max_batches=args.max_val_batches,
            instruction=instruction,
            image_ablation=ablation,
            eval_seed=args.eval_seed,
            collect_diagnostics=args.collect_diagnostics and ablation == "normal",
            diagnostic_batches=args.diagnostic_batches,
        )

    metrics = run_one(args.image_ablation)
    report = {
        "schema": "clearvla-rdt2-control-interface-eval-v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "interface_mode": model_config.interface_mode,
        "condition_mode": mode,
        "image_ablation": args.image_ablation,
        "metrics": metrics,
        "skipped": skipped,
    }
    if args.compare_image_ablations:
        modes: list[str] = []
        for value in ["normal", *args.compare_image_ablations]:
            if value not in modes:
                modes.append(value)
        rows = {value: (metrics if value == args.image_ablation else run_one(value)) for value in modes}
        normal = rows["normal"]
        keys = ("full_mse", "arm_first_rmse", "first4_rmse", "first8_rmse")
        report["ablation_metrics"] = rows
        report["degradation_vs_normal"] = {
            value: {
                key: {
                    "absolute": float(data[key] - normal[key]),
                    "relative": float(data[key] / normal[key] - 1.0) if normal[key] else None,
                }
                for key in keys
            }
            for value, data in rows.items()
            if value != "normal"
        }
    text = json.dumps(serializable(report), indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
