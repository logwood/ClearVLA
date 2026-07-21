from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clearvla.experiments.classic_policy_lab.act_reference import ACTReference, ACTReferenceConfig
from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    resolve_device,
    serializable,
)
from clearvla.experiments.classic_policy_lab.dataset import ACTDatasetConfig, ACTWindowDataset
from clearvla.experiments.classic_policy_lab.evaluation import evaluate_act
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an ACT reference checkpoint")
    add_data_args(p, default_resize=(128, 128))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _model_config(data: dict) -> ACTReferenceConfig:
    values = dict(data)
    values["camera_names"] = tuple(values["camera_names"])
    values["resnet18_weights"] = (
        None if values.get("resnet18_weights") is None else Path(values["resnet18_weights"])
    )
    return ACTReferenceConfig(**values)


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    context = payload["context"]
    normalizer_action = ArrayNormalizer.from_dict(payload["action_normalizer"])
    normalizer_state = ArrayNormalizer.from_dict(payload["state_normalizer"])
    model_config = _model_config(context["model"])
    data_config = ACTDatasetConfig(**context["data"])
    episodes, train_ids, val_ids, test_ids, _, _, store, skipped = load_data(
        args,
        min_length=data_config.chunk_len + 1,
        normalizer_mode="zscore",
        action_normalizer=normalizer_action,
        state_normalizer=normalizer_state,
        splits=context["splits"],
    )
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    ds = ACTWindowDataset(
        episodes,
        ids,
        image_store=store,
        camera_names=model_config.camera_names,
        state_normalizer=normalizer_state,
        action_normalizer=normalizer_action,
        config=data_config,
    )
    loader = make_loader(
        ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device
    )
    model = ACTReference(model_config).to(device)
    model.load_state_dict(payload["model"])
    metrics = evaluate_act(
        model,
        loader,
        device=device,
        action_normalizer=normalizer_action,
        max_batches=args.max_val_batches,
    )
    report = {
        "schema": "clearvla-act-reference-eval-v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "metrics": metrics,
        "skipped": skipped,
    }
    text = json.dumps(serializable(report), indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
