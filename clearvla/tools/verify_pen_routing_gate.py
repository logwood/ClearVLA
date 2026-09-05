"""Inspect the two real B8 smoke artifacts without constructing an optimizer.

Validates recorded numerics, per-owner gradients, loss ledger, shared data
identity and checkpoint model/ABI readback before the paired fresh Pen runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from clearvla.mainline.checkpoint import checkpoint_identity_from_mapping
from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.manifest import architecture_manifest_for_bspine_implementation
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import load_checkpoint_for_validation
from clearvla.mainline.runtime.deployment import validate_deployment_abi


def inspect_run(directory: Path) -> dict:
    context = json.loads((directory / "run_context.json").read_text())
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line]
    epochs = [row for row in rows if row.get("kind") == "epoch"]
    if len(epochs) != 1:
        raise ValueError(f"{directory}: smoke must have one completed epoch")
    epoch = epochs[0]
    train, validation = epoch["train"], epoch["validation"]
    for scope, metrics in (("train", train), ("validation", validation)):
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise FloatingPointError(f"{directory}: {scope}.{key} is nonfinite")
    for key in ("loss_ledger_gap", "loss_contribution_gap"):
        if abs(train[key]) > 1e-5:
            raise ValueError(f"{directory}: loss ledger does not close")
    positive = ["gradient_raw_bottom_spine_l2", "gradient_raw_bottom_spine_coarse_l2"]
    if "private_reader" in context["config"]["bottom"]["bspine_implementation"]:
        positive.append("gradient_raw_bottom_spine_private_reader_l2")
    for key in positive:
        if not train.get(key, 0.0) > 0.0:
            raise ValueError(f"{directory}: missing active owner gradient {key}")
    peak = train["runtime_cuda_peak_process_estimate_gib"]
    if peak > 22.0:
        raise RuntimeError(f"{directory}: process peak {peak} GiB exceeds budget")

    checkpoint = directory / "checkpoints" / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = config_from_mapping(payload["config"])
    identity = checkpoint_identity_from_mapping(payload["identity"])
    if identity.as_dict() != context["identity"]:
        raise ValueError("checkpoint and run context identities differ")
    expected = architecture_manifest_for_bspine_implementation(config.bottom.bspine_implementation)
    if expected.as_dict() != identity.manifest:
        raise ValueError("checkpoint variant manifest differs")
    if ComponentSelection.from_config(config).as_dict() != payload["component_selection"]:
        raise ValueError("checkpoint component selection differs")
    validate_deployment_abi(payload["data_state"]["deployment_abi"])
    model = ClearVLAMainlinePolicy(config)
    # Existing read-only loader verifies complete keys, dtypes, shapes and
    # finite weights. It restores neither optimizer nor any RNG state.
    loaded = load_checkpoint_for_validation(checkpoint, model=model, config=config, identity=identity)
    if loaded.global_step != epoch["step"] or loaded.epoch != 1:
        raise ValueError("checkpoint scalar progress differs from metrics")
    return {
        "run": str(directory), "variant": config.bottom.bspine_implementation,
        "step": loaded.global_step, "git": identity.git_commit,
        "dataset": identity.as_dict()["dataset"],
        "normalizer": context["normalizer_fingerprints"],
        "loss_total": train["loss_total"], "action_rmse": validation["validation_action_rmse_physical"],
        "gradients": {key: train[key] for key in positive},
        "peak_process_gib": peak, "seconds_per_batch": train["runtime_seconds_per_batch"],
        "checkpoint_readback": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = [inspect_run(path) for path in args.runs]
    if results[0]["dataset"] != results[1]["dataset"] or results[0]["normalizer"] != results[1]["normalizer"]:
        raise ValueError("paired experiment data/normalizer identities differ")
    if results[0]["git"] != results[1]["git"]:
        raise ValueError("paired experiment source commits differ")
    print(json.dumps({"passed": True, "runs": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
