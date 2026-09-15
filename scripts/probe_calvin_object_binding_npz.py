#!/usr/bin/env python3
"""Matched CALVIN language-to-object binding probe.

This is a read-only, deployment-only probe.  It reuses the exported causal
CALVIN snapshots and a frozen checkpoint, holds the RGB/state/history fixed,
and changes only the instruction.  In addition to the public S tensors it
captures the two object reads that are otherwise easy to miss:

* ``StatelessObjectIntentOrganizer.interval_object``;
* ``CoarseActionIntent.object_read``.

The probe records their object-axis attention/update, object tokens, and the
typed relevance masses.  It does not alter model parameters, gradients,
checkpoints, caches, simulator state, or the training process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

from scripts.probe_calvin_internal_layers import _build_online
from scripts.probe_calvin_internal_layers_npz import _fingerprint, _history


DEFAULT_INSTRUCTIONS = (
    "push the blue block to the right",
    "push the red block to the right",
    "push the pink block to the right",
    "push the blue block to the left",
)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _rmse(left: Any, right: Any) -> float:
    a = _array(left).astype(np.float64)
    b = _array(right).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.square(a - b))))


def _cosine_per_slot(left: Any, right: Any) -> list[float]:
    a = _array(left).reshape(-1, _array(left).shape[-1]).astype(np.float64)
    b = _array(right).reshape(-1, _array(right).shape[-1]).astype(np.float64)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    values = np.divide(
        np.sum(a * b, axis=-1), denominator, out=np.zeros_like(denominator), where=denominator > 1e-12
    )
    return [float(item) for item in values.tolist()]


def _capture_hook(store: dict[str, Any], name: str):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 3:
            raise RuntimeError(f"{name} returned an unexpected value")
        # The third result is the diagnostic attention map.  Clone detached
        # values immediately so the hook cannot retain a live graph/tensor.
        store[name] = {
            "value": output[0].detach().float().cpu().numpy(),
            "update": output[1].detach().float().cpu().numpy(),
            "attention": output[2].detach().float().cpu().numpy(),
        }

    return hook


def _read_capture(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    return {
        "present": True,
        "value": value["value"].tolist(),
        "update": value["update"].tolist(),
        "attention": value["attention"].tolist(),
    }


def _instruction_record(intent: Any, capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_tokens": _array(intent.object_tokens).tolist(),
        "interval_object_attention": _array(intent.interval_object_attention).tolist(),
        "typed_relevance_mass": _array(intent.typed_relevance_mass).tolist(),
        "public_interval_carrier": _array(intent.public_interval_carrier).tolist(),
        "organizer_interval_object": _read_capture(capture.get("organizer_interval_object")),
        "coarse_object_read": _read_capture(capture.get("coarse_object_read")),
    }


def _pairwise(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "object_tokens",
        "interval_object_attention",
        "typed_relevance_mass",
        "public_interval_carrier",
    ):
        result[name] = {"rmse": _rmse(left[name], right[name])}
    for name in ("organizer_interval_object", "coarse_object_read"):
        a = left[name]
        b = right[name]
        if not a.get("present") or not b.get("present"):
            result[name] = {"present": False}
            continue
        result[name] = {
            "present": True,
            "value_rmse": _rmse(a["value"], b["value"]),
            "update_rmse": _rmse(a["update"], b["update"]),
            "attention_rmse": _rmse(a["attention"], b["attention"]),
            "left_attention_argmax": np.argmax(np.asarray(a["attention"]), axis=-1).tolist(),
            "right_attention_argmax": np.argmax(np.asarray(b["attention"]), axis=-1).tolist(),
        }
    return result


def run_probe(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    observation_dir: Path,
    output: Path,
    layouts: tuple[str, ...],
    instructions: tuple[str, ...],
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    if not layouts or not instructions:
        raise ValueError("at least one layout and instruction are required")
    torch_device = torch.device(device)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch_device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    model = policy.bundle.model
    model.eval()
    records: dict[str, Any] = {}
    for layout in layouts:
        snapshot = observation_dir / f"{layout}.npz"
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        history = _history(snapshot)
        layout_record: dict[str, Any] = {
            "observation": _fingerprint(snapshot),
            "instructions": {},
            "pairwise_vs_first": {},
            "per_slot_cosine_vs_first": {},
        }
        raw: dict[str, dict[str, Any]] = {}
        for instruction in instructions:
            capture: dict[str, Any] = {}
            hooks = [
                model.intent.organizer.interval_object.register_forward_hook(
                    _capture_hook(capture, "organizer_interval_object")
                ),
                model.intent.coarse_action.object_read.register_forward_hook(
                    _capture_hook(capture, "coarse_object_read")
                ),
            ]
            try:
                online = _build_online(policy, history, instruction)
                with torch.no_grad():
                    cache, _metrics = deployment_cache(
                        model,
                        online,
                        policy.bundle.config,
                        collect_diagnostics=True,
                    )
                value = _instruction_record(cache.top.intent, capture)
            finally:
                for hook in hooks:
                    hook.remove()
            layout_record["instructions"][instruction] = value
            raw[instruction] = value
        baseline_name = instructions[0]
        baseline = raw[baseline_name]
        for instruction in instructions[1:]:
            layout_record["pairwise_vs_first"][instruction] = _pairwise(
                baseline, raw[instruction]
            )
            layout_record["per_slot_cosine_vs_first"][instruction] = {
                "object_tokens": _cosine_per_slot(
                    baseline["object_tokens"], raw[instruction]["object_tokens"]
                ),
                "organizer_object_value": _cosine_per_slot(
                    baseline["organizer_interval_object"]["value"],
                    raw[instruction]["organizer_interval_object"]["value"],
                ),
                "coarse_object_value": _cosine_per_slot(
                    baseline["coarse_object_read"]["value"],
                    raw[instruction]["coarse_object_read"]["value"],
                ),
            }
        records[layout] = layout_record
    result = {
        "schema": "clearvla-calvin-object-binding-npz-v1",
        "checkpoint": str(checkpoint),
        "observation_dir": str(observation_dir),
        "layouts": records,
        "instructions": list(instructions),
        "scope": {
            "parameters_changed": False,
            "optimizer_step": False,
            "checkpoint_write": False,
            "simulator_state_changed": False,
            "interpretation": (
                "object-token changes under language should be zero because they are visual-only; "
                "language-dependent object-axis attention/update is the relevant binding signal. "
                "A large public/S change with invariant object reads indicates direction/goal response "
                "without a language-conditioned object pointer."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=["standard", "blue_on_slider_left", "blue_on_slider_right", "red_on_slider_left"])
    parser.add_argument("--instructions", nargs="+", default=list(DEFAULT_INSTRUCTIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    args = parser.parse_args()
    result = run_probe(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        observation_dir=args.observation_dir,
        output=args.output,
        layouts=tuple(str(value) for value in args.layouts),
        instructions=tuple(str(value) for value in args.instructions),
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
    )
    print(json.dumps({"output": str(args.output), "layouts": list(result["layouts"])}, indent=2))


if __name__ == "__main__":
    main()
