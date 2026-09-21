#!/usr/bin/env python3
"""Prepare (never launch) a matched CALVIN S-query experiment from audited data."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from clearvla.mainline.config import config_from_mapping, load_config


def prepare(reference: dict, *, output_root: str) -> tuple[dict, dict]:
    base = load_config(Path(__file__).resolve().parents[1] / "configs/mainline/calvin_object_intent_dynamics_323_binary.json")
    payload = base.as_dict()
    # Preserve the complete data contract, including seed, sampler, split,
    # stride and normalizer. Unknown fields fail in the normal config parser.
    payload["data"] = dict(reference["config"]["data"])
    base = config_from_mapping(payload)
    if base.data.data_profile != "calvin_relative_7d_v1":
        raise ValueError("pilot reference must be CALVIN")
    base = replace(
        base,
        optimizer=replace(base.optimizer, epochs=5, batch_size=8),
        runtime=replace(
            base.runtime, max_train_batches=1000, max_val_batches=64,
            validation_panel="evenly_spaced", log_every=25,
            eval_sampling_diagnostic_batches=4, eval_proposal_ablation_batches=4,
            eval_execution_ablation_batches=2,
        ),
    )
    configs = {}
    for mode in ("goal_history", "history_only"):
        config = replace(
            base,
            data=replace(base.data, output_dir=f"{output_root.rstrip('/')}-{mode.replace('_', '-')}"),
            top=replace(base.top, interval_object_query_mode=mode),
        )
        config.validate()
        configs[mode] = config.as_dict()
    manifest = {
        "schema": "clearvla-calvin-shared-s-pilot-v1",
        "factor": "goal innovation in S interval K query only",
        "arms": list(configs),
        "source_reference_commit": reference["identity"]["git_commit"],
        "expected_dataset": reference["identity"]["dataset"],
        "expected_normalizers": reference["normalizer_fingerprints"],
        "steps_per_short_epoch": 1000,
        "epochs": 5,
        "retain_checkpoint_epochs": [2, 5],
        "expected_retained_steps": [2000, 5000],
        "interpretation": (
            "Fresh initialization with the current shared codec; not an exact resume or a "
            "controlled comparison with legacy V1/V2. Training reshuffles at each 1000-step "
            "short epoch. The same 64 evenly spaced validation batches are used for both arms."
        ),
    }
    return configs, manifest


def compare_contexts(left: dict, right: dict) -> None:
    """Reject a purported single-factor comparison if other run contracts differ."""
    normalized = []
    for context, expected_mode in ((left, "goal_history"), (right, "history_only")):
        config = config_from_mapping(context["config"])
        if config.top.interval_object_query_mode != expected_mode:
            raise ValueError(f"expected {expected_mode} arm")
        normalized.append(replace(
            config, data=replace(config.data, output_dir="paired"),
            top=replace(config.top, interval_object_query_mode="goal_history"),
        ).as_dict())
    if normalized[0] != normalized[1]:
        raise ValueError("pilot configs differ beyond the named query factor/output")
    for name in ("dataset", "language", "source"):
        if left["identity"][name] != right["identity"][name]:
            raise ValueError(f"pilot {name} identities differ")
    if left["normalizer_fingerprints"] != right["normalizer_fingerprints"]:
        raise ValueError("pilot normalizers differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    args = parser.parse_args()
    configs, manifest = prepare(
        json.loads(args.reference_context.read_text(encoding="utf-8")), output_root=args.run_prefix
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {**configs, "pilot_manifest": manifest}.items():
        destination = args.output_dir / f"{name}.json"
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
