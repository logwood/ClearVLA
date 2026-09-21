#!/usr/bin/env python3
"""Audit what P2's KxC-to-2D geometry value pooling can and cannot identify.

The input is the tensor artifact emitted by
``probe_calvin_latest_checkpoint_target_y_replay.py``.  This script performs
no model forward and changes no checkpoint.  It first reproduces the saved P2
selected geometry values from the frozen spatial posterior and W transport,
then distinguishes:

* the unavoidable nullspace of one query's weighted 2-D value read; and
* a single shared transport perturbation that must remain invisible to every
  recorded Q5 query and to both common and interval-residual reads.

The second claim is admitted only when the actual all-query constraint matrix
has a non-zero numerical nullspace.  A full-rank matrix explicitly withdraws
the stronger "complete Q5 cannot distinguish it" interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "clearvla-calvin-p2-geometry-nullspace-audit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rms(value: np.ndarray) -> float:
    value64 = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(value64))))


def _source_add(left: np.ndarray, right: np.ndarray, *, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return (
            torch.from_numpy(left)
            .to(torch.bfloat16)
            .add(torch.from_numpy(right).to(torch.bfloat16))
            .float()
            .numpy()
        )
    if dtype == "fp32":
        return np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)
    raise ValueError(f"unsupported source value dtype {dtype!r}")


def _source_select(
    posterior: np.ndarray,
    value: np.ndarray,
    *,
    dtype: str,
) -> np.ndarray:
    posterior_t = torch.from_numpy(np.array(posterior, copy=True))
    value_t = torch.from_numpy(np.array(value, copy=True))
    if dtype == "bf16":
        posterior_t = posterior_t.to(torch.bfloat16)
        value_t = value_t.to(torch.bfloat16)
    elif dtype == "fp32":
        posterior_t = posterior_t.float()
        value_t = value_t.float()
    else:
        raise ValueError(f"unsupported source value dtype {dtype!r}")
    return torch.einsum(
        "sbtqin,binv->sbtqiv",
        posterior_t,
        value_t,
    ).float().numpy()


def _variant_prefixes(keys: list[str]) -> list[str]:
    suffix = "__p2_geometry_spatial_posterior"
    prefixes = sorted(key[: -len(suffix)] for key in keys if key.endswith(suffix))
    if not prefixes:
        raise ValueError("tensor artifact contains no P2 geometry posterior")
    return prefixes


def _constraint_matrix(posterior: np.ndarray) -> np.ndarray:
    """Return A for common and interval-residual preservation.

    Variables are a shared physical perturbation Delta[I,N] to the complete
    interval transport.  The frozen producer decomposition is
    common=mean_I(Delta) and residual_i=Delta_i-common.  Every recorded query
    contributes one common-read and one residual-read constraint.
    """

    if posterior.ndim != 6:
        raise ValueError("posterior must be [calls,B,T,Q,I,N]")
    intervals = int(posterior.shape[-2])
    spatial = int(posterior.shape[-1])
    flat = posterior.reshape(-1, intervals, spatial).astype(np.float64)
    rows: list[np.ndarray] = []
    for node in flat:
        for interval in range(intervals):
            weight = node[interval]
            common = np.tile(weight / float(intervals), intervals)
            residual = -common.reshape(intervals, spatial)
            residual[interval] += weight
            rows.append(common)
            rows.append(residual.reshape(-1))
    return np.stack(rows, axis=0)


def _single_query_counterexample(
    posterior: np.ndarray,
    transport: np.ndarray,
    *,
    cameras: int,
) -> dict[str, Any]:
    calls, batch, horizon, basis, intervals, spatial = posterior.shape
    if batch != 1 or spatial % cameras:
        raise ValueError("single-query audit expects batch one and complete KxC")
    objects = spatial // cameras
    reshaped = posterior.reshape(calls, batch, horizon, basis, intervals, objects, cameras)
    if cameras < 2:
        raise ValueError("camera-mixture audit requires at least two cameras")

    # Choose the actually recorded node/K with the strongest two-camera joint
    # weight.  This is an algebraic interface witness, not a target identity.
    pair_mass = reshaped[..., 0] + reshaped[..., 1]
    index = np.unravel_index(int(np.argmax(pair_mass)), pair_mass.shape)
    call, batch_index, time, query, interval, object_index = index
    p1 = float(reshaped[index + (0,)])
    p2 = float(reshaped[index + (1,)])
    if p1 <= 0.0 or p2 <= 0.0:
        raise ValueError("selected camera pair has no two-sided posterior support")

    field_rms = _rms(transport)
    d = np.array([field_rms, 0.0], dtype=np.float64)
    delta_first = p2 * d
    delta_second = -p1 * d
    mixture = p1 * delta_first + p2 * delta_second
    perturbation = np.stack((delta_first, delta_second), axis=0)
    residual = float(np.linalg.norm(mixture))
    perturbation_norm = float(np.linalg.norm(perturbation))
    tolerance = max(1e-12, np.finfo(np.float64).eps * 100.0 * perturbation_norm)
    minimum_nonzero = max(100.0 * tolerance, field_rms * 0.01)
    return {
        "node": {
            "call": int(call),
            "batch": int(batch_index),
            "time": int(time),
            "basis": int(query),
            "interval": int(interval),
            "object_index_not_identity": int(object_index),
            "camera_indices": [0, 1],
        },
        "posterior_weights": [p1, p2],
        "transport_field_rms": field_rms,
        "d_norm": float(np.linalg.norm(d)),
        "perturbation_norm": perturbation_norm,
        "minimum_nonzero": minimum_nonzero,
        "mixture_residual": residual,
        "residual_tolerance": tolerance,
        "nontrivial": bool(perturbation_norm > minimum_nonzero),
        "mixture_preserved": bool(residual <= tolerance),
        "scope": "one recorded query and one common-or-residual value read",
    }


def _audit_variant(
    payload: Any,
    prefix: str,
    *,
    source_value_dtype: str,
) -> dict[str, Any]:
    posterior = np.asarray(
        payload[f"{prefix}__p2_geometry_spatial_posterior"], dtype=np.float64
    )
    selected = np.asarray(
        payload[f"{prefix}__p2_geometry_selected_preterminal"], dtype=np.float64
    )
    transport = np.asarray(payload[f"{prefix}__w_transport_mean"], dtype=np.float64)
    if posterior.ndim != 6 or selected.ndim != 6 or transport.ndim != 5:
        raise ValueError("unexpected P2/W tensor rank")
    calls, batch, horizon, basis, intervals, spatial = posterior.shape
    if tuple(selected.shape) != (calls, batch, horizon, basis, intervals, 2):
        raise ValueError("selected geometry tensor does not align with posterior")
    if tuple(transport.shape[:2]) != (batch, intervals) or transport.shape[-1] != 2:
        raise ValueError("W transport tensor does not align with P2 intervals")
    objects = int(transport.shape[2])
    cameras = int(transport.shape[3])
    if objects * cameras != spatial:
        raise ValueError("P2 flattened spatial axis is not KxC")
    if calls % 2:
        raise ValueError("Q5 trace must split evenly into proposal/refined calls")
    proposal_calls = calls // 2

    flat_transport = transport.reshape(batch, intervals, spatial, 2)
    # The artifact stores the initial cached W tensor. Calls in the second
    # half use the one rebuilt W generation, so field-level parity is limited
    # to proposal calls. Newer artifacts separately preserve common/residual
    # outputs for all calls, allowing their actual source-dtype sum to be
    # checked even when the rebuilt W tensor itself is unavailable.
    proposal_posterior = posterior[:proposal_calls]
    proposal_selected = selected[:proposal_calls]
    rebuilt = _source_select(
        proposal_posterior,
        flat_transport,
        dtype=source_value_dtype,
    )
    parity_delta = rebuilt - proposal_selected
    parity_max_abs = float(np.max(np.abs(parity_delta)))
    parity_rms = _rms(parity_delta)
    parity_tolerance = 1e-7

    common = flat_transport.mean(axis=1)
    residual = flat_transport - common[:, None]
    selected_common = _source_select(
        proposal_posterior,
        np.broadcast_to(common[:, None], flat_transport.shape),
        dtype=source_value_dtype,
    )
    selected_residual = _source_select(
        proposal_posterior,
        residual,
        dtype=source_value_dtype,
    )
    approximate_sum = _source_add(
        selected_common,
        selected_residual,
        dtype=source_value_dtype,
    )
    decomposition_delta = approximate_sum - proposal_selected
    decomposition_max_abs = float(np.max(np.abs(decomposition_delta)))

    split_keys = {
        "selected_common": f"{prefix}__p2_geometry_selected_common",
        "selected_residual": f"{prefix}__p2_geometry_selected_residual",
        "w_common": f"{prefix}__w_transport_common",
        "w_residual": f"{prefix}__w_transport_interval_innovation",
    }
    split_available = all(key in payload.files for key in split_keys.values())
    exact_split: dict[str, Any]
    if split_available:
        captured_common = np.asarray(payload[split_keys["selected_common"]])
        captured_residual = np.asarray(payload[split_keys["selected_residual"]])
        captured_sum = _source_add(
            captured_common,
            captured_residual,
            dtype=source_value_dtype,
        )
        sum_delta = captured_sum - selected
        w_common = np.asarray(payload[split_keys["w_common"]]).reshape(
            batch,
            spatial,
            2,
        )
        w_common_by_interval = np.broadcast_to(
            w_common[:, None],
            (batch, intervals, spatial, 2),
        )
        w_residual = np.asarray(payload[split_keys["w_residual"]]).reshape(
            batch,
            intervals,
            spatial,
            2,
        )
        rebuilt_common = _source_select(
            proposal_posterior,
            w_common_by_interval,
            dtype=source_value_dtype,
        )
        rebuilt_residual = _source_select(
            proposal_posterior,
            w_residual,
            dtype=source_value_dtype,
        )
        common_delta = rebuilt_common - captured_common[:proposal_calls]
        residual_delta = rebuilt_residual - captured_residual[:proposal_calls]
        exact_split = {
            "available": True,
            "selected_common_plus_residual_max_abs_all_calls": float(
                np.max(np.abs(sum_delta))
            ),
            "selected_sum_passed": bool(np.max(np.abs(sum_delta)) <= 1e-7),
            "initial_w_common_read_max_abs": float(np.max(np.abs(common_delta))),
            "initial_w_residual_read_max_abs": float(np.max(np.abs(residual_delta))),
            "initial_w_reads_passed": bool(
                np.max(np.abs(common_delta)) <= 1e-7
                and np.max(np.abs(residual_delta)) <= 1e-7
            ),
            "tolerance": 1e-7,
        }
    else:
        exact_split = {
            "available": False,
            "reason": (
                "legacy tensor artifact omitted W/selected common and residual; "
                "full-transport reconstruction is only an approximate ordering check"
            ),
        }

    matrix = _constraint_matrix(posterior)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = 1e-8 * largest
    rank = int(np.sum(singular_values > rank_tolerance))
    variable_count = int(matrix.shape[1])
    nullity = variable_count - rank
    all_query: dict[str, Any] = {
        "matrix_shape": list(matrix.shape),
        "variable_shape_per_coordinate": [intervals, spatial],
        "rank": rank,
        "nullity": nullity,
        "rank_tolerance": rank_tolerance,
        "largest_singular_value": largest,
        "smallest_singular_value": float(singular_values[-1]),
        "complete_query_set_invariance_admitted": bool(nullity > 0),
    }
    if nullity > 0:
        vector = vh[-1].astype(np.float64)
        vector_norm = float(np.linalg.norm(vector))
        scale = _rms(transport) / max(vector_norm, 1e-30)
        perturbation = vector * scale
        residual_vector = matrix @ perturbation
        residual_norm = float(np.linalg.norm(residual_vector))
        perturbation_norm = float(np.linalg.norm(perturbation))
        tolerance = max(
            1e-12,
            1e-8 * max(float(np.linalg.norm(matrix)) * perturbation_norm, 1e-30),
        )
        all_query.update(
            {
                "witness_perturbation_norm": perturbation_norm,
                "witness_residual_norm": residual_norm,
                "witness_residual_tolerance": tolerance,
                "witness_passed": bool(residual_norm <= tolerance),
            }
        )
    else:
        all_query["interpretation"] = (
            "the recorded Q5 posterior family has full column rank for a shared "
            "transport perturbation; the one-query value nullspace cannot be "
            "promoted to complete-Q5 invariance"
        )

    return {
        "shape": {
            "calls": calls,
            "batch": batch,
            "horizon": horizon,
            "basis": basis,
            "intervals": intervals,
            "objects": objects,
            "cameras": cameras,
            "value_width": 2,
        },
        "saved_value_parity": {
            "scope": "proposal calls using the saved initial W generation",
            "calls_checked": proposal_calls,
            "refined_calls_unchecked_without_rebuilt_w_tensor": calls - proposal_calls,
            "max_abs": parity_max_abs,
            "rms": parity_rms,
            "tolerance": parity_tolerance,
            "passed": bool(parity_max_abs <= parity_tolerance),
            "source_value_dtype": source_value_dtype,
            "exact_split_artifact": exact_split,
        },
        "common_residual_decomposition": {
            "max_abs": decomposition_max_abs,
            "tolerance": parity_tolerance,
            "passed": bool(decomposition_max_abs <= parity_tolerance),
        },
        "single_query_equal_mixture": _single_query_counterexample(
            proposal_posterior,
            transport,
            cameras=cameras,
        ),
        "all_recorded_queries_shared_transport": all_query,
    }


def run(
    *,
    tensor_input: Path,
    output: Path,
    source_value_dtype: str,
) -> dict[str, Any]:
    with np.load(tensor_input, allow_pickle=False) as payload:
        prefixes = _variant_prefixes(list(payload.files))
        variants = {
            prefix: _audit_variant(
                payload,
                prefix,
                source_value_dtype=source_value_dtype,
            )
            for prefix in prefixes
        }
    result = {
        "schema": SCHEMA,
        "tensor_input": {
            "path": str(tensor_input.resolve()),
            "sha256": _sha256(tensor_input),
        },
        "source_value_dtype": source_value_dtype,
        "variants": variants,
        "interpretation_contract": {
            "single_query": (
                "proves only a local value-interface non-identifiability under a "
                "fixed posterior; it is not a new physical scene or a full action claim"
            ),
            "all_queries": (
                "tests one shared interval/K/camera transport perturbation against every "
                "recorded Q5 posterior and both producer common/residual reads"
            ),
            "not_tested": (
                "address/key changes, P1/S factual paths and action consequences are not "
                "held invariant by this algebra-only audit"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-value-dtype",
        choices=("bf16", "fp32"),
        required=True,
    )
    args = parser.parse_args()
    result = run(
        tensor_input=args.tensor_input,
        output=args.output,
        source_value_dtype=str(args.source_value_dtype),
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "variants": sorted(result["variants"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
