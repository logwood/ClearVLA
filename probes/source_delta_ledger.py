"""Offline symmetric source/value/route decomposition; no model execution.

Input: source_values [J,B,T,Q,H], beta [B,T,Q], value_scale [B,T,Q,1],
reader_output [B,T,H], source_names [J], and metadata_json scalar text.
All calculations below use float64. Numerical residuals remain separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def stats(x, reference=None):
    x = np.asarray(x, dtype=np.float64)
    result = {"rms": rms(x), "max_abs": float(np.max(np.abs(x)))}
    if reference is not None:
        y = np.asarray(reference, dtype=np.float64)
        dot = float(np.sum(x * y))
        result["cosine_to_observed_delta"] = dot / max(
            float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-30
        )
        result["signed_projection_on_observed_delta"] = dot / max(
            float(np.sum(y * y)), 1e-30
        )
    return result


def load(path):
    with np.load(path, allow_pickle=False) as p:
        return {k: np.array(p[k], copy=True) for k in p.files}


def analyze(left, right):
    names = [str(x) for x in left["source_names"]]
    if names != [str(x) for x in right["source_names"]]:
        raise ValueError("source axes differ")
    m0 = json.loads(str(left["metadata_json"]))
    m1 = json.loads(str(right["metadata_json"]))
    for k in ("checkpoint", "source_commit", "layout", "instruction", "stage",
              "integration_index", "integration_call_index", "time"):
        if m0[k] != m1[k]:
            raise ValueError(f"unaligned states: {k}")
    for k in ("checkpoint_sha256", "observation_sha256", "probe_sha256", "noise_seed"):
        if m0.get(k) != m1.get(k):
            raise ValueError(f"unaligned input identity: {k}")
    x0, x1 = (p["source_values"].astype(np.float64) for p in (left, right))
    b0, b1 = (p["beta"].astype(np.float64)[..., None] for p in (left, right))
    l0, l1 = (p["value_scale"].astype(np.float64) for p in (left, right))
    y0, y1 = (p["reader_output"].astype(np.float64) for p in (left, right))
    if x0.shape != x1.shape or x0.ndim != 5 or x0.shape[0] != len(names):
        raise ValueError("source schema must be [J,B,T,Q,H]")
    expected_beta = x0.shape[1:-1] + (1,)
    expected_output = x0.shape[1:-2] + (x0.shape[-1],)
    if any(t.shape != expected_beta for t in (b0, b1, l0, l1)):
        raise ValueError("probability/contract axes do not match the source basis")
    if any(t.shape != expected_output for t in (y0, y1)):
        raise ValueError("reader output axes do not match the source axes")
    if not all(np.isfinite(t).all() for t in (x0, x1, b0, b1, l0, l1, y0, y1)):
        raise ValueError("non-finite source decomposition input")
    a0, a1 = b0 * l0, b1 * l1
    d0, d1 = (x0 * a0).sum(-2), (x1 * a1).sum(-2)
    source = ((a0 + a1) * 0.5 * (x1 - x0)).sum(-2)
    xmean = (x0 + x1) * 0.5
    beta = ((l0 + l1) * 0.5 * (b1 - b0) * xmean).sum(-2)
    contract = ((b0 + b1) * 0.5 * (l1 - l0) * xmean).sum(-2)
    actual = y1 - y0
    residual0, residual1 = y0 - d0.sum(0), y1 - d1.sum(0)
    residual_delta = residual1 - residual0
    reconstruction = (source + beta + contract).sum(0) + residual_delta
    source_rows = {}
    for j, name in enumerate(names):
        source_rows[name] = {
            "producer_delta": stats(x1[j] - x0[j]),
            "routed_delta": stats(d1[j] - d0[j], actual),
            "source_value_term": stats(source[j], actual),
            "basis_probability_term": stats(beta[j], actual),
            "value_contract_term": stats(contract[j], actual),
            "symmetric_identity_error": stats(d1[j] - d0[j] - source[j] - beta[j] - contract[j]),
        }
    return {
        "schema": "clearvla-source-delta-ledger-v1",
        "units": "bottom action hidden update; not native command or TCP displacement",
        "interpretation": "symmetric two-state accounting, not independent causal contributions",
        "metadata": m0,
        "observed_reader_delta": stats(actual),
        "summed_source_value_term": stats(source.sum(0), actual),
        "summed_basis_probability_term": stats(beta.sum(0), actual),
        "summed_value_contract_term": stats(contract.sum(0), actual),
        "numeric_residual_baseline": stats(residual0),
        "numeric_residual_swapped": stats(residual1),
        "numeric_residual_delta": stats(residual_delta, actual),
        "paired_reader_closure_error": stats(actual - reconstruction),
        "beta_delta": stats(b1 - b0),
        "value_scale_delta": stats(l1 - l0),
        "sources": source_rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("swapped", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(load(a.baseline), load(a.swapped))
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
