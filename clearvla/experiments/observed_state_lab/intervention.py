from __future__ import annotations

"""Formal V35 intervention-branch tensor contract.

The NPZ stores episode/frame keys into the ordinary DINO cache plus raw physical
state/action arrays. This avoids duplicating image tokens while allowing short
counterfactual branches from the same initial state.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer


REQUIRED_FIELDS = (
    "branch_group",
    "history_keys",
    "target_history_keys",
    "state_raw",
    "history_state_raw",
    "executed_action_history_raw",
    "target_history_state_raw",
    "target_executed_action_history_raw",
    "action_raw",
    "future_state_raw",
    "segment_state_raw",
)


class InterventionBranchDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        action_normalizer: ArrayNormalizer,
        state_normalizer: ArrayNormalizer,
        policy_horizon: int,
    ) -> None:
        data = np.load(path, allow_pickle=False)
        missing = [name for name in REQUIRED_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"intervention NPZ missing fields: {missing}")
        self.data = {name: np.asarray(data[name]) for name in data.files}
        count = len(self.data["branch_group"])
        for name, value in self.data.items():
            if len(value) != count:
                raise ValueError(f"intervention field {name!r} length mismatch")
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.policy_horizon = int(policy_horizon)

    def __len__(self) -> int:
        return len(self.data["branch_group"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        d = self.data
        state_raw = d["state_raw"][index].astype(np.float32)
        history_state_raw = d["history_state_raw"][index].astype(np.float32)
        executed_raw = d["executed_action_history_raw"][index].astype(np.float32)
        target_state_raw = d["target_history_state_raw"][index].astype(np.float32)
        target_executed_raw = d["target_executed_action_history_raw"][index].astype(np.float32)
        action_raw = d["action_raw"][index].astype(np.float32)
        future_state_raw = d["future_state_raw"][index].astype(np.float32)
        segment_state_raw = d["segment_state_raw"][index].astype(np.float32)
        return {
            "branch_group": torch.tensor(int(d["branch_group"][index]), dtype=torch.long),
            "branch_index": torch.tensor(index, dtype=torch.long),
            "state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "state_raw": torch.from_numpy(state_raw),
            "action_state": torch.from_numpy(self.action_normalizer.encode(state_raw)),
            "history_state": torch.from_numpy(self.state_normalizer.encode(history_state_raw)),
            "executed_action_history": torch.from_numpy(self.action_normalizer.encode(executed_raw)),
            "target_history_state": torch.from_numpy(self.state_normalizer.encode(target_state_raw)),
            "target_executed_action_history": torch.from_numpy(self.action_normalizer.encode(target_executed_raw)),
            "action": torch.from_numpy(self.action_normalizer.encode(action_raw)),
            "action_raw": torch.from_numpy(action_raw),
            "policy_action": torch.from_numpy(self.action_normalizer.encode(action_raw[: self.policy_horizon])),
            "policy_action_raw": torch.from_numpy(action_raw[: self.policy_horizon]),
            "future_state": torch.from_numpy(self.state_normalizer.encode(future_state_raw)),
            "future_state_raw": torch.from_numpy(future_state_raw),
            "segment_state": torch.from_numpy(self.state_normalizer.encode(segment_state_raw)),
            "segment_state_raw": torch.from_numpy(segment_state_raw),
            "history_keys": torch.from_numpy(d["history_keys"][index].astype(np.int64)),
            "target_history_keys": torch.from_numpy(d["target_history_keys"][index].astype(np.int64)),
        }


def validate_intervention_groups(groups: np.ndarray) -> None:
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    bad = unique[counts < 2]
    if len(bad):
        raise ValueError(f"every intervention group needs at least two branches; invalid={bad.tolist()}")


__all__ = ["InterventionBranchDataset", "validate_intervention_groups", "REQUIRED_FIELDS"]
