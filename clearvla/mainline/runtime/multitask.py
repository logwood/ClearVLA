"""CPU-task-stratified reporting over one shared model evaluation.

Task identity selects rows only after deployment prediction has completed.  It
is never passed to model forward, loss construction or an optimizer owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from ..data.normalizer import ArrayNormalizer
from ..interfaces import TrainingBatch
from .evaluation import ValidationAccumulator

TASK_VALIDATION_METRICS = (
    "validation_action_rmse_physical",
    "validation_band_1_4_rmse_physical",
    "validation_band_5_12_rmse_physical",
    "validation_band_13_24_rmse_physical",
    "validation_arm_rmse_physical",
    "validation_gripper_rmse_physical",
    "validation_decoded_gripper_event_precision",
    "validation_decoded_gripper_event_recall",
    "validation_decoded_gripper_event_f1",
    "validation_decoded_gripper_events_predicted",
    "validation_decoded_gripper_events_target",
)

TASK_MACRO_METRICS = tuple(
    name
    for name in TASK_VALIDATION_METRICS
    if name
    not in {
        "validation_decoded_gripper_events_predicted",
        "validation_decoded_gripper_events_target",
    }
)


def _project(metrics: Mapping[str, float], *, samples: int) -> dict[str, float]:
    missing = [name for name in TASK_VALIDATION_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"multitask validation is missing required metrics: {missing}")
    return {
        "validation_sample_count": float(samples),
        **{name: float(metrics[name]) for name in TASK_VALIDATION_METRICS},
    }


@dataclass
class TaskValidationAccumulators:
    task_order: tuple[str, ...]
    accumulators: tuple[ValidationAccumulator, ...]

    @classmethod
    def from_action_normalizer(
        cls,
        task_order: tuple[str, ...],
        normalizer: ArrayNormalizer,
        *,
        device: torch.device,
        gripper_event_threshold: float,
        arm_motion_threshold: float,
    ) -> "TaskValidationAccumulators":
        if not task_order or len(set(task_order)) != len(task_order):
            raise ValueError("multitask validation requires an ordered unique task registry")
        return cls(
            task_order=tuple(task_order),
            accumulators=tuple(
                ValidationAccumulator.from_action_normalizer(
                    normalizer,
                    device=device,
                    gripper_event_threshold=gripper_event_threshold,
                    arm_motion_threshold=arm_motion_threshold,
                )
                for _ in task_order
            ),
        )

    def update(
        self,
        task_indices: Tensor,
        prediction: Tensor,
        batch: TrainingBatch,
        *,
        motion_logits: Tensor | None = None,
        motion_target: Tensor | None = None,
        physical_field: Tensor | None = None,
        gripper_decode_delta_blend: float | None = None,
    ) -> None:
        tasks = task_indices.detach().to(device="cpu", dtype=torch.long)
        if tasks.ndim != 1 or int(tasks.numel()) != int(prediction.shape[0]):
            raise ValueError("validation task indices must align with prediction rows")
        if not tasks.numel():
            raise ValueError("multitask validation cannot consume an empty batch")
        if int(tasks.min()) < 0 or int(tasks.max()) >= len(self.task_order):
            raise IndexError("validation task index is outside the task registry")
        for task in torch.unique(tasks, sorted=True).tolist():
            rows = torch.nonzero(tasks == int(task), as_tuple=False).flatten()
            self.accumulators[int(task)].update(
                prediction,
                batch,
                motion_logits=motion_logits,
                motion_target=motion_target,
                physical_field=physical_field,
                gripper_decode_delta_blend=gripper_decode_delta_blend,
                row_indices=rows,
            )

    def report(self, micro_metrics: Mapping[str, float]) -> dict[str, object]:
        task_metrics: dict[str, dict[str, float]] = {}
        for name, accumulator in zip(
            self.task_order,
            self.accumulators,
            strict=True,
        ):
            if accumulator.samples:
                task_metrics[name] = _project(
                    accumulator.means(),
                    samples=accumulator.samples,
                )
        observed = tuple(task_metrics)
        missing = tuple(name for name in self.task_order if name not in task_metrics)
        sample_count = sum(
            accumulator.samples for accumulator in self.accumulators
        )
        if not observed or sample_count <= 0:
            raise ValueError("multitask validation did not observe any registered task")
        macro = {
            name: float(
                sum(task_metrics[task][name] for task in observed) / len(observed)
            )
            for name in TASK_MACRO_METRICS
        }
        return {
            "schema": "clearvla-multitask-validation-v1",
            "metric_space": "shared_raw_physical_action_chart",
            "expected_task_count": len(self.task_order),
            "observed_task_count": len(observed),
            "task_coverage": float(len(observed) / len(self.task_order)),
            "observed_tasks": list(observed),
            "missing_tasks": list(missing),
            "sample_count": sample_count,
            "micro": _project(micro_metrics, samples=sample_count),
            "macro": macro,
            "tasks": task_metrics,
        }


__all__ = [
    "TASK_MACRO_METRICS",
    "TASK_VALIDATION_METRICS",
    "TaskValidationAccumulators",
]
