"""New source revisions cannot hide inside the original A/B admission name."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from test_calvin_optimizer_moment_initialization import Tiny

from clearvla.mainline.runtime.causal_identity_migration import (
    CAUSAL_IDENTITY_AB_V1,
    CAUSAL_UNIFIED_SOURCE_V1,
    SOURCE_DIGEST,
    SOURCE_PATHS,
    allowed_source_paths,
    validate_selection,
)
from clearvla.mainline.runtime.checkpoints import InitializationState
from clearvla.mainline.train import _initialize_training_clock, _parser
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule


def test_old_allowlist_is_not_broadened():
    typed_only = "clearvla/mainline/v120_core/decoder.py"
    assert typed_only not in allowed_source_paths(CAUSAL_IDENTITY_AB_V1)
    assert typed_only in allowed_source_paths(CAUSAL_UNIFIED_SOURCE_V1)
    extra = "clearvla/vision/source_weights.py"
    assert extra not in SOURCE_PATHS
    assert extra not in allowed_source_paths(CAUSAL_IDENTITY_AB_V1)
    assert extra in allowed_source_paths(CAUSAL_UNIFIED_SOURCE_V1)
    assert "clearvla/mainline/identity_pairs.py" in allowed_source_paths(CAUSAL_UNIFIED_SOURCE_V1)
    assert "clearvla/mainline/v120_core/time_domain_mmdit.py" not in allowed_source_paths(
        CAUSAL_UNIFIED_SOURCE_V1
    )
    with pytest.raises(ValueError, match="unknown"):
        allowed_source_paths("silently_accept_everything")


def test_cli_accepts_only_explicit_new_migration():
    args = _parser().parse_args(["--init-model-contract-migration", CAUSAL_UNIFIED_SOURCE_V1])
    assert args.init_model_contract_migration == CAUSAL_UNIFIED_SOURCE_V1
    assert args.init_optimizer_state == "fresh"


def test_mature_execution_clock_does_not_skip_new_optimizer_warmup():
    model = Tiny(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=100, total_steps=1024, minimum_ratio=0.1, update_origin=11012
    )
    engine = SimpleNamespace(global_step=0, model=model)
    source = SimpleNamespace(model_contract_migration=CAUSAL_UNIFIED_SOURCE_V1, global_step=11012)
    _initialize_training_clock(
        cast(MainlineTrainingEngine, engine),
        schedule,
        cast(InitializationState, source),
        mode="checkpoint",
    )
    assert engine.global_step == model.completed == schedule.step_index == 11012
    assert optimizer.param_groups[0]["lr"] == pytest.approx(8e-7)
    assert not optimizer.state


def test_unrelated_source_digest_remains_rejected():
    with pytest.raises(ValueError, match="source identity"):
        validate_selection(None, None, "0" * len(SOURCE_DIGEST))
