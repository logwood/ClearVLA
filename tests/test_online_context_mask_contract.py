"""Unit checks for the A/B online-context mask control.

These tests intentionally avoid constructing the production V120 encoder.  The
contract being checked here is the boundary between the mainline compiler and
the encoder: the current-context switch is forwarded explicitly, while the
legacy config payload remains byte-compatible.
"""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from clearvla.mainline.config import ExperimentConfig, config_from_mapping
from clearvla.mainline.model.restored_observation import (
    RestoredV120ObservationCompiler,
)


def test_online_context_mask_default_is_legacy_and_omitted_from_identity() -> None:
    config = ExperimentConfig()
    assert config.observation.online_context_mask_mode == "legacy"
    payload = config.as_dict()
    assert "online_context_mask_mode" not in payload["observation"]

    disabled = replace(
        config,
        observation=replace(config.observation, online_context_mask_mode="disabled"),
    )
    disabled_payload = disabled.as_dict()
    assert disabled_payload["observation"]["online_context_mask_mode"] == "disabled"
    assert (
        config_from_mapping(disabled_payload).observation.online_context_mask_mode
        == "disabled"
    )
    assert disabled.digest() != config.digest()


def test_restored_compiler_forwards_online_mask_without_using_eval_mode() -> None:
    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []
            self.training = True

        def forward(self, visual, *, raw_visual=None, online_context_mask=True):
            self.calls.append(
                {
                    "training": self.training,
                    "online_context_mask": online_context_mask,
                    "raw_visual": raw_visual,
                }
            )
            return SimpleNamespace()

    compiler = RestoredV120ObservationCompiler.__new__(
        RestoredV120ObservationCompiler
    )
    nn.Module.__init__(compiler)
    compiler.config = SimpleNamespace(
        observation=SimpleNamespace(online_context_mask_mode="disabled")
    )
    compiler.encoder = FakeEncoder()
    observation = SimpleNamespace(
        dino_history=torch.zeros(1, 4, 2, 16, 16, 8),
        raw_rgb=torch.zeros(1, 4, 2, 3, 32, 32),
        validate=lambda config: None,
    )

    prepared = compiler.prepare(observation, training_mask=True)
    assert prepared.training_mask is True
    assert len(compiler.encoder.calls) == 1
    call = compiler.encoder.calls[0]
    assert call["training"] is True
    assert call["online_context_mask"] is False
    assert call["raw_visual"] is observation.raw_rgb
    # prepare restores the module mode; the switch must never be implemented
    # by changing model.eval() and thereby disabling unrelated train behavior.
    assert compiler.encoder.training is True
