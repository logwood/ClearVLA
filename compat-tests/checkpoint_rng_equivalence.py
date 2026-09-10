"""Check that raw-flow checkpoint RNG preservation is numerically redundant.

The production raw pyramid/refiner contains no stochastic layers.  This probe
compares the checkpoint wrapper with and without RNG-state preservation on the
same module, inputs, parameters, and backward seed.  It is a diagnostic proof
for the capture workaround; it is not a training benchmark.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import h5py  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("h5py")
    stub.File = type("File", (), {})
    stub.Dataset = type("Dataset", (), {})
    sys.modules["h5py"] = stub

from clearvla.mainline.v120_core import flow_dino_evidence as flow


def _run(
    preserve: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(1827)
    module = flow._RawImagePyramid(base=8, activation_checkpoint=True).train()
    value = torch.randn(2, 3, 32, 32, requires_grad=True)
    original = flow.checkpoint

    def wrapped(function, *args, **kwargs):
        kwargs["preserve_rng_state"] = preserve
        return original(function, *args, **kwargs)

    flow.checkpoint = wrapped
    try:
        high, mid = module(value)
        loss = high.square().mean() + mid.square().mean()
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        return (high.detach(), mid.detach()), gradients
    finally:
        flow.checkpoint = original


def main() -> None:
    torch.use_deterministic_algorithms(True)
    preserved, preserved_grad = _run(True)
    unpreserved, unpreserved_grad = _run(False)
    for left, right in zip(preserved, unpreserved, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    if preserved_grad.keys() != unpreserved_grad.keys():
        raise AssertionError("gradient parameter sets differ")
    for name in preserved_grad:
        torch.testing.assert_close(
            preserved_grad[name], unpreserved_grad[name], rtol=0.0, atol=0.0
        )
    print("raw_checkpoint_rng_equivalence_ok")


if __name__ == "__main__":
    main()
