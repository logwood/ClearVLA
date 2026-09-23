"""Check the two-stage TargetFact microgrid contraction against the old form."""

from __future__ import annotations

import json

import torch

from clearvla.mainline.model.v120_p1 import LateRawDetailPolicyReader


def old_object(camera_route, fine_weights, micro_basis, value):
    rows = []
    basis = micro_basis.to(device=fine_weights.device, dtype=torch.float32)
    value_f = value.float()
    for index in range(int(basis.shape[1])):
        local = fine_weights.float() * basis[:, index]
        local = local / local.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        rows.append(torch.einsum(
            "bqgcijmn,bkcijm,bcijmnv->bqgkcv", local, camera_route, value_f
        ))
    return torch.stack(rows, dim=-2)


def old_camera(camera_route, fine_weights, micro_basis, value):
    rows = []
    basis = micro_basis.to(device=fine_weights.device, dtype=torch.float32)
    value_f = value.float()
    for index in range(int(basis.shape[1])):
        local = fine_weights.float() * basis[:, index]
        local = local / local.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        rows.append(torch.einsum(
            "bqgcijm,bqgcijmn,bcijmnr->bqgcr", camera_route, local, value_f
        ))
    return torch.stack(rows, dim=-2)


def compare(kind: str, seed: int) -> dict[str, float | str]:
    torch.manual_seed(seed)
    b, q, g, k, c, y, x, m, n, v = 2, 3, 2, 4, 2, 3, 3, 2, 4, 7
    fine = torch.rand(b, q, g, c, y, x, m, n, requires_grad=True)
    basis = torch.rand(n, 3)
    value = torch.randn(b, c, y, x, m, n, v, requires_grad=True)
    route = (
        torch.rand(b, k, c, y, x, m, requires_grad=True)
        if kind == "object"
        else torch.rand(b, q, g, c, y, x, m, requires_grad=True)
    )
    if kind == "object":
        old = old_object(route, fine, basis, value)
        new = LateRawDetailPolicyReader._microgrid_by_object_camera(
            camera_route=route, fine_weights=fine, micro_basis=basis, value=value
        )
    elif kind == "camera":
        old = old_camera(route, fine, basis, value)
        new = LateRawDetailPolicyReader._microgrid_by_camera(
            camera_route=route, fine_weights=fine,
            micro_basis=basis, value=value
        )
    else:
        raise ValueError(kind)
    old_loss = old.square().mean()
    new_loss = new.square().mean()
    old_grads = torch.autograd.grad(old_loss, (fine, route, value), retain_graph=True)
    new_grads = torch.autograd.grad(new_loss, (fine, route, value), retain_graph=True)
    def maxdiff(a, b):
        return float((a.detach().float() - b.detach().float()).abs().max())
    return {
        "kind": kind,
        "forward_max_abs": maxdiff(old, new),
        "fine_grad_max_abs": maxdiff(old_grads[0], new_grads[0]),
        "route_grad_max_abs": maxdiff(old_grads[1], new_grads[1]),
        "value_grad_max_abs": maxdiff(old_grads[2], new_grads[2]),
    }


def zero_case() -> dict[str, object]:
    b, q, g, k, c, y, x, m, n, v = 1, 2, 1, 4, 2, 2, 2, 1, 4, 3
    fine = torch.zeros(b, q, g, c, y, x, m, n, requires_grad=True)
    basis = torch.ones(n, 2)
    value = torch.randn(b, c, y, x, m, n, v, requires_grad=True)
    route = torch.zeros(b, k, c, y, x, m, requires_grad=True)
    output = LateRawDetailPolicyReader._microgrid_by_object_camera(
        camera_route=route, fine_weights=fine, micro_basis=basis, value=value
    )
    grads = torch.autograd.grad(output.square().sum(), (fine, route, value))
    return {
        "output_max_abs": float(output.abs().max()),
        "finite": bool(torch.isfinite(output).all()),
        "gradient_max_abs": [float(grad.abs().max()) for grad in grads],
    }


def main() -> None:
    report = {
        "schema": "clearvla-target-fact-microgrid-parity-v1",
        "checks": [compare("object", 1101), compare("camera", 1102)],
        "zero_case": zero_case(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
