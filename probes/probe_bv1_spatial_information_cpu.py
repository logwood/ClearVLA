"""CPU capability audit with saved B-v1 weights; no policy/optimizer updates.

Constructed probability laws are not natural images or physical object labels.
All other object attributes are held fixed to localize one information boundary.
New feature offsets / contextual reads below are diagnostic alternatives only.
"""
from pathlib import Path
import argparse
import copy
from dataclasses import fields
import hashlib
import inspect
import json
import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch import nn
import torch.nn.functional as F

from clearvla.mainline.model.compiler import ObjectFutureEffectReader
from clearvla.mainline.model.task_execution import JointTaskRelationEncoder
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.model.types import ObjectFactSet, ObjectWorldBelief, FutureObjectDynamics
from clearvla.mainline.model.view_geometry import ViewConditionedTransport
from clearvla.mainline.task_execution import JOINT_SPATIAL_TASK_EXECUTION
from clearvla.vision.entity_chart import current_image_grid


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(8 * 1024**2), b""): h.update(x)
    return h.hexdigest()


def tensor_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


class CenteredOffsetCoordinates(nn.Module):
    """Audit-only: W2[SiLU(W1 x + b) - SiLU(b)], b starts at exact zero."""
    def __init__(self, original):
        super().__init__()
        self.first = copy.deepcopy(original[0])
        self.last = copy.deepcopy(original[2])
        self.offset = nn.Parameter(torch.zeros(self.first.out_features))

    def forward(self, x):
        return self.last(F.silu(self.first(x) + self.offset) - F.silu(self.offset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(481)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert payload["global_step"] == 12036
    assert payload["identity"]["git_commit"] == "3a84399926d50831470518e1b7f105131d3f1056"
    assert payload["config"]["top"]["task_execution_mode"] == "joint_object_scene_v1"
    saved = payload["model"]
    prefix = "intent.organizer.task_relation_encoder."
    state = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
    hidden, state_dim = state["robot.weight"].shape
    cameras = tuple(payload["config"]["data"]["camera_names"])
    model = JointTaskRelationEncoder(hidden=hidden, state_dim=state_dim, camera_names=cameras)
    model.load_state_dict(state, strict=True)
    model.eval()
    before = tensor_sha(model.state_dict())
    grid = current_image_grid(16, 16, device=torch.device("cpu")).flatten(0, 1)

    # Four positive masses, reflected to disjoint cells. First and second
    # moments agree. The signed difference is odd with zero linear moment.
    a = torch.zeros(16, 16)
    a[10, 9] = .25; a[14, 13] = .25
    a[1, 6] = .25; a[5, 2] = .25
    b = a.flip((0, 1)).contiguous()
    pa, pb = a.flatten(), b.flatten()
    delta = pa - pb
    ma, mb = pa @ grid, pb @ grid
    ca = torch.einsum("n,ni,nj->ij", pa, grid, grid)
    cb = torch.einsum("n,ni,nj->ij", pb, grid, grid)
    assert torch.allclose(ma, mb, atol=1e-7, rtol=0)
    assert torch.allclose(ca, cb, atol=1e-7, rtol=0)
    assert delta.abs().sum().item() == 2

    features = model.coordinates(grid)
    pooled_a, pooled_b = pa @ features, pb @ features
    w1 = model.coordinates[0].weight.double()
    w2 = model.coordinates[2].weight.double()
    f64 = F.linear(F.silu(F.linear(grid.double(), w1)), w2)
    reflection_identity_error = (f64 - f64.flip(0) - F.linear(F.linear(grid.double(), w1), w2)).abs().max()

    binding = TargetBinding.from_logits(torch.zeros(1, 4), torch.zeros(1, 1), torch.ones(1, 4, dtype=torch.bool))
    common = dict(task_intervals=torch.randn(1, 4, hidden),
                  attributes=torch.randn(1, 4, 2, 4, hidden), position_grid=grid,
                  state=torch.randn(1, state_dim), history_context=torch.randn(1, hidden),
                  view_observed=torch.ones(1, 4, 2, dtype=torch.bool), binding=binding,
                  current_content=torch.randn(1, 4, hidden))
    def relation(module, probability):
        return module(position_probability=probability[None, None, None].expand(1, 4, 2, -1), **common).values
    va, vb = relation(model, pa), relation(model, pb)
    repeat = relation(model, pa)
    assert torch.equal(va, repeat)
    cotangent = torch.randn_like(va)
    alpha = torch.tensor(0., requires_grad=True)
    law = (pa + pb) / 2 + alpha * delta
    derivative = torch.autograd.grad((relation(model, law) * cotangent).sum(), alpha)[0]
    # Strictly positive laws exclude a hard-zero/support artefact.
    soft_a, soft_b = .95 * pa + .05 / 256, .95 * pb + .05 / 256
    with torch.no_grad(): sa, sb = relation(model, soft_a), relation(model, soft_b)
    assert soft_a.min() > 0 and soft_b.min() > 0

    # Low-cost candidate mechanism; kept strictly inside this CPU experiment.
    shifted = copy.deepcopy(model)
    shifted.coordinates = CenteredOffsetCoordinates(model.coordinates)
    za, zb = relation(shifted, pa), relation(shifted, pb)
    assert torch.equal(za, va) and torch.equal(zb, vb)
    original_pull = torch.autograd.grad((va * cotangent).sum(), tuple(model.parameters()))
    inherited = [p for n, p in shifted.named_parameters() if n != "coordinates.offset"]
    new_pull = torch.autograd.grad((za * cotangent).sum(), inherited, retain_graph=True)
    inherited_vjp_error = max(float((x-y).abs().max()) for x, y in zip(original_pull, new_pull))
    assert inherited_vjp_error == 0
    offset_pull = torch.autograd.grad(((za-zb) * cotangent).sum(), shifted.coordinates.offset)[0]
    offset_rows = []
    direction = torch.randn(hidden)
    direction /= direction.square().mean().sqrt()
    for amplitude in (.1, -.1):
        with torch.no_grad(): shifted.coordinates.offset.copy_(direction * amplitude)
        xa, xb = relation(shifted, pa), relation(shifted, pb)
        offset_rows.append(dict(audit_offset_rms=amplitude,
                               relation_max_abs=float((xa-xb).abs().max()),
                               relation_rms=float((xa-xb).square().mean().sqrt())))
    assert shifted.coordinates(torch.zeros(1, 2)).abs().max().item() == 0

    # The already present v2 S operator is a separate, untrained capability
    # control. Switching the global mode also changes P2, so this is not an
    # approved config toggle or a matched trained model.
    contextual = JointTaskRelationEncoder(hidden=hidden, state_dim=state_dim,
        camera_names=cameras, task_execution_mode=JOINT_SPATIAL_TASK_EXECUTION)
    missing = contextual.load_state_dict(state, strict=False)
    assert missing.missing_keys == ["spatial_context.weight"] and not missing.unexpected_keys
    contextual.eval()
    with torch.no_grad(): qa, qb = relation(contextual, pa), relation(contextual, pb)

    # Actual current P2 context consumes the first moment and W covariance.
    prefix_p2 = "policy_compiler.effect_reader.view_geometry."
    p2 = ViewConditionedTransport(hidden=hidden, camera_names=cameras)
    p2.load_state_dict({k[len(prefix_p2):]: v for k, v in saved.items() if k.startswith(prefix_p2)}, strict=True)
    coords_a = ma.expand(1, 4, 2, 2); coords_b = mb.expand_as(coords_a)
    cov = torch.zeros(1, 4, 4, 2, 3); support = torch.ones(1, 4, 4, 2, dtype=torch.bool)
    context_a = p2.context_features(coords_a, cov, support)
    context_b = p2.context_features(coords_b, cov, support)
    metric = ObjectFutureEffectReader._covariance_aware_distance
    legacy_a = (-.25 * metric(grid - ma, torch.zeros(256, 3))).clamp(-1, 0)
    legacy_b = (-.25 * metric(grid - mb, torch.zeros(256, 3))).clamp(-1, 0)
    # Direct full-law kernel integral proposed by the other thread, scored
    # with our actual covariance metric; no model inference is changed.
    kernel = (-.25 * metric(grid[:, None] - grid[None], torch.zeros(256, 256, 3))).clamp(-1, 0).exp()
    integral_a, integral_b = (kernel @ pa).log(), (kernel @ pb).log()

    assert tensor_sha(model.state_dict()) == before
    assert reflection_identity_error.item() < 1e-12
    assert (pooled_a-pooled_b).abs().max().item() < 1e-6
    assert (va-vb).abs().max().item() < 1e-6
    assert offset_pull.norm().item() > 0
    report = dict(
        complete=True, device="cpu", cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        checkpoint_source=payload["identity"]["git_commit"], global_step=payload["global_step"],
        runtime_source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(inspect.getfile(ObjectFactSet)).parents[3], text=True).strip(),
        probe_sha256=sha(__file__),
        supported_laws=dict(tv=float(delta.abs().sum()/2), first_moment_max_abs=float((ma-mb).abs().max()),
            second_moment_max_abs=float((ca-cb).abs().max()), shared_support=True,
            scope="constructed current image laws; fixed other facts; not recorded object trajectories"),
        S=dict(mode="joint_object_scene_v1", trained_hidden=hidden,
            reflection_identity_fp64_error=float(reflection_identity_error),
            expectation_max_abs=float((pooled_a-pooled_b).abs().max()),
            relation_max_abs=float((va-vb).abs().max()), relation_rms=float((va-vb).square().mean().sqrt()),
            exact_repeat=True, signed_law_directional_vjp=float(derivative)),
        positive_support_control=dict(min_probability=float(soft_a.min()),
            tv=float((soft_a-soft_b).abs().sum()/2),
            relation_max_abs=float((sa-sb).abs().max()),
            relation_rms=float((sa-sb).square().mean().sqrt())),
        centered_offset_audit=dict(zero_init_exact=True, inherited_parameter_vjp_max_error=inherited_vjp_error,
            contrast_new_offset_vjp_l2=float(offset_pull.norm()), added_parameters=hidden,
            zero_coordinate_exact_zero=True, fixed_offset_controls=offset_rows,
            trained=False, production_admitted=False),
        existing_contextual_S_audit=dict(relation_max_abs=float((qa-qb).abs().max()),
            relation_rms=float((qa-qb).square().mean().sqrt()), new_context_trained=False,
            added_parameters=hidden*4*hidden, production_admitted=False),
        W_P2=dict(world_belief_fields=[f.name for f in fields(ObjectWorldBelief)],
            future_fields=[f.name for f in fields(FutureObjectDynamics)],
            actual_p2_context_max_abs=float((context_a-context_b).abs().max()),
            centroid_score_max_abs=float((legacy_a-legacy_b).abs().max()),
            direct_full_law_score_max_abs=float((integral_a-integral_b).abs().max()),
            extra_current_law_bytes_bs8=8*4*2*16*16*4,
            scope="geometry route only; implicit content and other policy paths not held equivalent by this result"),
        checkpoint_weights_unchanged=True, optimizer_updates=0,
        natural_behavior_improvement_proven=False,
        source_files={inspect.getfile(t): sha(inspect.getfile(t)) for t in
            [ObjectFactSet, JointTaskRelationEncoder, ObjectFutureEffectReader, ViewConditionedTransport]},
    )
    (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ["complete", "supported_laws", "S", "centered_offset_audit", "existing_contextual_S_audit", "W_P2"]}, indent=2))


if __name__ == "__main__":
    main()
