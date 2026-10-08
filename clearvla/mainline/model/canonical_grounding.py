"""Candidate canonical image ownership before observed-value mixing.

This is a new discrete model J[k,u]=m[u]q[k|u], not a refactor claiming
equivalence to local-mixture assignment. The old local chart remains available
to P1; its obsolete global-K marginals are never manufactured or reused.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from clearvla.vision.canonical_transport import rasterize_source, rasterize_value
from clearvla.vision.entity_chart import CanonicalImageReadSource, current_image_grid
from clearvla.vision.entity_motion import CURRENT_ENTITY_MOTION, current_entity_motion

from .routing import smooth_rms_contract
from .types import ObjectFactSet


class SharedIdentityViewDecoder(nn.Module):
    """Shared K identity plus camera role; no free [K,C] prototype or pixel input."""

    def __init__(self, hidden, content_dim, camera_names):
        super().__init__()
        self.identity = nn.Linear(hidden, hidden, bias=False)
        self.view = nn.Linear(len(camera_names), hidden, bias=False)
        self.output = nn.Linear(hidden, content_dim, bias=False)
        self.background = nn.Linear(len(camera_names), content_dim, bias=False)
        names = sorted(camera_names)
        self.register_buffer(
            "role", torch.eye(len(names))[[names.index(n) for n in camera_names]], persistent=False
        )

    def forward(self, identity):
        role = self.role.to(identity)
        value = self.output(
            F.silu(self.identity(identity)[:, :, None] + self.view(role)[None, None])
        )
        return value, self.background(role)


def encode_owners(module, candidates, mass, legal):
    """One shared slot transition and one K+null competition on canonical cells."""
    b, n, h = candidates.shape
    prior = mass.float()[..., None]
    log_prior = torch.where(prior > 0, prior, 1.0).log()
    valid = legal.float()[..., None]
    slots = module.slot_seed.to(candidates).expand(b, -1, -1)
    for _ in range(module.iterations):
        _, _, _, read, _, _ = module._competition(slots, candidates, valid, prior, log_prior, legal)
        update = read.to(candidates) @ candidates
        update, _ = smooth_rms_contract(update, module.maximum_update_rms)
        nxt = (
            module.gru(update.reshape(-1, h).float(), slots.reshape(-1, h).float())
            .reshape(b, module.objects, h)
            .to(slots)
        )
        ffn, _ = smooth_rms_contract(module.update_ffn(nxt), module.maximum_update_rms)
        with torch.autocast(device_type=slots.device.type, enabled=False):
            identity = F.normalize(
                module.slot_seed.float() - module.slot_seed.float().mean(1, keepdim=True),
                dim=-1,
                eps=1e-6,
            )
            scale = (
                torch.linalg.vector_norm((nxt + ffn).float(), dim=-1, keepdim=True)
                .detach()
                .clamp_min(1e-6)
            )
        slots = nxt + ffn + (0.5 * scale * identity).to(nxt)
    _, _, _, _, parent_log, _ = module._competition(
        slots, candidates, valid, prior, log_prior, legal
    )
    pair = torch.cat(
        (
            slots[:, None].expand(-1, n, -1, -1),
            candidates[:, :, None].expand(-1, -1, module.objects, -1),
        ),
        -1,
    )
    residual = 0.5 * torch.tanh(module.g3_residual(pair).squeeze(-1).float())
    with torch.autocast(device_type=slots.device.type, enabled=False):
        real_log = parent_log[..., : module.objects]
        # Quarantine an empty real event BEFORE logsumexp. Masking its -inf
        # result later still creates 0*NaN in LogsumexpBackward.
        total = torch.logsumexp(torch.where(legal[..., None], real_log, 0.0), -1, keepdim=True)
        safe_total = torch.where(legal[..., None], total, 0.0)
        conditional = torch.log_softmax(
            torch.where(legal[..., None], real_log - safe_total, 0.0) + residual, -1
        )
        corrected = torch.cat((conditional + safe_total, parent_log[..., module.objects :]), -1)
        corrected = torch.where(
            legal[..., None],
            corrected,
            torch.cat((torch.full_like(real_log, -torch.inf), torch.zeros_like(total)), -1),
        )
    return slots, corrected


def canonical_grounding(module, local, chart, history_context, *, collect_diagnostics):
    from .grounding import _coordinate_basis, _finite_log_measure

    full = local.current_observed_content
    if full is None or full.ndim != 5 or chart.current_image_support is None:
        raise ValueError(
            "canonical identity needs actual full-current observed values and producer support"
        )
    b, c, h, w, d = full.shape
    u = h * w
    k = module.objects
    if c != len(module.camera_names) or d != module.content_dim:
        raise ValueError("canonical observed chart differs from decoder camera/content contract")
    observed = (
        F.interpolate(
            chart.cell_observed[..., 0].float().reshape(b * c, 1, *chart.dino_content.shape[2:4]),
            size=(h, w),
            mode="nearest",
        )
        .reshape(b, c, u)
        .bool()
    )
    full = torch.where(observed.reshape(b, c, h, w, 1), full, 0.0)
    # Preserve every local source before rasterization. Typed/context/history
    # fields are weighted by their actual producer priors, not unweighted means.
    conditional, _ = rasterize_source(
        chart.current_image_support,
        torch.ones_like(chart.candidate_owner_prior),
        chart.candidate_validity,
        rows=h,
        columns=w,
    )
    source = conditional * chart.candidate_owner_prior.reshape(b, c, -1, 1).float()
    mass = source.sum(-2)
    legal = (mass > 0) & observed
    mass = torch.where(legal, mass, 0.0)
    fields = {}
    for name in ("semantic", "appearance", "geometry"):
        typed = (
            conditional
            * getattr(chart, "candidate_" + name + "_prior").reshape(b, c, -1, 1).float()
        )
        fields[name] = rasterize_value(
            typed, typed.sum(-2), getattr(chart, "candidate_" + name), chart.candidate_validity
        )
    grid = current_image_grid(h, w, device=full.device).reshape(u, 2)
    value = module.content_key(full.reshape(b, c, u, d))
    value = value + (
        module.semantic_key(fields["semantic"].to(value))
        + module.appearance_key(fields["appearance"].to(value))
        + module.geometry_key(fields["geometry"].to(value))
    ) / math.sqrt(3.0)
    value = value + module.coordinate_key(_coordinate_basis(grid.to(value), 16))[None, None]
    if module.context_key is not None:
        if chart.candidate_context is None:
            raise ValueError("canonical source lost completed G context")
        context = rasterize_value(source, mass, chart.candidate_context, chart.candidate_validity)
        value = value + module.context_key(context.to(value))
    if module.history_key is not None:
        if history_context is None:
            raise ValueError("canonical source lost causal history")
        history = rasterize_value(source, mass, history_context, chart.candidate_validity)
        value = value + module.history_key(history.to(value))
    candidates = torch.where(legal[..., None], module.candidate_norm(value), 0.0).reshape(
        b, c * u, -1
    )
    identity, owner_log = encode_owners(
        module, candidates, mass.reshape(b, -1), legal.reshape(b, -1)
    )
    owner_log = owner_log.reshape(b, c, h, w, k + 1).permute(0, 4, 1, 2, 3)
    log_mass = torch.where(mass > 0, mass, 1.0).log().reshape(b, 1, c, h, w)
    supported = legal.reshape(b, 1, c, h, w).expand(-1, k, -1, -1, -1)
    joint_log = torch.where(supported, owner_log[:, :k] + log_mass, 0.0)
    src = CanonicalImageReadSource(joint_log, supported, chart.current_image_support)
    native = src.on_image(rows=h, columns=w)
    read, _ = native.normalized((2, 3, 4))
    view_read, _ = native.normalized((-2, -1))
    read = read.flatten(-2)
    view_read = view_read.flatten(-2)

    def aggregate(field):
        with torch.autocast(device_type=field.device.type, enabled=False):
            return torch.einsum("bkcu,bcud->bkd", read, field.float())

    def by_view(field):
        with torch.autocast(device_type=field.device.type, enabled=False):
            return torch.einsum("bkcu,bcud->bkcd", view_read, field.float())

    content = aggregate(full.reshape(b, c, u, d))
    views = {name: by_view(field) for name, field in fields.items()}
    camera_content = by_view(full.reshape(b, c, u, d))
    # This is raw allocated source mass; camera validity remains independently
    # Boolean physical support, including cameras with very small allocation.
    view_mass = torch.where(supported, joint_log.exp(), 0.0).sum((-2, -1))
    has_view = supported.any((-2, -1), keepdim=True)
    safe_log = torch.where(supported, joint_log, -torch.inf)
    view_log_mass = torch.logsumexp(torch.where(has_view, safe_log, 0.0), (-2, -1))
    view_log_mass = torch.where(has_view[..., 0, 0], view_log_mass, -torch.inf)
    camera_valid = supported.flatten(-2).any(-1)[..., None].float()
    valid = camera_valid.any(2).float()
    image = src.on_image(rows=chart.dino_content.shape[2], columns=chart.dino_content.shape[3])
    chart_read, _ = image.normalized((2, 3, 4))
    safe_observed = torch.where(chart.cell_observed.bool(), chart.dino_content, 0.0)
    observed_content = torch.einsum("bkcyx,bcyxd->bkd", chart_read, safe_observed.float())
    coordinates = native.camera_centers().to(value.dtype)
    if module.entity_motion_mode == CURRENT_ENTITY_MOTION:
        transport = current_entity_motion(image, local.observed_history).to(value.dtype)
    else:
        motion = rasterize_value(
            source, mass, chart.candidate_transport_prior, chart.candidate_validity
        )
        transport = by_view(motion)
    support_field = rasterize_value(
        source, mass, chart.candidate_support[..., None], chart.candidate_validity
    )
    camera_support = by_view(support_field)
    support = aggregate(support_field)
    conditional_owner, _ = image.normalized((1,))
    decoded, background = module.canonical_decoder(identity)
    rows, columns = chart.dino_content.shape[2:4]
    pos = module.decode_position(
        _coordinate_basis(current_image_grid(rows, columns, device=full.device).to(value), 16)
    )
    reconstructed = torch.einsum("bkcyx,bkcd->bcyxd", conditional_owner.to(decoded), decoded)
    reconstructed = reconstructed + conditional_owner.sum(1)[..., None].to(decoded) * (
        pos[None, None] + background[None, :, None, None]
    )
    mask = chart.cell_observed.bool()
    reconstructed = torch.where(mask, reconstructed, 0.0).to(chart.dino_content)
    target = torch.where(mask, chart.dino_content.detach(), 0.0)
    error = (reconstructed.float() - target.float()).square().mean(
        -1, keepdim=True
    ).sum() / mask.sum().clamp_min(1)
    q = owner_log[:, :k].exp().flatten(-2)
    null = owner_log[:, k:].exp().flatten(-2)
    confidence = q / (q + null).clamp_min(torch.finfo(q.dtype).tiny)
    existence = (read * confidence).sum((2, 3))[..., None]
    facts = ObjectFactSet(
        dense_chart=chart,
        content=content,
        semantic=aggregate(fields["semantic"]),
        appearance=aggregate(fields["appearance"]),
        geometry=aggregate(fields["geometry"]),
        identity_state=identity,
        image_ownership=owner_log.exp(),
        observed_content=observed_content,
        view_mass=view_mass,
        view_log_mass=view_log_mass,
        camera_content=camera_content,
        camera_semantic=views["semantic"],
        camera_appearance=views["appearance"],
        camera_geometry=views["geometry"],
        camera_coordinates=coordinates,
        camera_transport_prior=transport,
        camera_support=camera_support,
        camera_validity=camera_valid,
        log_camera_validity=_finite_log_measure(camera_valid),
        support=support,
        existence=existence,
        validity=valid,
        log_validity=_finite_log_measure(valid),
        object_to_chart=chart_read,
        candidate_assignment=None,
        semantic_candidate_assignment=None,
        appearance_candidate_assignment=None,
        geometry_candidate_assignment=None,
        null_assignment=None,
        reconstructed_dino=reconstructed,
        reconstruction_error=error,
        latest_flow_steps=local.latest_flow_steps,
        object_chart_mode=module.entity_chart_mode,
        current_image_measure=image,
        current_image_source=src,
    )
    facts.validate()
    if not collect_diagnostics:
        return facts, {}
    total = mass.sum().clamp_min(1.0)
    conserved = ((q.sum(1) + null[:, 0]) - 1) * legal
    metrics = dict(
        object_grounding_reconstruction_mse=error.detach(),
        object_grounding_dense_objective_count=error.new_ones(()),
        object_grounding_validity_mean=valid.mean(),
        object_grounding_existence_mean=existence.detach().mean(),
        object_grounding_null_mass=(null[:, 0] * mass).sum().detach() / total.detach(),
        object_grounding_mass_conservation_error=conserved.detach().abs().max(),
        object_grounding_slot_pair_cosine=module._pair_cosine(identity),
        object_grounding_object_content_pair_cosine=module._pair_cosine(content),
    )
    return facts, metrics
