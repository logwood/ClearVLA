"""Geometry/temporal identity pressure and genuinely held-source prediction.

All auxiliary inputs stay on the training label plane. A source-only encoding
is rebuilt from that source's raw observed descriptors before any G1/G2 fusion;
target-view/context/history values cannot leak through the full online slots.
"""

import math

import torch
import torch.nn.functional as F

from ..identity_pairs import validate_identity_clock
from ..model.canonical_grounding import encode_owners
from ..model.grounding import _coordinate_basis


def sample(field, xy):
    """B,D,Y,X -> B,N,D, same endpoint chart as actual online image reads."""
    return (
        F.grid_sample(field.float(), xy.float()[:, None], align_corners=True, padding_mode="border")
        .squeeze(2)
        .transpose(1, 2)
    )


def js_divergence(left, right):
    with torch.autocast(device_type=left.device.type, enabled=False):
        left = left.float()
        right = right.float()
        middle = (left + right) / 2
        a = left * (left.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())
        b = right * (right.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())
        return 0.5 * (a + b).sum(-1)


def conditional_real_law(log_measure, supported, *, dim=1):
    """Identity given a real entity; abstention is not an identity label.

    Normalize BEFORE spatial interpolation so a varying real/null fraction
    cannot reweight the correspondence stencil. Producer support alone owns
    missing cells. The old joint K+null objective remains a separate version.
    """
    with torch.autocast(device_type=log_measure.device.type, enabled=False):
        legal = supported.any(dim, keepdim=True)
        safe = torch.where(supported, log_measure.float(), -torch.inf)
        safe = torch.where(legal, safe, 0.0)
        conditional = torch.softmax(safe, dim=dim)
        return torch.where(legal, conditional, 0.0), legal


def identity_terms(model, online, facts, labels):
    module = model.grounding.grounder
    if module.canonical_decoder is None or facts.image_ownership is None:
        raise ValueError("identity training requires the integrated canonical graph")
    labels.validate(batch=online.batch, device=online.device)
    validate_identity_clock(labels, online)
    raw = online.observation.dino_history[:, -2:].detach()
    b, t, c, n, d = raw.shape
    side = math.isqrt(n)
    k = module.objects
    if side * side != n or t != 2 or c != len(labels.camera_names):
        raise ValueError("identity sources lost their declared camera/time chart")
    if tuple(labels.camera_names) != tuple(module.camera_names):
        raise ValueError("identity labels and online camera order differ")
    with torch.autocast(device_type=raw.device.type, enabled=False):
        observed = model.observation.compiler.encoder.teacher_norm(raw.float())
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, side, device=raw.device),
                torch.linspace(-1, 1, side, device=raw.device),
                indexing="ij",
            ),
            -1,
        )
        .flip(-1)
        .reshape(n, 2)
    )
    # B*T*C independent encodings: every source excludes all target values,
    # producer priors, context fusion and history learned from other sources.
    source = observed.reshape(b * t * c, n, d)
    key = (
        module.content_key(source)
        + module.coordinate_key(_coordinate_basis(grid.to(source), 16))[None]
    )
    key = module.candidate_norm(key)
    mass = torch.ones(b * t * c, n, device=source.device)
    state, log_owner = encode_owners(module, key, mass, mass.bool())
    owner = log_owner.exp().reshape(b, t, c, side, side, k + 1).permute(0, 1, 2, 5, 3, 4)
    conditional = (
        torch.softmax(log_owner[..., :k].float(), -1)
        .reshape(b, t, c, side, side, k)
        .permute(0, 1, 2, 5, 3, 4)
    )
    state = state.reshape(b, t, c, k, module.hidden)
    target = observed[:, -1].reshape(b, c, side, side, d).permute(0, 1, 4, 2, 3)
    full = facts.image_ownership
    if full.shape != (b, k + 1, c, side, side):
        raise ValueError("online ownership differs from admitted identity chart")
    conditional_mode = model.config.top.identity_supervision_mode == "rgbd_temporal_conditional_v2"
    if conditional_mode:
        image_source = facts.current_image_source
        full_real, full_supported = conditional_real_law(
            image_source.log_measure, image_source.supported
        )
    pair_losses = []
    prediction_losses = []
    counts = []
    without = []
    identity_counts = []
    null_source = []
    null_target = []
    pairs = labels.pair_groups()
    counts_by_kind = {kind: [] for kind in ("cross", "temporal")}
    for group in pairs:
        kind, time = group.kind, group.source_time
        camera, destination = group.source_camera, group.target_camera
        if group.valid.shape[-1] == 0:
            # Empty pair groups contribute no fake observations or dilution.
            continue
        valid = group.valid
        xy_source = torch.where(valid[..., None], group.source, 0.0)
        xy_target = torch.where(valid[..., None], group.target, 0.0)
        source_law = sample(owner[:, time, camera], xy_source)
        target_law = sample(full[:, :, destination], xy_target)
        count = valid.float().sum()
        counts.append(count)
        counts_by_kind[kind].append(count)
        null_source.append(torch.where(valid, source_law[..., -1], 0.0).sum() / count.clamp_min(1))
        null_target.append(torch.where(valid, target_law[..., -1], 0.0).sum() / count.clamp_min(1))
        identity_valid = valid
        if conditional_mode:
            source_law = sample(conditional[:, time, camera], xy_source)
            target_law = sample(full_real[:, :, destination], xy_target)
            identity_valid = valid & (
                sample(full_supported[:, :, destination].float(), xy_target)[..., 0] > 1 - 1e-6
            )
        # Also align actual online ownership across cameras. Slot index
        # equality alone never supplies a positive without a sensor match.
        if kind == "cross":
            current_law = sample(
                full_real[:, :, camera] if conditional_mode else full[:, :, camera], xy_source
            )
            if conditional_mode:
                identity_valid = identity_valid & (
                    sample(full_supported[:, :, camera].float(), xy_source)[..., 0] > 1 - 1e-6
                )
            divergence = (
                js_divergence(source_law, target_law) + js_divergence(current_law, target_law)
            ) / 2
        else:
            divergence = js_divergence(source_law, target_law)
        identity_count = identity_valid.float().sum()
        identity_counts.append(identity_count)
        pair_losses.append(
            torch.where(identity_valid, divergence, 0.0).sum() / identity_count.clamp_min(1)
        )
        prototype, background = module.canonical_decoder(state[:, time, camera])
        q = sample(conditional[:, time, camera], xy_source)
        predicted = torch.einsum("bnk,bkd->bnd", q.to(prototype), prototype[:, :, destination])
        position = module.decode_position(_coordinate_basis(xy_target.to(prototype), 16))
        predicted = predicted + position + background[destination][None, None]
        truth = sample(target[:, destination], xy_target).detach()
        error = (predicted.float() - truth).square().mean(-1)
        prediction_losses.append(torch.where(valid, error, 0.0).sum() / count.clamp_min(1))
        with torch.no_grad():
            zero_value, zero_background = module.canonical_decoder(
                torch.zeros_like(state[:, time, camera])
            )
            zero_prediction = (
                torch.einsum("bnk,bkd->bnd", q.to(zero_value), zero_value[:, :, destination])
                + position
                + zero_background[destination][None, None]
            )
            zero_error = (zero_prediction.float() - truth).square().mean(-1)
            without.append(torch.where(valid, zero_error, 0.0).sum() / count.clamp_min(1))
    zero = state.sum() * 0

    def available_group_mean(items, support_counts):
        """Equal weight per observed group, not per padded/unknown group.

        Producer masks alone choose the denominator. We deliberately do not
        use learned null/allocation/confidence or change the positive-pair
        budget; source-prediction and correspondence have different support.
        """
        if not items:
            return zero
        active = torch.stack(support_counts) > 0
        values = torch.stack(items)
        numerator = torch.where(active, values, 0.0).sum()
        return numerator / active.to(values.dtype).sum().clamp_min(1)

    def total(items):
        return torch.stack(items).sum() if items else zero.detach()

    result = dict(
        identity_correspondence=available_group_mean(pair_losses, identity_counts),
        identity_source_prediction=available_group_mean(prediction_losses, counts),
        identity_cross_admitted_pairs=total(counts_by_kind["cross"]),
        identity_temporal_admitted_pairs=total(counts_by_kind["temporal"]),
        identity_source_removed_prediction_mse=available_group_mean(without, counts),
    )
    if conditional_mode:
        result.update(
            identity_conditional_supported_pairs=total(identity_counts).detach(),
            identity_conditional_support_fraction=(
                total(identity_counts) / total(counts).clamp_min(1)
            ).detach(),
            identity_source_null_mass=available_group_mean(null_source, counts).detach(),
            identity_target_null_mass=available_group_mean(null_target, counts).detach(),
        )
    return result
