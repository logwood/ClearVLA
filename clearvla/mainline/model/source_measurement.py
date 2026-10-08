"""Observed current-to-successor differences before any learned object value.

The shared G law chooses the source; feature correspondence transports each
source cell separately. Unknown correspondence contributes no invented change
and is retained in the separate null/uncertainty ledger.
"""

import math

import torch

from clearvla.vision.entity_chart import current_image_grid
from clearvla.vision.observed_correspondence import observed_feature_correspondence


def source_consistent_measurement(
    module, facts, observations, offsets, observed, target_cell_observed=None
):
    from .observation_association import ObjectObservationMeasurement

    b, f, c, h, w, d = observations.shape
    source = facts.current_image_source
    if source is None:
        raise ValueError("source measurement requires an authoritative current image source")
    current = facts.dense_chart.dino_content.detach().float()
    if current.shape != (b, c, h, w, d):
        raise ValueError("source and measured observations must use the same declared chart")
    source_mask = facts.dense_chart.cell_observed[..., 0]
    current = torch.where(source_mask[..., None], current, 0.0).flatten(2, 3)
    observed = (
        torch.ones(b, f, dtype=torch.bool, device=current.device) if observed is None else observed
    )
    # The successor encoder owns this image. A mask on the CURRENT image
    # cannot rule out observing a moved object at a different successor cell.
    # Optional target masks have already been validated by the public entry.
    target_mask = observed[:, :, None, None, None].expand(b, f, c, h, w)
    if target_cell_observed is not None:
        target_mask = target_mask & target_cell_observed
    target = torch.where(target_mask[..., None], observations.float(), 0.0).flatten(3, 4)
    kernel = observed_feature_correspondence(
        current[:, None].expand(b, f, c, h * w, d),
        target,
        source_mask.flatten(2)[:, None].expand(b, f, c, h * w),
        target_mask.flatten(3),
    )
    image = source.on_image(rows=h, columns=w).restrict(source_mask[:, None])
    joint, _ = image.normalized((2, 3, 4))
    conditional, _ = image.normalized((-2, -1))
    joint, conditional = joint.flatten(-2), conditional.flatten(-2)
    real = kernel[..., :-1]
    real_sum = real.sum(-1)
    posterior = torch.einsum("bkcn,bfcnm->bfkcm", joint, real)
    null = torch.einsum("bkcn,bfcn->bfk", joint, kernel[..., -1])[..., None]
    has_source = joint.flatten(2).sum(-1) > 0
    null = torch.where(has_source[:, None, :, None], null, 1.0)
    reference = torch.einsum("bkcn,bcnd->bkd", joint, current)
    per_source_delta = real @ target - real_sum[..., None] * current[:, None]
    delta = torch.einsum("bkcn,bfcnd->bfkd", joint, per_source_delta)
    xy = current_image_grid(h, w, device=current.device).reshape(h * w, 2)
    first = real @ xy
    pair_delta = first - real_sum[..., None] * xy
    transport = torch.einsum("bkcn,bfcnd->bfkcd", conditional, pair_delta)
    xx = real @ xy[:, 0].square() - 2 * xy[:, 0] * first[..., 0] + real_sum * xy[:, 0].square()
    yy = real @ xy[:, 1].square() - 2 * xy[:, 1] * first[..., 1] + real_sum * xy[:, 1].square()
    xy2 = (
        real @ (xy[:, 0] * xy[:, 1])
        - xy[:, 0] * first[..., 1]
        - xy[:, 1] * first[..., 0]
        + real_sum * (xy[:, 0] * xy[:, 1])
    )
    second = torch.einsum("bkcn,bfcnd->bfkcd", conditional, torch.stack((xx, xy2, yy), -1))
    variance_x = (second[..., 0] - transport[..., 0].square()).clamp_min(0)
    variance_y = (second[..., 2] - transport[..., 1].square()).clamp_min(0)
    limit = (variance_x * variance_y).sqrt()
    covariance_xy = (second[..., 1] - transport[..., 0] * transport[..., 1]).clamp(-limit, limit)
    covariance = torch.stack((variance_x, covariance_xy, variance_y), -1)
    law = torch.cat((posterior.flatten(3), null), -1)
    safe = torch.where(law > 0, law, 1.0)
    entropy = -(law * safe.log()).sum(-1, keepdim=True) / math.log(c * h * w + 1)
    object_validity = facts.validity.detach().float()
    camera_validity = facts.camera_validity.detach().float()
    legal = (camera_validity > 0) & (object_validity[:, :, None] > 0)
    # These metric fields describe the actual transported distribution. There
    # is no fabricated random-projection semantic/appearance score.
    diagnostic = posterior.reshape(b, f, facts.objects, c, h, w)
    return ObjectObservationMeasurement(
        successor_per_support=reference[:, None] + delta,
        transport_per_support=transport,
        covariance_per_support=covariance,
        candidate_posterior=diagnostic,
        null_probability=null,
        current_reference=reference[:, None],
        object_validity=object_validity,
        camera_validity=camera_validity,
        geometry_coordinate=image.camera_centers(),
        camera_legal=legal,
        offsets=offsets,
        association_real_mass_per_support=1 - null,
        uncertainty_per_support=null + (1 - null) * entropy,
        reliability_per_support=(1 - null) * (1 - entropy),
        association_confidence=1 - entropy,
        semantic=diagnostic,
        appearance=diagnostic,
        geometry=diagnostic,
        fraction=module._flow_horizon_scale(offsets),
    )
