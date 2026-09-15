"""Shared policy contracts with no dependency on experiment facades."""

from torch import Tensor


LAYER_CONTRACT_KEYS = (
    "rollout_tokens",
    "trajectory_pooled",
    "rollout_effect_pred",
    "rollout_delta_pred",
    "policy_effect_tokens",
    "policy_effect_time_tokens",
    "unified_intervention_latent_pred",
    "neutral_latent_pred",
    "milestone_step_delta_pred",
)


def scaled_contract_view(x: Tensor, scale: float) -> Tensor:
    """Return x with gradients scaled by ``scale`` but values unchanged."""

    if scale >= 1.0:
        return x
    if scale <= 0.0:
        return x.detach()
    return x.detach() + float(scale) * (x - x.detach())
