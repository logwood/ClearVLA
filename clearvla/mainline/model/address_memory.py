"""Dense address state across existing G iterations, local to one observation.

Only the real-K conditional law is updated. At a fixed competition input its
real total/null and producer support remain those of the existing competition.
Subsequent recurrent slots can of course change those totals indirectly.
"""
import torch
from torch import Tensor


def retain_address(competition, previous: Tensor, gain: Tensor, prior: Tensor,
                   log_prior: Tensor, legal: Tensor):
    from .grounding import _masked_log_softmax, _finite_log_measure
    _, _, null_mass, _, owner_log, _ = competition
    with torch.autocast(device_type=owner_log.device.type, enabled=False):
        real = owner_log[..., :-1]
        safe = torch.where(legal[..., None], real, 0.)
        old = torch.where(legal[..., None], previous.float(), 0.)
        # Remove only the log-law gauge on K; N/camera are never pooled.
        old = old - old.mean(-1, keepdim=True)
        # Subtract identical operations at gain=0 for bitwise legacy forward.
        # This keeps ordinary gradients to the old address and the new gain.
        delta = torch.log_softmax(safe + torch.tanh(gain.float()) * old, -1) - torch.log_softmax(safe, -1)
        updated_real = real + delta
        updated_log = torch.cat((updated_real, owner_log[..., -1:]), -1)
        owner = updated_log.exp()
        valid = legal.float()
        object_mass = owner[..., :-1] * valid[..., None] * prior
        read_measure = updated_real.transpose(1, 2) + log_prior[..., 0][:, None] + _finite_log_measure(valid)[:, None]
        support = legal[:, None].expand(-1, real.shape[-1], -1)
        read, read_log, _ = _masked_log_softmax(read_measure, support, dim=-1)
    return owner, object_mass, null_mass, read, updated_log, read_log


def competition_with_memory(module, slots, candidates, valid, prior, log_prior,
                            legal, previous, index):
    result = module._competition(slots, candidates, valid, prior, log_prior, legal)
    if previous is not None:
        result = retain_address(result, previous, module.address_memory_gain[index - 1],
                                prior, log_prior, legal)
    return result
