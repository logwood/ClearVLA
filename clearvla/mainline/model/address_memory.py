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
                            legal, previous, index, *, diagnostics=None):
    result = module._competition(slots, candidates, valid, prior, log_prior, legal)
    if previous is not None:
        baseline = result
        result = retain_address(result, previous, module.address_memory_gain[index - 1],
                                prior, log_prior, legal)
        if diagnostics is not None:
            diagnostics.update(address_boundary_metrics(baseline, result, prior, legal, index))
    return result


@torch.no_grad()
def address_boundary_metrics(baseline, updated, prior, legal, index):
    """Immediate same-input routing change; not object identity or downstream gain.

    No extra competition/encoder call. Producer mass weights only legal cells.
    Empty support contributes zero and its mass is reported explicitly.
    """
    with torch.autocast(device_type=prior.device.type, enabled=False):
        weight = torch.where(legal, prior.detach().float()[..., 0], 0.)
        total = weight.sum()
        def conditional(value):
            logits = torch.where(legal[..., None], value[4].detach().float()[..., :-1], 0.)
            return torch.softmax(logits, -1)
        tv = .5 * (conditional(updated) - conditional(baseline)).abs().sum(-1)
        read_tv = .5 * (updated[3].detach().float() - baseline[3].detach().float()).abs().sum(-1)
        read_support = legal.any(-1, keepdim=True).expand_as(read_tv).float()
        prefix = f"object_grounding_address_step{index}_"
        return {
            prefix + "conditional_tv": (tv * weight).sum() / total.clamp_min(torch.finfo(weight.dtype).tiny),
            prefix + "read_tv": (read_tv * read_support).sum() / read_support.sum().clamp_min(1),
            prefix + "producer_mass": total.clone(),
            prefix + "null_change_max": (updated[0][..., -1].detach() - baseline[0][..., -1].detach()).abs().max(),
            prefix + "real_mass_change_max": (updated[0][..., :-1].detach().sum(-1) - baseline[0][..., :-1].detach().sum(-1)).abs().max(),
        }


@torch.no_grad()
def address_parameter_metrics(parameter):
    """Snapshot ordinary preclip gradients; disconnected is distinct from zero.

    Clone before optimizer.step so archived values cannot alias live parameters.
    """
    value = parameter.detach().float().clone()
    gradient = None if parameter.grad is None else parameter.grad.detach().float().clone()
    metrics = {
        "gradient_parameter_address_memory_connected": value.new_tensor(float(gradient is not None)),
    }
    for index in range(value.numel()):
        metrics[f"object_grounding_address_gain{index + 1}"] = value[index].clone()
        metrics[f"object_grounding_address_effective_gain{index + 1}"] = value[index].tanh()
        if gradient is not None:
            metrics[f"gradient_parameter_address_gain{index + 1}_raw"] = gradient[index].clone()
    if gradient is not None:
        metrics["gradient_parameter_address_memory_raw_l2"] = gradient.norm()
    return metrics
