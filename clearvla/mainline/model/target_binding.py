"""One soft operated-object law, with object-internal evidence reads.

This is a per-observation contract, not a persistent physical object ID or a
stop controller. Object mass is applied AFTER object-internal attention; null
mass is never divided away by attention or by a normalization of selected keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing import smooth_rms_contract

LOCAL_TARGET_READERS = "reader_local_v1"
SHARED_TARGET_BINDING = "shared_operation_v1"


def target_binding_metadata() -> dict[str, object]:
    return {
        "schema": "shared-operated-object-v1",
        "identity": "one-soft-K-plus-null-law-per-online-observation",
        "internal_read": "attribute-and-declared-view-tokens-within-each-K",
        "mass_application": "after-internal-read-without-real-mass-renormalization",
        "null": "no-target-evidence-not-environment-stop",
        "lifetime": "immutable-within-proposal-refined-and-ODE",
        "persistent_physical_ids": False,
    }


def masked_probability(logits: Tensor, valid: Tensor) -> Tensor:
    """Last-axis softmax with exact empty support, including finite backward."""
    if logits.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("probability support must be boolean and match logits")
    safe = torch.where(valid, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("nonfinite score on supported evidence")
    masked = safe.masked_fill(~valid, -torch.inf)
    nonempty = valid.any(-1, keepdim=True)
    masked = torch.where(nonempty, masked, torch.zeros_like(masked))
    return torch.where(valid, masked.softmax(-1), torch.zeros_like(masked))


@dataclass(frozen=True)
class TargetBinding:
    """Normalized log probabilities [B,K+1], last entry is an explicit null."""

    log_probability: Tensor
    supported: Tensor  # bool [B,K], physical source support, not learned confidence

    @classmethod
    def from_logits(cls, logits: Tensor, null_logit: Tensor, supported: Tensor) -> TargetBinding:
        if logits.ndim != 2 or null_logit.shape != (logits.shape[0], 1):
            raise ValueError("target logits must be [B,K] and null [B,1]")
        if supported.shape != logits.shape or supported.dtype != torch.bool:
            raise ValueError("target support must be boolean [B,K]")
        if logits.device != supported.device or logits.device != null_logit.device:
            raise ValueError("target source device mismatch")
        safe = torch.where(supported, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
        if not bool(torch.isfinite(safe).all()) or not bool(torch.isfinite(null_logit).all()):
            raise ValueError("nonfinite supported target or null score")
        combined = torch.cat((safe.masked_fill(~supported, -torch.inf), null_logit.float()), -1)
        return cls(combined.log_softmax(-1), supported)

    @property
    def mass(self) -> Tensor:
        return self.log_probability[..., :-1].exp()

    @property
    def null_mass(self) -> Tensor:
        return self.log_probability[..., -1:].exp()

    def validate(self, *, batch: int, objects: int, device: torch.device) -> None:
        if self.log_probability.shape != (batch, objects + 1) or self.supported.shape != (
            batch,
            objects,
        ):
            raise ValueError("target binding lost B/K/null axes")
        if self.log_probability.dtype != torch.float32 or self.supported.dtype != torch.bool:
            raise TypeError("target law must be FP32 with boolean physical support")
        if self.log_probability.device != device or self.supported.device != device:
            raise ValueError("target binding and evidence must share a device")
        # Exact zero mass (-inf log mass) is valid even on supported objects.
        if bool(torch.isnan(self.log_probability).any()) or bool(
            torch.isposinf(self.log_probability).any()
        ):
            raise ValueError("target law has nonfinite positive/NaN log mass")
        if not bool(torch.isneginf(self.log_probability[:, :-1][~self.supported]).all()):
            raise ValueError("target law assigned mass outside physical support")
        if not torch.allclose(
            self.log_probability.logsumexp(-1), torch.zeros(batch, device=device), atol=2e-5, rtol=0
        ):
            raise ValueError("target law is not normalized including null")

    def permute(self, index: Tensor) -> TargetBinding:
        return TargetBinding(
            torch.cat((self.log_probability[:, :-1][:, index], self.log_probability[:, -1:]), -1),
            self.supported[:, index],
        )


@dataclass(frozen=True)
class TargetEvidence:
    """Task-independent current values, with distinct attribute/view tokens."""

    tokens: Tensor  # [B,K,L,H]; four attribute tokens then declared cameras
    valid: Tensor  # bool [B,K,L]

    def validate(self, *, hidden: int) -> None:
        if self.tokens.ndim != 4 or self.tokens.shape[-1] != hidden:
            raise ValueError("target evidence must be [B,K,L,H]")
        if self.tokens.shape[2] < 2:
            raise ValueError("target evidence must retain multiple internal sources")
        if self.valid.shape != self.tokens.shape[:-1] or self.valid.dtype != torch.bool:
            raise ValueError("target evidence lost per-source support")
        if self.valid.device != self.tokens.device:
            raise ValueError("target evidence support device mismatch")
        safe = torch.where(self.valid[..., None], self.tokens, torch.zeros_like(self.tokens))
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("nonfinite target evidence on observed support")

    def permute(self, index: Tensor) -> TargetEvidence:
        return TargetEvidence(self.tokens[:, index], self.valid[:, index])


class SharedTargetBinder(nn.Module):
    """Language/current-history query selects one distribution over G entities."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden, bias=False))
        self.key = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden, bias=False))
        self.null = nn.Linear(hidden, 1)
        self.scale = hidden**-0.5

    def forward(self, task_context: Tensor, objects: Tensor, supported: Tensor) -> TargetBinding:
        safe = torch.where(supported[..., None], objects, torch.zeros_like(objects))
        query = self.query(task_context)
        score = torch.einsum("bh,bkh->bk", query.float(), self.key(safe).float()) * self.scale
        return TargetBinding.from_logits(score, self.null(task_context), supported)


class BoundTargetRead(nn.Module):
    """Learn where/what to read INSIDE each object, never another K softmax.

    Unlike single-key attention this retains genuine Q/K dependence. Its value
    path has no query-only FFN/bias that could survive an empty or null read.
    """

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("target read heads must divide hidden")
        self.hidden, self.heads, self.width = hidden, heads, hidden // heads
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

    def forward(self, query: Tensor, evidence: TargetEvidence, binding: TargetBinding) -> Tensor:
        evidence.validate(hidden=self.hidden)
        batch, objects, sources, _ = evidence.tokens.shape
        binding.validate(batch=batch, objects=objects, device=query.device)
        if (
            query.shape[0] != batch
            or query.shape[-1] != self.hidden
            or query.device != evidence.tokens.device
        ):
            raise ValueError("target query and current facts do not align")
        if not torch.equal(binding.supported, evidence.valid.any(-1)):
            raise ValueError("binding and current evidence disagree about object support")
        safe = torch.where(
            evidence.valid[..., None], evidence.tokens, torch.zeros_like(evidence.tokens)
        )
        q = self.query(self.query_norm(query)).reshape(batch, -1, self.heads, self.width)
        k = self.key(self.key_norm(safe)).reshape(batch, objects, sources, self.heads, self.width)
        v = self.value(safe).reshape(batch, objects, sources, self.heads, self.width)
        logits = torch.einsum("bpad,bklad->bpkal", q.float(), k.float()) * self.width**-0.5
        support = evidence.valid[:, None, :, None, :].expand_as(logits)
        probability = masked_probability(logits, support)
        per_object = torch.einsum("bpkal,bklad->bpkad", probability.to(v.dtype), v)
        per_object = self.output(per_object.flatten(-2))
        # Keep FP32 multiplication even in AMP; no normalization of real mass.
        result = (per_object.float() * binding.mass[:, None, :, None]).sum(2)
        result, _ = smooth_rms_contract(result, 0.35)
        return result.to(query.dtype).reshape_as(query)
