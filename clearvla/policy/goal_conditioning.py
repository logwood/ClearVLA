"""Compact precomputed-T5 goal conditioning for the policy canvas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class GoalResamplerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(hidden * expansion)),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden),
        )

    def forward(self, query: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=~mask,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(query)


class GoalTokenResampler(nn.Module):
    """Resample frozen language embeddings into a few trainable goal tokens."""

    def __init__(
        self,
        *,
        language_dim: int,
        hidden: int,
        goal_tokens: int,
        heads: int,
        depth: int,
        expansion: float,
    ) -> None:
        super().__init__()
        if min(language_dim, hidden, goal_tokens, heads, depth) <= 0:
            raise ValueError("goal resampler dimensions must be positive")
        self.language_dim = int(language_dim)
        self.hidden = int(hidden)
        self.goal_tokens = int(goal_tokens)
        self.input = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, hidden),
        )
        self.query = nn.Parameter(torch.randn(1, goal_tokens, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [
                GoalResamplerBlock(hidden, heads, expansion)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, language_tokens: Tensor, language_mask: Tensor) -> Tensor:
        if language_tokens.ndim != 3 or int(language_tokens.shape[-1]) != self.language_dim:
            raise ValueError(
                f"language_tokens must be [B,L,{self.language_dim}], got "
                f"{tuple(language_tokens.shape)}"
            )
        if tuple(language_mask.shape) != tuple(language_tokens.shape[:2]):
            raise ValueError("language_mask must align with language_tokens as [B,L]")
        mask = language_mask.to(device=language_tokens.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one valid language token")
        memory = self.input(language_tokens)
        query = self.query.expand(language_tokens.shape[0], -1, -1).to(
            device=memory.device, dtype=memory.dtype
        )
        for block in self.blocks:
            query = block(query, memory, mask)
        return self.output_norm(query)


class StatelessPhaseAdapter(nn.Module):
    """Infer an ordered soft phase belief without recurrent deployment state.

    The returned context is selector-only: callers may add it to world or
    spatial-address queries, but it must not be registered as a global semantic
    value or a direct action writer.  Ordered sinusoidal phase bases keep the
    phase axis meaningful instead of turning it into another unconstrained
    bank of free value tokens.
    """

    def __init__(self, hidden: int, phase_count: int) -> None:
        super().__init__()
        if int(hidden) < 1 or int(phase_count) < 2:
            raise ValueError("stateless phase dimensions are invalid")
        self.hidden = int(hidden)
        self.phase_count = int(phase_count)
        self.condition = nn.Sequential(
            nn.LayerNorm(4 * self.hidden),
            nn.Linear(4 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.phase_count),
        )
        self.context_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.selector_condition_proj = nn.Sequential(
            nn.LayerNorm(2 * self.hidden),
            nn.Linear(2 * self.hidden, self.hidden, bias=False),
        )
        phase = torch.linspace(0.0, 1.0, self.phase_count)
        half = max(self.hidden // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(max(half - 1, 1))
        frequency = torch.exp(
            -torch.log(torch.tensor(10_000.0, dtype=torch.float32)) * exponent
        )
        angle = phase[:, None] * frequency[None] * (2.0 * torch.pi)
        basis = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if int(basis.shape[-1]) < self.hidden:
            basis = torch.nn.functional.pad(
                basis, (0, self.hidden - int(basis.shape[-1]))
            )
        self.register_buffer("ordered_phase_basis", basis[:, : self.hidden])

    @staticmethod
    def _summary(value: Tensor, *, batch: int, hidden: int, name: str) -> Tensor:
        if (
            value.ndim != 3
            or int(value.shape[0]) != int(batch)
            or int(value.shape[-1]) != int(hidden)
            or int(value.shape[1]) <= 0
        ):
            raise ValueError(f"{name} must be non-empty [B,N,{hidden}]")
        return value.mean(dim=1)

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        history_tokens: Tensor,
        state_tokens: Tensor,
        visual_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch = int(state_tokens.shape[0])
        summaries = (
            self._summary(
                goal_tokens, batch=batch, hidden=self.hidden, name="phase goal"
            ),
            self._summary(
                history_tokens,
                batch=batch,
                hidden=self.hidden,
                name="phase history",
            ),
            self._summary(
                state_tokens, batch=batch, hidden=self.hidden, name="phase state"
            ),
            self._summary(
                visual_tokens, batch=batch, hidden=self.hidden, name="phase visual"
            ),
        )
        logits = self.condition(torch.cat(summaries, dim=-1))
        belief = torch.softmax(logits.float(), dim=-1)
        basis = self.ordered_phase_basis.to(
            device=logits.device, dtype=logits.dtype
        )
        phase_context = belief.to(dtype=logits.dtype) @ basis
        phase_context = self.context_proj(phase_context)
        selector_condition = self.selector_condition_proj(
            torch.cat((summaries[0], summaries[1]), dim=-1)
        )
        entropy = -(
            belief.detach() * belief.detach().clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy = entropy / torch.log(
            belief.new_tensor(float(self.phase_count))
        )
        phase_index = torch.arange(
            self.phase_count, device=belief.device, dtype=belief.dtype
        )
        expectation = (belief * phase_index[None]).sum(dim=-1)
        metrics = {
            "flow_jepa_phase_entropy": entropy.mean(),
            "flow_jepa_phase_max": belief.detach().amax(dim=-1).mean(),
            "flow_jepa_phase_expected_index": expectation.detach().mean(),
            "flow_jepa_phase_expected_index_std": expectation.detach().std(
                unbiased=False
            ),
            "flow_jepa_phase_context_norm": (
                phase_context.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_condition_selector_context_norm": (
                selector_condition.detach().float().norm(dim=-1).mean()
            ),
        }
        for index in range(self.phase_count):
            metrics[f"flow_jepa_phase_mass_{index}"] = (
                belief.detach()[:, index].mean()
            )
        return phase_context, selector_condition, metrics


def load_precomputed_t5_condition(
    *,
    condition_path: Path,
    max_tokens: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Load one precomputed T5 condition without instantiating a text model.

    The canonical format is a tensor shaped ``[L,D]``.  ``[1,L,D]`` and dict
    wrappers using common embedding/mask names are also accepted.  A missing
    mask means every stored token is valid.
    """

    if max_tokens <= 0:
        raise ValueError("goal language max_tokens must be positive")
    path = Path(condition_path).expanduser().resolve()
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(f"T5 condition must be a .pt/.pth file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"T5 condition file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mask: Any = None
    if isinstance(payload, dict):
        tokens: Any = None
        for key in (
            "tokens",
            "embeddings",
            "embedding",
            "language_embedding",
            "last_hidden_state",
        ):
            if key in payload:
                tokens = payload[key]
                break
        for key in ("mask", "attention_mask", "language_mask"):
            if key in payload:
                mask = payload[key]
                break
        if tokens is None:
            mask_keys = {"mask", "attention_mask", "language_mask"}
            tensor_values = [
                value
                for key, value in payload.items()
                if key not in mask_keys and torch.is_tensor(value)
            ]
            if len(tensor_values) == 1:
                tokens = tensor_values[0]
            else:
                raise ValueError(
                    "T5 condition dict needs tokens/embeddings/embedding/"
                    "language_embedding/last_hidden_state"
                )
    else:
        tokens = payload
    raw_tokens = torch.as_tensor(tokens)
    original_shape = tuple(int(value) for value in raw_tokens.shape)
    original_dtype = str(raw_tokens.dtype).replace("torch.", "")
    tokens = raw_tokens.detach().to(device="cpu", dtype=torch.float32)
    if tokens.ndim == 2:
        tokens = tokens[None]
    if tokens.ndim != 3 or int(tokens.shape[0]) != 1:
        raise ValueError(
            f"T5 condition tokens must be [L,D] or [1,L,D], got {original_shape}"
        )
    if int(tokens.shape[1]) < 1 or int(tokens.shape[2]) < 1:
        raise ValueError("T5 condition must contain at least one finite token and feature")
    tokens = tokens[:, :max_tokens]
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("T5 condition contains NaN or infinity")
    if mask is None:
        mask_tensor = torch.ones(tokens.shape[:2], dtype=torch.bool)
    else:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor[None]
        mask_tensor = mask_tensor[:, : tokens.shape[1]]
    if tuple(mask_tensor.shape) != tuple(tokens.shape[:2]):
        raise ValueError("T5 condition mask must align with tokens as [1,L]")
    if not bool(mask_tensor.any()):
        raise ValueError("T5 condition mask must retain at least one valid token")
    metadata = {
        "source": "precomputed_t5_condition",
        "path": str(path),
        "original_shape": list(original_shape),
        "original_dtype": original_dtype,
        "effective_tokens": int(tokens.shape[1]),
    }
    return tokens.contiguous(), mask_tensor.contiguous(), metadata


__all__ = [
    "GoalTokenResampler",
    "StatelessPhaseAdapter",
    "load_precomputed_t5_condition",
]
