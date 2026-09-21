"""Fixed-temperature, reparameterized residual SAC; no flow-policy gradient."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SACConfig
from .networks import ResidualActor, TwinQ
from .replay import ReplayBuffer


def bellman_target(
    reward: Tensor,
    terminated: Tensor,
    next_q: Tensor,
    next_log_prob: Tensor,
    *,
    gamma: float,
    alpha: float,
) -> Tensor:
    # Timeout is NOT terminal. next_features must be the final observation,
    # not an autoreset state. True termination removes both Q and entropy.
    return reward + gamma * (1.0 - terminated) * (next_q - alpha * next_log_prob)


class ResidualSAC:
    def __init__(
        self, feature_dim: int, config: SACConfig, device: torch.device = torch.device("cpu")
    ) -> None:
        config.validate()
        self.config, self.device, self.feature_dim = config, device, feature_dim
        # Construct on CPU without perturbing a caller's base-policy RNG.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also resets all CUDA generators. Only touch
            # the CPU generator whose state this context actually restores.
            torch.random.default_generator.manual_seed(config.seed)
            self.actor = ResidualActor(feature_dim, config.hidden, config.initial_log_std)
            self.critic = TwinQ(feature_dim, config.hidden)
        self.actor.to(device)
        self.critic.to(device)
        self.target = deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator(device=device).manual_seed(config.seed + 1)
        self.replay = ReplayBuffer(config.replay_capacity, feature_dim, seed=config.seed + 2)
        self.updates = 0

    @torch.no_grad()
    def act(self, features: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.shape != (self.feature_dim,) or not np.isfinite(value).all():
            raise ValueError("invalid actor observation")
        residual, _ = self.actor.sample(
            torch.from_numpy(value[None]).to(self.device),
            generator=self.generator,
            deterministic=deterministic,
        )
        result = residual[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite residual")
        return result

    def _step(self, loss: Tensor, model, optimizer) -> float:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite RL objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"missing/nonfinite raw gradient: {name}")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        return float(norm)

    def update(self) -> dict[str, float]:
        cfg = self.config
        batch = self.replay.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            u_next, log_next = self.actor.sample(batch["next_features"], generator=self.generator)
            q1, q2 = self.target(batch["next_features"], u_next)
            target = bellman_target(
                batch["reward"],
                batch["terminated"],
                torch.minimum(q1, q2),
                log_next,
                gamma=cfg.gamma,
                alpha=cfg.alpha,
            )
        q1, q2 = self.critic(batch["features"], batch["residual"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_grad = self._step(critic_loss, self.critic, self.critic_optimizer)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        try:
            u, log_prob = self.actor.sample(batch["features"], generator=self.generator)
            q1, q2 = self.critic(batch["features"], u)
            actor_loss = (cfg.alpha * log_prob - torch.minimum(q1, q2)).mean()
            actor_grad = self._step(actor_loss, self.actor, self.actor_optimizer)
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for target_param, param in zip(
                self.target.parameters(), self.critic.parameters(), strict=True
            ):
                target_param.lerp_(param, cfg.tau)
        self.updates += 1
        return dict(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            q_mean=float(torch.minimum(q1, q2).detach().mean()),
            target_mean=float(target.mean()),
            residual_entropy=float(-log_prob.detach().mean()),
            alpha=cfg.alpha,
            critic_grad_norm=critic_grad,
            actor_grad_norm=actor_grad,
        )

    def state_dict(self) -> dict:
        return dict(
            config=self.config.as_dict(),
            feature_dim=self.feature_dim,
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target=self.target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            generator=self.generator.get_state(),
            replay=self.replay.state_dict(),
            updates=self.updates,
        )

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != self.config.as_dict() or state["feature_dim"] != self.feature_dim:
            raise ValueError("SAC config/feature identity differs")
        if type(state.get("updates")) is not int or state["updates"] < 0:
            raise ValueError("invalid learner update count")
        for owner in ("actor", "critic", "target"):
            model = getattr(self, owner)
            saved = state[owner]
            current = model.state_dict()
            if set(saved) != set(current):
                raise ValueError(f"{owner} parameter ownership differs")
            for key, value in saved.items():
                if value.shape != current[key].shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid {owner} checkpoint tensor {key}")

        def finite_tree(value):
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite optimizer checkpoint state")
            if isinstance(value, dict):
                for child in value.values():
                    finite_tree(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite_tree(child)

        for name in ("actor_optimizer", "critic_optimizer"):
            saved = state[name]
            current_groups = getattr(self, name).state_dict()["param_groups"]
            if saved["param_groups"] != current_groups:
                raise ValueError("optimizer ownership/hyperparameters differ")
            finite_tree(saved["state"])
        for owner in ("actor", "critic", "target"):
            getattr(self, owner).load_state_dict(state[owner], strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.replay.load_state_dict(state["replay"])
        self.updates = int(state["updates"])
