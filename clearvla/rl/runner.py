"""One environment, one executed row, and one cached base decision per step."""

import time

import numpy as np
import torch

from clearvla.simulation.history import CausalHistory

from .features import Decision
from .networks import ResidualActionMap
from .sac import ResidualSAC


def check_cuda_memory(device: torch.device) -> float:
    """Conservative existing 22-GiB process gate; shared GPU use cannot hide it."""
    if device.type != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_reserved(device)
    estimate = (peak + max(total - free - reserved, 0)) / (1024**3)
    if estimate > 22.0:
        raise RuntimeError(f"RL CUDA process estimate {estimate:.3f} GiB exceeds 22 GiB gate")
    return estimate


def run_episode(
    environment, reader, learner: ResidualSAC, *, seed: int, step_budget: int, mode: str
) -> dict:
    if step_budget <= 0 or mode not in {"train", "base", "adapter"}:
        raise ValueError("invalid episode budget/mode")
    started = time.perf_counter()
    reset = environment.reset(seed=seed)
    reset.observation.validate()
    reader.reset()
    history = CausalHistory()
    history.reset(reset.observation)
    action_map = ResidualActionMap(*environment.action_bounds(), learner.config.residual_fraction)
    decision: Decision = reader.decide(history.snapshot())
    peak_gib = check_cuda_memory(learner.device)
    total_reward, success = 0.0, False
    clip_total = base_clip_total = residual_total = 0.0
    update_metrics = None
    terminated = truncated = False
    for step in range(1, step_budget + 1):
        u = (
            np.zeros(7, np.float32)
            if mode == "base"
            else learner.act(decision.features, deterministic=mode == "adapter")
        )
        action, diagnostics = action_map.execute(decision.base_action, u)
        if not np.array_equal(action, environment.clip_action(action)):
            raise ValueError("residual action map differs from the native environment boundary")
        result = environment.step(action)
        result.validate()
        if not np.allclose(result.observation.action_state, action, rtol=0, atol=1e-6):
            raise ValueError("next action_state must be the actual executed residual command")
        history.append(action, result.observation)
        terminated = bool(result.terminated)
        # A collector budget is a timeout, not an absorbing MDP termination.
        truncated = bool(result.truncated) or (step == step_budget and not terminated)
        # Final timeout observations are retained for bootstrap. True terminal
        # observations are never replaced with a reset observation either.
        next_decision = (
            Decision(np.zeros(learner.feature_dim, np.float32), np.zeros(7, np.float32))
            if terminated
            else reader.decide(history.snapshot())
        )
        if mode == "train":
            learner.replay.add(
                features=decision.features,
                next_features=next_decision.features,
                residual=u,
                executed=action,
                reward=float(result.reward),
                terminated=terminated,
                truncated=truncated,
            )
            if learner.replay.size >= learner.config.learning_starts:
                update_metrics = learner.update()
        peak_gib = max(peak_gib, check_cuda_memory(learner.device))
        decision = next_decision
        total_reward += float(result.reward)
        # This evaluator fact is reported only, never supplied to reader/actor/Q.
        success = success or bool(result.evaluation.metrics.get("success", False))
        clip_total += diagnostics["residual_clip_fraction"]
        base_clip_total += diagnostics["base_clip_fraction"]
        residual_total += diagnostics["residual_native_l2"]
        if terminated or truncated:
            break
    return dict(
        seed=seed,
        steps=step,
        mode=mode,
        reward_sum=total_reward,
        success=success,
        terminated=terminated,
        truncated=truncated,
        mean_base_clip_fraction=base_clip_total / step,
        mean_residual_clip_fraction=clip_total / step,
        mean_residual_native_l2=residual_total / step,
        seconds=time.perf_counter() - started,
        updates=learner.updates,
        last_update=update_metrics,
        cuda_peak_process_estimate_gib=peak_gib,
    )
