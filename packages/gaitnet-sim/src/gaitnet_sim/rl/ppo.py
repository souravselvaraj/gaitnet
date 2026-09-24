"""RSL-RL's PPO with learning-rate and entropy schedules, and a measure of how far each update
moved the policy.

RSL-RL offers a fixed learning rate or its KL-adaptive one. The adaptive one raises the rate
whenever an update's KL is under half its target, and a GaitNet policy held stochastic by the
entropy bonus (entropy ~0.75 nats all run long) makes small per-update changes, so it would
mostly push the rate up; its KL also covers only a tick's first footstep. Runs instead hold a
fixed rate and entropy bonus, then decay both over a set window of iterations, so late
training settles instead of wandering (the baseline's evals got worse from iteration 5000 to
8000).

Each update also reports the change it made over the whole tick, every footstep of it: an
estimate of KL(old || new) from the stored actions' log-probabilities (`approx_kl`, the
k3 estimator (ratio - 1) - log ratio, unbiased and never negative) and the share of those
actions whose probability ratio left PPO's clip range (`clip_fraction`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from gaitnet_core.action_layout import EnvAction
from gaitnet_core.rounds import TickSelection

SHAPES = ("cosine", "linear")


@dataclass(frozen=True)
class Schedule:
    """A value held at `initial` until iteration `start`, then moved to `final` by iteration
    `end` along `shape`, and held there."""

    initial: float
    start: int
    end: int
    final: float
    shape: str = "cosine"

    def __post_init__(self):
        if not 0 <= self.start < self.end:
            raise ValueError(f"a schedule needs 0 <= start < end, got start {self.start}, end {self.end}")
        if self.shape not in SHAPES:
            raise ValueError(f"schedule shape must be one of {SHAPES}, got '{self.shape}'")
        if self.final < 0:
            raise ValueError(f"a schedule's final value must be >= 0, got {self.final}")

    @classmethod
    def from_cfg(cls, cfg: dict | None, initial: float) -> "Schedule | None":
        """None (or an empty dict) for a constant value; else `{start, end, final[, shape]}`."""
        if not cfg:
            return None
        unknown = set(cfg) - {"start", "end", "final", "shape"}
        if unknown:
            raise ValueError(f"unknown schedule keys {sorted(unknown)}; a schedule has start, end, final, shape")
        return cls(initial, int(cfg["start"]), int(cfg["end"]), float(cfg["final"]), cfg.get("shape", "cosine"))

    def __call__(self, iteration: int) -> float:
        if iteration <= self.start:
            return self.initial
        if iteration >= self.end:
            return self.final
        progress = (iteration - self.start) / (self.end - self.start)
        if self.shape == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.initial + (self.final - self.initial) * progress


class ScheduledPPO(PPO):
    """See the module docstring."""

    def __init__(
        self,
        *args,
        lr_schedule: dict | None = None,
        entropy_schedule: dict | None = None,
        policy_change_samples: int = 8192,
        **kwargs,
    ) -> None:
        """
        Args:
            lr_schedule: `{start, end, final[, shape]}` in training iterations: the learning
                rate stays at `learning_rate` until `start`, then moves to `final` by `end`
                (cosine by default). None for a constant rate. Not with `schedule="adaptive"`.
            entropy_schedule: the same for `entropy_coef`.
            policy_change_samples: stored transitions the policy change is measured on after
                each update; 0 turns the measurement off.
        """
        super().__init__(*args, **kwargs)
        if lr_schedule and self.schedule == "adaptive":
            raise ValueError("lr_schedule sets the learning rate itself; use it with schedule='fixed'")
        self.lr_schedule = Schedule.from_cfg(lr_schedule, self.learning_rate)
        self.entropy_schedule = Schedule.from_cfg(entropy_schedule, self.entropy_coef)
        self.policy_change_samples = int(policy_change_samples)
        self.iteration = 0
        """The training iteration the next update belongs to (the runner's count, restored on
        resume), which the schedules are functions of."""

    def apply_schedules(self) -> None:
        if self.lr_schedule is not None:
            self.learning_rate = self.lr_schedule(self.iteration)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        if self.entropy_schedule is not None:
            self.entropy_coef = self.entropy_schedule(self.iteration)

    def update(self) -> dict[str, float]:
        self.apply_schedules()
        change: dict[str, float] = {}
        if self.policy_change_samples > 0:
            # measured after the optimization, on the rollout it learned from, which PPO's
            # update clears as its last step
            storage_clear = self.storage.clear

            def measure_then_clear() -> None:
                change.update(self.policy_change())
                storage_clear()

            self.storage.clear = measure_then_clear
        try:
            losses = super().update()
        finally:
            self.storage.__dict__.pop("clear", None)
        self.iteration += 1
        losses.update(change)
        losses["entropy_coef"] = self.entropy_coef
        return losses

    @torch.no_grad()
    def policy_change(self) -> dict[str, float]:
        """How far the latest update moved the policy, on up to `policy_change_samples` of
        the stored transitions: `approx_kl` and `clip_fraction` (see the module docstring)."""
        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        rows = torch.randperm(actions.shape[0], device=actions.device)[: self.policy_change_samples]
        new_log_prob = self._log_prob(observations[rows], actions[rows])
        log_ratio = new_log_prob - old_log_prob[rows]
        ratio = log_ratio.exp()
        return {
            "approx_kl": ((ratio - 1.0) - log_ratio).mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > self.clip_param).float().mean().item(),
        }

    def _log_prob(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        tick_distribution = getattr(self.actor, "tick_distribution", None)
        if tick_distribution is not None:
            # a GaitNet actor: score without acting, so feedback observers don't advance
            action = EnvAction.decode(actions)
            return tick_distribution(obs).log_prob(TickSelection(index=action.choice_index.long(), duration=action.duration))
        self.actor(obs, stochastic_output=True)
        return self.actor.get_output_log_prob(actions)

    def save(self) -> dict:
        saved = super().save()
        saved["ppo_iteration"] = self.iteration
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration:
            # the runner resumes at the checkpoint's iteration, and so do the schedules
            self.iteration = int(loaded_dict.get("iter", loaded_dict.get("ppo_iteration", 0)))
            self.apply_schedules()
        return load_iteration
