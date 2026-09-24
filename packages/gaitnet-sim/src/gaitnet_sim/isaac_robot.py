"""The simulated robots behind `gaitnet_core.interfaces.RobotInterface`, so evaluation runs
the same `PlannerRuntime` as deployment.

Observations come from the env's footstep action term. A command is one env step: the
footsteps (as many as the env's rounds per tick) and the nudge go through the action vector,
like a policy's; any further footsteps go straight to the low-level controller first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.state import Observation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

    from gaitnet_sim.env.actions import FootstepControlAction


class IsaacRobot:
    def __init__(self, env: "ManagerBasedRLEnv", action_name: str = "footstep"):
        self.env = env
        self.term: "FootstepControlAction" = env.action_manager.get_term(action_name)  # type: ignore[assignment]
        self.spec = self.term.spec
        self.last_step: "VecEnvStepReturn | None" = None
        """What the last `command` returned from `env.step`: observations, rewards,
        terminated, truncated, extras. Robots that terminated have already been reset."""

    @property
    def num_robots(self) -> int:
        return self.env.num_envs

    def observe(self) -> Observation:
        """What a policy would see in training: with the env's observation noise, if any."""
        return self.term.planner_observation()

    def command(self, footsteps: FootstepCommand | Sequence[FootstepCommand], nudge: Nudge | None = None) -> None:
        if isinstance(footsteps, FootstepCommand):
            footsteps = [footsteps]
        n, device = self.num_robots, self.env.device
        # the env executes up to its rounds per tick through the action; any more go
        # straight to the controller first
        rounds = self.term.rounds
        in_action = list(footsteps[:rounds])
        for extra in footsteps[rounds:]:
            self.term.controller.command_footsteps(extra)
        in_action += [FootstepCommand.none(n, device)] * (rounds - len(in_action))
        action = EnvAction(
            # the choice index only matters for training's log-probabilities
            choice_index=torch.zeros(n, rounds, dtype=torch.long, device=device),
            duration=torch.stack([f.duration for f in in_action], dim=1),
            leg=torch.stack([torch.where(f.active, f.leg, torch.full_like(f.leg, NO_STEP_LEG)) for f in in_action], dim=1),
            target=torch.stack([f.target for f in in_action], dim=1),
            nudge=nudge.command_delta if nudge is not None else torch.zeros(n, 3, device=device),
        )
        self.last_step = self.env.step(action.encode())

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        self.env.reset(env_ids=robot_ids)
