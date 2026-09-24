"""The simulator's action vector.

The RL library stores and replays actions, so the vector carries both what the policy
chose (the candidate index and duration, needed to recompute log-probabilities) and what
the environment should do (a concrete footstep and a nudge). The environment reads only
the latter, so it never needs the candidate set.

A tick may start up to R footsteps, chosen one after another (`gaitnet_core.rounds`). The
vector is R blocks of the step fields, round-major, then the nudge: with R = 1 it is the
original nine numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaitnet_core.interfaces import FootstepCommand, Nudge

STEP_FIELDS: tuple[str, ...] = ("choice_index", "duration", "leg", "x", "y", "z")
NUDGE_FIELDS: tuple[str, ...] = ("nudge_vx", "nudge_vy", "nudge_wz")
FIELDS: tuple[str, ...] = STEP_FIELDS + NUDGE_FIELDS
"""The fields of a one-round action."""
NO_STEP_LEG = -1


def dim(rounds: int = 1) -> int:
    """Length of the action vector for `rounds` footsteps per tick."""
    if rounds < 1:
        raise ValueError(f"need at least one round, got {rounds}")
    return len(STEP_FIELDS) * rounds + len(NUDGE_FIELDS)


def rounds_for_dim(action_dim: int) -> int:
    """The number of rounds an action vector of `action_dim` numbers holds."""
    rounds, rest = divmod(action_dim - len(NUDGE_FIELDS), len(STEP_FIELDS))
    if rest or rounds < 1:
        raise ValueError(f"{action_dim} is not an action length: {len(STEP_FIELDS)} per round plus {len(NUDGE_FIELDS)}")
    return rounds


DIM = dim(1)


@dataclass
class EnvAction:
    choice_index: torch.Tensor
    """(N, R) flat candidate index the policy chose in each round, the no-op index for no
    step. (N,) is accepted for a single round."""
    duration: torch.Tensor
    """(N, R) swing duration (s)."""
    leg: torch.Tensor
    """(N, R) leg to step, -1 for no step."""
    target: torch.Tensor
    """(N, R, 3) foothold in the leg's hip yaw frame (m)."""
    nudge: torch.Tensor
    """(N, 3) velocity command delta."""

    def __post_init__(self):
        if self.choice_index.dim() == 1:
            self.choice_index = self.choice_index.unsqueeze(-1)
            self.duration = self.duration.unsqueeze(-1)
            self.leg = self.leg.unsqueeze(-1)
            self.target = self.target.unsqueeze(-2)

    @property
    def rounds(self) -> int:
        return self.choice_index.shape[-1]

    def encode(self) -> torch.Tensor:
        """(N, dim(R)) float tensor."""
        steps = torch.cat(
            [
                self.choice_index.float().unsqueeze(-1),
                self.duration.float().unsqueeze(-1),
                self.leg.float().unsqueeze(-1),
                self.target.float(),
            ],
            dim=-1,
        )
        return torch.cat([steps.flatten(1), self.nudge.float()], dim=-1)

    @classmethod
    def decode(cls, action: torch.Tensor) -> "EnvAction":
        rounds = rounds_for_dim(action.shape[-1])
        steps = action[:, : len(STEP_FIELDS) * rounds].reshape(action.shape[0], rounds, len(STEP_FIELDS))
        return cls(
            choice_index=steps[..., 0].round().long(),
            duration=steps[..., 1],
            leg=steps[..., 2].round().long(),
            target=steps[..., 3:6],
            nudge=action[:, -len(NUDGE_FIELDS) :],
        )

    def footstep_command(self, round: int = 0) -> FootstepCommand:
        """The footsteps of one round."""
        leg = self.leg[:, round]
        return FootstepCommand(
            active=leg != NO_STEP_LEG,
            leg=leg.clamp(min=0),
            target=self.target[:, round],
            duration=self.duration[:, round],
        )

    def footstep_commands(self) -> list[FootstepCommand]:
        """The footsteps of every round, in order."""
        return [self.footstep_command(r) for r in range(self.rounds)]

    def nudge_command(self) -> Nudge:
        return Nudge(command_delta=self.nudge)
