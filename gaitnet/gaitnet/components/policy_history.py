"""A short history of the robot's own motion for the policy: the last T policy ticks of D = 26
features per env, appended after the candidates in the policy observation (GaitNet-T's layout:
[robot state 25 | candidates 65 x 8 | history T x 26]).

Per tick, body frame, legs in GaitNet's slot order FR, FL, RR, RL:

    base linear velocity (3) | base angular velocity (3) | foot position relative to its hip,
    xy (4 x 2) | scheduled contact (4) | last-touchdown error: achieved touchdown position
    relative to the hip, xy, minus the hip-frame foothold commanded for it (4 x 2; 0 before a
    leg's first touchdown)

Nothing privileged: on the robot these are the odometry twist, the feet from joint encoders
(TF), and the gait node's own schedule and commanded footholds. The buffer is zeroed for an env
when its episode starts, and holds ticks oldest first.
"""

from __future__ import annotations

import torch

HISTORY_DIM = 26
SLOT_FROM_LEG = [1, 0, 3, 2]
"""Controller leg order FL, FR, RL, RR -> GaitNet slot order FR, FL, RR, RL (its own inverse)."""


def tick_features(
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    foot_position_hip: torch.Tensor,
    contact: torch.Tensor,
    touchdown_error: torch.Tensor,
) -> torch.Tensor:
    """(N, 26) one tick's features from controller-order inputs.

    Args:
        base_lin_vel, base_ang_vel: (N, 3) body frame (m/s, rad/s).
        foot_position_hip: (N, 4, 3) feet relative to their hips, body frame, FL, FR, RL, RR (m).
        contact: (N, 4) bool scheduled contact, FL, FR, RL, RR.
        touchdown_error: (N, 4, 2) achieved minus commanded touchdown xy, FL, FR, RL, RR (m).
    """
    n = base_lin_vel.shape[0]
    return torch.cat(
        [
            base_lin_vel,
            base_ang_vel,
            foot_position_hip[:, SLOT_FROM_LEG, :2].reshape(n, 8),
            contact[:, SLOT_FROM_LEG].to(base_lin_vel.dtype),
            touchdown_error[:, SLOT_FROM_LEG].reshape(n, 8),
        ],
        dim=-1,
    )


class PolicyHistory:
    """(N, T, D) ring of the last T ticks, oldest first."""

    def __init__(self, num_envs: int, length: int, device: torch.device | str, dim: int = HISTORY_DIM):
        self.length = length
        self.dim = dim
        self.buffer = torch.zeros(num_envs, length, dim, device=device)
        self._last_tick: int | None = None

    def push(self, features: torch.Tensor, tick: int) -> None:
        """Append one tick's (N, D) features, once per `tick`: a second call for the same tick
        (an observation recomputed without stepping) is ignored."""
        if tick == self._last_tick:
            return
        self._last_tick = tick
        self.buffer = torch.cat([self.buffer[:, 1:], features.to(self.buffer.dtype).unsqueeze(1)], dim=1)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        self.buffer[slice(None) if env_ids is None else env_ids] = 0.0

    def flat(self) -> torch.Tensor:
        """(N, T * D) as it goes into the observation."""
        return self.buffer.flatten(start_dim=1)
