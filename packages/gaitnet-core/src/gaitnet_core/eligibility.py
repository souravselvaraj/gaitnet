"""Which legs may be given a new footstep."""

from __future__ import annotations

import torch


def scheduled_stance(gait_timing: torch.Tensor) -> torch.Tensor:
    """(N, L) bool, whether the controller's schedule has each leg in stance.

    This uses the schedule rather than measured contact: a leg the controller still has
    in swing (e.g. after an early touchdown) can't take a new footstep.
    """
    return gait_timing[..., 1] <= 0


def step_eligible(gait_timing: torch.Tensor, min_stance_after_step: int = 2, min_stance_time: float = 0.0) -> torch.Tensor:
    """(N, L) bool, True for legs that may start a swing this tick.

    A leg is eligible if it is in scheduled stance, has been for at least `min_stance_time`
    s, and, after it lifts off, at least `min_stance_after_step` legs remain in stance.
    """
    stance = scheduled_stance(gait_timing)
    num_stance = stance.sum(dim=-1, keepdim=True)
    settled = gait_timing[..., 2] >= min_stance_time
    return stance & settled & (num_stance - 1 >= min_stance_after_step)
