"""Reward terms that depend on the footstep action term (the nudged command, the step
taken) or that isaaclab's stock mdp doesn't provide."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from gaitnet_sim.env.observations import footstep_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: "ManagerBasedRLEnv", command: Literal["effective", "base"], action_name: str) -> torch.Tensor:
    term = footstep_action(env, action_name)
    return term.effective_command() if command == "effective" else term.base_command()


def track_lin_vel_xy_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command: Literal["effective", "base"] = "effective",
    action_name: str = "footstep",
) -> torch.Tensor:
    """exp(-|v_xy - command_xy|^2 / std^2), base frame.

    Args:
        command: track the nudged command the controller follows ("effective") or the
            command before the nudge ("base").
    """
    velocity = footstep_action(env, action_name).io.robot.data.root_link_lin_vel_b.torch[:, :2]
    error = torch.sum(torch.square(_command(env, command, action_name)[:, :2] - velocity), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command: Literal["effective", "base"] = "effective",
    action_name: str = "footstep",
) -> torch.Tensor:
    """exp(-(yaw rate - commanded yaw rate)^2 / std^2)."""
    yaw_rate = footstep_action(env, action_name).io.robot.data.root_link_ang_vel_b.torch[:, 2]
    error = torch.square(_command(env, command, action_name)[:, 2] - yaw_rate)
    return torch.exp(-error / std**2)


def step_taken(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """Footsteps the policy started this env step: 0 for the no-op, up to the contract's
    `max_steps_per_tick`, so every step costs the same however many share a tick."""
    return footstep_action(env, action_name).steps_started().float()


def foot_slip(
    env: "ManagerBasedRLEnv",
    threshold: float = 1.0,
    settle_time: float = 0.04,
    stance_only: bool = True,
    action_name: str = "footstep",
) -> torch.Tensor:
    """Sum of the horizontal speed of feet that should be planted (m/s).

    A foot counts if it is in contact (normal force over `threshold` N) and, with
    `stance_only`, in the controller's scheduled stance for at least `settle_time` s. Feet that
    are lifting off or still landing move by design; counting them charged every footstep a
    slip cost on top of `step_taken`, which taught the policy to step less rather than to stop
    feet sliding."""
    term = footstep_action(env, action_name)
    io = term.io
    forces = io.contact_sensor.data.net_normal_forces_w.torch[:, io.contact_ids]
    planted = forces.norm(dim=-1) > threshold
    if stance_only:
        timing = term.controller.gait_timing()
        planted = planted & (timing[..., 1] <= 0) & (timing[..., 2] >= settle_time)
    speed = io.robot.data.body_link_lin_vel_w.torch[:, io.foot_ids, :2].norm(dim=-1)
    return torch.sum(planted * speed, dim=1)
