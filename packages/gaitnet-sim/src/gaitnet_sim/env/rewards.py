"""Reward terms that depend on the footstep action term (the nudged command, the step
taken) or that isaaclab's stock mdp doesn't provide.

The long-horizon terms (`presets=horizon`) judge what a step leads to rather than the instant:
`WindowTracking` compares where the base got to over the last second with where the commands
asked it to go (and the heading it drifted to), `foothold_edge` charges footholds that leave
little margin to a hole edge, and `short_stance` charges lifting a leg again right after it
landed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

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


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class CommandWindow:
    """Per robot, the last `window_s` of base motion against the operator's command: where the
    base went (world xy, unwrapped heading) and where the commands asked it to go. Shared by the
    window reward (`WindowTracking`) and the heading-drift constraint."""

    def __init__(self, env: "ManagerBasedRLEnv", window_s: float):
        self.window = max(1, round(window_s / env.step_dt))
        n, device, slots = env.num_envs, env.device, self.window + 1
        # per robot, the latest `window` + 1 values of: position, cumulative commanded
        # displacement, unwrapped heading, cumulative commanded heading change
        self.position = torch.zeros(n, slots, 2, device=device)
        self.commanded = torch.zeros(n, slots, 2, device=device)
        self.heading = torch.zeros(n, slots, device=device)
        self.commanded_heading = torch.zeros(n, slots, device=device)
        self.last_yaw = torch.zeros(n, device=device)
        self.count = torch.zeros(n, dtype=torch.long, device=device)
        self.head = 0

    def reset(self, env_ids=None) -> None:
        self.count[slice(None) if env_ids is None else env_ids] = 0

    def update(self, env: "ManagerBasedRLEnv", action_name: str = "footstep") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record this env step; returns (N,) steps of history in the window, the (N, 2)
        displacement error (actual - commanded, m) and the (N,) heading error (rad) over it."""
        term = footstep_action(env, action_name)
        data = term.io.robot.data
        dt = env.step_dt
        position = data.root_link_pos_w.torch[:, :2]
        x, y, z, w = data.root_link_quat_w.torch.unbind(-1)  # xyzw
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        command = term.base_command()

        slots = self.window + 1
        now, previous = self.head, (self.head - 1) % slots
        fresh = self.count == 0
        heading = torch.where(fresh, yaw, self.heading[:, previous] + _wrap(yaw - self.last_yaw))
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        velocity = torch.stack([cos * command[:, 0] - sin * command[:, 1], sin * command[:, 0] + cos * command[:, 1]], -1)
        commanded = torch.where(fresh.unsqueeze(-1), torch.zeros_like(velocity), self.commanded[:, previous] + velocity * dt)
        commanded_heading = torch.where(fresh, torch.zeros_like(yaw), self.commanded_heading[:, previous] + command[:, 2] * dt)
        self.position[:, now] = position
        self.commanded[:, now] = commanded
        self.heading[:, now] = heading
        self.commanded_heading[:, now] = commanded_heading
        self.last_yaw = yaw
        self.head = (now + 1) % slots
        self.count = (self.count + 1).clamp(max=slots)

        back = (self.count - 1).clamp(max=self.window)  # steps of history in the window
        old = (now - back) % slots
        rows = torch.arange(position.shape[0], device=position.device)
        displacement_error = (position - self.position[rows, old]) - (commanded - self.commanded[rows, old])
        heading_error = (heading - self.heading[rows, old]) - (commanded_heading - self.commanded_heading[rows, old])
        return back, displacement_error, heading_error


class WindowTracking(ManagerTermBase):
    """Command tracking over a window: how far the base went against how far the commands
    asked it to go, over the last `window_s`.

    With `quantity="xy"` it is exp(-(|displacement error| / window)^2 / std^2), world xy: a
    robot that tracks every instant but keeps slowing down or drifting sideways scores low,
    where the instantaneous tracking terms forgive it step by step. With
    `quantity="heading"` it is the squared heading error accumulated over the window (rad^2),
    against the commanded yaw rate: use it as a penalty. Both read the operator's command
    (before any nudge), and give 0 until an episode has `min_window_s` of history.
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.tracker = CommandWindow(env, cfg.params.get("window_s", 1.0))

    def reset(self, env_ids=None) -> None:
        self.tracker.reset(env_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        window_s: float = 1.0,
        std: float = 0.1,
        quantity: Literal["xy", "heading"] = "xy",
        min_window_s: float = 0.2,
        action_name: str = "footstep",
    ) -> torch.Tensor:
        back, displacement_error, heading_error = self.tracker.update(env, action_name)
        if quantity == "xy":
            rate = displacement_error.norm(dim=-1) / (back.clamp(min=1) * env.step_dt)
            value = torch.exp(-torch.square(rate / std))
        elif quantity == "heading":
            value = torch.square(heading_error)
        else:
            raise ValueError(f"quantity must be 'xy' or 'heading', got {quantity!r}")
        return torch.where(back >= round(min_window_s / env.step_dt), value, torch.zeros_like(value))


def terminated_by(env: "ManagerBasedRLEnv", term_keys: list[str]) -> torch.Tensor:
    """1 where the episode ended on one of the named termination terms (e.g. the falls, and not
    the constraint terminations of `gaitnet_sim.env.constraints`)."""
    manager = env.termination_manager
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_keys:
        ended |= manager.get_term(name)
    return ended.float()


def _step_quality(env: "ManagerBasedRLEnv", action_name: str):
    term = footstep_action(env, action_name)
    if not term.cfg.step_quality:
        raise ValueError("this reward reads the footstep term's step quality: set env.actions.footstep.step_quality=True")
    return term


def foothold_edge(env: "ManagerBasedRLEnv", margin_cells: int = 6, action_name: str = "footstep") -> torch.Tensor:
    """Sum over the footsteps started this env step of (1 - clearance / margin_cells), where
    clearance is the foothold's distance to the nearest edge cell of the true terrain (cells,
    Chebyshev; see `FootstepControlAction.step_clearance`): 1 on an edge, 0 at `margin_cells`
    or more. A foothold with room around it keeps the next steps' options open."""
    term = _step_quality(env, action_name)
    return (1.0 - term.step_clearance / margin_cells).clamp(0.0, 1.0).sum(dim=1)


def short_stance(env: "ManagerBasedRLEnv", min_stance_s: float = 0.1, action_name: str = "footstep") -> torch.Tensor:
    """Footsteps started this env step on a leg that touched down less than `min_stance_s`
    ago: lifting a leg again right away (a leg "chattering") gains little ground and leaves
    the base on fewer feet."""
    term = _step_quality(env, action_name)
    return (term.step_stance_time < min_stance_s).float().sum(dim=1)
