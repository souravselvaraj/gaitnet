"""Constraints as terminations (CaT, Chane-Sane et al., IROS 2024).

Most of the reward terms were limits written as weighted penalties (don't slip, don't tilt,
don't bounce, don't veer, don't lift a leg the moment it lands), and weighted penalties trade
against each other and against the task in ways no one can predict: every weight interacts
with every other. Here each limit is a threshold in physical units instead. Each env step a
constraint's violation v >= 0 (how far past its limit the robot is) ends the episode with
probability

    p = p_max * clip(v / v_max, 0, 1)

where v_max is a running estimate of the largest violation seen (an exponential moving average
of each step's batch maximum), so a constraint that is broken badly is almost always fatal and
a slight overshoot rarely is. Ending the episode forfeits the rest of its return, which is what
PPO learns to avoid; the task rewards stay, and the tuning is limits the robot and task set.

Constraint terminations are not falls: they are not time-outs either, and the curriculum and
the fall penalty count only the fall terminations (`FALL_TERMS`). Evaluation removes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from gaitnet_sim.env.observations import footstep_action
from gaitnet_sim.env.rewards import CommandWindow

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FALL_TERMS: list[str] = ["bad_orientation", "bad_height", "foot_below_ground"]
"""The termination terms that are falls."""

Quantity = Literal["foot_slip", "tilt", "vertical_speed", "body_rates", "heading_drift", "short_stance"]


class ConstraintTermination(ManagerTermBase):
    """One constraint as a stochastic termination, see the module docstring.

    Params:
        quantity: what is limited, all in physical units:
            foot_slip: the fastest horizontal speed of a foot that should be planted (m/s)
            tilt: the base's angle from upright (rad)
            vertical_speed: the base's vertical speed (m/s)
            body_rates: the base's roll/pitch rate (rad/s)
            heading_drift: the heading error accumulated over the last `window_s` against the
                commanded turning (rad)
            short_stance: for a footstep started this step, how much less than `limit` its
                leg had been down (s); needs the footstep term's `step_quality`
        limit: the threshold
        p_max: the termination probability of the largest violations
        tau: the running maximum's decay per env step
    """

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.running_max = torch.zeros((), device=env.device)
        self.window = CommandWindow(env, cfg.params.get("window_s", 1.0)) if cfg.params.get("quantity") == "heading_drift" else None
        self.violating = torch.zeros((), device=env.device)
        """Share of robots over the limit on the latest step (for logging)."""

    def reset(self, env_ids=None) -> None:
        if self.window is not None:
            self.window.reset(env_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        quantity: Quantity,
        limit: float,
        p_max: float = 0.1,
        tau: float = 0.95,
        window_s: float = 1.0,
        slip_contact_threshold: float = 1.0,
        slip_settle_time: float = 0.04,
        action_name: str = "footstep",
    ) -> torch.Tensor:
        violation = self._violation(env, quantity, limit, slip_contact_threshold, slip_settle_time, action_name)
        self.violating = (violation > 0).float().mean()
        self.running_max = torch.maximum(tau * self.running_max + (1 - tau) * violation.max(), torch.tensor(1e-6, device=violation.device))
        probability = p_max * (violation / self.running_max).clamp(0.0, 1.0)
        return torch.rand_like(probability) < probability

    def _violation(self, env, quantity, limit, contact_threshold, settle_time, action_name) -> torch.Tensor:
        term = footstep_action(env, action_name)
        data = term.io.robot.data
        if quantity == "foot_slip":
            io = term.io
            forces = io.contact_sensor.data.net_normal_forces_w.torch[:, io.contact_ids]
            timing = term.controller.gait_timing()
            planted = (forces.norm(dim=-1) > contact_threshold) & (timing[..., 1] <= 0) & (timing[..., 2] >= settle_time)
            speed = data.body_link_lin_vel_w.torch[:, io.foot_ids, :2].norm(dim=-1)
            value = torch.where(planted, speed, torch.zeros_like(speed)).amax(dim=1)
        elif quantity == "tilt":
            value = torch.acos((-data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0))
        elif quantity == "vertical_speed":
            value = data.root_link_lin_vel_b.torch[:, 2].abs()
        elif quantity == "body_rates":
            value = data.root_link_ang_vel_b.torch[:, :2].norm(dim=-1)
        elif quantity == "heading_drift":
            back, _, heading_error = self.window.update(env, action_name)
            value = torch.where(back >= self.window.window, heading_error.abs(), torch.zeros_like(heading_error))
        elif quantity == "short_stance":
            if not term.cfg.step_quality:
                raise ValueError("the short_stance constraint reads step quality: set env.actions.footstep.step_quality=True")
            shortfall = (limit - term.step_stance_time).clamp(min=0.0)  # inf stance (no step) -> 0
            return shortfall.amax(dim=1)
        else:
            raise ValueError(f"unknown constraint quantity {quantity!r}")
        return (value - limit).clamp(min=0.0)


def constraint_terms(env_cfg) -> list[str]:
    """Names of the constraint terminations in an env cfg's terminations."""
    names = []
    for name in env_cfg.terminations.__dataclass_fields__:
        term = getattr(env_cfg.terminations, name)
        if term is not None and getattr(term, "func", None) is ConstraintTermination:
            names.append(name)
    return names


def strip_constraints(env_cfg) -> None:
    """Remove the constraint terminations, e.g. for evaluation, where a robot runs until it falls."""
    for name in constraint_terms(env_cfg):
        setattr(env_cfg.terminations, name, None)
