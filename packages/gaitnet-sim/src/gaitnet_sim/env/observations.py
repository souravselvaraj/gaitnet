"""Observation terms. Each builds on `gaitnet_core`, so the simulator and the deployment
runtime compute the same quantities.

Groups (see `env_cfg.ObservationsCfg`): `state` (the robot state vector, features chosen by
name), `candidates` (the footholds the policy scores this tick), and, when a preset turns
them on, `terrain` (per-leg height patches, for networks that read them), `privileged`
(sim-only inputs for the critic), `base_command` (the command before any nudge, for
feedback observers) and `teacher_state` / `teacher_candidates` (a distillation teacher's
view, see below). Candidates are an observation so the RL library stores them with the
step and can recompute log-probabilities on exactly the set the action was drawn from.

The policy's groups read the action term's `planner_observation()`, which carries the
observation noise and the camera map; privileged terms read the truth. A distillation
teacher reads the truth too, but scores the student's own candidates, re-judged on the true
terrain (`teacher_candidates`), so the two policies are distributions over the same set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg

from gaitnet_core.candidates import reassess
from gaitnet_core.features import state_vector
from gaitnet_core.terrain import fill_unknown, inner_heights, valid_footholds

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster

    from gaitnet_sim.env.actions import FootstepControlAction


def footstep_action(env: "ManagerBasedRLEnv", name: str = "footstep") -> "FootstepControlAction":
    """The footstep action term, which owns the controller and reads the robot's state."""
    return env.action_manager.get_term(name)  # type: ignore[return-value]


def robot_state(env: "ManagerBasedRLEnv", features: list[str], action_name: str = "footstep") -> torch.Tensor:
    """(N, D) the named `gaitnet_core.features`, concatenated in order."""
    return state_vector(footstep_action(env, action_name).planner_observation().state, features)


def terrain_heights(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """(N, L, *patch_size) terrain heights relative to each hip (m).

    Unknown cells (no ray returned) read `gaitnet_core.terrain.UNKNOWN_HEIGHT` rather than
    -inf, as the networks that read terrain see them at deployment too.
    """
    return fill_unknown(footstep_action(env, action_name).planner_observation().terrain.heights)


def footstep_candidates(
    env: "ManagerBasedRLEnv",
    sampler: str = "uniform_jitter",
    sampler_kwargs: dict | None = None,
    action_name: str = "footstep",
) -> torch.Tensor:
    """(N, L, K, 5) packed `gaitnet_core.candidates.Candidates` from the named sampler.

    Only valid footholds (terrain rules and leg eligibility, `env.cfg.gaitnet`) are
    sampled, so the policy never scores a foothold it isn't allowed to take.
    """
    return footstep_action(env, action_name).candidates(sampler, sampler_kwargs).pack()


def base_command(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """(N, 3) the velocity command before any nudge, which feedback observers scale."""
    return footstep_action(env, action_name).base_command()


##
# Distillation: the teacher's view of the same tick
##


def teacher_robot_state(env: "ManagerBasedRLEnv", features: list[str], action_name: str = "footstep") -> torch.Tensor:
    """(N, D) the named features of the true robot state, without observation noise."""
    return state_vector(footstep_action(env, action_name).observation().state, features)


def teacher_terrain_heights(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """(N, L, *patch_size) the true terrain heights relative to each hip (m), for a teacher whose
    network reads terrain (the student's `terrain` group reads the camera map instead)."""
    return fill_unknown(footstep_action(env, action_name).terrain().heights)


def teacher_candidates(
    env: "ManagerBasedRLEnv",
    candidates_group: str = "candidates",
    candidates_term: str = "candidates",
    action_name: str = "footstep",
) -> torch.Tensor:
    """(N, L, K, 5) the student's candidates (the `candidates_group` group's term, same
    draw) judged on the true terrain and state: a slot the truth rules out is invalid, and z is
    the true height (`gaitnet_core.candidates.reassess`). Flat indices mean the same foothold in
    both groups, so the teacher's distribution is over the student's own choices."""
    term = footstep_action(env, action_name)
    params = getattr(getattr(env.cfg.observations, candidates_group), candidates_term).params
    candidates = term.candidates(params.get("sampler", "uniform_jitter"), params.get("sampler_kwargs"))
    truth = term.observation()
    rules = env.cfg.gaitnet.foothold_rules()
    valid = rules.valid(truth, term.spec)
    heights = inner_heights(rules.heights(truth), term.grid)
    return reassess(candidates, valid, term.grid, heights).pack()


##
# Privileged: what the critic may know and the policy can't
##


def terrain_height_summary(env: "ManagerBasedRLEnv", cells: int = 5, action_name: str = "footstep") -> torch.Tensor:
    """(N, L * cells^2) each leg's candidate grid of terrain heights, averaged down to
    cells x cells. Unknown cells count as `UNKNOWN_HEIGHT`."""
    term = footstep_action(env, action_name)
    heights = fill_unknown(inner_heights(term.terrain().heights, term.grid))
    n, l, size_x, size_y = heights.shape
    pooled = F.adaptive_avg_pool2d(heights.reshape(n * l, 1, size_x, size_y), cells)
    return pooled.reshape(n, -1)


def foothold_validity_summary(env: "ManagerBasedRLEnv", cells: int = 5, action_name: str = "footstep") -> torch.Tensor:
    """(N, L * cells^2) fraction of valid footholds (terrain rules only, not leg eligibility)
    in each of cells x cells blocks of each leg's grid."""
    term = footstep_action(env, action_name)
    rules = env.cfg.gaitnet.foothold_rules()
    valid = valid_footholds(term.terrain().heights, term.spec, term.grid, rules.step_threshold, rules.edge_margin)
    n, l, size_x, size_y = valid.shape
    pooled = F.adaptive_avg_pool2d(valid.float().reshape(n * l, 1, size_x, size_y), cells)
    return pooled.reshape(n, -1)


def base_terrain_clearance(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_scanner"),
    max_clearance: float = 1.0,
) -> torch.Tensor:
    """(N, 1) the base's height above the highest surface under the trunk (m), at most
    `max_clearance` (also where no ray hit anything).

    `sensor_cfg` is a ray caster on the base, yaw aligned, whose pattern covers the trunk's
    footprint. On flat ground this is the base's height above the ground.
    """
    scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = scanner.data.ray_hits_w.torch[..., 2]
    hits = torch.where(torch.isfinite(hits), hits, torch.full_like(hits, float("-inf")))
    clearance = scanner.data.pos_w.torch[:, 2] - hits.amax(dim=1)
    return clearance.clamp(max=max_clearance).unsqueeze(-1)


def foot_contact_forces(env: "ManagerBasedRLEnv", scale: float = 0.01, action_name: str = "footstep") -> torch.Tensor:
    """(N, L) each foot's contact force magnitude, times `scale` (1/N)."""
    io = footstep_action(env, action_name).io
    return io.contact_sensor.data.net_normal_forces_w.torch[:, io.contact_ids].norm(dim=-1) * scale
