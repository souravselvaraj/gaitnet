"""Terrain curriculum: which difficulty level each robot's next episode runs on.

Isaac Lab calls the term for robots whose episode just ended, before any reset, so the action
term's per-episode progress is still there to read."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def terrain_levels_survival(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    p_up_given_success: float = 0.1,
    p_down_given_failure: float = 0.5,
    p_random: float = 0.02,
) -> torch.Tensor:
    """Move robots that survived their episode up a level, and robots that fell down one,
    each with some probability.

    The level settles where P(fall) = p_up / (p_up + p_down), and the probabilities set how
    fast it moves and how much it spreads. `p_random` moves a robot up and down alike,
    which leaves it in place; it keeps some robots on levels the rule wouldn't pick.

    Returns:
        The mean difficulty over all robots.
    """
    terrain: TerrainImporter = env.scene.terrain
    termination = env.termination_manager
    n = len(env_ids)
    noise = torch.rand(n, device=env.device) < p_random
    move_up = (termination.time_outs[env_ids] & (torch.rand(n, device=env.device) < p_up_given_success)) | noise
    move_down = (termination.terminated[env_ids] & (torch.rand(n, device=env.device) < p_down_given_failure)) | noise
    terrain.update_env_origins(env_ids, move_up, move_down)

    low, high = terrain.cfg.terrain_generator.difficulty_range
    level = torch.mean(terrain.terrain_levels.float()) / terrain.max_terrain_level
    return low + level * (high - low)


def terrain_levels_progress(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    required_fraction: float = 0.5,
    min_commanded: float = 0.2,
    p_up_given_success: float = 0.1,
    p_down_given_failure: float = 0.5,
    p_random: float = 0.02,
    action_name: str = "footstep",
) -> torch.Tensor:
    """`terrain_levels_survival`, except that surviving is not enough to move up: the robot
    must also have walked `required_fraction` of the distance its command asked for.

    Promoting on survival alone rewards creeping: a robot that stands still survives, and the
    curriculum then pushes it to harder terrain than a robot that walks and sometimes falls.
    A robot that survived without making progress stays where it is. Episodes whose command
    asked for less than `min_commanded` m (standing still) count survival as success.

    Returns:
        The mean difficulty over all robots.
    """
    from gaitnet_sim.env.observations import footstep_action

    terrain: TerrainImporter = env.scene.terrain
    termination = env.termination_manager
    term = footstep_action(env, action_name)
    n = len(env_ids)
    commanded = term.episode_commanded[env_ids]
    walked = term.episode_progress[env_ids]
    progressed = (commanded < min_commanded) | (walked >= required_fraction * commanded)
    success = termination.time_outs[env_ids] & progressed
    noise = torch.rand(n, device=env.device) < p_random
    move_up = (success & (torch.rand(n, device=env.device) < p_up_given_success)) | noise
    move_down = (termination.terminated[env_ids] & (torch.rand(n, device=env.device) < p_down_given_failure)) | noise
    terrain.update_env_origins(env_ids, move_up, move_down)

    low, high = terrain.cfg.terrain_generator.difficulty_range
    level = torch.mean(terrain.terrain_levels.float()) / terrain.max_terrain_level
    return low + level * (high - low)
