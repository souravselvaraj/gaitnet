from isaaclab.managers import EventTermCfg as EventTerm, termination_manager
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp  # type: ignore
from isaaclab.envs import mdp

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, terrain_generator
from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain  # type: ignore
    termination_manager = env.termination_manager

    move_down_given_failure = torch.rand(len(env_ids), device=env.device) < 0.5  # P_d
    move_up_given_success = torch.rand(len(env_ids), device=env.device) < 0.1  # P_u
    # Given the two parameters above, we can expect the system to settle into
    # a steady state success rate while the terrain levels are changing.
    # The level will be $\frac{P_d}{P_u + P_d}$
    # The magnitude of these combined with the magnitude of noise will determine
    # how quickly the terrain levels change, and the variance around the mean level.
    
    # random chance to move up or down regardless of success/failure
    random_noise = torch.rand(len(env_ids), device=env.device) < 0.02

    # move up with some probability if you survived the whole time
    success = termination_manager.time_outs[env_ids]
    move_up = success & move_up_given_success
    # randomly move up with a small probability
    move_up = move_up | random_noise

    # move down with some probability if you terminated early
    terminated_early = termination_manager.terminated[env_ids]
    move_down = terminated_early & move_down_given_failure
    # randomly move down with a small probability
    move_down = move_down | random_noise

    # update terrain levels
    # note that move_up and move_down are not mutually exclusive because of the noise
    # in this case terrain.update_env_origins won't move
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean difficulty
    difficulty_range: tuple[float, float] = terrain.cfg.terrain_generator.difficulty_range  # type: ignore
    mean_terrain_level = torch.mean(terrain.terrain_levels.float())
    max_terrain_level = terrain.max_terrain_level
    # scale terrain level from [0, max_terrain_level] to [difficulty_range[0], difficulty_range[1]]
    mean_difficulty = difficulty_range[0] + (
        mean_terrain_level / max_terrain_level
    ) * (difficulty_range[1] - difficulty_range[0])
    return mean_difficulty


def terrain_levels_walk(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    required_fraction: float = 0.3,
    min_commanded: float = 0.2,
    p_up_given_success: float = 0.1,
    p_down_given_failure: float = 0.5,
    p_random: float = 0.02,
    random_fraction: float = 0.0,
) -> torch.Tensor:
    """Up a level for surviving and walking `required_fraction` of the commanded distance, down for
    falling or stalling (components/walk_curriculum.walk_moves, which says why vmdp.terrain_levels_vel
    kept every robot on level 0). 0.3 rather than 0.5 because the 1035 checkpoint makes about half
    its commanded speed even on level 0 (the teacher recipe's Episode_Reward/progress, 0.44-0.50):
    a 0.5 bar would demote half the robots with no voids in the way. Reads GaitNetEnv's per-episode distances, which are still this
    episode's: Isaac Lab runs the curriculum before anything is reset. Returns the mean terrain
    level, the same quantity terrain_levels_vel logs; with `random_fraction` > 0 that many of the
    episodes run on a uniformly random level instead (walk_curriculum.next_levels), and the logged
    level is the curriculum's, not the random ones."""
    from gaitnet.gaitnet.components.walk_curriculum import next_levels, walk_moves

    terrain: TerrainImporter = env.scene.terrain  # type: ignore
    termination = env.termination_manager
    env_ids = torch.as_tensor(env_ids, device=env.device)
    failed = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for name in termination.active_terms:
        if not termination.get_term_cfg(name).time_out:
            failed |= termination.get_term(name)[env_ids]
    walked, commanded = env.walk_progress[env_ids], env.walk_commanded[env_ids]  # type: ignore[attr-defined]
    move_up, move_down, progressed = walk_moves(
        termination.time_outs[env_ids], failed, walked, commanded, required_fraction, min_commanded,
        p_up_given_success, p_down_given_failure, p_random,
    )
    if random_fraction <= 0.0:
        terrain.update_env_origins(env_ids, move_up, move_down)
    else:
        if not hasattr(env, "walk_curriculum_level"):
            env.walk_curriculum_level = terrain.terrain_levels.clone()  # type: ignore[attr-defined]
            env.walk_explored = torch.zeros_like(terrain.terrain_levels, dtype=torch.bool)  # type: ignore[attr-defined]
        level, explore, run_on = next_levels(
            env.walk_curriculum_level[env_ids], env.walk_explored[env_ids], move_up, move_down,  # type: ignore[attr-defined]
            terrain.max_terrain_level, random_fraction,
        )
        env.walk_curriculum_level[env_ids] = level  # type: ignore[attr-defined]
        env.walk_explored[env_ids] = explore  # type: ignore[attr-defined]
        terrain.terrain_levels[env_ids] = run_on
        terrain.env_origins[env_ids] = terrain.terrain_origins[run_on, terrain.terrain_types[env_ids]]
        env.walk_metrics["explore_level"] = terrain.terrain_levels[env.walk_explored].float().mean().item() \
            if env.walk_explored.any() else 0.0  # type: ignore[attr-defined]

    walking = commanded >= min_commanded
    if walking.any():
        env.walk_metrics["walked_fraction"] = (walked[walking] / commanded[walking]).mean().item()  # type: ignore[attr-defined]
        env.walk_metrics["progressed"] = progressed[walking].float().mean().item()  # type: ignore[attr-defined]
    if random_fraction > 0.0:
        return torch.mean(env.walk_curriculum_level.float())  # type: ignore[attr-defined]
    return torch.mean(terrain.terrain_levels.float())


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # in the beginning we reward always moving a foot to teach
    # the model when to use/not the no-op action. Once it has
    # learned this, we remove the reward.
    # remove_foot_in_swing_reward = CurrTerm(
    #     func=mdp.modify_reward_weight,  # type: ignore
    #     params={
    #         "term_name": "a_foot_in_swing",
    #         "weight": 0,
    #         "num_steps": 10000,
    #     },
    # )

    # once the model has learned to move a foot, we add a small penalty on joint torques
    # to encourage energy efficiency
    # add_joint_torque_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,  # type: ignore
    #     params={
    #         "term_name": "joint_accelerations",
    #         "weight": -3e-8,
    #         "num_steps": 10000,
    #     },
    # )

    # # once the robot learns to walk, reward it for moving it's feet less
    # add_no_op_reward = CurrTerm(
    #     func=mdp.modify_reward_weight,  # type: ignore
    #     params={
    #         "term_name": "no_op",
    #         "weight": 0.1,
    #         "num_steps": 15000,
    #     },
    # )

    terrain_levels = CurrTerm(func=vmdp.terrain_levels_vel)
    # terrain_levels = CurrTerm(func=terrain_levels_vel)  # type: ignore
