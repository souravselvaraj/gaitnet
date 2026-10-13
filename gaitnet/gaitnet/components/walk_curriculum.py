"""The rule behind the `walk` terrain curriculum (env_cfg/curriculum.terrain_levels_walk): which robots
move up or down a terrain level when their episode ends. Pure torch, so it can be tested without
Isaac Sim; GaitNetEnv (gaitnet_env.py) keeps the per-episode distances it reads.

Isaac Lab's terrain_levels_vel, the curriculum every run here used, moves a robot up only when it
ends its episode more than half a tile (2 m) from where it started, and down when that net
distance is under half of |final command| x episode length. With commands of at most 0.2 m/s that
change direction every 2.5-10 s, even a robot tracking every command perfectly ends up 1.4 m away
(median) and is moved down 50% of the time and up 22%, so every run sank to level 0 (0-4% voids)
within ~10 iterations whatever the policy did. This rule judges a robot against its own command
instead: the distance walked along the commanded direction, against the distance commanded.
"""
from __future__ import annotations

import torch


def walk_moves(
    time_outs: torch.Tensor,
    failed: torch.Tensor,
    walked: torch.Tensor,
    commanded: torch.Tensor,
    required_fraction: float,
    min_commanded: float = 0.2,
    p_up_given_success: float = 0.1,
    p_down_given_failure: float = 0.5,
    p_random: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(move_up, move_down, progressed), each (n,) bool, for n robots whose episode just ended.

    A robot succeeded when it survived (`time_outs`) and walked at least `required_fraction` of its
    `commanded` distance along the command (episodes that asked for under `min_commanded` m, i.e.
    standing, count survival alone); it moves up with probability `p_up_given_success`. One that
    fell (`failed`), or survived without walking that far, moves down with probability
    `p_down_given_failure`, so the levels settle where about p_up / (p_up + p_down) of the robots
    fall or stall (1/6 at the defaults) rather than where they can merely stand. `p_random` moves
    a robot up and down alike, which leaves it in place.

    walked, commanded: (n,) distance walked along the commanded direction, and commanded (m).
    """
    n = time_outs.shape[0]
    rand = lambda: torch.rand(n, device=time_outs.device)  # noqa: E731
    progressed = (commanded < min_commanded) | (walked >= required_fraction * commanded)
    noise = rand() < p_random
    move_up = (time_outs & progressed & (rand() < p_up_given_success)) | noise
    stalled_or_fell = failed | (time_outs & ~progressed)
    move_down = (stalled_or_fell & (rand() < p_down_given_failure)) | noise
    return move_up, move_down, progressed


def next_levels(
    curriculum_level: torch.Tensor,
    explored: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
    num_levels: int,
    random_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(curriculum level, explore flag, level to run on) for the next episode of n robots.

    Each robot keeps a curriculum level, moved by `walk_moves` only after episodes it ran on that
    level (one it spent on a random level says nothing about its own). A robot that climbs past the
    top goes to a random level, as Isaac Lab's update_env_origins does. Independently, each next
    episode is an exploration episode with probability `random_fraction`, run on a level drawn
    uniformly from all of them: the curriculum settles where few robots fall, which, with falls that
    come from speed more than terrain, kept every run at 5-9% voids while the evaluation asks for
    10-40%; exploration episodes keep the dense terrain in the training mix.

    curriculum_level, explored, move_up, move_down: (n,) long / bool, for the episodes just ended.
    """
    n = curriculum_level.shape[0]
    device = curriculum_level.device
    stepped = curriculum_level + (move_up.long() - move_down.long()) * (~explored).long()
    climbed_out = stepped >= num_levels
    stepped = torch.where(climbed_out, torch.randint(0, num_levels, (n,), device=device), stepped.clamp_min(0))
    explore = torch.rand(n, device=device) < random_fraction
    run_on = torch.where(explore, torch.randint(0, num_levels, (n,), device=device), stepped)
    return stepped, explore, run_on
