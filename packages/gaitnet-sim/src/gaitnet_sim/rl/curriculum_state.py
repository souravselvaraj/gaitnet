"""Carry the terrain curriculum across a resume.

Isaac Lab keeps each robot's terrain level in the TerrainImporter, not in anything RSL-RL
checkpoints, so every resumed run used to start from the importer's initial spread (levels
uniform over all rows, a mean difficulty of about 0.23) and spend its first 1500-2000 iterations
climbing back to where it had been (t3 ended at 0.29 and resumed at 0.23). This stores the
levels in each checkpoint's `infos` and restores their distribution on load.

Levels are restored as a distribution, not robot by robot: a checkpoint is written by rank 0
alone, and a resumed run may have another robot count or rank count. Every rank draws its
robots' levels from the saved empirical distribution. The terrain type (column) of a robot is
fixed by its index, so only the level (row) is carried.

A checkpoint written before this existed has no levels; `initial_levels(lo, hi)` then spreads
the robots uniformly over rows lo..hi instead (e.g. 3..11 of 12 rows for a mean near level 7).
"""

from __future__ import annotations

import torch

INFOS_KEY = "terrain_levels"


def terrain_of(env):
    """The env's curriculum terrain (an Isaac Lab TerrainImporter), or None if it has no
    level curriculum (grid spawning, flat ground, or a vec-env wrapper around one)."""
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    terrain = getattr(scene, "terrain", None) if scene is not None else None
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
        return None
    if getattr(terrain, "terrain_levels", None) is None:
        return None
    return terrain


def saved_levels(env) -> torch.Tensor | None:
    """This env's current terrain levels on the CPU, for a checkpoint."""
    terrain = terrain_of(env)
    return None if terrain is None else terrain.terrain_levels.detach().to("cpu", torch.long).clone()


def resample(levels: torch.Tensor, num_envs: int, max_level: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """num_envs levels drawn from the empirical distribution of `levels`, clipped to the rows."""
    levels = levels.to(torch.long).flatten()
    if levels.numel() == 0:
        raise ValueError("no saved terrain levels to resample")
    picks = torch.randint(0, levels.numel(), (num_envs,), generator=generator)
    return levels[picks].clamp(0, max_level - 1)


def initial_levels(lo: int, hi: int, num_envs: int, max_level: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """num_envs levels uniform over rows lo..hi (inclusive), clipped to the rows."""
    lo, hi = max(0, int(lo)), min(max_level - 1, int(hi))
    if lo > hi:
        raise ValueError(f"empty terrain level range {lo}..{hi} for {max_level} rows")
    return torch.randint(lo, hi + 1, (num_envs,), generator=generator)


def apply_levels(env, levels: torch.Tensor) -> bool:
    """Put the env's robots on these levels: their next reset spawns there. False if the env has
    no level curriculum."""
    terrain = terrain_of(env)
    if terrain is None:
        return False
    levels = levels.to(terrain.terrain_levels.device, terrain.terrain_levels.dtype)
    if levels.shape != terrain.terrain_levels.shape:
        raise ValueError(f"{levels.shape[0]} levels for {terrain.terrain_levels.shape[0]} robots")
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return True


def parse_range(text: str) -> tuple[int, int]:
    """'3:11' -> (3, 11)."""
    try:
        lo, hi = (int(part) for part in text.split(":"))
    except ValueError as error:
        raise ValueError(f"expected LO:HI terrain levels, got {text!r}") from error
    return lo, hi
