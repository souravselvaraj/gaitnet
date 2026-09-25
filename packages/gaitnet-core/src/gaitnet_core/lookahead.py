"""The terrain ahead of the robot, coarsely: what the planner can know beyond its foothold patches.

Each leg's foothold patch ends about 0.25 m ahead of the front feet, while the camera sees
metres further. This summarizes a strip ahead of the base as a small grid of cells, each with
its mean height, the fraction of it that is a hole, and the fraction of it that is known, so the
policy can set its steps up for what is coming (shorten or lengthen strides, choose which leg
reaches an obstacle) without scoring anything there.

Geometry, in the base's yaw frame: x from -0.2 m to 1.4 m (from under the body to well past
the front feet), y from -0.6 m to 0.6 m. The terrain is sampled on a fine grid of FINE_SHAPE
points at FINE_RESOLUTION (cell centres), pooled POOL x POOL into COARSE_SHAPE cells. The
simulator samples it with a ray caster; a robot samples its elevation map at the same points.
"""

from __future__ import annotations

import torch

AHEAD_CENTRE = (0.6, 0.0)
"""Centre of the strip in the base's yaw frame (m)."""
FINE_RESOLUTION = 0.05
FINE_SHAPE = (32, 24)
"""Fine sample points along (x, y): 1.6 m x 1.2 m of 0.05 m cells, x outer, y inner."""
POOL = 4
COARSE_SHAPE = (FINE_SHAPE[0] // POOL, FINE_SHAPE[1] // POOL)
"""8 x 6 cells of 0.2 m."""
CHANNELS = ("height", "holes", "known")
FEATURE_DIM = len(CHANNELS) * COARSE_SHAPE[0] * COARSE_SHAPE[1]
"""144 numbers: channel-major, then x, then y."""
HOLE_DEPTH = 0.05
"""A point this far below the ground reference counts as a hole (m)."""
HEIGHT_CLIP = 0.3
"""Heights relative to the ground reference are clipped to +-this (m)."""


def fine_points(device: torch.device | str | None = None) -> torch.Tensor:
    """(*FINE_SHAPE, 2) the sample points' (x, y) in the base's yaw frame (m)."""
    half_x = (FINE_SHAPE[0] - 1) * FINE_RESOLUTION / 2
    half_y = (FINE_SHAPE[1] - 1) * FINE_RESOLUTION / 2
    x = torch.linspace(-half_x, half_x, FINE_SHAPE[0], device=device) + AHEAD_CENTRE[0]
    y = torch.linspace(-half_y, half_y, FINE_SHAPE[1], device=device) + AHEAD_CENTRE[1]
    return torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)


def terrain_ahead(heights: torch.Tensor, known: torch.Tensor, ground: torch.Tensor) -> torch.Tensor:
    """(N, FEATURE_DIM) the strip's summary.

    Args:
        heights: (N, *FINE_SHAPE) terrain height at each sample point (m, any vertical datum)
        known: (N, *FINE_SHAPE) bool, the height is known (seen, and a ray returned)
        ground: (N,) the height the robot stands on, in the same datum (e.g. its feet)

    Returns, per coarse cell: the mean height of its known points relative to `ground`
    (clipped to +-HEIGHT_CLIP, 0 if none is known), the fraction of its known points more than
    HOLE_DEPTH below `ground`, and the fraction of its points that are known.
    """
    n = heights.shape[0]
    if tuple(heights.shape[1:]) != FINE_SHAPE:
        raise ValueError(f"expected heights of shape (N, {FINE_SHAPE}), got {tuple(heights.shape)}")
    known = known & torch.isfinite(heights)
    relative = heights - ground.view(n, 1, 1)
    relative = torch.where(known, relative, torch.zeros_like(relative))
    hole = known & (relative < -HOLE_DEPTH)
    relative = relative.clamp(-HEIGHT_CLIP, HEIGHT_CLIP)

    def pool(x: torch.Tensor) -> torch.Tensor:  # (N, 32, 24) -> (N, 8, 6) sums
        return x.reshape(n, COARSE_SHAPE[0], POOL, COARSE_SHAPE[1], POOL).sum(dim=(2, 4))

    count = pool(known.float())
    per_cell = count.clamp(min=1.0)
    height = pool(relative) / per_cell
    holes = pool(hole.float()) / per_cell
    known_fraction = count / (POOL * POOL)
    return torch.stack([height, holes, known_fraction], dim=1).reshape(n, FEATURE_DIM)
