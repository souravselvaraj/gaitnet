"""Privileged observations for a GaitNet-T teacher, and the terrain geometry behind the teacher's
edge reward. None of this exists on the robot; a deployable student is distilled without it.

Two kinds of features, appended at the very end of the policy observation so that everything
before them ([state 25 | candidates 65 x 8 | history]) is identical for teacher and student:

- controller knobs (13): the per-env randomisation the controller is running with
  (sim2real/gpu_mpc.ControllerKnobs, plus the liftoff delay), scaled to roughly [-1, 1];
- per-candidate terrain (65 x 2): for every candidate foothold, the distance (m, clipped to
  0.15) to the nearest invalid cell of that leg's height scan, and the terrain height there
  relative to the hip (m, clipped to [-0.6, 0.1]). NO_STEP and padding slots get zeros.

Geometry: each leg's scanner is a 41 x 41 grid of vertical rays at 1 cm around its hip, yaw
aligned. The observation keeps it max-pooled to 37 x 37 (2 cm c-space dilation) and the sampler
draws candidates from cells whose value hip_z - hit_z - 0.5 lies in the valid range (-0.5, 0); a
candidate at observation cell (i, j) (footstep_scanner_constants.xy_to_idx) is raw ray
(i + 2, j + 2). Here "invalid" uses the same range on the RAW (undilated) scan, and distances are
taken between ray positions in the world, so they are exact to the 1 cm ray spacing.
"""

from __future__ import annotations

import torch

import gaitnet.constants as const
from gaitnet.simulation.cfg.footstep_scanner_constants import xy_to_idx

SCANNERS = ("FR_foot_scanner", "FL_foot_scanner", "RR_foot_scanner", "RL_foot_scanner")
"""Height scanners in GaitNet slot order FR, FL, RR, RL."""
SCAN_OFFSET = 0.5
"""The offset the observation subtracts from hip_z - hit_z (observations.cspace_height_scan)."""
MAX_EDGE_DISTANCE = 0.15
HEIGHT_CLIP = (-0.6, 0.1)
KNOB_DIM = 13
TERRAIN_DIM = 2
SLOT_FROM_LEG = [1, 0, 3, 2]


def privileged_dim(num_options: int) -> int:
    return KNOB_DIM + num_options * TERRAIN_DIM


def knob_features(controllers) -> torch.Tensor:
    """(N, 13) the controller knobs, roughly in [-1, 1]: command tracking, planar weight scale,
    swing apex, touchdown bias (4 x 2, slot order), touchdown noise std, liftoff delay."""
    knobs = controllers.controller.knobs
    n = knobs.command_tracking.shape[0]
    return torch.cat(
        [
            (2.0 * knobs.command_tracking - 1.0).unsqueeze(-1),              # [0, 1]
            (2.0 * knobs.planar_weight_scale - 1.0).unsqueeze(-1),           # [0, 1]
            ((knobs.swing_apex - 0.095) / 0.025).unsqueeze(-1),             # [0.07, 0.12] m
            (knobs.touchdown_bias[:, SLOT_FROM_LEG] / 0.02).reshape(n, 8),  # +/- 0.02 m
            (knobs.touchdown_noise / 0.01 - 1.0).unsqueeze(-1),            # [0, 0.02] m
            (controllers.liftoff_delay.to(knobs.swing_apex.dtype) - 1.0).unsqueeze(-1),  # {0, 1, 2}
        ],
        dim=-1,
    )


def scan_geometry(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per leg (slot order), from the raw ray casts:
    ray xy in the world (N, 4, R, 2), terrain height relative to the hip (N, 4, R; NaN where a
    ray hit nothing) and the invalid mask (N, 4, R) with the sampler's valid range."""
    xy, height = [], []
    for name in SCANNERS:
        sensor = env.scene[name]
        hits = sensor.data.ray_hits_w                       # (N, R, 3), refreshes the sensor
        xy.append(sensor._ray_starts_w[..., :2])            # vertical rays: start xy = hit xy
        height.append(hits[..., 2] - sensor.data.pos_w[:, 2:3])
    xy, height = torch.stack(xy, 1), torch.stack(height, 1)
    height = torch.where(torch.isfinite(height), height, torch.full_like(height, float("nan")))
    scan = -height - SCAN_OFFSET                            # hip_z - hit_z - 0.5, as observed
    low, high = const.gait_net.valid_height_range
    invalid = ~((scan > low) & (scan < high))               # NaN (no hit) is invalid too
    return xy, height, invalid


def nearest_invalid(points: torch.Tensor, ray_xy: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
    """(N, 4, P) distance (m) from each of P world xy points per leg to the nearest invalid ray of
    that leg's scan, clipped to MAX_EDGE_DISTANCE (also when the scan has no invalid ray)."""
    n, legs, p, _ = points.shape
    d = torch.cdist(points.reshape(n * legs, p, 2), ray_xy.reshape(n * legs, -1, 2))
    d = d.masked_fill(~invalid.reshape(n * legs, 1, -1), float("inf"))
    return d.min(dim=-1).values.clamp(max=MAX_EDGE_DISTANCE).view(n, legs, p)


def candidate_terrain(env, options: torch.Tensor) -> torch.Tensor:
    """(N, M, 2) per-candidate [edge distance, height relative to hip] for the sampler's options
    (N, M, 4) = (slot leg, dx, dy, cost); zeros for NO_STEP and padding."""
    ray_xy, height, invalid = scan_geometry(env)
    n, m = options.shape[:2]
    grid = int(const.footstep_scanner.sensor_grid_size[1])
    pad = const.gait_net.cspace_dialation
    leg = options[..., 0].long()
    real = (leg >= 0) & (leg < 4)
    indices, _ = xy_to_idx(options[..., 1:3])
    ray = ((indices[..., 0].long() + pad) * grid + indices[..., 1].long() + pad)   # (N, M)
    leg_c = leg.clamp(0, 3)
    batch = torch.arange(n, device=options.device).unsqueeze(-1)
    point = ray_xy[batch, leg_c, ray]                                           # (N, M, 2)
    cand_height = height[batch, leg_c, ray].nan_to_num(HEIGHT_CLIP[0]).clamp(*HEIGHT_CLIP)

    # distances per leg: group the candidates by their leg slot (M = 4K + 1, K per leg in order)
    per_leg = const.gait_net.num_footstep_options
    pts = point[:, : 4 * per_leg].reshape(n, 4, per_leg, 2)
    dist = nearest_invalid(pts, ray_xy, invalid).reshape(n, 4 * per_leg)
    dist = torch.cat([dist, torch.zeros(n, m - 4 * per_leg, device=dist.device)], dim=1)
    # candidates are generated per leg in slot order, but trust the leg column, not the position
    assert per_leg * 4 + 1 == m
    features = torch.stack([dist, cand_height], dim=-1)
    return torch.where(real.unsqueeze(-1), features, torch.zeros_like(features))


def foot_edge_distance(env, foot_xy_w: torch.Tensor) -> torch.Tensor:
    """(N, 4) distance (m, clipped to 0.15) from each foot's world xy, (N, 4, 2) in SLOT order,
    to the nearest invalid cell of its own leg's scan."""
    ray_xy, _, invalid = scan_geometry(env)
    return nearest_invalid(foot_xy_w.unsqueeze(2), ray_xy, invalid).squeeze(2)
