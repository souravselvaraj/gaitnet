"""What a front depth camera's elevation map knows of the terrain, for sim2real.

The deployed robot maps terrain with one forward-looking depth camera (an Intel D435i) into
an elevation map. The planner's terrain patches are cut from that map, so a cell is known
only if it was once in the camera's view: the ground right under the front feet sits below
the camera's view (it was seen while still ahead), and the ground under the hind feet, or
beside and behind the robot, is only ever memory. Training's ray casters see every cell
exactly, which would teach the policy to trust terrain the robot never saw.

`FrontCameraMap` keeps, per robot, a robot-centric map, like the robot's own (elevation
mapping keeps a window around the robot and forgets what falls out of it): which cells the
camera has seen and the height error its depth measurements left there. Each env step it marks
the cells inside the camera's view (assuming the ground lies at the height of the robot's feet,
and ignoring occlusion) and fuses a fresh depth error into them, whose standard deviation grows
with the square of the range, as a stereo camera's does. The planner's patches then read the
truth plus that error where the map knows the cell, and unknown (-inf) elsewhere. Rewards,
terminations and privileged observations keep reading the truth.

The window rolls with the robot: a cell is stored at its world index modulo the window size,
tagged with that index, so it stays known until the robot has moved a whole window away and a
new cell takes its slot.

Pure torch, so it runs without the simulator; the action term owns one.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass


@configclass
class FrontCameraCfg:
    """A front depth camera and the elevation map it feeds. The defaults are the D435 mount
    in legged_perceptive's robot description (0.36 m ahead of the base, 0.38 rad down) and a
    D435i's depth specification."""

    mount_pos: tuple[float, float, float] = (0.36, 0.0, 0.01)
    """Camera position in the base frame (m)."""
    pitch_down: float = 0.38
    """Optical axis below the base's forward axis (rad)."""
    horizontal_fov: float = math.radians(87.0)
    vertical_fov: float = math.radians(58.0)
    min_range: float = 0.28
    """Closer than this the camera measures nothing (m)."""
    max_range: float = 3.0
    """Farther than this the map ignores measurements (m)."""
    depth_noise: float = 0.004
    """Standard deviation of one measurement's height error per metre squared of range
    (m / m^2): about 4 mm at 1 m and 16 mm at 2 m. Repeated measurements are fused."""
    map_resolution: float = 0.03
    """Cell size of the elevation map (m), as on the robot."""
    map_size: float = 5.0
    """Side of the square window the map keeps around the robot (m), as on the robot."""
    known_at_reset: float = 0.6
    """Half-side of the square around the spawn point the map starts out knowing (m): the
    spawn platform, as the robot's map starts with the ground it stands on. A little over the
    1 m platform, out to where the camera's view first reaches the ground, or slow walking
    leaves a strip no frame ever saw."""


class FrontCameraMap:
    """Per-robot rolling map of seen cells and their fused height error, see the module
    docstring."""

    def __init__(self, cfg: FrontCameraCfg, num_robots: int, device: torch.device | str):
        self.cfg = cfg
        self.cells = int(round(cfg.map_size / cfg.map_resolution))
        self.device = torch.device(device)
        shape = (num_robots, self.cells, self.cells)
        self.key = torch.full((*shape, 2), torch.iinfo(torch.int64).min, dtype=torch.int64, device=device)
        """(N, C, C, 2) world cell index held in each slot; a lookup of another cell there is
        unknown."""
        self.known = torch.zeros(shape, dtype=torch.bool, device=device)
        """(N, C, C) the camera has seen the cell in the slot."""
        self.error = torch.zeros(shape, device=device)
        """(N, C, C) fused height error the map holds for the cell (m)."""
        self.variance = torch.full(shape, float("inf"), device=device)
        """(N, C, C) variance of that error (m^2), inf where unseen."""
        steps = torch.arange(self.cells, device=device) - self.cells // 2
        # (C, C, 2) window offsets in cells, index 0 along x
        self._window = torch.stack(torch.meshgrid(steps, steps, indexing="ij"), dim=-1)

    def _index(self, xy: torch.Tensor) -> torch.Tensor:
        """World cell index of world points (..., 2)."""
        return torch.floor(xy / self.cfg.map_resolution).long()

    def _window_cells(self, centre_xy: torch.Tensor) -> torch.Tensor:
        """(N, C, C, 2) world cell indices of the window around `centre_xy` (N, 2)."""
        return self._index(centre_xy).view(-1, 1, 1, 2) + self._window

    def _slots(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row, x and y storage indices of world cell indices (N, ..., 2)."""
        rows = torch.arange(index.shape[0], device=index.device).view(-1, *([1] * (index.dim() - 2)))
        slot = torch.remainder(index, self.cells)
        return rows.expand(index.shape[:-1]), slot[..., 0], slot[..., 1]

    def reset(self, robot_ids: torch.Tensor, origins_xy: torch.Tensor) -> None:
        """Forget everything the robots `robot_ids` saw, knowing only the spawn platform
        around `origins_xy` (len(ids), 2), exactly."""
        self.key[robot_ids] = torch.iinfo(torch.int64).min
        self.known[robot_ids] = False
        self.error[robot_ids] = 0.0
        self.variance[robot_ids] = float("inf")
        index = self._window_cells(origins_xy.to(torch.float32))
        centres = (index.float() + 0.5) * self.cfg.map_resolution
        platform = ((centres - origins_xy.view(-1, 1, 1, 2)).abs() <= self.cfg.known_at_reset).all(dim=-1)
        rows, sx, sy = self._slots(index)
        rows = torch.as_tensor(robot_ids, device=self.device).view(-1, 1, 1).expand_as(sx)
        self.key[rows[platform], sx[platform], sy[platform]] = index[platform]
        self.known[rows[platform], sx[platform], sy[platform]] = True
        self.variance[rows[platform], sx[platform], sy[platform]] = 0.0

    def visible(self, base_pos: torch.Tensor, base_quat: torch.Tensor, ground_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Which cells of the window around the robot the camera sees now.

        Args:
            base_pos: (N, 3) world position of the base (m)
            base_quat: (N, 4) world orientation of the base, xyzw
            ground_z: (N,) height the ground is taken to be at (m), e.g. the feet's

        Returns:
            (N, C, C, 2) world cell indices of the window, (N, C, C) bool seen, and
            (N, C, C) range from the camera (m)
        """
        cfg = self.cfg
        n = base_pos.shape[0]
        index = self._window_cells(base_pos[:, :2])
        cells_w = (index.to(base_pos.dtype) + 0.5) * cfg.map_resolution
        points_w = torch.cat([cells_w, ground_z.view(n, 1, 1, 1).expand(n, self.cells, self.cells, 1)], dim=-1)
        rotation = _quat_xyzw_to_matrix(base_quat)  # (N, 3, 3) base -> world
        camera_w = base_pos + torch.einsum("nij,j->ni", rotation, torch.tensor(cfg.mount_pos, device=base_pos.device))
        # the point relative to the camera, in the base frame
        relative = torch.einsum("nji,nabj->nabi", rotation, points_w - camera_w.view(n, 1, 1, 3))
        c, s = math.cos(cfg.pitch_down), math.sin(cfg.pitch_down)
        forward = relative[..., 0] * c - relative[..., 2] * s
        up = relative[..., 0] * s + relative[..., 2] * c
        left = relative[..., 1]
        distance = relative.norm(dim=-1)
        seen = (
            (forward > 0)
            & (torch.atan2(left.abs(), forward) <= cfg.horizontal_fov / 2)
            & (torch.atan2(up.abs(), forward) <= cfg.vertical_fov / 2)
            & (distance >= cfg.min_range)
            & (distance <= cfg.max_range)
        )
        return index, seen, distance

    def update(self, base_pos: torch.Tensor, base_quat: torch.Tensor, ground_z: torch.Tensor) -> None:
        """Fuse one depth frame: every cell in view gets a fresh measurement error."""
        index, seen, distance = self.visible(base_pos, base_quat, ground_z)
        rows, sx, sy = self._slots(index)
        # a slot holding another cell is a cell never seen
        current = (self.key[rows, sx, sy] == index).all(dim=-1)
        variance = torch.where(current, self.variance[rows, sx, sy], torch.full_like(distance, float("inf")))
        error = torch.where(current, self.error[rows, sx, sy], torch.zeros_like(distance))
        measured = (self.cfg.depth_noise * distance.square()).square().clamp(min=1e-10)
        sample = torch.randn_like(distance) * measured.sqrt()
        # inverse-variance fusion, as an elevation map's Kalman update; an unseen cell takes
        # the measurement as it is
        unseen = torch.isinf(variance)
        fused_variance = torch.where(unseen, measured, variance * measured / (variance + measured))
        fused_error = torch.where(unseen, sample, (error * measured + sample * variance) / (variance + measured))
        r, x, y = rows[seen], sx[seen], sy[seen]
        self.key[r, x, y] = index[seen]
        self.known[r, x, y] = True
        self.error[r, x, y] = fused_error[seen]
        self.variance[r, x, y] = fused_variance[seen]

    def lookup(self, points_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What the map holds at world points.

        Args:
            points_xy: (N, ..., 2) world xy (m)

        Returns:
            (N, ...) bool known, and (N, ...) height error (m), 0 where unknown
        """
        index = self._index(points_xy)
        rows, sx, sy = self._slots(index)
        known = self.known[rows, sx, sy] & (self.key[rows, sx, sy] == index).all(dim=-1)
        error = torch.where(known, self.error[rows, sx, sy], torch.zeros_like(known, dtype=self.error.dtype))
        return known, error


def _quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) rotation matrices from (N, 4) xyzw quaternions."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)  # fmt: skip
