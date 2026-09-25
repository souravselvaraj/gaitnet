"""Reads the planner's view of the robot (gaitnet_core.state) out of an Isaac Lab scene.

Frames follow the core contract: feet relative to the base in its gravity-aligned yaw
frame, velocities in the base frame, terrain heights relative to each hip in the hip's
yaw frame. Isaac Lab 3 quaternions are xyzw, which is also what the controllers take.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster

from gaitnet_core import lookahead
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import RobotSpec
from gaitnet_core.state import RobotState, TerrainPatch


class RobotIO:
    def __init__(
        self,
        scene: InteractiveScene,
        spec: RobotSpec,
        grid: FootholdGrid,
        robot_name: str,
        joint_names: Sequence[str],
        foot_names: Sequence[str],
        contact_sensor_name: str,
        scanner_names: Sequence[str],
        contact_threshold: float,
        ahead_scanner_name: str | None = None,
    ):
        """
        Args:
            joint_names: in the core's leg-major order
            foot_names: in leg order
            scanner_names: one hip-mounted, yaw-aligned grid ray caster per leg, in leg order,
                whose pattern covers `grid.patch_size`
            contact_threshold: normal force above which a foot is in contact (N)
            ahead_scanner_name: the lookahead ray caster (`gaitnet_sim.env.scene.ahead_scanner_cfg`),
                if the scene has one: then states carry `terrain_ahead`
        """
        self.spec = spec
        self.grid = grid
        self.robot: Articulation = scene[robot_name]
        self.contact_sensor: ContactSensor = scene[contact_sensor_name]
        self.scanners: list[RayCaster] = [scene[name] for name in scanner_names]
        self.contact_threshold = contact_threshold
        self.ahead_scanner: RayCaster | None = None
        if ahead_scanner_name is not None and ahead_scanner_name in scene.sensors:
            self.ahead_scanner = scene.sensors[ahead_scanner_name]
            points = lookahead.FINE_SHAPE[0] * lookahead.FINE_SHAPE[1]
            if self.ahead_scanner.num_rays != points:
                raise ValueError(f"the lookahead scanner casts {self.ahead_scanner.num_rays} rays, needs {points}")

        self.joint_ids, _ = self.robot.find_joints(list(joint_names), preserve_order=True)
        self.foot_ids, _ = self.robot.find_bodies(list(foot_names), preserve_order=True)
        self.contact_ids, _ = self.contact_sensor.find_bodies(list(foot_names), preserve_order=True)

        if len(self.scanners) != spec.num_legs:
            raise ValueError(f"expected one scanner per leg, got {list(scanner_names)}")
        rays = grid.patch_size[0] * grid.patch_size[1]
        for name, scanner in zip(scanner_names, self.scanners):
            if scanner.num_rays != rays:
                raise ValueError(
                    f"scanner {name} casts {scanner.num_rays} rays, the foothold grid needs {grid.patch_size}"
                )

    @property
    def num_robots(self) -> int:
        return self.robot.num_instances

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 12) joint positions and velocities, leg-major."""
        data = self.robot.data
        return data.joint_pos.torch[:, self.joint_ids], data.joint_vel.torch[:, self.joint_ids]

    def base_pose(self) -> torch.Tensor:
        """(N, 7) base position and xyzw orientation, world frame."""
        return self.robot.data.root_link_pose_w.torch

    def base_vel(self) -> torch.Tensor:
        """(N, 6) base linear then angular velocity, world frame."""
        return self.robot.data.root_link_vel_w.torch

    def robot_state(self, gait_timing: torch.Tensor, command: torch.Tensor, base_command: torch.Tensor) -> RobotState:
        """
        Args:
            gait_timing: (N, L, 3) the controller's schedule
            command: (N, 3) the velocity command the controller is tracking
            base_command: (N, 3) the command before any nudge
        """
        data = self.robot.data
        base_pos = data.root_link_pos_w.torch
        base_quat = data.root_link_quat_w.torch
        num_legs = len(self.foot_ids)

        foot_offset_w = data.body_link_pos_w.torch[:, self.foot_ids] - base_pos.unsqueeze(1)
        yaw = math_utils.yaw_quat(base_quat).unsqueeze(1).expand(-1, num_legs, -1)
        foot_pos = math_utils.quat_apply_inverse(yaw, foot_offset_w)

        # relative to the base, as leg kinematics measure it: excludes the base's own motion
        base_ang_vel_w = data.root_link_ang_vel_w.torch.unsqueeze(1).expand_as(foot_offset_w)
        foot_vel_w = (
            data.body_link_lin_vel_w.torch[:, self.foot_ids]
            - data.root_link_lin_vel_w.torch.unsqueeze(1)
            - torch.cross(base_ang_vel_w, foot_offset_w, dim=-1)
        )
        foot_vel = math_utils.quat_apply_inverse(base_quat.unsqueeze(1).expand(-1, num_legs, -1), foot_vel_w)

        forces = self.contact_sensor.data.net_normal_forces_w.torch[:, self.contact_ids]
        return RobotState(
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            base_lin_vel=data.root_link_lin_vel_b.torch,
            base_ang_vel=data.root_link_ang_vel_b.torch,
            projected_gravity=data.projected_gravity_b.torch,
            contact=forces.norm(dim=-1) > self.contact_threshold,
            gait_timing=gait_timing,
            command=command,
            base_command=base_command,
            terrain_ahead=self.terrain_ahead() if self.ahead_scanner is not None else None,
        )

    FOOT_RADIUS = 0.02
    """Go1's foot sphere (m): a planted foot's centre is this far above the ground."""

    def ground_height(self) -> torch.Tensor:
        """(N,) the height the robot stands on (m, world): its feet, less their radius."""
        return self.robot.data.body_link_pos_w.torch[:, self.foot_ids, 2].mean(dim=1) - self.FOOT_RADIUS

    def ahead_samples(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, *lookahead.FINE_SHAPE, 2) world xy and (N, *FINE_SHAPE) world z of the lookahead
        points, z -inf where a ray found nothing."""
        hits = self.ahead_scanner.data.ray_hits_w.torch.reshape(-1, *lookahead.FINE_SHAPE, 3)
        z = torch.where(torch.isfinite(hits[..., 2]), hits[..., 2], torch.full_like(hits[..., 2], float("-inf")))
        return hits[..., :2], z

    def terrain_ahead(self) -> torch.Tensor:
        """(N, lookahead.FEATURE_DIM) the exact terrain ahead (every returned point known)."""
        _, z = self.ahead_samples()
        return lookahead.terrain_ahead(z, torch.isfinite(z), self.ground_height())

    def foot_heights(self) -> torch.Tensor:
        """(N, L) each foot's height relative to its hip (m), vertical, as `terrain()` measures
        heights."""
        feet_z = self.robot.data.body_link_pos_w.torch[:, self.foot_ids, 2]
        hips_z = torch.stack([scanner.data.pos_w.torch[:, 2] for scanner in self.scanners], dim=1)
        return feet_z - hips_z

    def terrain_points_xy(self) -> torch.Tensor:
        """(N, L, *patch_size, 2) world xy of every terrain patch cell (m), where its ray hit."""
        points = [scanner.data.ray_hits_w.torch[..., :2] for scanner in self.scanners]
        return torch.stack(points, dim=1).reshape(-1, len(self.scanners), *self.grid.patch_size, 2)

    def terrain(self) -> TerrainPatch:
        """Terrain heights relative to each hip, -inf where a ray found nothing."""
        patches = []
        for scanner in self.scanners:
            hits_z = scanner.data.ray_hits_w.torch[..., 2]
            # the scanner's frame is its hip link; its 20 m offset only lifts the ray starts
            heights = hits_z - scanner.data.pos_w.torch[:, 2:3]
            heights = torch.where(torch.isfinite(heights), heights, torch.full_like(heights, float("-inf")))
            patches.append(heights.reshape(-1, *self.grid.patch_size))
        return TerrainPatch(heights=torch.stack(patches, dim=1), grid=self.grid)
