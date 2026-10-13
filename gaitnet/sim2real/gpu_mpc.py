"""The low-level controller batched on the GPU, set up to stand in for the CPU one here.

`gaitnet/control` (rl-mpc-locomotion @ dbe115f) runs one convex MPC per robot in a CPU
process pool (`VectorPool[SimInterface]`), which bounds training throughput by the CPU
count. `gaitnet_core.control` (from the gaitnet repo, installed as a pinned snapshot of
commit c79ab67, see cluster/setup_train_env.sh) is a batched torch copy of the same
controller that runs every robot at once. It copies gaitnet-mpc, which is this
rl-mpc-locomotion plus some later changes, so `VARIANTS["gaitnet"]` switches those back:

| gaitnet-mpc / gaitnet_core                         | here (dbe115f), `variant="gaitnet"`       |
| -------------------------------------------------- | ----------------------------------------- |
| mass 13.10 kg, measured inertia x2                 | mass 10.41 kg, older inertia x5           |
| weights as listed, wx/wy weighted 0.4              | weights x10, wx/wy weighted 0             |
| velocity command rate limited to 3 m/s^2           | no rate limit                             |
| target shifted back by body travel over the swing  | no shift                                  |
| touchdown z from the command (+ foot radius)       | touchdown z on the stance plane           |
| swing apex above the higher of liftoff/touchdown   | swing apex above liftoff                  |

What stays different whichever variant is chosen is what gaitnet_core changes on
purpose (its control/README.md): a fixed budget of batched ADMM iterations instead of
qpOASES, float32 instead of float16 rotations, and one MPC update phase for the batch.

Everything here is in the controller's own conventions: leg order FL, FR, RL, RR,
leg-major joints (hip, thigh, calf), xyzw quaternions, base-frame velocities. The Isaac
Lab side (joint order, wxyz, footstep slots, resets) is `robot_controllers.py`.

On top of gaitnet_core this adds per-robot knobs for controller randomisation, all of
which reproduce the unrandomised controller exactly at their defaults, and the per-step
telemetry a history buffer would read.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from gaitnet_core.control import BatchedMpcConfig, BatchedMpcController, BatchedConvexMpc, kinematics
from gaitnet_core.control.model import GO1_SRBD, SrbdModel
from gaitnet_core.interfaces import FootstepCommand

GO1_DBE115F = dataclasses.replace(
    GO1_SRBD,
    # Quadruped(RobotType.GO1) in gaitnet/control/mpc/common/Quadruped.py
    mass=5.204 * 2,
    inertia=(0.0168128557 * 5, 0.063009565 * 5, 0.0716547275 * 5),
    weights=tuple(
        10.0 * w for w in (1.0, 1.5, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.1, 0.0)
    ),
    # ConvexMPCLocomotion.setup_command takes the command as it is
    max_linear_accel=math.inf,
)
"""The SRBD model and gains of the CPU controller vendored in gaitnet/control."""


@dataclass(frozen=True)
class Variant:
    """Which controller the GPU one reproduces."""

    model: SrbdModel
    compensate_body_travel: bool
    """Shift the touchdown target back by the body's travel over the swing."""
    touchdown_on_stance_plane: bool
    """Put the touchdown target's z on the stance plane (z = 0 in the estimator frame,
    `foot_position_global[2] = 0` in SpecifiedFootstepLocomotion) instead of the
    commanded height."""
    apex_above_start: bool
    """Swing apex clears the liftoff point rather than the higher end of the swing."""


VARIANTS: dict[str, Variant] = {
    "gaitnet": Variant(
        GO1_DBE115F,
        compensate_body_travel=False,
        touchdown_on_stance_plane=True,
        apex_above_start=True,
    ),
    "gaitnet_core": Variant(
        GO1_SRBD,
        compensate_body_travel=True,
        touchdown_on_stance_plane=False,
        apex_above_start=False,
    ),
}
"""`gaitnet`: the CPU controller in this repo, which every GaitNet checkpoint here was
trained against. `gaitnet_core`: gaitnet_core.control unchanged (gaitnet-mpc's tuning)."""

PLANAR_STATES = (2, 3, 4, 8, 9, 10)
"""MPC state indices of the planar motion: yaw, x, y, yaw rate, vx, vy."""


@dataclass
class ControllerKnobs:
    """Per-robot randomisation knobs. Each default reproduces the plain controller."""

    command_tracking: torch.Tensor
    """(N,) scale on the (vx, vy, yaw rate) reference the stance MPC tracks. 1 tracks
    the command (the MPC pulls the body along with the stance feet); 0 references zero
    velocity, so the stance feet hold the body where it is."""
    planar_weight_scale: torch.Tensor
    """(N,) scale on the MPC's planar tracking weights (yaw, x, y, yaw rate, vx, vy).
    1 as tuned; 0 leaves planar motion untracked (the body coasts, only height and
    attitude are held)."""
    swing_apex: torch.Tensor
    """(N,) swing apex clearance (m). Default nominal height / 3 = 0.0867 m."""
    touchdown_bias: torch.Tensor
    """(N, L, 2) added to every commanded hip-frame (x, y) foothold (m). Default 0."""
    touchdown_noise: torch.Tensor
    """(N,) std of Gaussian noise added to each commanded (x, y) foothold (m). Default
    0. Only drawn when `noise_enabled`, from the controller's own generator."""
    travel_compensation: torch.Tensor | None = None
    """(N,) fraction of the body's travel over the swing taken off the touchdown target,
    for a variant that does not compensate it: 0 (default) executes the foothold against
    the hip at touchdown, so the foot lands about speed x swing time past it; 1 lands it
    where the hip was when it was commanded, the convention of gaitnet_core and OCS2."""
    noise_enabled: bool = False

    @classmethod
    def defaults(
        cls, num_robots: int, legs: int, apex: float, device: torch.device, dtype: torch.dtype
    ) -> ControllerKnobs:
        def full(value: float, *shape: int) -> torch.Tensor:
            return torch.full((num_robots, *shape), value, device=device, dtype=dtype)

        return cls(
            command_tracking=full(1.0),
            planar_weight_scale=full(1.0),
            swing_apex=full(apex),
            touchdown_bias=full(0.0, legs, 2),
            touchdown_noise=full(0.0),
            travel_compensation=full(0.0),
        )


class _KnobbedConvexMpc(BatchedConvexMpc):
    """`BatchedConvexMpc` with per-robot command tracking and planar weights."""

    def __init__(
        self,
        model: SrbdModel,
        settings,
        num_robots: int,
        control_dt: float,
        device: torch.device,
        dtype: torch.dtype,
        knobs: ControllerKnobs,
    ) -> None:
        super().__init__(model, settings, num_robots, control_dt, device, dtype)
        self.knobs = knobs
        self._base_weights = self._weights.clone()

    def solve(self, *, desired_velocity: torch.Tensor, desired_yaw_rate: torch.Tensor, **state):
        knobs = self.knobs
        scale = knobs.command_tracking
        weight_scale = torch.ones_like(self._base_weights)
        weight_scale[:, PLANAR_STATES] = knobs.planar_weight_scale.unsqueeze(-1)
        self._weights = self._base_weights * weight_scale
        return super().solve(
            desired_velocity=desired_velocity * scale.unsqueeze(-1),
            desired_yaw_rate=desired_yaw_rate * scale,
            **state,
        )


class GaitnetMpcController(BatchedMpcController):
    """gaitnet_core's batched controller as a `Variant`, with per-robot knobs and telemetry.

    Telemetry, refreshed by every `compute_torques` (one physics step), all (N, ...) on
    the controller's device, leg order FL, FR, RL, RR, base frame = the body frame:

    - `foot_position_hip` (N, L, 3): each foot (sphere centre) relative to its hip (m),
      from the joint encoders.
    - `foot_velocity` (N, L, 3): its velocity relative to the body (m/s).
    - `base_linear_velocity`, `base_angular_velocity` (N, 3): the body twist, base frame.
    - `touchdown_position_hip` (N, L, 3): where each foot was, relative to its hip, on
      the control step its last scheduled swing ended; `touchdown_count` (N, L) counts
      those touchdowns since reset, so a reader can tell a new one from an old one.
    - `touchdown_command` (N, L, 2): `commanded_foothold` as it was at that touchdown, and
      `touchdown_error` (N, L, 2) the achieved minus that commanded hip-frame (x, y), 0
      for a leg that has not touched down since reset.
    - `commanded_foothold` (N, L, 2): the last hip-frame (x, y) commanded per leg,
      before bias and noise; `applied_foothold` (N, L, 2) is it after them.
    - `touchdown_target` (N, L, 3): where each foot is aimed, estimator frame (the
      origin is under the body, z = 0 is the stance plane).
    """

    def __init__(
        self,
        cfg: BatchedMpcConfig,
        variant: Variant,
        num_robots: int,
        dt: float,
        device: str | torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(dataclasses.replace(cfg, model=variant.model), num_robots, dt, device)
        self.variant = variant
        model = self.model
        legs = model.num_legs

        self.knobs = ControllerKnobs.defaults(
            num_robots,
            legs,
            model.spec.nominal_height * model.swing_apex_fraction,
            self.device,
            self.dtype,
        )
        self._mpc = _KnobbedConvexMpc(
            model, self.cfg.mpc, num_robots, dt, self.device, self.dtype, self.knobs
        )
        self.generator = generator

        def per_foot(width: int = 3) -> torch.Tensor:
            return torch.zeros(num_robots, legs, width, device=self.device, dtype=self.dtype)

        self.foot_position_hip = per_foot()
        self.foot_velocity = per_foot()
        self.base_linear_velocity = torch.zeros(num_robots, 3, device=self.device, dtype=self.dtype)
        self.base_angular_velocity = torch.zeros_like(self.base_linear_velocity)
        self.touchdown_position_hip = per_foot()
        self.touchdown_count = torch.zeros(num_robots, legs, device=self.device, dtype=torch.long)
        self.touchdown_command = per_foot(2)
        self.commanded_foothold = self._footstep_offsets[..., :2].clone()
        self._was_swinging = torch.zeros(num_robots, legs, device=self.device, dtype=torch.bool)

    # -- schedule ------------------------------------------------------------------------

    def contact_state(self) -> torch.Tensor:
        """(N, L) bool, the schedule's stance legs: what `CalculatedGait.getContactPhase`
        returns. A commanded leg reads False from the moment it is commanded until its
        scheduled touchdown, however the foot is actually doing."""
        return self._gait.in_contact()

    def swing_phase(self) -> torch.Tensor:
        """(N, L) progress through the current swing in [0, 1), 0 in stance."""
        return self._gait.swing_phase()

    @property
    def touchdown_target(self) -> torch.Tensor:
        """(N, L, 3) where each foot is aimed, estimator frame (origin under the body,
        z = 0 the stance plane for the "gaitnet" variant) (m)."""
        return self._swing_end

    @property
    def applied_foothold(self) -> torch.Tensor:
        """(N, L, 2) the hip-frame (x, y) each leg is aimed at: `commanded_foothold` plus
        the touchdown bias and noise (m)."""
        return self._footstep_offsets[..., :2]

    @property
    def touchdown_error(self) -> torch.Tensor:
        """(N, L, 2) achieved minus commanded hip-frame (x, y) at each leg's last
        touchdown (m); 0 before a leg's first touchdown since reset."""
        error = self.touchdown_position_hip[..., :2] - self.touchdown_command
        return torch.where((self.touchdown_count > 0).unsqueeze(-1), error, torch.zeros_like(error))

    def foot_positions(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """(N, L, 3) foot positions relative to their hips, body frame, from (N, 12)
        leg-major joint angles: the encoder-only kinematics the controller uses."""
        position, _ = kinematics.foot_position_and_jacobian(
            joint_pos.to(self.dtype).view(-1, self.model.num_legs, 3), self._leg_offsets
        )
        return position

    @property
    def height_estimate(self) -> torch.Tensor:
        """(N,) base height above the stance feet the MPC tracks from (m)."""
        return self._height

    # -- overrides -----------------------------------------------------------------------

    def command_footsteps(self, footsteps: FootstepCommand) -> None:
        index = footsteps.leg.view(-1, 1, 1).expand(-1, 1, 2)
        target = footsteps.target.to(self.dtype)
        held = self.commanded_foothold.gather(1, index)
        chosen = torch.where(footsteps.active.view(-1, 1, 1), target[:, None, :2], held)
        self.commanded_foothold.scatter_(1, index, chosen)

        knobs = self.knobs
        bias = knobs.touchdown_bias.gather(1, index).squeeze(1)
        xy = target[:, :2] + bias
        if knobs.noise_enabled:
            noise = torch.randn(
                xy.shape, device=self.device, dtype=self.dtype, generator=self.generator
            )
            xy = xy + noise * knobs.touchdown_noise.unsqueeze(-1)
        target = torch.cat([xy, target[:, 2:]], dim=-1)
        super().command_footsteps(dataclasses.replace(footsteps, target=target))

    def _estimate(self, base_pose: torch.Tensor, base_vel: torch.Tensor):
        estimate = super()._estimate(base_pose, base_vel)
        self.base_linear_velocity = estimate[3]
        self.base_angular_velocity = estimate[4]
        return estimate

    def _touchdown_targets(
        self, linear_velocity: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        if self.variant.compensate_body_travel:
            targets = super()._touchdown_targets(linear_velocity, position)
        else:
            targets = self._hip_offsets + self._footstep_offsets + position.unsqueeze(1)
            alpha = self.knobs.travel_compensation
            if alpha is not None:  # 0 leaves the target exactly as it was
                # gaitnet_core's travel term (controller.py), scaled per robot
                travel = linear_velocity[:, None, :2] * self._gait.swing_duration.unsqueeze(-1)
                limit = self.model.max_body_travel_compensation
                travel = travel * (limit / travel.norm(dim=-1, keepdim=True).clamp_min(limit))
                targets = targets.clone()
                targets[..., :2] -= alpha.view(-1, 1, 1).to(targets.dtype) * travel
        if self.variant.touchdown_on_stance_plane:
            targets = torch.cat([targets[..., :2], torch.zeros_like(targets[..., 2:])], dim=-1)
        return targets

    def _track_swing(
        self, swinging: torch.Tensor, swing_phase: torch.Tensor, foot_in_world: torch.Tensor
    ) -> None:
        starting = swinging & self._first_swing
        self._swing_start = torch.where(starting.unsqueeze(-1), foot_in_world, self._swing_start)
        self._first_swing = ~swinging

        position, velocity = swing_trajectory(
            self._swing_start,
            self._swing_end,
            self.knobs.swing_apex.unsqueeze(-1),
            swing_phase,
            self._gait.swing_duration,
            apex_above_start=self.variant.apex_above_start,
        )
        airborne = swinging.unsqueeze(-1)
        self._foot_target = torch.where(airborne, position, self._foot_target)
        self._foot_target_velocity = torch.where(airborne, velocity, self._foot_target_velocity)

    def _leg_torques(
        self,
        swinging: torch.Tensor,
        position: torch.Tensor,
        foot_position: torch.Tensor,
        foot_velocity: torch.Tensor,
        jacobian: torch.Tensor,
        joint_vel: torch.Tensor,
        linear_velocity: torch.Tensor,
    ) -> torch.Tensor:
        self.foot_position_hip = foot_position
        self.foot_velocity = foot_velocity
        landed = self._was_swinging & ~swinging
        self.touchdown_position_hip = torch.where(
            landed.unsqueeze(-1), foot_position, self.touchdown_position_hip
        )
        self.touchdown_count = self.touchdown_count + landed.long()
        self.touchdown_command = torch.where(
            landed.unsqueeze(-1), self.commanded_foothold, self.touchdown_command
        )
        self._was_swinging = swinging
        return super()._leg_torques(
            swinging, position, foot_position, foot_velocity, jacobian, joint_vel, linear_velocity
        )

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        """Controller state back to standing, nothing commanded. Knobs are left alone;
        `GpuRobotControllers.reset` resamples them."""
        if robot_ids is None:
            robot_ids = torch.arange(self.num_robots, device=self.device)
        super().reset(robot_ids)
        self.commanded_foothold[robot_ids] = self._nominal_offsets[0, :, :2]
        self.foot_position_hip[robot_ids] = 0.0
        self.foot_velocity[robot_ids] = 0.0
        self.base_linear_velocity[robot_ids] = 0.0
        self.base_angular_velocity[robot_ids] = 0.0
        self.touchdown_position_hip[robot_ids] = 0.0
        self.touchdown_count[robot_ids] = 0
        self.touchdown_command[robot_ids] = 0.0
        self._was_swinging[robot_ids] = False


def _bezier(start: torch.Tensor, end: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return start + (phase**3 + 3.0 * phase**2 * (1.0 - phase)) * (end - start)


def _bezier_rate(start: torch.Tensor, end: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return 6.0 * phase * (1.0 - phase) * (end - start)


def swing_trajectory(
    start: torch.Tensor,
    end: torch.Tensor,
    apex_clearance: torch.Tensor | float,
    phase: torch.Tensor,
    duration: torch.Tensor,
    apex_above_start: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`gaitnet_core.control.swing.swing_trajectory`, op for op, with the apex optionally
    above the liftoff point (FootSwingTrajectory in gaitnet/control) and a per-robot
    clearance ((N, 1) broadcasts over legs)."""
    phase = phase.unsqueeze(-1)
    inverse_duration = 1.0 / duration.clamp_min(1e-6).unsqueeze(-1)

    position = _bezier(start, end, phase)
    velocity = _bezier_rate(start, end, phase) * inverse_duration

    base = start[..., 2] if apex_above_start else torch.maximum(start[..., 2], end[..., 2])
    apex = base + apex_clearance
    rising = phase[..., 0] < 0.5
    half_start = torch.where(rising, start[..., 2], apex)
    half_end = torch.where(rising, apex, end[..., 2])
    half_phase = torch.where(rising, phase[..., 0] * 2.0, phase[..., 0] * 2.0 - 1.0)
    height = _bezier(half_start, half_end, half_phase)
    height_rate = _bezier_rate(half_start, half_end, half_phase) * 2.0 * inverse_duration[..., 0]

    position = torch.cat([position[..., :2], height.unsqueeze(-1)], dim=-1)
    velocity = torch.cat([velocity[..., :2], height_rate.unsqueeze(-1)], dim=-1)
    return position, velocity
