"""The env's low-level controllers for all robots at once, behind one batched-tensor API.

`GaitNetEnvCfg.robot_controllers` holds one of these. The action terms, observations and
rewards talk to it only through `RobotControllers`, so they do not care where the MPC
runs:

- `PooledRobotControllers`: the CPU path, one `SimInterface` (rl-mpc-locomotion @ dbe115f)
  per robot in a `VectorPool` of worker processes. Default; behaviour unchanged.
- `GpuRobotControllers`: `gpu_mpc.GaitnetMpcController`, every robot solved together on
  the env's device. Selected with `train.py --gpu_mpc`.

Conventions at this boundary: tensors on the env device, first dimension N = num_envs.
Joints arrive and torques leave in Isaac Lab's articulation order; legs are FL, FR, RL,
RR (the controller's order, not GaitNet's FR, FL, RR, RL slot order, which the callers
convert); footholds are (x, y) offsets from the leg's hip in the body frame (m).
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch

from gaitnet import get_logger

logger = get_logger()

LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("hip", "thigh", "calf")
ISAAC_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for joint in JOINTS for leg in LEGS)
"""Isaac Lab's Go1 joint order, joint-type major. `simulation/util.py` undoes it with a
Fortran-order reshape; the GPU path checks the articulation's names against it."""


def isaac_joints_to_interface(
    joint_pos_isaac: np.ndarray, joint_vel_isaac: np.ndarray
) -> np.ndarray:
    """
    Convert Isaac Gym joint positions and velocities to the interface format.

    Parameters:
    - joint_pos_isaac: np.ndarray of shape (num_envs, 12)
    - joint_vel_isaac: np.ndarray of shape (num_envs, 12)

    Returns:
    - joint_states: np.ndarray of shape (num_envs, 4, 3, 2)
    """
    # Stack positions and velocities along the last axis
    joint_pos_vel = np.stack(
        [joint_pos_isaac, joint_vel_isaac], axis=-1
    )  # shape (:, 12, 2)

    # Reshape to (num_envs, 4, 3, 2) with Fortran-like index order
    joint_states_interface = joint_pos_vel.reshape(-1, 4, 3, 2, order="F")

    return joint_states_interface


def isaac_body_to_interface(body_state_isaac: np.ndarray) -> np.ndarray:
    """
    Convert Isaac Gym body position, orientation (quaternion), and velocity to the interface format.

    Parameters:
    - body_state_isaac: np.ndarray of shape (num_envs, 13)
        [
            pos_x, pos_y, pos_z,
            quat_w, quat_x, quat_y, quat_z,
            vel_x, vel_y, vel_z,
            omega_x, omega_y, omega_z
        ]

    Returns:
    - body_states_interface: np.ndarray of shape (num_envs, 13)
        [
            pos_x, pos_y, pos_z,
            quat_x, quat_y, quat_z, quat_w
            vel_x, vel_y, vel_z,
            omega_x, omega_y, omega_z
        ]
    """
    # move quatw to end and shift xyz to left
    body_states_interface = np.concatenate(
        [
            body_state_isaac[:, :3],  # pos_x, pos_y, pos_z
            body_state_isaac[:, 4:7],  # quat_x, quat_y, quat_z
            body_state_isaac[:, 3:4],  # quat_w
            body_state_isaac[:, 7:],  # rest
        ],
        axis=1,
    )

    return body_states_interface


def interface_to_isaac_torques(torques_interface: np.ndarray) -> np.ndarray:
    """
    Convert torques from the interface format back to Isaac Gym format.

    Parameters:
    - torques_interface: np.ndarray of shape (num_envs, 4, 3)

    Returns:
    - torques_isaac: np.ndarray of shape (num_envs, 12)
    """
    # Reshape to (num_envs, 12) with Fortran-like index order
    torques_isaac = torques_interface.reshape(-1, 12, order="F")

    return torques_isaac


class RobotControllers(ABC):
    """Footsteps and velocity commands in, joint torques and the contact schedule out."""

    num_envs: int

    @abstractmethod
    def initiate_footsteps(
        self,
        active: torch.Tensor,
        leg: torch.Tensor,
        location_hip: torch.Tensor,
        duration: torch.Tensor,
    ) -> None:
        """Start at most one swing per env, once per policy tick.

        Args:
            active: (N,) bool, False where the env takes no step this tick.
            leg: (N,) long leg index in FL, FR, RL, RR order; any value where inactive.
            location_hip: (N, 2) touchdown (x, y) relative to that leg's hip, body frame (m).
            duration: (N,) swing duration (s).
        """

    @abstractmethod
    def compute_torques(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_state_w: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """(N, 12) joint torques in Isaac joint order, for one physics step.

        Args:
            joint_pos, joint_vel: (N, 12) Isaac joint order (rad, rad/s).
            root_state_w: (N, 13) Isaac Lab `root_state_w`: position, wxyz quaternion,
                world linear and angular velocity.
            command: (N, 3) vx, vy (m/s), yaw rate (rad/s), base frame.
        """

    @abstractmethod
    def contact_state(self) -> torch.Tensor:
        """(N, 4) bool scheduled contact, FL, FR, RL, RR. True in stance."""

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None) -> None:
        """Called by the `reset_robot_controllers` event for envs starting a new episode."""

    def close(self) -> None:
        """Release worker processes, if any."""


class PooledRobotControllers(RobotControllers):
    """The CPU path: `VectorPool[SimInterface]`, converted exactly as before."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        dt: float,
        iterations_between_mpc: int,
        reset_on_episode_end: bool = False,
    ) -> None:
        """
        Args:
            reset_on_episode_end: re-create a robot's controller when its episode ends.
                False (default) keeps what training has always done: the reset call
                (`MPCActionTerm.reset`) sat behind `HierarchicalActionTerm`, which never
                forwarded it, so controllers ran on across episodes.
        """
        from gaitnet.sim2real.siminterface import SimInterface
        from gaitnet.util import VectorPool

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset_on_episode_end = reset_on_episode_end
        self.pool = VectorPool(
            instances=num_envs,
            cls=SimInterface,
            dt=dt,
            iterations_between_mpc=iterations_between_mpc,
            debug_logging=False,
        )

    def call(self, *args, **kwargs):
        """The underlying `VectorPool.call`, for scripts that still use it."""
        return self.pool.call(*args, **kwargs)

    def initiate_footsteps(self, active, leg, location_hip, duration) -> None:
        from gaitnet.sim2real.abstractinterface import Sim2RealInterface

        self.pool.call(
            function=Sim2RealInterface.initiate_footstep,
            mask=active.cpu().numpy(),
            leg=leg.to(torch.int32).cpu().numpy(),
            location_hip=location_hip.to(torch.float32).cpu().numpy(),
            duration=duration.to(torch.float32).cpu().numpy(),
        )

    def compute_torques(self, joint_pos, joint_vel, root_state_w, command) -> torch.Tensor:
        from gaitnet.sim2real.siminterface import SimInterface

        joint_states = isaac_joints_to_interface(joint_pos.cpu().numpy(), joint_vel.cpu().numpy())
        body_state = isaac_body_to_interface(root_state_w.cpu().numpy())
        torques = self.pool.call(
            function=SimInterface.get_torques,
            mask=None,
            joint_states=joint_states,
            body_state=body_state,
            command=command.cpu().numpy(),
        )
        return torch.from_numpy(interface_to_isaac_torques(torques)).to(self.device)

    def contact_state(self) -> torch.Tensor:
        from gaitnet.sim2real.abstractinterface import Sim2RealInterface

        contacts: np.ndarray = self.pool.call(Sim2RealInterface.get_contact_state, mask=None)
        return torch.from_numpy(contacts).to(self.device)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if not self.reset_on_episode_end:
            return
        from gaitnet.sim2real.siminterface import SimInterface

        mask = np.zeros(self.num_envs, dtype=bool)
        mask[slice(None) if env_ids is None else env_ids.cpu().numpy()] = True
        self.pool.call(function=SimInterface.reset, mask=mask)

    def close(self) -> None:
        self.pool._cleanup()

    def _cleanup(self) -> None:
        self.close()


@dataclass
class ControllerRandomizationCfg:
    """Per-env controller randomisation, resampled for each env when its episode starts.

    Every default reproduces the unrandomised controller exactly and draws nothing
    from any random generator. Ranges are (low, high), sampled uniformly per env.
    See `gpu_mpc.ControllerKnobs` for what each knob does.
    """

    command_tracking: tuple[float, float] = (1.0, 1.0)
    """Scale on the velocity command the stance MPC tracks: 1 = pulls the body along at
    the commanded velocity (current), 0 = holds the body still."""
    planar_weight_scale: tuple[float, float] = (1.0, 1.0)
    """Scale on the MPC's planar tracking weights: 1 = as tuned, 0 = untracked."""
    swing_apex: tuple[float, float] | None = None
    """Swing apex clearance (m); None keeps the controller's 0.26 / 3 = 0.0867 m."""
    touchdown_bias: float = 0.0
    """Per env and leg, a constant (x, y) offset on every commanded foothold, uniform in
    [-touchdown_bias, touchdown_bias] per axis (m)."""
    touchdown_noise: tuple[float, float] = (0.0, 0.0)
    """Per env, the std of fresh Gaussian noise on each commanded foothold (m)."""
    liftoff_delay_ticks: tuple[int, int] = (0, 0)
    """Policy ticks between a footstep command and the controller starting it,
    integer uniform in [low, high]."""
    travel_compensation: tuple[float, float] = (0.0, 0.0)
    """Per env, the fraction of body travel over the swing taken off the target
    (`gpu_mpc.ControllerKnobs.travel_compensation`): 0 = this controller, 1 = OCS2's and
    gaitnet_core's convention of landing where the hip was when commanded."""
    touchdown_along: tuple[float, float] = (0.0, 0.0)
    """Per env, a constant offset along x (forward) on every commanded foothold (m); OCS2
    in Gazebo lands SC's feet 3-5 cm short, i.e. about -0.04."""
    touchdown_outboard: tuple[float, float] = (0.0, 0.0)
    """Per env, a constant offset away from the body's centre line on every commanded
    foothold (m, positive = outboard); OCS2 in Gazebo lands them 3-5 cm outboard."""
    mix: ControllerRandomizationCfg | None = None
    """Another randomisation, drawn instead of this one for an env's episode with probability
    `mix_fraction` (e.g. the recipe's, so part of the episodes keep the controller the policy
    learned the terrain on)."""
    mix_fraction: float = 0.0


@dataclass
class GpuMpcCfg:
    """What `GpuRobotControllers` builds the GPU controller with."""

    variant: str = "gaitnet"
    """`gpu_mpc.VARIANTS` key: "gaitnet" reproduces the CPU controller here (the one
    the existing checkpoints were trained on); "gaitnet_core" is gaitnet-mpc's tuning."""
    horizon: int = 10
    solver_iterations: int = 80
    """ADMM iterations per MPC solve; gaitnet_core's tuned default."""
    solver_rho_update_interval: int = 20
    """Tuned together with `solver_iterations`; change them together."""
    foot_radius: float = 0.02
    """Only used by the "gaitnet_core" variant, which aims at the terrain surface."""
    double_precision: bool = False
    seed: int = 0
    """Seed of the controller's own generator (randomisation and touchdown noise)."""
    randomization: ControllerRandomizationCfg = field(default_factory=ControllerRandomizationCfg)


class GpuRobotControllers(RobotControllers):
    """The GPU path: `GaitnetMpcController` for every env, Isaac conventions converted.

    Per-env reset (`reset`, run by the `reset_robot_controllers` event when an episode
    starts) puts that env's controller back to standing with nothing commanded: gait
    clock and swing schedule to zero (all legs in stance), MPC warm start dropped, every
    foothold back to nominal, height estimate to nominal, swing anchors, telemetry and
    any delayed footstep commands cleared, and its randomisation knobs resampled.
    Other envs are untouched; the MPC solves on the same physics steps for every env.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        dt: float,
        iterations_between_mpc: int,
        cfg: GpuMpcCfg | None = None,
        joint_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        from gaitnet_core.control import AdmmSettings, BatchedMpcConfig, MpcSettings

        from gaitnet.sim2real.gpu_mpc import VARIANTS, GaitnetMpcController

        self.cfg = cfg = cfg or GpuMpcCfg()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(cfg.seed)
        self.controller = GaitnetMpcController(
            BatchedMpcConfig(
                mpc=MpcSettings(
                    horizon=cfg.horizon,
                    control_steps_per_update=iterations_between_mpc,
                    admm=AdmmSettings(
                        iterations=cfg.solver_iterations,
                        rho_update_interval=cfg.solver_rho_update_interval,
                    ),
                ),
                foot_radius=cfg.foot_radius,
                dtype=torch.float64 if cfg.double_precision else torch.float32,
            ),
            VARIANTS[cfg.variant],
            num_robots=num_envs,
            dt=dt,
            device=self.device,
            generator=self.generator,
        )
        self.dtype = self.controller.dtype

        names = tuple(joint_names) if joint_names is not None else ISAAC_JOINT_NAMES
        if names != ISAAC_JOINT_NAMES:
            logger.warning(
                f"articulation joint order {names} is not the {ISAAC_JOINT_NAMES} the CPU "
                "path assumes; the GPU path follows the names"
            )
        # controller (leg-major) index -> Isaac index
        self._isaac_index = torch.tensor(
            [names.index(f"{leg}_{joint}_joint") for leg in LEGS for joint in JOINTS],
            device=self.device,
        )

        rnd = cfg.randomization
        self._max_delay = int(max(rnd.liftoff_delay_ticks[1], rnd.mix.liftoff_delay_ticks[1] if rnd.mix else 0))
        self.liftoff_delay = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        """(N,) policy ticks between a footstep command and the swing starting."""
        slots = self._max_delay + 1
        self._queue_active = torch.zeros(slots, num_envs, device=self.device, dtype=torch.bool)
        self._queue_leg = torch.zeros(slots, num_envs, device=self.device, dtype=torch.long)
        self._queue_xy = torch.zeros(slots, num_envs, 2, device=self.device, dtype=self.dtype)
        self._queue_duration = torch.zeros(slots, num_envs, device=self.device, dtype=self.dtype)
        self._tick = 0
        self.controller.knobs.noise_enabled = rnd.touchdown_noise[1] > 0.0
        self._resample(torch.arange(num_envs, device=self.device))

        # GAITNET_PROFILE_MPC=1: time compute_torques (synchronising around it), report on close
        self._profile = os.environ.get("GAITNET_PROFILE_MPC", "0") == "1"
        self._profile_seconds = 0.0
        self._profile_calls = 0

    # -- footsteps -----------------------------------------------------------------------

    def initiate_footsteps(self, active, leg, location_hip, duration) -> None:
        leg = leg.to(device=self.device, dtype=torch.long).clamp(0, 3)
        xy = location_hip.to(device=self.device, dtype=self.dtype)
        duration = duration.to(device=self.device, dtype=self.dtype)
        active = active.to(self.device)
        if self._max_delay == 0:
            self._command(active, leg, xy, duration)
            return

        # a ring of max_delay + 1 slots; a command due `delay` ticks from now goes in the
        # slot fired then. Each env's delay is fixed between resets, so a slot is always
        # empty again before that env can write it.
        slots = self._max_delay + 1
        envs = torch.arange(self.num_envs, device=self.device)
        due = (self._tick + self.liftoff_delay) % slots
        self._queue_active[due, envs] = active
        self._queue_leg[due, envs] = leg
        self._queue_xy[due, envs] = xy
        self._queue_duration[due, envs] = duration

        now = self._tick % slots
        self._command(
            self._queue_active[now],
            self._queue_leg[now],
            self._queue_xy[now],
            self._queue_duration[now],
        )
        self._queue_active[now] = False
        self._tick += 1

    def _command(self, active, leg, xy, duration) -> None:
        from gaitnet_core.interfaces import FootstepCommand

        # z for the "gaitnet_core" variant, which takes a terrain height: the stance
        # plane, one foot radius below the foot centres. "gaitnet" overrides it with z = 0.
        z = -(self.controller.height_estimate + self.cfg.foot_radius)
        self.controller.command_footsteps(
            FootstepCommand(
                active=active,
                leg=leg,
                target=torch.cat([xy, z.unsqueeze(-1)], dim=-1),
                duration=duration,
            )
        )

    def pending_footsteps(self) -> torch.Tensor:
        """(N, 4) bool, legs commanded but not yet started (only with a liftoff delay)."""
        pending = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        if self._max_delay == 0:
            return pending
        for slot in range(self._max_delay + 1):
            one_hot = torch.nn.functional.one_hot(self._queue_leg[slot], 4).bool()
            pending |= one_hot & self._queue_active[slot].unsqueeze(-1)
        return pending

    # -- per physics step ----------------------------------------------------------------

    def compute_torques(self, joint_pos, joint_vel, root_state_w, command) -> torch.Tensor:
        if not self._profile:
            return self._compute_torques(joint_pos, joint_vel, root_state_w, command)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        torques = self._compute_torques(joint_pos, joint_vel, root_state_w, command)
        torch.cuda.synchronize(self.device)
        self._profile_seconds += time.perf_counter() - start
        self._profile_calls += 1
        return torques

    def _compute_torques(self, joint_pos, joint_vel, root_state_w, command) -> torch.Tensor:
        index = self._isaac_index
        root = root_state_w.to(self.dtype)
        base_pose = torch.cat([root[:, :3], root[:, [4, 5, 6, 3]]], dim=-1)  # wxyz -> xyzw
        torques = self.controller.compute_torques(
            joint_pos[:, index],
            joint_vel[:, index],
            base_pose,
            root[:, 7:13],
            command,
        )
        isaac = torch.empty_like(torques)
        isaac[:, index] = torques
        return isaac.to(joint_pos.dtype)

    def contact_state(self) -> torch.Tensor:
        """The controller's schedule (`GaitnetMpcController.contact_state`), with legs
        whose command is still waiting out a liftoff delay reported as already lifted,
        so the policy cannot commit them twice and the three-in-stance rule counts them."""
        contact = self.controller.contact_state()
        if self._max_delay == 0:
            return contact
        return contact & ~self.pending_footsteps()

    def swing_phase(self) -> torch.Tensor:
        return self.controller.swing_phase()

    def history_features(
        self, joint_pos: torch.Tensor, base_lin_vel_b: torch.Tensor, base_ang_vel_b: torch.Tensor
    ) -> torch.Tensor:
        """(N, 26) this tick's `policy_history.tick_features`, slot order FR, FL, RR, RL.

        Feet come from the current Isaac-order joint angles through the controller's own
        kinematics (so they are current even on the tick an env was reset); contact is
        `contact_state()`; the touchdown error is the controller's.
        """
        from gaitnet.gaitnet.components.policy_history import tick_features

        controller = self.controller
        return tick_features(
            base_lin_vel_b.to(self.dtype),
            base_ang_vel_b.to(self.dtype),
            controller.foot_positions(joint_pos[:, self._isaac_index]),
            self.contact_state(),
            controller.touchdown_error,
        )

    def close(self) -> None:
        if self._profile and self._profile_calls:
            logger.info(
                f"GPU MPC profile: {self._profile_calls} control steps for {self.num_envs} envs, "
                f"{1e3 * self._profile_seconds / self._profile_calls:.2f} ms per step, "
                f"{self._profile_seconds:.1f} s total"
            )
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / 2**20
            logger.info(f"torch peak allocated on {self.device}: {peak:.0f} MiB")

    # -- episodes ------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.controller.reset(env_ids)
        self._queue_active[:, env_ids] = False
        self._resample(env_ids)

    def _resample(self, env_ids: torch.Tensor) -> None:
        rnd = self.cfg.randomization
        if rnd.mix is None or rnd.mix_fraction <= 0:
            self._sample(env_ids, rnd)
            return
        mixed = torch.rand(env_ids.numel(), device=self.device, generator=self.generator) < rnd.mix_fraction
        self._sample(env_ids[mixed], rnd.mix)
        self._sample(env_ids[~mixed], rnd)

    def _sample(self, env_ids: torch.Tensor, rnd: ControllerRandomizationCfg) -> None:
        """Draw `rnd`'s knobs for `env_ids`."""
        knobs = self.controller.knobs
        count = env_ids.numel()
        if count == 0:
            return

        def uniform(low: float, high: float, *shape: int) -> torch.Tensor | float:
            if low == high:
                return low
            sample = torch.rand(
                (count, *shape), device=self.device, dtype=self.dtype, generator=self.generator
            )
            return low + (high - low) * sample

        knobs.command_tracking[env_ids] = uniform(*rnd.command_tracking)
        knobs.planar_weight_scale[env_ids] = uniform(*rnd.planar_weight_scale)
        if rnd.swing_apex is not None:
            knobs.swing_apex[env_ids] = uniform(*rnd.swing_apex)
        bias = rnd.touchdown_bias
        offset = torch.zeros(count, 4, 2, device=self.device, dtype=self.dtype)
        if bias > 0:
            offset += uniform(-bias, bias, 4, 2)
        along = torch.as_tensor(uniform(*rnd.touchdown_along), device=self.device, dtype=self.dtype).reshape(-1, 1)
        outboard = torch.as_tensor(uniform(*rnd.touchdown_outboard), device=self.device, dtype=self.dtype).reshape(-1, 1)
        side = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device, dtype=self.dtype)   # FL, FR, RL, RR: +y is left
        offset[..., 0] += along
        offset[..., 1] += outboard * side
        knobs.touchdown_bias[env_ids] = offset
        if knobs.travel_compensation is not None:
            knobs.travel_compensation[env_ids] = uniform(*rnd.travel_compensation)
        knobs.touchdown_noise[env_ids] = uniform(*rnd.touchdown_noise)
        low, high = rnd.liftoff_delay_ticks
        if low == high:
            self.liftoff_delay[env_ids] = low
        else:
            self.liftoff_delay[env_ids] = torch.randint(
                low, high + 1, (count,), device=self.device, generator=self.generator
            )
