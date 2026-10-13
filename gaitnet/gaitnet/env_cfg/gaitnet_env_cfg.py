from typing import TypeVar
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from gaitnet.sim2real.robot_controllers import (
    GpuMpcCfg,
    GpuRobotControllers,
    PooledRobotControllers,
    RobotControllers,
)
from gaitnet.gaitnet.env_cfg.curriculum import CurriculumCfg
from gaitnet.simulation.cfg.scene import SceneCfg
import numpy as np

from gaitnet import get_logger
from gaitnet.gaitnet.env_cfg.observations import ObservationsCfg
from gaitnet.gaitnet.env_cfg.actions import ActionsCfg
from gaitnet.gaitnet.env_cfg.events import EventsCfg
from gaitnet.gaitnet.env_cfg.terminations import TerminationsCfg
from gaitnet.gaitnet.env_cfg.rewards import RewardsCfg
from gaitnet.gaitnet.env_cfg.commands import CommandsCfg

logger = get_logger()


@configclass
class GaitNetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(env_spacing=2.5)  # type: ignore

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: ActionsCfg = ActionsCfg()  # type: ignore
    events: EventsCfg = EventsCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    rewards: RewardsCfg = RewardsCfg()  # type: ignore
    commands: CommandsCfg = CommandsCfg()  # type: ignore
    curriculum: CurriculumCfg = CurriculumCfg()  # type: ignore

    robot_controllers: RobotControllers = None  # type: ignore set by update_controllers

    policy_history_len: int = 0
    """Policy ticks of `policy_history` features appended after the candidates (0 = none, the
    original layout). Needs the GPU controller."""

    policy_privileged: bool = False
    """Append the teacher's privileged features (components/privileged.py) at the very end of the
    policy observation. Needs the GPU controller."""

    # TODO: do this the right way
    # ['trunk', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    hip_indices = np.asarray([1, 2, 3, 4], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    thigh_indices = np.asarray([5, 6, 7, 8], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    calf_indices = np.asarray([9, 10, 11, 12], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    foot_indices = np.asarray([13, 14, 15, 16], dtype=np.int32)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # env decimation -> 25 Hz footstep planning
        self.render_interval = (
            self.decimation / 2
        )  # render faster than footstep planning rate
        # simulation settings
        self.sim.dt = 0.004  # simulation timestep -> 250 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material

        self.episode_length_s = 20


RECIPES = ("thesis", "step", "teacher")
"""Reward recipes. thesis: RewardsCfg as is. step: op_penalty -2.5 -> -0.4 and the velocity
tracking terms 0.25 -> 1.0 (no curriculum term touches these weights), so the policy is paid
for tracking rather than charged for stepping; train.py pairs it with controller randomisation.
teacher: step plus progress along the command (1.0), a penalty per foot touching down within
5 cm of a void edge (-25, i.e. -1.0 of return per touchdown after the step_dt scaling), and mild body stability (lin_vel_z_l2 -0.5, ang_vel_xy_l2 -0.05,
flat_orientation_l2 -2.0). No gait-shaping terms. Like every term, these are scaled by step_dt."""

COMMANDS = ("omni", "forward")
"""Velocity commands. omni: x and y in [-0.2, 0.2] m/s, yaw rate in [-0.4, 0.4] rad/s (env_cfg/commands.py),
what every run before 2026-10-12 17:00 used. forward: x in [0, 0.2] m/s, no lateral command, yaw rate
within +-forward_yaw_rate."""

CURRICULA = ("vel", "walk")
"""Terrain curricula. vel: Isaac Lab's terrain_levels_vel, what every run so far used; with this
command range it keeps every robot on level 0 (components/walk_curriculum.py says why). walk:
env_cfg/curriculum.terrain_levels_walk, up for surviving and walking half the commanded distance,
down for falling or stalling."""


def get_env_cfg(
    num_envs: int,
    device: str,
    recipe: str = "thesis",
    history_len: int = 0,
    privileged: bool = False,
    curriculum: str = "vel",
    tracking_std: float | None = None,
    yaw_tracking_std: float | None = None,
    walk_p_down: float | None = None,
    walk_p_up: float | None = None,
    walk_random_fraction: float | None = None,
    max_init_terrain_level: int | None = None,
    commands: str = "omni",
    forward_yaw_rate: float = 0.2,
    idm_inverse: str | None = None,
) -> GaitNetEnvCfg:
    cfg = GaitNetEnvCfg()
    if curriculum not in CURRICULA:
        raise ValueError(f"unknown curriculum {curriculum!r}, expected one of {CURRICULA}")
    if curriculum == "walk":
        from isaaclab.managers import CurriculumTermCfg as CurrTerm

        from gaitnet.gaitnet.env_cfg import curriculum as curricula

        params = {} if walk_p_down is None else {"p_down_given_failure": walk_p_down}
        if walk_p_up is not None:
            params["p_up_given_success"] = walk_p_up
        if walk_random_fraction is not None:
            params["random_fraction"] = walk_random_fraction
        cfg.curriculum.terrain_levels = CurrTerm(func=curricula.terrain_levels_walk, params=params)
    cfg.policy_history_len = history_len
    cfg.policy_privileged = privileged
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}, expected one of {RECIPES}")
    if recipe in ("step", "teacher"):
        cfg.rewards.op_penalty.weight = -0.4
        cfg.rewards.xy_tracking.weight = 1.0
        cfg.rewards.yaw_tracking.weight = 1.0
    if recipe == "teacher":
        from isaaclab.envs import mdp
        from isaaclab.managers import RewardTermCfg as RewTerm

        from gaitnet.gaitnet.env_cfg import rewards

        cfg.rewards.progress = RewTerm(func=rewards.progress, weight=1.0)
        # every term is scaled by step_dt (0.04 s): -25 makes one edge touchdown cost 1.0 of return
        # (at -1.0 it cost 0.04 and the first iteration logged -0.0025 per episode, i.e. nothing)
        cfg.rewards.edge_touchdowns = RewTerm(func=rewards.edge_touchdowns, weight=-25.0)
        cfg.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
        cfg.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        cfg.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
        cfg.scene.contact_forces.track_air_time = True  # for first-contact detection

    # the 0.5 default is sized for commands of 1-2 m/s; with commands of at most 0.2 m/s per axis it
    # pays a robot that stands still 0.90 of the tracking reward, and walking at full speed only 0.10 more
    if max_init_terrain_level is not None:
        cfg.scene.terrain.max_init_terrain_level = max_init_terrain_level
    cfg.actions.footstep_controller.idm_inverse = idm_inverse
    if commands not in COMMANDS:
        raise ValueError(f"unknown commands {commands!r}, expected one of {COMMANDS}")
    if commands == "forward":
        # the deployed robot maps terrain with a forward-facing camera: nothing beside or behind it is
        # known, so it is only ever commanded forward (and the evals and Gazebo trials are straight)
        ranges = cfg.commands.base_velocity.ranges
        ranges.lin_vel_x = (0.0, max(ranges.lin_vel_x))
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (-forward_yaw_rate, forward_yaw_rate)
    if tracking_std is not None:
        cfg.rewards.xy_tracking.params["std"] = tracking_std
    if yaw_tracking_std is not None:
        cfg.rewards.yaw_tracking.params["std"] = yaw_tracking_std

    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    # update sensor update periods
    # we tick all the sensors based on the smallest update period (physics update period)
    # update all properties of cfg.scene which subclass SensorBaseCfg
    for attr_name in cfg.scene.__dir__():
        attr = getattr(cfg.scene, attr_name)
        if not issubclass(type(attr), SensorBaseCfg):
            continue
        sensor: SensorBaseCfg = attr  # type: ignore
        sensor.update_period = cfg.decimation * cfg.sim.dt  # control rate
    return cfg


def update_controllers(
    cfg: GaitNetEnvCfg,
    num_envs: int,
    gpu_mpc: GpuMpcCfg | None = None,
    joint_names: list[str] | None = None,
) -> None:
    """Build the robots' low-level controllers into `cfg.robot_controllers`.

    Args:
        cfg: The environment configuration.
        num_envs: Number of robots.
        gpu_mpc: None (default) for the CPU process pool, one rl-mpc-locomotion
            controller per robot; otherwise the batched GPU controller built with this.
        joint_names: the articulation's joint names, which the GPU path checks its
            joint order against (default: the order the CPU path assumes).
    """
    if gpu_mpc is None:
        controllers: RobotControllers = PooledRobotControllers(
            num_envs=num_envs,
            device=cfg.sim.device,
            dt=cfg.sim.dt,  # 250 Hz leg PD control
            iterations_between_mpc=5,  # 50 Hz MPC
        )
    else:
        controllers = GpuRobotControllers(
            num_envs=num_envs,
            device=cfg.sim.device,
            dt=cfg.sim.dt,
            iterations_between_mpc=5,
            cfg=gpu_mpc,
            joint_names=joint_names,
        )
    cfg.robot_controllers = controllers  # type: ignore

# generic for manager_class
T = TypeVar("T", bound=ManagerBasedRLEnv)

def get_env(
    num_envs: int,
    device: str,
    manager_class: type[T] = ManagerBasedRLEnv,
    gpu_mpc: GpuMpcCfg | None = None,
    recipe: str = "thesis",
    history_len: int = 0,
    privileged: bool = False,
    curriculum: str = "vel",
    tracking_std: float | None = None,
    yaw_tracking_std: float | None = None,
    walk_p_down: float | None = None,
    walk_p_up: float | None = None,
    walk_random_fraction: float | None = None,
    max_init_terrain_level: int | None = None,
    commands: str = "omni",
    forward_yaw_rate: float = 0.2,
    idm_inverse: str | None = None,
) -> T:
    """Get the environment configuration and the environment instance.

    `gpu_mpc`: None for the CPU controller pool (default), else the GPU controller's cfg.
    `recipe`: see RECIPES. `history_len`: see GaitNetEnvCfg.policy_history_len.
    `curriculum`: see CURRICULA. `tracking_std`, `yaw_tracking_std`: the velocity tracking
    kernels' widths (m/s, rad/s); None keeps RewardsCfg's 0.5. `walk_p_down`: the walk
    curriculum's p_down_given_failure; None keeps its 0.5; `walk_p_up` its p_up_given_success (None: 0.1). `walk_random_fraction`: share of episodes
    run on a random level (walk_curriculum.next_levels). `max_init_terrain_level`: highest level a
    robot can start on (None: any), e.g. to resume a trained policy near where its curriculum was.
    """
    if (history_len > 0 or privileged) and gpu_mpc is None:
        raise ValueError("the policy history and privileged features need the GPU controller (gpu_mpc)")
    env_cfg = get_env_cfg(
        num_envs, device, recipe=recipe, history_len=history_len, privileged=privileged, curriculum=curriculum,
        tracking_std=tracking_std, yaw_tracking_std=yaw_tracking_std, walk_p_down=walk_p_down, walk_p_up=walk_p_up,
        walk_random_fraction=walk_random_fraction, max_init_terrain_level=max_init_terrain_level,
        commands=commands, forward_yaw_rate=forward_yaw_rate, idm_inverse=idm_inverse,
    )
    env = manager_class(cfg=env_cfg)
    update_controllers(
        env_cfg, num_envs, gpu_mpc=gpu_mpc, joint_names=env.scene["robot"].joint_names
    )
    return env
