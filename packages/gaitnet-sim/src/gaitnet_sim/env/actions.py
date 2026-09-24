"""The one action term: execute the planner's footstep and nudge through a low-level controller.

The action vector is `gaitnet_core.action_layout`: for each of the tick's rounds, the policy's
choice (candidate index, duration) and the concrete footstep it resolves to, then a velocity
command nudge. The term reads only the footsteps and nudge, so it never needs the candidate
set. The number of rounds is the contract's `max_steps_per_tick`.

The term owns the controller. Observation and reward terms reach the controller, the
nudged command and the robot's state through it (`footstep_action(env)`). Its cfg is in
`gaitnet_sim.env.actions_cfg`, which names this class by string so task cfgs import
without the simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm

from gaitnet_core import action_layout
from gaitnet_core.action_layout import EnvAction
from gaitnet_core.interfaces import FootstepCommand, LowLevelController
from gaitnet_core.state import Observation, RobotState, TerrainPatch
from gaitnet_sim.env.noise import corrupt
from gaitnet_sim.env.perception import FrontCameraMap
from gaitnet_sim.robot_io import RobotIO

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from gaitnet_sim.env.actions_cfg import FootstepControlActionCfg


class FootstepControlAction(ActionTerm):
    cfg: "FootstepControlActionCfg"
    _asset: Articulation
    _env: "ManagerBasedRLEnv"

    def __init__(self, cfg: "FootstepControlActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        contract = env.cfg.gaitnet
        self.spec = contract.robot_spec()
        self.grid = contract.foothold_grid()
        self.io = RobotIO(
            env.scene,
            self.spec,
            self.grid,
            robot_name=cfg.asset_name,
            joint_names=cfg.joint_names,
            foot_names=cfg.foot_names,
            contact_sensor_name=cfg.contact_sensor_name,
            scanner_names=cfg.scanner_names,
            contact_threshold=cfg.contact_threshold,
        )
        self.controller: LowLevelController = cfg.controller.class_type(
            cfg.controller, num_robots=self.num_envs, dt=env.physics_dt, device=self.device
        )
        self.rounds = contract.foothold_rules().max_steps_per_tick
        self._raw_actions = torch.zeros(self.num_envs, action_layout.dim(self.rounds), device=self.device)
        self._nudge = torch.zeros(self.num_envs, 3, device=self.device)
        self._footsteps = [FootstepCommand.none(self.num_envs, device=self.device) for _ in range(self.rounds)]
        self._planner_observation: Observation | None = None
        self.episode_progress = torch.zeros(self.num_envs, device=self.device)
        """(N,) distance walked this episode along the operator's command direction (m)."""
        self.episode_commanded = torch.zeros(self.num_envs, device=self.device)
        """(N,) distance the operator's command asked for this episode (m)."""
        self.camera_map = (
            FrontCameraMap(cfg.front_camera, self.num_envs, self.device) if cfg.front_camera is not None else None
        )
        if self.camera_map is not None:
            self.camera_map.reset(torch.arange(self.num_envs, device=self.device), env.scene.env_origins[:, :2])

    def __del__(self):
        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.close()
        super().__del__()

    @property
    def action_dim(self) -> int:
        return action_layout.dim(self.rounds)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def footsteps(self) -> list[FootstepCommand]:
        """The footsteps started on the latest env step, one command per round."""
        return self._footsteps

    def steps_started(self) -> torch.Tensor:
        """(N,) footsteps started on the latest env step."""
        return torch.stack([footsteps.active for footsteps in self._footsteps], dim=0).sum(dim=0)

    @property
    def nudge(self) -> torch.Tensor:
        """(N, 3) the velocity command delta applied this env step."""
        return self._nudge

    def base_command(self) -> torch.Tensor:
        """(N, 3) the command manager's velocity command, before the nudge."""
        return self._env.command_manager.get_command(self.cfg.command_name)

    def effective_command(self) -> torch.Tensor:
        """(N, 3) the velocity command the controller tracks: the base command plus the nudge."""
        return self.base_command() + self._nudge

    def robot_state(self) -> RobotState:
        return self.io.robot_state(self.controller.gait_timing(), self.effective_command(), self.base_command())

    def terrain(self) -> TerrainPatch:
        return self.io.terrain()

    def observation(self) -> Observation:
        """The truth: what terminations, rewards and privileged observations read."""
        return Observation(self.robot_state(), self.terrain())

    def planner_observation(self) -> Observation:
        """What the planner sees: the truth with `cfg.observation_noise` added, if any. One draw
        per env step, so every observation group sees the same corrupted world."""
        if self._planner_observation is None:
            observation = self.observation()
            if self.camera_map is not None:
                observation = self._through_camera(observation)
            if self.cfg.observation_noise is not None:
                observation = corrupt(observation, self.cfg.observation_noise)
            self._planner_observation = observation
        return self._planner_observation

    def _track_progress(self) -> None:
        """Add one env step to the episode's walked and commanded distances: the base's
        velocity along the operator's (vx, vy) command, and the command's speed."""
        command = self.base_command()[:, :2]
        speed = command.norm(dim=-1)
        direction = command / speed.clamp(min=1e-6).unsqueeze(-1)
        velocity = self.io.robot.data.root_link_lin_vel_b.torch[:, :2]
        dt = self._env.step_dt
        self.episode_progress += (velocity * direction).sum(dim=-1) * dt
        self.episode_commanded += speed * dt

    def _through_camera(self, observation: Observation) -> Observation:
        """The terrain as the front camera's map knows it, after fusing this step's frame."""
        pose = self.io.base_pose()
        feet_z = self.io.robot.data.body_link_pos_w.torch[:, self.io.foot_ids, 2]
        self.camera_map.update(pose[:, :3], pose[:, 3:], feet_z.mean(dim=1))
        known, error = self.camera_map.lookup(self.io.terrain_points_xy())
        heights = observation.terrain.heights
        heights = torch.where(known, heights + error, torch.full_like(heights, float("-inf")))
        return Observation(observation.state, TerrainPatch(heights=heights, grid=observation.terrain.grid))

    def process_actions(self, actions: torch.Tensor):
        # the robots are about to move
        self._planner_observation = None
        self._track_progress()
        # copied into the buffers made at init: tensors made here under torch.inference_mode
        # (RSL-RL, evaluation) would refuse the in-place updates of a later reset outside it
        self._raw_actions[:] = actions
        action = EnvAction.decode(actions)
        for buffer, footsteps in zip(self._footsteps, action.footstep_commands()):
            buffer.active[:] = footsteps.active
            buffer.leg[:] = footsteps.leg
            buffer.target[:] = footsteps.target
            duration = footsteps.duration
            if self.cfg.clamp_duration:
                duration = duration.clamp(*self.spec.swing_duration_range)
            buffer.duration[:] = torch.where(footsteps.active, duration, footsteps.duration)
        if self.cfg.apply_nudge:
            self._nudge[:] = action.nudge
        # rounds step different legs (the policy only offers eligible ones), and both
        # controllers keep each leg's swing apart, so they are started one after another
        for buffer in self._footsteps:
            self.controller.command_footsteps(buffer)

    def apply_actions(self):
        joint_pos, joint_vel = self.io.joint_state()
        torques = self.controller.compute_torques(
            joint_pos, joint_vel, self.io.base_pose(), self.io.base_vel(), self.effective_command()
        )
        self._asset.actuators.target_command.set_effort_index(value=torques, joint_ids=self.io.joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[ids] = 0.0
        self._nudge[ids] = 0.0
        for footsteps in self._footsteps:
            footsteps.active[ids] = False
        self.controller.reset(ids)
        if self.camera_map is not None:
            self.camera_map.reset(ids, self._env.scene.env_origins[ids, :2])
        self.episode_progress[ids] = 0.0
        self.episode_commanded[ids] = 0.0
        self._planner_observation = None
