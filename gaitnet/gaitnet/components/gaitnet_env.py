from typing import Any
import torch
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    RecorderManager,
    RewardManager,
    TerminationManager,
)

from gaitnet import get_logger
from gaitnet.gaitnet.actions.mpc_action import ManagerBasedEnv
from gaitnet.gaitnet.components.footstep_candidate_sampler import FootstepCandidateSampler
from gaitnet.gaitnet.components.gaitnet_observation_manager import (
    GaitNetObservationManager,
)
import gaitnet.constants as const

logger = get_logger()


class GaitNetEnv(ManagerBasedRLEnv):
    def __init__(self, episode_info: dict[str, Any] | None = None, *args, **kwargs):
        self.episode_info = episode_info
        self.walk_metrics: dict[str, float] = {}
        """What the `walk` terrain curriculum saw at the latest episode ends, logged every step."""
        super().__init__(*args, **kwargs)
        self.walk_progress = torch.zeros(self.num_envs, device=self.device)
        """(N,) distance walked this episode along the commanded direction (m), for the `walk`
        terrain curriculum (env_cfg/curriculum.terrain_levels_walk)."""
        self.walk_commanded = torch.zeros(self.num_envs, device=self.device)
        """(N,) distance the velocity command asked for this episode (m)."""

    def _track_walk(self) -> None:
        """Add this step to the episode's walked and commanded distances. Read at the start of the
        step, so the velocity is the one the previous step ended with."""
        command = self.command_manager.get_command("base_velocity")[:, :2]
        speed = command.norm(dim=-1)
        velocity = self.scene["robot"].data.root_lin_vel_b[:, :2]
        along = (velocity * command).sum(dim=-1) / speed.clamp_min(1e-6)
        self.walk_progress += along * self.step_dt
        self.walk_commanded += speed * self.step_dt

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)  # the curriculum reads the walked distances first
        if hasattr(self, "walk_progress"):
            self.walk_progress[env_ids] = 0.0
            self.walk_commanded[env_ids] = 0.0

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # prepare the managers
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        # Note that this is the one change from the parent class
        self.observation_manager = GaitNetObservationManager(
            cfg=self.cfg.observations,
            env=self,
            footstep_option_generator=FootstepCandidateSampler(
                env=self, options_per_leg=const.gait_net.num_footstep_options
            ),
            num_footstep_options=const.gait_net.num_footstep_options,
            history_len=getattr(self.cfg, "policy_history_len", 0),
            privileged=getattr(self.cfg, "policy_privileged", False),
        )
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if (
            self.__class__ == ManagerBasedEnv
            and "startup" in self.event_manager.available_modes
        ):
            self.event_manager.apply(mode="startup")

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if self.episode_info is not None:
            for key, val in self.episode_info.items():
                self.extras["log"][f"Custom_Metrics/{key}"] = val
            self.episode_info.clear()
        for key, val in self.walk_metrics.items():
            self.extras["log"][f"Curriculum/{key}"] = val
        self._track_walk()
        return super().step(action)
