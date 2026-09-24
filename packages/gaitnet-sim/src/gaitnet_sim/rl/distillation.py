"""Teacher-student distillation of the GaitNet actor (DAgger), as an RSL-RL algorithm.

The teacher is a trained GaitNet actor that sees the truth: the true robot state and the true
terrain (the env's `distill` preset groups). The student is a GaitNet actor, usually a smaller
network, that sees what the robot will: the camera map and observation noise. Both score the
same candidate set, the student's, which the teacher sees judged on the true terrain
(`gaitnet_sim.env.observations.teacher_candidates`); so on every tick the two policies are
distributions over the same footsteps and the loss can be their exact KL divergence:

    KL(teacher || student) = KL of the categorical choice (candidate or no-op)
                           + sum over footsteps of P_teacher(footstep) * KL of its swing duration

summed over the tick's rounds along the footsteps that were actually taken (a later round's
distribution depends on the earlier rounds' choices, see `gaitnet_core.rounds`). The student
acts, so it learns on the states it reaches itself (DAgger); early on, a decaying fraction of
robots execute the teacher's footsteps instead, which keeps the first rollouts from being all
falls.

The teacher is loaded from a PPO checkpoint (`teacher_checkpoint`); its network, duration
noise floor and state features come from that run's saved agent cfg, and the run's contract
must match this env's (rounds per tick, state features). Resuming with `--checkpoint` restores
the student, the teacher, the optimizer and the DAgger schedule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups

from gaitnet_core import action_layout
from gaitnet_core.action_layout import EnvAction
from gaitnet_core.rounds import Round, TickDistribution, TickSelection
from gaitnet_sim.rl.model import GaitNetActor

TEACHER_CFG_KEYS = ("network", "duration_std_floor", "state_features", "min_stance_after_step")
"""What the teacher takes from its run's agent cfg: what its weights were trained with."""


def load_teacher_run(checkpoint: str | Path) -> tuple[dict, dict, dict]:
    """(checkpoint dict, the run's agent cfg, the run's env cfg) of a PPO checkpoint in an
    Isaac Lab run directory (`<run>/model_<i>.pt` next to `<run>/params/`)."""
    from gaitnet_sim.rl.export import load_params

    checkpoint = Path(checkpoint)
    run = checkpoint.parent
    params = run / "params"
    if not (params / "agent.yaml").is_file() or not (params / "env.yaml").is_file():
        raise FileNotFoundError(f"the teacher checkpoint {checkpoint} has no params/agent.yaml and params/env.yaml next to it")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return saved, load_params(params / "agent.yaml"), load_params(params / "env.yaml")


def teacher_state_dict(saved: dict) -> dict:
    """The teacher's weights from a PPO checkpoint (`actor_state_dict`) or a distillation one."""
    for key in ("actor_state_dict", "teacher_state_dict"):
        if key in saved:
            return saved[key]
    raise KeyError(f"no actor_state_dict or teacher_state_dict in the teacher checkpoint (keys {sorted(saved)})")


def round_kl(teacher: Round, student: Round, duration: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,) KL(teacher || student) of one round's categorical choice, and of the swing
    durations weighted by the teacher's probability of each footstep.

    The teacher's candidates are the student's re-judged on the truth, so its support is (up
    to the state's noise in eligibility) inside the student's; the teacher is renormalized over
    the student's support so the KL is always finite."""
    student_logp = student.distribution.categorical.logits
    support = torch.isfinite(student_logp)
    teacher_logits = teacher.distribution.categorical.logits.detach()
    teacher_logp = F.log_softmax(teacher_logits.masked_fill(~support, float("-inf")), dim=-1)
    teacher_p = teacher_logp.exp()
    taken = teacher_p > 0
    zero = torch.zeros_like(student_logp)
    categorical = (teacher_p * (torch.where(taken, teacher_logp, zero) - torch.where(taken, student_logp, zero))).sum(-1)

    if not duration or teacher.distribution.duration_std is None or student.distribution.duration_std is None:
        return categorical, torch.zeros_like(categorical)
    # the last entry is the no-op, which has no duration
    mean_t = teacher.distribution._duration_mean[:, :-1].detach()
    mean_s = student.distribution._duration_mean[:, :-1]
    std_t = teacher.distribution.duration_std.detach()
    std_s = student.distribution.duration_std
    gaussian = torch.log(std_s / std_t) + (std_t**2 + (mean_t - mean_s) ** 2) / (2 * std_s**2) - 0.5
    weights = torch.where(taken[:, :-1], teacher_p[:, :-1], zero[:, :-1])
    return categorical, (weights * torch.where(taken[:, :-1], gaussian, zero[:, :-1])).sum(-1)


def tick_kl(
    teacher: TickDistribution, student: TickDistribution, selection: TickSelection, duration: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,) categorical and duration KL(teacher || student), summed over the rounds a robot
    was still choosing in, both walked along `selection` (the footsteps actually taken). The
    teacher is a constant target: its later rounds are scored without gradients."""
    student.log_prob(selection)
    with torch.no_grad():
        teacher.log_prob(selection)
    categorical = torch.zeros(selection.index.shape[0], device=selection.index.device)
    durations = torch.zeros_like(categorical)
    for t, s in zip(teacher.path, student.path):
        cat, dur = round_kl(t, s, duration)
        categorical = categorical + torch.where(s.active, cat, torch.zeros_like(cat))
        durations = durations + torch.where(s.active, dur, torch.zeros_like(dur))
    return categorical, durations


class CandidateDistillation(Distillation):
    """See the module docstring. RSL-RL's `DistillationRunner` drives it like its own
    `Distillation`; the loss, the rollout's action mixing and the minibatching differ."""

    student: GaitNetActor
    teacher: GaitNetActor

    def __init__(
        self,
        student: GaitNetActor,
        teacher: GaitNetActor,
        storage: RolloutStorage,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = 1.0,
        duration_coef: float = 1.0,
        student_stochastic: bool = True,
        teacher_action_prob: float = 0.5,
        teacher_action_iterations: int = 200,
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            num_mini_batches: each epoch shuffles the rollout's transitions into this many
            duration_coef: weight of the swing duration KL
            student_stochastic: the student samples its footsteps (else takes its most likely
                ones); sampling visits the states its mistakes lead to
            teacher_action_prob: the fraction of robots, per step, that execute the teacher's
                footsteps at the first iteration (DAgger's beta)
            teacher_action_iterations: that fraction falls linearly to 0 over these iterations
            kwargs: RSL-RL's and Isaac Lab's cfg keys this algorithm doesn't use
                (`gradient_length`, `loss_type`, `teacher_checkpoint`, ...)
        """
        kwargs.pop("gradient_length", None)
        kwargs.pop("loss_type", None)
        kwargs.pop("teacher_checkpoint", None)
        super().__init__(
            student,
            teacher,
            storage,
            num_learning_epochs=num_learning_epochs,
            gradient_length=1,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
            **kwargs,
        )
        if num_mini_batches < 1:
            raise ValueError(f"need at least one minibatch, got {num_mini_batches}")
        if not 0.0 <= teacher_action_prob <= 1.0:
            raise ValueError(f"teacher_action_prob must be in [0, 1], got {teacher_action_prob}")
        self.num_mini_batches = num_mini_batches
        self.duration_coef = duration_coef
        self.student_stochastic = student_stochastic
        self.teacher_action_prob = teacher_action_prob
        self.teacher_action_iterations = max(int(teacher_action_iterations), 0)
        self._executed = torch.zeros((), device=self.device)
        self._teacher_executed = torch.zeros((), device=self.device)

    # --- acting ---

    @property
    def beta(self) -> float:
        """The fraction of robots executing the teacher's footsteps this iteration."""
        if self.teacher_action_iterations == 0:
            return 0.0
        return self.teacher_action_prob * max(0.0, 1.0 - self.num_updates / self.teacher_action_iterations)

    def act(self, obs: TensorDict) -> torch.Tensor:
        student_actions = self.student(obs, stochastic_output=self.student_stochastic)
        teacher_actions = self.teacher(obs)
        actions = student_actions
        beta = self.beta
        if beta > 0.0:
            use_teacher = torch.rand(student_actions.shape[0], device=student_actions.device) < beta
            actions = torch.where(use_teacher.unsqueeze(-1), teacher_actions, student_actions)
            self._teacher_executed += use_teacher.sum()
        self._executed += student_actions.shape[0]
        self.transition.actions = actions.detach()
        self.transition.privileged_actions = teacher_actions.detach()
        self.transition.observations = obs
        return self.transition.actions

    # --- learning ---

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        teacher_actions = self.storage.privileged_actions.flatten(0, 1)
        total = actions.shape[0]
        batch = max(total // self.num_mini_batches, 1)

        sums = {"kl": 0.0, "kl_categorical": 0.0, "kl_duration": 0.0}
        batches = 0
        for _ in range(self.num_learning_epochs):
            order = torch.randperm(total, device=actions.device)
            for start in range(0, batch * self.num_mini_batches, batch):
                index = order[start : start + batch]
                obs = observations[index]
                action = EnvAction.decode(actions[index])
                selection = TickSelection(index=action.choice_index.long(), duration=action.duration)
                with torch.no_grad():
                    teacher = self.teacher.tick_distribution(obs)
                student = self.student.tick_distribution(obs)
                categorical, duration = tick_kl(teacher, student, selection)
                # the teacher's scores are constants; only the student's carry gradients
                loss = (categorical + self.duration_coef * duration).mean()

                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.optimizer.step()

                sums["kl"] += loss.item()
                sums["kl_categorical"] += categorical.mean().item()
                sums["kl_duration"] += duration.mean().item()
                batches += 1

        stats = {key: value / max(batches, 1) for key, value in sums.items()}
        stats.update(self._agreement(observations, teacher_actions))
        stats["teacher_action_prob"] = self.beta
        stats["teacher_executed"] = float(self._teacher_executed / self._executed.clamp(min=1))
        self._executed.zero_()
        self._teacher_executed.zero_()
        self.storage.clear()
        return stats

    @torch.no_grad()
    def _agreement(self, observations: TensorDict, teacher_actions: torch.Tensor, max_rows: int = 8192) -> dict[str, float]:
        """How often the student's most likely first footstep is the teacher's, on (up to
        `max_rows` of) this rollout's states: the same leg or the no-op ("gate"), and the
        same foothold too ("footstep")."""
        rows = torch.randperm(teacher_actions.shape[0], device=teacher_actions.device)[:max_rows]
        student = self.student.tick_distribution(observations[rows])
        first = student.first.deterministic()
        teacher_index = EnvAction.decode(teacher_actions[rows]).choice_index[:, 0].long()
        per_leg = student.candidates.per_leg
        noop = student.candidates.noop_index
        gate_s = torch.where(first.index < noop, first.index // per_leg, torch.full_like(first.index, -1))
        gate_t = torch.where(teacher_index < noop, teacher_index // per_leg, torch.full_like(teacher_index, -1))
        return {
            "agreement_gate": (gate_s == gate_t).float().mean().item(),
            "agreement_footstep": (first.index == teacher_index).float().mean().item(),
            "teacher_step_rate": (teacher_index < noop).float().mean().item(),
            "student_step_rate": (first.index < noop).float().mean().item(),
        }

    # --- checkpoints ---

    def save(self) -> dict:
        saved = super().save()
        saved["distillation_updates"] = self.num_updates
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "student_state_dict" in loaded_dict and (load_cfg is None or load_cfg.get("iteration", True)):
            self.num_updates = int(loaded_dict.get("distillation_updates", 0))
        return load_iteration

    # --- construction ---

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "CandidateDistillation":
        """Like `Distillation.construct_algorithm`, with the teacher built and loaded from
        `cfg["algorithm"]["teacher_checkpoint"]` and its run's cfg."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        checkpoint = alg_cfg.get("teacher_checkpoint")
        if not checkpoint:
            raise ValueError("set agent.algorithm.teacher_checkpoint to the teacher's PPO checkpoint (<run>/model_<i>.pt)")
        saved, teacher_agent, teacher_env = load_teacher_run(checkpoint)

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])
        env_cfg = getattr(env, "cfg", None)
        check_teacher_contract(teacher_env, teacher_agent, env_cfg, env.num_actions, cfg["obs_groups"]["teacher"])

        student_class, student_cfg = resolve_class(cfg["student"])
        teacher_class, teacher_cfg = resolve_class(cfg["teacher"])
        for key in TEACHER_CFG_KEYS:
            if key in teacher_agent["actor"]:
                teacher_cfg[key] = teacher_agent["actor"][key]

        student = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **student_cfg).to(device)
        teacher = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **teacher_cfg).to(device)
        teacher.load_state_dict(teacher_state_dict(saved))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f"Student: {sum(p.numel() for p in student.parameters())} parameters\n{student}")
        print(f"Teacher from {checkpoint} (iteration {saved.get('iter')}): {sum(p.numel() for p in teacher.parameters())} parameters")

        storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(student, teacher, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])
        alg.teacher_loaded = True
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


def check_teacher_contract(teacher_env: dict, teacher_agent: dict, env_cfg, num_actions: int, teacher_groups: list[str]) -> None:
    """Refuse a teacher trained under a contract this env doesn't offer it: another number of
    footsteps per tick, or state features other than its state group builds."""
    rounds = action_layout.rounds_for_dim(num_actions)
    teacher_rounds = int(teacher_env.get("gaitnet", {}).get("max_steps_per_tick", 1))
    if teacher_rounds != rounds:
        raise ValueError(
            f"the teacher was trained with {teacher_rounds} footsteps per tick, this env offers {rounds};"
            f" set env.gaitnet.max_steps_per_tick={teacher_rounds}"
        )
    if list(teacher_agent.get("obs_groups", {}).get("actor", ["state"])) != ["state"]:
        raise ValueError(f"the teacher read {teacher_agent['obs_groups']['actor']}; only a 'state' actor can be a teacher")
    trained = list(teacher_env["observations"]["state"]["robot_state"]["params"]["features"])
    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group in teacher_groups:
            term = getattr(getattr(observations, group, None), "robot_state", None)
            if term is not None and list(term.params["features"]) != trained:
                raise ValueError(
                    f"the teacher was trained on features {trained}, the '{group}' group builds"
                    f" {list(term.params['features'])}"
                )
