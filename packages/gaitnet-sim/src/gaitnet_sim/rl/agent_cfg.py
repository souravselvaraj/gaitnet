"""RSL-RL runner configs for the GaitNet tasks.

PPO settings carry over from the Isaac Lab 2 training script. The actor is
`gaitnet_sim.rl.model.GaitNetActor` over a `gaitnet_core.networks` scoring network; the
critic is RSL-RL's own MLP. Metrics go to MLflow (`MlflowLogWriter`) and to a TensorBoard
log in the run directory.

Presets (`presets=<name>[,<name>...]`), each matched by the env cfg's observation groups:
- spatial: the dense spatial CNN (D1) actor, reading the terrain group
- crop: the candidate scorer with the local-crop encoder, reading the terrain group
- privileged: the critic also reads the privileged group
- slowdown: the actor runs the step-confidence slowdown observer, reading base_command

`GaitNetDistillationRunnerCfg` (agent entry point `rsl_rl_distill_cfg_entry_point`, env preset
`distill`) distills a trained actor into a smaller student that sees the camera map, see
`gaitnet_sim.rl.distillation`.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_tasks.utils import preset

from gaitnet_core.features import DEFAULT_FEATURES
from gaitnet_sim.env.contract import GaitNetCfg


class WriterCfg(dict):
    """An RSL-RL log writer, `{"class_name": "module.Class", **kwargs}`.

    Hashable because Isaac Lab 3.0's train entry point checks
    `agent_cfg.logger in {"wandb", "neptune"}`, assuming the plain string form, while RSL-RL
    needs this dict form for any writer of its own. `to_dict()` turns it back into a dict.

    Name the class `module.Class`, not `module:Class`: configclass wraps "module:Class"
    strings for lazy import and rebuilds the dict around them as a plain, unhashable one.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)


_CONTRACT = GaitNetCfg()
FOOTHOLD_GRID = {
    "resolution": _CONTRACT.grid_resolution,
    "size": list(_CONTRACT.grid_size),
    "border": _CONTRACT.grid_border,
}
"""The env's default foothold grid, which networks that read terrain are built for. A run
with a different grid needs e.g. `agent.actor.network.grid.resolution=...` too; export
refuses a mismatch."""

CANDIDATE_SCORER = {
    "class_name": "CandidateScorer",
    "candidate_features": "xyz",
    "shared_sizes": [128, 128, 128],
    "candidate_sizes": [64, 64],
    "trunk_sizes": [128, 128, 128],
}
SMALL_CANDIDATE_SCORER = {
    **CANDIDATE_SCORER,
    "shared_sizes": [64, 64],
    "candidate_sizes": [32, 32],
    "trunk_sizes": [64, 64],
}
"""About a fifth of the default scorer's parameters (19k vs 103k), for a distilled onboard student."""
SMALL_CROP_SCORER = {**SMALL_CANDIDATE_SCORER, "candidate_features": "xyz_crop", "grid": FOOTHOLD_GRID, "crop_radius": 2}
"""The small scorer with the 5 x 5 cells of terrain around each candidate, for a student that has
to see hole edges on the camera map."""
FIXED_SWING_DURATION = 0.25
"""Swing duration (s) of the `swing_duration_ablation` preset, overridable as
`agent.actor.network.fixed_duration=0.3`."""
CROP_SCORER ={**CANDIDATE_SCORER, "candidate_features": "xyz_crop", "grid": FOOTHOLD_GRID, "crop_radius": 2}
DENSE_SPATIAL_CNN = {
    "class_name": "DenseSpatialCNN",
    "grid": FOOTHOLD_GRID,
    "channels": [16, 16, 16],
    "state_sizes": [128, 64],
    "noop_sizes": [64],
}
SLOWDOWN_OBSERVERS = {"step_confidence_slowdown": {"patience": 10, "margin": 0.0, "scale": 0.5}}


@configclass
class ScheduledPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """RSL-RL's PPO cfg plus `gaitnet_sim.rl.ppo.ScheduledPPO`'s schedules, see there."""

    class_name: str = "gaitnet_sim.rl.ppo:ScheduledPPO"
    lr_schedule: dict | None = None
    """`{start, end, final[, shape]}` in iterations, None for a constant learning rate."""
    entropy_schedule: dict | None = None
    """The same for the entropy coefficient."""
    policy_change_samples: int = 8192
    """Transitions `approx_kl` and `clip_fraction` are measured on after each update."""


LR_DECAY = {"start": 4000, "end": 8000, "final": 1e-5, "shape": "cosine"}
"""Hold the learning rate to iteration 4000, then decay it to 1e-5 by 8000. The baseline's evals
peaked near 5000 and drifted after; override per run, e.g. agent.algorithm.lr_schedule.end=9000."""
ENTROPY_DECAY = {"start": 4000, "end": 8000, "final": 0.005, "shape": "linear"}
"""Lower the entropy bonus from 0.02 to 0.005 over the same window, so the policy commits."""


@configclass
class GaitNetActorCfg:
    """Keyword arguments of `GaitNetActor`, see there."""

    class_name: str = "gaitnet_sim.rl.model:GaitNetActor"
    network = preset(
        default=CANDIDATE_SCORER,
        spatial=DENSE_SPATIAL_CNN,
        crop=CROP_SCORER,
        swing_duration_ablation={**CANDIDATE_SCORER, "fixed_duration": FIXED_SWING_DURATION},
    )
    candidates_group: str = "candidates"
    terrain_group: str = "terrain"
    observers = preset(default={}, slowdown=SLOWDOWN_OBSERVERS)
    base_command_group: str = "base_command"
    duration_std: float = 0.05
    duration_std_floor: float = 0.01
    # the actor can't read the env's contract, so these repeat it for the later rounds of a
    # tick (env.gaitnet.max_steps_per_tick > 1); export checks that they still agree
    state_features: list[str] = list(DEFAULT_FEATURES)
    min_stance_after_step: int = 2
    # Isaac Lab's cfg handling reads these on every model cfg; the actor's distribution is fixed
    distribution_cfg: None = None
    stochastic: bool = False


@configclass
class GaitNetPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 250  # half an episode per rollout
    max_iterations = 10000
    save_interval = 5
    experiment_name = "gaitnet_holes"
    obs_groups = {"actor": ["state"], "critic": preset(default=["state"], privileged=["state", "privileged"])}
    actor = GaitNetActorCfg()
    critic = RslRlMLPModelCfg(hidden_dims=[64] * 6, activation="relu", obs_normalization=False)
    algorithm = ScheduledPpoAlgorithmCfg(
        lr_schedule=dict(LR_DECAY),
        entropy_schedule=dict(ENTROPY_DECAY),
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.3,
        entropy_coef=0.02,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="fixed",  # the lr_schedule above; see gaitnet_sim.rl.ppo for why not "adaptive"
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,  # only used by schedule="adaptive"
        max_grad_norm=1.0,
    )
    logger = WriterCfg(class_name="gaitnet_sim.rl.mlflow_writer.MlflowLogWriter", experiment_name="gaitnet")


@configclass
class GaitNetPillarsPpoRunnerCfg(GaitNetPpoRunnerCfg):
    experiment_name = "gaitnet_pillars"


@configclass
class CandidateDistillationAlgorithmCfg:
    """Keyword arguments of `gaitnet_sim.rl.distillation.CandidateDistillation`, see there."""

    class_name: str = "gaitnet_sim.rl.distillation:CandidateDistillation"
    teacher_checkpoint: str = ""
    """The teacher's PPO checkpoint, `<run dir>/model_<i>.pt`; its run's `params/` give the
    teacher's network. Required, e.g. `agent.algorithm.teacher_checkpoint=/path/model_5000.pt`."""
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    duration_coef: float = 1.0
    student_stochastic: bool = True
    teacher_action_prob: float = 0.5
    teacher_action_iterations: int = 200
    optimizer: str = "adam"


@configclass
class GaitNetDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill a trained GaitNet actor (the teacher, on the truth) into a small student that
    sees the camera map and observation noise. Run with `--agent rsl_rl_distill_cfg_entry_point
    presets=distill` (the env's teacher groups) and the teacher's contract, e.g.
    `env.gaitnet.max_steps_per_tick=2`."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "gaitnet_holes"
    obs_groups = {"student": ["state"], "teacher": ["teacher_state"]}
    student = GaitNetActorCfg(network=preset(default=SMALL_CANDIDATE_SCORER, crop=SMALL_CROP_SCORER))
    """`presets=distill,crop` gives the student the terrain around each candidate (and the env its
    terrain group); the teacher needs neither, whatever its network."""
    teacher = GaitNetActorCfg(candidates_group="teacher_candidates")
    """The teacher's network, duration floor and state features are replaced by its run's."""
    algorithm = CandidateDistillationAlgorithmCfg()
    logger = WriterCfg(class_name="gaitnet_sim.rl.mlflow_writer.MlflowLogWriter", experiment_name="gaitnet")


@configclass
class GaitNetPillarsDistillationRunnerCfg(GaitNetDistillationRunnerCfg):
    experiment_name = "gaitnet_pillars"
