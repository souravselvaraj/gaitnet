from isaaclab.app import AppLauncher
import argparse
import dataclasses

from gaitnet import PROJECT_ROOT
from gaitnet.control.mpc.convex_MPC import Gait

PRESETS: dict[str, dict] = {
    "sc_idm": dict(
        gpu_mpc=True,
        num_envs=1024,
        arch="gaitnet_t",
        recipe="step",
        idm_inverse=str(PROJECT_ROOT / "training/gaitnet/idm_models/gaitnet_inverse.pt"),
        curriculum="walk",
        walk_p_down=0.3,
        walk_random_fraction=0.25,
        max_init_terrain_level=2,
        commands="forward",
        tracking_std=0.1,
        yaw_tracking_std=0.2,
        critic_noop_fix=True,
        num_steps_per_env=100,
        num_mini_batches=8,
        init_at_random_ep_len=True,
    ),
}
"""Named training settings for --preset. sc_idm: the run behind the best policy so far (SC-idm,
model_1590): GaitNet-T on the step recipe, its footsteps sent through the training controller's
inverse footstep model (idm.py), the walk curriculum, forward commands. That policy got there in
stages (890 iterations without the IDM, on omni commands, before this setting); the preset is the
final setting, trained from scratch."""

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--num_envs", type=int, default=100, help="Number of environments to simulate."
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seeds the environment (Isaac Lab seeds torch, numpy, random and warp with it, so the network's "
    "initial weights too) and the GPU controller's randomisation. Default: unseeded environment, controller 0.",
)
parser.add_argument(
    "--preset",
    choices=sorted(PRESETS),
    default=None,
    help="Start from a named training setting (PRESETS); flags given on the command line still override it.",
)
parser.add_argument(
    "--max_iterations", type=int, default=10000, help="RL Policy training iterations."
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from.",
)
parser.add_argument(
    "--num_steps_per_env",
    type=int,
    default=500,
    help="Rollout length per PPO iteration (env steps per robot). 500 = one 20 s episode; lower only for smoke tests.",
)
parser.add_argument(
    "--num_mini_batches",
    type=int,
    default=4,
    help="PPO minibatches per epoch. Scale with --num_envs to keep the minibatch (and the critic's "
    "activation memory, ~2 GB per layer at 128k samples) bounded.",
)
parser.add_argument(
    "--gpu_mpc",
    action="store_true",
    default=False,
    help="Run the low-level MPC batched on the GPU (gaitnet/sim2real/gpu_mpc.py) instead of one "
    "CPU process per robot. Off by default.",
)
parser.add_argument(
    "--gpu_mpc_variant",
    choices=["gaitnet", "gaitnet_core"],
    default="gaitnet",
    help="gaitnet: reproduce the CPU controller in this repo (what the checkpoints were trained "
    "against). gaitnet_core: gaitnet-mpc's tuning. Only with --gpu_mpc.",
)
parser.add_argument(
    "--mpc_solver_iterations",
    type=int,
    default=80,
    help="ADMM iterations per GPU MPC solve. Only with --gpu_mpc.",
)
parser.add_argument(
    "--arch",
    choices=["gaitnet", "gaitnet_t"],
    default="gaitnet",
    help="Actor: gaitnet (GaitnetActor) or gaitnet_t (GaitnetTActor, gaitnet_t.py). With --resume of a "
    "GaitnetActor checkpoint, gaitnet_t converts it (same outputs at start) and starts a fresh optimizer.",
)
parser.add_argument(
    "--history_len",
    type=int,
    default=None,
    help="Policy ticks of history appended to the observation (components/policy_history.py). "
    "Default: 25 with --arch gaitnet_t, else 0. Needs --gpu_mpc.",
)
parser.add_argument(
    "--recipe",
    choices=["thesis", "step", "teacher"],
    default="thesis",
    help="thesis: current rewards, no controller randomisation. step: op_penalty -0.4, velocity "
    "tracking 1.0, and per-env controller randomisation (needs --gpu_mpc). teacher: step plus "
    "progress, void-edge touchdown penalty and mild body stability (env_cfg/gaitnet_env_cfg.py).",
)
parser.add_argument(
    "--teacher",
    type=str,
    default=None,
    help="Privileged GaitNet-T teacher checkpoint (a --privileged run) to distill into this run's "
    "GaitNet-T student (needs --arch gaitnet_t and --distill). The student sees no privileged inputs.",
)
parser.add_argument(
    "--distill",
    choices=["dagger_bc", "dagger_ppo"],
    default=None,
    help="dagger_bc: KL to the teacher's masked categorical + duration MSE, no RL loss. dagger_ppo: PPO + "
    "beta * the same, beta 1 -> 0 over --distill_beta_iterations. Rollouts use the student's actions.",
)
parser.add_argument(
    "--curriculum",
    choices=["vel", "walk"],
    default="vel",
    help="Terrain curriculum. vel: Isaac Lab's terrain_levels_vel, which with this command range keeps "
    "every robot on level 0 (0-4%% voids). walk: up for surviving and walking half the commanded "
    "distance, down for falling or stalling (components/walk_curriculum.py).",
)
parser.add_argument(
    "--tracking_std",
    type=float,
    default=None,
    help="Width of the planar velocity tracking reward's kernel, exp(-err^2/std^2) (m/s). Default 0.5, "
    "which with commands of at most 0.2 m/s per axis pays a robot standing still 0.90 of the reward; "
    "0.1 pays it 0.19.",
)
parser.add_argument(
    "--yaw_tracking_std",
    type=float,
    default=None,
    help="Width of the yaw rate tracking reward's kernel (rad/s). Default 0.5.",
)
parser.add_argument(
    "--walk_p_down",
    type=float,
    default=None,
    help="--curriculum walk: probability that a robot that fell or stalled moves down a level (default "
    "0.5). With p_up 0.1 the levels settle where p_up / (p_up + p_down) of episodes fail: 1/6 at 0.5, "
    "1/4 at 0.3.",
)
parser.add_argument(
    "--walk_p_up",
    type=float,
    default=None,
    help="--curriculum walk: probability that a robot that survived and walked far enough moves up a level "
    "(default 0.1). With p_down 0.3: 0.1 settles where 1/4 of episodes fail, 0.2 where 2/5 do.",
)
parser.add_argument(
    "--walk_random_fraction",
    type=float,
    default=None,
    help="--curriculum walk: share of episodes run on a uniformly random terrain level, outside the "
    "curriculum (components/walk_curriculum.next_levels). Default 0.",
)
parser.add_argument(
    "--max_init_terrain_level",
    type=int,
    default=None,
    help="Highest terrain level a robot can start on (default: any of the 12). Use a low one when resuming "
    "a trained policy, so the curriculum does not spend hours bringing every robot back down.",
)
parser.add_argument(
    "--commands",
    choices=["omni", "forward"],
    default="omni",
    help="omni: x, y in [-0.2, 0.2] m/s and yaw rate in [-0.4, 0.4] rad/s. forward: x in [0, 0.2] m/s only, "
    "yaw rate within --forward_yaw_rate; the deployed camera sees nothing beside or behind the robot.",
)
parser.add_argument("--forward_yaw_rate", type=float, default=0.2, help="rad/s, with --commands forward.")
parser.add_argument(
    "--max_forward_speed",
    type=float,
    default=None,
    help="With --commands forward, the top of the x range (m/s); default keeps 0.2. The training controller's IDM "
    "was fitted on commands up to 0.2 m/s (body speed p99 0.25), so past ~0.25 it extrapolates.",
)
parser.add_argument(
    "--idm_inverse",
    type=str,
    default=None,
    help="Inverse footstep model (idm.py) of the training controller: the chosen candidate is sent through "
    "it so the foot lands on it, instead of about speed x swing time past it (needs --gpu_mpc).",
)
parser.add_argument(
    "--controller_randomization",
    choices=["recipe", "none", "agnostic", "agnostic_ahead", "agnostic_mix", "agnostic_ahead_mix"],
    default="recipe",
    help="Low-level controller randomisation (--gpu_mpc). recipe: what --recipe sets (step/teacher: "
    "STEP_RANDOMIZATION). agnostic: AGNOSTIC_RANDOMIZATION, for a policy that has to run on other "
    "controllers (OCS2): travel compensation 0-1, landing offsets and noise of several cm, latency. "
    "agnostic_ahead: the same with every step lifting 3-6 ticks after it is chosen, for deployment where each "
    "step is announced to OCS2 0.1-0.2 s ahead (gaitnet_long_swing_lead). *_mix: half the episodes on the "
    "step recipe's controller instead (with the same commit-ahead for agnostic_ahead_mix), so the void skill "
    "learned there is kept.",
)
parser.add_argument("--distill_beta_iterations", type=int, default=200)
parser.add_argument("--distill_duration_coef", type=float, default=1.0)
parser.add_argument(
    "--lr_warmup_iterations",
    type=int,
    default=None,
    help="Ramp the learning rate linearly from 1%% to 100%% over this many iterations. A fresh Adam takes "
    "sign-sized steps on every weight at first, which wrecks an already-trained network (a converted "
    "GaitNet-T collapsed to always-no-op after one update without it). Default: 20 when a GaitnetActor "
    "checkpoint is converted or when distilling (fresh optimizer either way), else 0.",
)
parser.add_argument(
    "--privileged",
    action="store_true",
    default=False,
    help="Teacher inputs (needs --arch gaitnet_t and --gpu_mpc): the controller knobs and per-candidate "
    "terrain features appended to the observation, read by a widened GaitNet-T (teacher.py) that "
    "starts out identical to the plain one.",
)
parser.add_argument(
    "--init_at_random_ep_len",
    action="store_true",
    default=False,
    help="Start every env at a random point of its episode. Use with --num_steps_per_env below 500, or "
    "every rollout covers the same stretch of every episode (they all start and time out together).",
)
parser.add_argument(
    "--critic_tail",
    action="store_true",
    default=False,
    help="Asymmetric critic: the critic also reads everything after the candidates (the policy history, "
    "and the privileged features with --privileged or a teacher). The actor is unchanged. Starts out "
    "identical to the loaded critic; resuming a checkpoint without the tail starts a fresh optimizer.",
)
parser.add_argument(
    "--critic_noop_fix",
    action="store_true",
    default=False,
    help="Make the critic read one-hot column 0 as no-op instead of column 4 (the RL leg, which every "
    "checkpoint's critic was trained with). Resume a run trained with it with it.",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="",
    help="Suffix for the run directory, training/gaitnet/runs/gaitnet_<timestamp>_<run_name>.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# a preset only replaces defaults, so anything given on the command line wins
preset = parser.parse_known_args()[0].preset
if preset is not None:
    parser.set_defaults(**PRESETS[preset])
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import multiprocessing
import signal
import datetime
import os
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
import gaitnet.simulation.cfg.footstep_scanner_constants as fs
from rsl_rl.runners import on_policy_runner
import rsl_rl.modules
import re
import torch
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env
from gaitnet.gaitnet.components.policy_history import HISTORY_DIM
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor
from gaitnet.gaitnet.teacher import PrivilegedActor, is_gaitnet_t_state_dict, is_privileged_state_dict
from gaitnet.gaitnet.distill import DistillationPPO
from gaitnet.sim2real.robot_controllers import ControllerRandomizationCfg, GpuMpcCfg
from gaitnet.util import log_exceptions
from gaitnet.gaitnet import gaitnet
import gaitnet.constants as const
from gaitnet import get_logger

logger = get_logger()

# # Global flag for graceful shutdown
# shutdown_requested = False


# def signal_handler(sig, frame):
#     global shutdown_requested
#     if multiprocessing.current_process().name == "MainProcess":
#         signal_name = signal.Signals(sig).name
#         logger.info(f"signal {signal_name} received in main process, shutting down...")
#     shutdown_requested = True


# # Set up the signal handler for SIGINT (Ctrl+C)
# signal.signal(signal.SIGINT, signal_handler)

###############################################################################
###############################################################################


STEP_RANDOMIZATION = ControllerRandomizationCfg(
    command_tracking=(0.3, 1.0),
    planar_weight_scale=(0.5, 1.0),
    swing_apex=(0.07, 0.12),
    touchdown_bias=0.02,
    touchdown_noise=(0.01, 0.01),
    liftoff_delay_ticks=(0, 1),
)
"""Controller randomisation of the `step` recipe."""


AGNOSTIC_RANDOMIZATION = ControllerRandomizationCfg(
    command_tracking=(0.3, 1.0),
    planar_weight_scale=(0.5, 1.0),
    swing_apex=(0.07, 0.12),
    touchdown_bias=0.02,
    touchdown_noise=(0.0, 0.05),
    liftoff_delay_ticks=(0, 2),
    travel_compensation=(0.0, 1.0),
    touchdown_along=(-0.05, 0.02),
    touchdown_outboard=(0.0, 0.05),
)
"""--controller_randomization agnostic: STEP_RANDOMIZATION widened to cover executors other than the
training controller. Per episode: how much body travel the target compensates (0 = this controller,
1 = OCS2 / gaitnet_core), a constant landing offset short/long and outboard (OCS2 in Gazebo lands SC's
feet 3-5 cm short and 3-5 cm outboard), per-step landing noise up to 5 cm (OCS2's foothold region
leaves 5-7 cm of spread), and up to 2 ticks of command latency."""


AGNOSTIC_AHEAD_RANDOMIZATION = dataclasses.replace(AGNOSTIC_RANDOMIZATION, liftoff_delay_ticks=(3, 6))
"""--controller_randomization agnostic_ahead: AGNOSTIC_RANDOMIZATION with every step committed 3-6 ticks (0.12-0.24 s)
before it lifts. In Gazebo, announcing each step 0.1-0.2 s ahead lets OCS2's MPC prepare for it (SC-agnostic survival
20 -> 47-60 %, decisions.md 2026-10-12 ~20:15); OCS2 adds its own ~50 ms on top."""


AGNOSTIC_MIX_RANDOMIZATION = dataclasses.replace(AGNOSTIC_RANDOMIZATION, mix=STEP_RANDOMIZATION, mix_fraction=0.5)
"""--controller_randomization agnostic_mix: half the episodes AGNOSTIC_RANDOMIZATION, half STEP_RANDOMIZATION. On the
agnostic executors alone the curriculum sank to ~4 % voids and SC-agnostic lost its void skill (it 1270 -> 1485, IDM
eval 10-30 % voids @0.2 0.97 -> 0.67, 40 % 0.82 -> 0.39) while gaining on the OCS2-like executor; the step half keeps
the terrain the policy learned on the training controller."""

AGNOSTIC_AHEAD_MIX_RANDOMIZATION = dataclasses.replace(
    AGNOSTIC_AHEAD_RANDOMIZATION, mix=dataclasses.replace(STEP_RANDOMIZATION, liftoff_delay_ticks=(3, 6)), mix_fraction=0.5
)
"""--controller_randomization agnostic_ahead_mix: AGNOSTIC_MIX with every step committed 3-6 ticks ahead in both halves."""


def curriculum_file(checkpoint: str) -> str | None:
    """training/.../model_<it>.pt -> .../curriculum_<it>.pt (not model_*.pt, which the checkpoint globs read)."""
    path = re.sub(r"model_(\d+)\.pt$", r"curriculum_\1.pt", checkpoint)
    return path if path != checkpoint else None


def carry_curriculum(runner: on_policy_runner.OnPolicyRunner, env, resume: str | None) -> None:
    """Keep the terrain curriculum across restarts: every checkpoint the runner saves gets the robots'
    curriculum levels beside it (curriculum_file), and --resume puts them back when that file is there and
    the robot count matches. Without it a resumed run starts again on levels 0..max_init_terrain_level."""
    terrain = env.scene.terrain
    saved = curriculum_file(resume) if resume else None
    if saved and os.path.exists(saved):
        level = torch.load(saved, map_location=env.device)["level"].to(env.device)
        if level.shape == terrain.terrain_levels.shape:
            env.walk_curriculum_level = level.clone()
            env.walk_explored = torch.zeros_like(level, dtype=torch.bool)
            terrain.terrain_levels[:] = level
            terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
            logger.info(f"terrain curriculum restored from {saved}: mean level {level.float().mean():.2f}")
        else:
            logger.warning(f"{saved} holds {level.numel()} robots, this run has {terrain.terrain_levels.numel()}: not restored")
    elif resume:
        logger.info(f"no {saved}: the curriculum starts on levels 0-{terrain.cfg.max_init_terrain_level}")
    save = runner.save

    def save_with_curriculum(path: str, infos: dict | None = None) -> None:
        save(path, infos)
        target = curriculum_file(path)
        if target:
            torch.save({"level": getattr(env, "walk_curriculum_level", terrain.terrain_levels).cpu()}, target)

    runner.save = save_with_curriculum


def make_gaitnet_actor() -> gaitnet.GaitnetActor:
    """The GaitnetActor every checkpoint here was trained with."""
    return gaitnet.GaitnetActor(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[128, 128, 128],
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[64, 64],
        trunk_layer_sizes=[128, 128, 128],
    ).to(args_cli.device)


def submodule_state(state: dict, prefix: str) -> dict:
    """The entries of a state dict under `prefix.`, with the prefix removed."""
    return {re.sub(rf"^{prefix}\.", "", k): v for k, v in state.items() if k.startswith(prefix + ".")}


def main():
    history_len = args_cli.history_len
    if history_len is None:
        history_len = 25 if args_cli.arch == "gaitnet_t" else 0
    if (history_len > 0 or args_cli.recipe != "thesis" or args_cli.privileged) and not args_cli.gpu_mpc:
        raise ValueError("--recipe step/teacher, --privileged and the policy history need --gpu_mpc")
    if args_cli.privileged and args_cli.arch != "gaitnet_t":
        raise ValueError("--privileged needs --arch gaitnet_t")
    if (args_cli.teacher is None) != (args_cli.distill is None):
        raise ValueError("--teacher and --distill go together")
    distilling = args_cli.teacher is not None
    if distilling and (args_cli.arch != "gaitnet_t" or args_cli.privileged or not args_cli.gpu_mpc):
        raise ValueError("distillation trains a non-privileged --arch gaitnet_t student with --gpu_mpc")

    gpu_mpc = None
    if args_cli.gpu_mpc:
        gpu_mpc = GpuMpcCfg(
            variant=args_cli.gpu_mpc_variant,
            solver_iterations=args_cli.mpc_solver_iterations,
            seed=args_cli.seed or 0,
        )
        if args_cli.recipe in ("step", "teacher"):
            gpu_mpc.randomization = STEP_RANDOMIZATION
        if args_cli.controller_randomization == "agnostic":
            gpu_mpc.randomization = AGNOSTIC_RANDOMIZATION
        elif args_cli.controller_randomization == "agnostic_ahead":
            gpu_mpc.randomization = AGNOSTIC_AHEAD_RANDOMIZATION
        elif args_cli.controller_randomization == "agnostic_mix":
            gpu_mpc.randomization = AGNOSTIC_MIX_RANDOMIZATION
        elif args_cli.controller_randomization == "agnostic_ahead_mix":
            gpu_mpc.randomization = AGNOSTIC_AHEAD_MIX_RANDOMIZATION
        elif args_cli.controller_randomization == "none":
            gpu_mpc.randomization = ControllerRandomizationCfg()
        logger.info(f"low-level controller: batched GPU MPC, {gpu_mpc}")
    else:
        logger.info("low-level controller: CPU MPC process pool")
    logger.info(
        f"arch {args_cli.arch}, recipe {args_cli.recipe}, history {history_len} x {HISTORY_DIM}, "
        f"privileged {args_cli.privileged}"
    )
    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=GaitNetEnv,
        gpu_mpc=gpu_mpc,
        recipe=args_cli.recipe,
        history_len=history_len,
        # a distilled student's env still carries the teacher's privileged features (it ignores them)
        privileged=args_cli.privileged or distilling,
        curriculum=args_cli.curriculum,
        tracking_std=args_cli.tracking_std,
        yaw_tracking_std=args_cli.yaw_tracking_std,
        walk_p_down=args_cli.walk_p_down,
        walk_p_up=args_cli.walk_p_up,
        walk_random_fraction=args_cli.walk_random_fraction,
        max_init_terrain_level=args_cli.max_init_terrain_level,
        commands=args_cli.commands,
        forward_yaw_rate=args_cli.forward_yaw_rate,
        idm_inverse=args_cli.idm_inverse,
        max_forward_speed=args_cli.max_forward_speed,
        seed=args_cli.seed,
    )
    logger.info(
        f"terrain curriculum: {args_cli.curriculum}; tracking std {args_cli.tracking_std or 0.5} m/s, "
        f"yaw {args_cli.yaw_tracking_std or 0.5} rad/s; commands {args_cli.commands}, "
        f"x up to {args_cli.max_forward_speed or 0.2} m/s"
    )
    episode_info = {}
    env.episode_info = episode_info

    gaitnet_actor = make_gaitnet_actor()

    gaitnet_critic = gaitnet.GaitnetCritic(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[64, 64, 64],
        num_unique_states=const.gait_net.num_footstep_options*const.robot.num_legs+1,  # +1 for the "no step" option
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[32, 32],
        trunk_layer_sizes=[64, 64, 64],
        trunk_combiner_head_sizes=[32, 32],
    ).to(args_cli.device)

    # GaitNet-T: convert a GaitnetActor (trained, from --resume, or fresh) so it starts out
    # computing exactly what that actor computes; the critic is loaded as is, the optimizer is new
    old_actor = None
    resume_iteration = None
    load_with_runner = args_cli.resume is not None
    if args_cli.arch == "gaitnet_t":
        if args_cli.resume is not None:
            checkpoint = torch.load(args_cli.resume, map_location=args_cli.device, weights_only=False)
            state = checkpoint["model_state_dict"]
            if is_gaitnet_t_state_dict(state):
                if is_privileged_state_dict(state) != args_cli.privileged:
                    raise ValueError("--resume's teacher/non-teacher actor does not match --privileged")
                logger.info("--resume is a GaitNet-T checkpoint: resuming it as is")
            else:
                gaitnet_actor.load_state_dict(submodule_state(state, "actor"))
                gaitnet_critic.load_state_dict(submodule_state(state, "critic"))
                resume_iteration = checkpoint["iter"]
                load_with_runner = False
                logger.info(f"converting the GaitnetActor in {args_cli.resume} (iteration {resume_iteration}) to GaitNet-T")
        converted_from = gaitnet_actor.eval()
        gaitnet_actor = GaitnetTActor.from_gaitnet_actor(
            converted_from, history_len=history_len, history_dim=HISTORY_DIM
        ).to(args_cli.device)
        if args_cli.privileged:
            gaitnet_actor = PrivilegedActor.widen(gaitnet_actor)
        if resume_iteration is not None:
            old_actor = converted_from

    if args_cli.critic_noop_fix:
        gaitnet_critic.noop_index = 0
    if args_cli.critic_tail:
        obs_dim = env.observation_manager.group_obs_dim["policy"][0]
        tail_dim = obs_dim - (const.gait_net.robot_state_dim + gaitnet.NUM_OPTIONS * const.gait_net.footstep_option_dim)
        if tail_dim <= 0:
            raise ValueError("--critic_tail needs something after the candidates: a history or privileged features")
        gaitnet_critic.add_tail(tail_dim)
        logger.info(f"asymmetric critic: reads the {tail_dim} observation values after the candidates")

    actor_critic_class = gaitnet.GaitnetActorCritic
    on_policy_runner.__dict__[actor_critic_class.__name__] = actor_critic_class

    teacher = None
    student_obs_dim = None
    if distilling:
        teacher = PrivilegedActor.widen(
            GaitnetTActor.from_gaitnet_actor(make_gaitnet_actor(), history_len=history_len, history_dim=HISTORY_DIM)
        ).to(args_cli.device)
        teacher_state = torch.load(args_cli.teacher, map_location=args_cli.device, weights_only=False)["model_state_dict"]
        if not is_privileged_state_dict(teacher_state):
            raise ValueError(f"--teacher {args_cli.teacher} is not a privileged GaitNet-T checkpoint")
        teacher.load_state_dict(submodule_state(teacher_state, "actor"))
        student_obs_dim = (
            const.gait_net.robot_state_dim
            + gaitnet.NUM_OPTIONS * const.gait_net.footstep_option_dim
            + history_len * HISTORY_DIM
        )
        on_policy_runner.__dict__[DistillationPPO.__name__] = DistillationPPO
        logger.info(f"distilling {args_cli.teacher} ({args_cli.distill}); the student reads {student_obs_dim} values")

    # wrap for RL training
    env = RslRlVecEnvWrapper(env)

    # Create unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gaitnet_{timestamp}" + (f"_{args_cli.run_name}" if args_cli.run_name else "")

    # Create unique log directory
    log_dir = f"./training/gaitnet/runs/{experiment_name}"
    save_dir = f"./training/gaitnet/checkpoints/{experiment_name}"

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare config dict for OnPolicyRunner
    train_cfg = {
        "algorithm": {
            "class_name": "DistillationPPO" if distilling else "PPO",
            "schedule": "fixed",  # we don't support adaptive
            # specified the proximal part of PPO - larger = faster at cost of stability
            "clip_param": 0.3,
            # how many times to use each batch of data for gradient updates
            "num_learning_epochs": 8,
            # number of minibatches to split one batch of data into
            "num_mini_batches": args_cli.num_mini_batches,
            "value_loss_coef": 0.5,
            # controls entropy - higher = more exploration, slower convergence
            "entropy_coef": 0.02,
            # learning rate
            "learning_rate": 3e-4,
            # clips gradients to prevent unstable training
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            # decay rate of future rewards. Closer to 1 = long horizon, closer to 0 = short horizon
            # H=1/(1-gamma); 0.98 corresponds to 2s, 0.99 to 5s
            "gamma": 0.99,
            "lam": 0.95,
        },
        "policy": {
            "class_name": actor_critic_class.__name__,
            "actor": gaitnet_actor,
            "critic": gaitnet_critic,
            "episode_info": episode_info,
            # GaitNet-T's attention: chunked + checkpointed over the 128k-sample PPO minibatches
            "actor_chunk_size": 16384 if args_cli.arch == "gaitnet_t" else None,
            "actor_obs_dim": student_obs_dim,
        },
        "obs_groups": {  # no private observations
            "policy": ["policy",],
            "critic": ["policy",],
        },
        "log_dir": log_dir,
        "num_steps_per_env": args_cli.num_steps_per_env,  # default 500: ~1 episode per batch (episode = 20s = 500 iterations)
        "save_interval": 5,
        "empirical_normalization": False,
        "logger": "tensorboard",  # Explicitly set TensorBoard as logger
    }

    if distilling:
        train_cfg["algorithm"].update(
            teacher=teacher,
            distill_mode=args_cli.distill,
            beta_iterations=args_cli.distill_beta_iterations,
            duration_coef=args_cli.distill_duration_coef,
        )

    runner = on_policy_runner.OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=train_cfg["log_dir"],
        device=args_cli.device,
    )

    # Resume from checkpoint if specified
    if load_with_runner:
        logger.info(f"Loading checkpoint from {args_cli.resume}")
        runner.load(args_cli.resume, load_optimizer=not distilling)
        if getattr(runner.alg.policy, "loaded_without_critic_tail", False):
            resume_iteration = torch.load(args_cli.resume, map_location="cpu", weights_only=False)["iter"]
            runner.current_learning_iteration = resume_iteration
    elif resume_iteration is not None:
        runner.current_learning_iteration = resume_iteration
    fresh_optimizer = (
        old_actor is not None or distilling or getattr(runner.alg.policy, "loaded_without_critic_tail", False)
    )
    warmup = args_cli.lr_warmup_iterations
    if warmup is None:  # a fresh optimizer on trained weights: converted, critic widened, or distilled
        warmup = 20 if fresh_optimizer else 0
    if warmup > 0:
        alg, base_lr, update = runner.alg, runner.alg.learning_rate, runner.alg.update
        updates_done = [0]

        def warmed_up_update():
            lr = base_lr * min(1.0, 0.01 + 0.99 * updates_done[0] / max(warmup - 1, 1))
            for group in alg.optimizer.param_groups:
                group["lr"] = lr
            alg.learning_rate = lr
            updates_done[0] += 1
            return update()

        alg.update = warmed_up_update
        logger.info(f"learning rate warm-up over {warmup} iterations to {base_lr}")
    if old_actor is not None:
        # the converted actor must reproduce the old one on real observations
        with torch.no_grad():
            obs = env.get_observations()["policy"]
            n_old = const.gait_net.robot_state_dim + gaitnet.NUM_OPTIONS * const.gait_net.footstep_option_dim
            logits_old, duration_old = old_actor(obs[:, :n_old])
            logits_new, duration_new = runner.alg.policy._actor_forward(obs)  # the actor's own view
        logger.info(
            f"GaitNet-T warm start on {obs.shape[0]} real observations ({obs.shape[1]} values): "
            f"max |d logits| {(logits_new - logits_old).abs().max().item():.2e}, "
            f"max |d duration| {(duration_new - duration_old).abs().max().item():.2e}"
        )

    carry_curriculum(runner, env.unwrapped, args_cli.resume)
    logger.info("Starting training...")

    # Let the runner handle the entire training loop
    # This ensures proper TensorBoard logging continuity
    try:
        runner.learn(
            num_learning_iterations=args_cli.max_iterations,
            init_at_random_ep_len=args_cli.init_at_random_ep_len,
        )
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Closing environments")
    env.unwrapped.cfg.robot_controllers.close()  # stop the MPC worker processes, if any
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
