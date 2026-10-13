"""Deterministic evaluation of trained GaitNet / GaitNet-T checkpoints at fixed void densities and
speeds, the way the policy would be deployed: argmax over the same masked candidate set PPO samples
from, no controller randomisation, a straight-ahead command.

Training curves can't say whether a policy is getting better while the terrain curriculum keeps
moving robots between levels; this can. Every (void fraction, speed) cell gets its own block of
1 m wide, `--strip_length` m long strips (the thesis evaluation's corridors, eval/eval_gaitnet.py),
several robots per strip (Isaac Lab filters collisions between envs), all spawned on the strip's
solid centre platform facing +x. One env serves every checkpoint: it is built with the policy
history and the privileged features, and each actor is handed the slice of the observation it
was trained on. The terrain and spawn are seeded, so two checkpoints see the same voids.

Per cell it reports survival (no fall), the distance walked along x as a fraction of the commanded
distance, success (survived and walked at least half the commanded distance), steps per tick and
why robots fell, with Wilson 95% intervals on the rates. It also measures where each foot lands
against the spot the policy chose: the chosen candidate is a point on the terrain relative to the hip
when the step was decided, so the error is the foot's touchdown position minus that point, in the
yaw frame of that moment (along / lateral). The controller's own `touchdown_error` is measured
against the hip at touchdown and cannot show this.

    python -m gaitnet.gaitnet.eval_policy --headless --checkpoints a.pt b.pt --labels A B --out <dir>
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Evaluate GaitNet checkpoints at fixed void densities.")
parser.add_argument("--checkpoints", nargs="+", required=True, help="model_<it>.pt files (rsl_rl runner checkpoints).")
parser.add_argument("--labels", nargs="*", default=None, help="A name per checkpoint (default: run dir + iteration).")
parser.add_argument("--difficulties", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4], help="Void fractions.")
parser.add_argument("--velocities", nargs="+", type=float, default=[0.1, 0.2], help="Forward commands (m/s).")
parser.add_argument("--envs_per_cell", type=int, default=150)
parser.add_argument("--strips_per_cell", type=int, default=30)
parser.add_argument("--strip_length", type=float, default=10.0, help="m along x; 0.2 m/s for 20 s needs 4 m each way.")
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--gpu_mpc_variant", choices=["gaitnet", "gaitnet_core"], default="gaitnet",
    help="Low-level controller (sim2real/gpu_mpc.py). gaitnet: what every policy here was trained on, which "
    "executes a foothold against the hip at touchdown, so the foot lands about speed x swing time past the "
    "chosen spot. gaitnet_core: takes that body travel back off the target.",
)
parser.add_argument(
    "--inference", nargs="+", choices=["argmax", "marginal", "sample"], default=["argmax"],
    help="How each tick's action is picked from the masked distribution. argmax: the single most likely "
    "slot (the deployed rule). marginal: step when the candidates together outweigh the no-op, to the "
    "most likely candidate. sample: draw from it, as in training. Each checkpoint runs once per rule.",
)
parser.add_argument("--out", type=str, required=True, help="Directory for summary.json and per-env CSVs.")
parser.add_argument("--ctrl", nargs="*", default=[], metavar="KNOB=VALUE",
                    help="Fix controller knobs for the whole eval (robot_controllers.ControllerRandomizationCfg), e.g. "
                    "travel_compensation=1 touchdown_along=-0.04 touchdown_outboard=0.04 touchdown_noise=0.05 "
                    "liftoff_delay_ticks=1: an OCS2-like executor in Isaac.")
parser.add_argument("--zero_history", type=int, nargs="*", default=[],
                    help="History feature indices (0-25, policy_history.py) to zero in every tick the actor sees, e.g. "
                    "18 19 20 21 22 23 24 25 for the touchdown errors: does the policy still work without them?")
parser.add_argument("--history_noise", nargs="*", default=[], metavar="INDEX=STD",
                    help="Gaussian noise added to history features (0-25) in the ticks the actor sees, drawn once per tick "
                    "and carried with it through the window, e.g. 5=0.67 for the yaw-rate spread SC sees in Gazebo.")
parser.add_argument("--log_intent", action="store_true",
                    help="Also save, per tick and env, the chosen leg (-1 no-op), the leg of the best non-no-op candidate and "
                    "the step-minus-no-op logit margin (intent_<label>.pt): how far ahead SC's next step can be read.")
parser.add_argument("--perturb_xy", type=float, default=0.0,
                    help="Std (m) of Gaussian noise on every commanded foothold, for collecting IDM data.")
parser.add_argument("--dump_events", action="store_true",
                    help="Save every landed step (state, leg, duration, intent, command, landing) to <out>/<label>.events.pt.")
parser.add_argument("--idm_inverse", type=str, default=None, help="Inverse footstep model (idm.py) of this controller.")
parser.add_argument("--idm_forward", type=str, default=None,
                    help="Forward footstep model of the controller the policy was trained on; with --idm_inverse, "
                    "each command u becomes I(s, F(s, u)).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

import gaitnet.constants as const
from gaitnet import get_logger
from gaitnet.eval.components.fixed_velocity_command import FixedVelocityCommandCfg
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
from gaitnet.gaitnet.components.policy_history import HISTORY_DIM
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env_cfg, update_controllers
from gaitnet.gaitnet.gaitnet import NUM_OPTIONS, GaitnetActor, candidate_block
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor
from gaitnet.gaitnet.idm import STATE_DIM, IdmAdapter
from gaitnet.gaitnet.teacher import PrivilegedActor, is_gaitnet_t_state_dict, is_privileged_state_dict
from gaitnet.sim2real.robot_controllers import ControllerRandomizationCfg, GpuMpcCfg
from gaitnet.simulation.cfg.terrain import HfVoidTerrainCfg, hole_terrain
from gaitnet.util import log_exceptions

logger = get_logger()

GENERATED: list[tuple[float, float]] = []
"""(void fraction asked for, void fraction generated outside the platform), one per strip, in the
order the terrain generator builds them: column 0, 1, ... (one row)."""


def _fixed_holes(difficulty: float, cfg: "FixedVoidTerrainCfg") -> np.ndarray:
    """hole_terrain at the strip's own void fraction instead of the generator's row difficulty."""
    heights = hole_terrain.__wrapped__(cfg.void_fraction, cfg)
    # hole_terrain's 12.5 cm cells, without the one-pixel void padding it adds along the far edges
    cells = heights[: heights.shape[0] // 5 * 5 : 5, : heights.shape[1] // 5 * 5 : 5]
    nx, ny = cells.shape
    platform = int(cfg.platform_size / cfg.horizontal_scale / 5)
    mask = np.ones_like(cells, dtype=bool)
    mask[nx // 2 - platform // 2 : nx // 2 + platform // 2, :] = False   # the strip is no wider than the platform
    GENERATED.append((cfg.void_fraction, float((cells[mask] < 0).mean()) if mask.any() else 0.0))
    return heights


@configclass
class FixedVoidTerrainCfg(HfVoidTerrainCfg):
    function = height_field_to_mesh(_fixed_holes)
    void_fraction: float = 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def build_actor(state: dict, device: str) -> tuple[torch.nn.Module, int | None]:
    """The checkpoint's actor and how many observation values it reads (None: all of them)."""
    actor_state = {re.sub(r"^actor\.", "", k): v for k, v in state.items() if k.startswith("actor.")}
    base = GaitnetActor(const.gait_net.robot_state_dim, [128, 128, 128], const.gait_net.footstep_option_dim, [64, 64], [128, 128, 128])
    if not is_gaitnet_t_state_dict(state):
        base.load_state_dict(actor_state)
        return base.to(device).eval(), const.gait_net.robot_state_dim + NUM_OPTIONS * const.gait_net.footstep_option_dim
    key = next((k for k in actor_state if k.endswith("history_encoder.0.weight")), None)
    history_len = actor_state[key].shape[1] // HISTORY_DIM if key else 0
    actor = GaitnetTActor.from_gaitnet_actor(base, history_len=history_len, history_dim=HISTORY_DIM)
    if is_privileged_state_dict(state):
        actor = PrivilegedActor.widen(actor)
        actor.load_state_dict(actor_state)
        return actor.to(device).eval(), None
    actor.load_state_dict(actor_state)
    return actor.to(device).eval(), actor.obs_dim


@torch.no_grad()
def act(actor: torch.nn.Module, obs: torch.Tensor, obs_dim: int | None, rule: str = "argmax") -> torch.Tensor:
    """(N, 2) (candidate index, swing duration) from the masked distribution PPO samples from
    (GaitnetActorCritic.update_distribution: padding slots excluded, the real no-op kept), by `rule`.

    argmax compares the no-op with each candidate on its own, so a policy that spreads its stepping
    probability over many footholds can step far less often under argmax than it did in training;
    marginal compares the no-op with all candidates together."""
    logits, duration = actor(obs if obs_dim is None else obs[:, :obs_dim])
    padding = candidate_block(obs)[:, :, 0] == 1
    padding[:, -1] = False
    masked = logits.masked_fill(padding, float("-inf"))
    if rule == "argmax":
        index = masked.argmax(dim=-1)
    elif rule == "marginal":
        probs = torch.softmax(masked, dim=-1)
        best = masked[:, :-1].argmax(dim=-1)
        step = probs[:, :-1].sum(dim=-1) > probs[:, -1]
        index = torch.where(step, best, torch.full_like(best, masked.shape[1] - 1))
    else:
        index = torch.distributions.Categorical(logits=masked).sample()
    return torch.stack([index.float(), duration.gather(1, index[:, None]).squeeze(1)], dim=-1)


def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)   # hole_terrain draws its voids from numpy's global generator
    torch.manual_seed(args_cli.seed)
    cells = [(d, v) for d in args_cli.difficulties for v in args_cli.velocities]
    num_cols = len(cells) * args_cli.strips_per_cell
    num_envs = len(cells) * args_cli.envs_per_cell
    assert args_cli.envs_per_cell % args_cli.strips_per_cell == 0, "envs_per_cell must be a multiple of strips_per_cell"
    # PhysX fails to cook the terrain mesh past a few million vertices, and then every robot falls
    # through the ground on its first step (seen at 480 strips of 10 m x 1 m: 7.7 M); 300 strips worked
    vertices = num_cols * (args_cli.strip_length / 0.025 + 1) * (1.0 / 0.025 + 1)
    if vertices > 5.0e6:
        raise ValueError(f"{num_cols} strips of {args_cli.strip_length} m make a {vertices / 1e6:.1f} M-vertex terrain mesh, "
                         "too large to cook; lower --strips_per_cell or --strip_length")

    cfg = get_env_cfg(num_envs, args_cli.device, recipe="step", history_len=25, privileged=True)
    generator: TerrainGeneratorCfg = cfg.scene.terrain.terrain_generator  # type: ignore
    generator.size = (args_cli.strip_length, 1.0)
    generator.num_rows = 1
    generator.num_cols = num_cols
    generator.curriculum = True       # columns get sub-terrain types in proportion, in order (not sampled)
    generator.seed = args_cli.seed
    generator.sub_terrains = {
        f"cell{i}": FixedVoidTerrainCfg(void_depth=-0.5, platform_size=1.0, void_fraction=d, proportion=1.0)
        for i, (d, _) in enumerate(cells)
    }
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity = FixedVelocityCommandCfg(command=(args_cli.velocities[0], 0.0, 0.0))  # type: ignore
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.episode_length_s = args_cli.episode_s

    env = GaitNetEnv(cfg=cfg)
    knobs = {}
    for item in args_cli.ctrl:
        key, value = item.split("=")
        knobs[key] = (int(value), int(value)) if key == "liftoff_delay_ticks" else (
            float(value) if key == "touchdown_bias" else (float(value), float(value)))
    update_controllers(cfg, num_envs, gpu_mpc=GpuMpcCfg(variant=args_cli.gpu_mpc_variant, seed=args_cli.seed,
                                                         randomization=ControllerRandomizationCfg(**knobs)),
                       joint_names=env.scene["robot"].joint_names)
    robot = env.scene["robot"]
    hip_ids = [robot.find_bodies(f"{leg}_hip")[0][0] for leg in ("FR", "FL", "RR", "RL")]    # slot order
    foot_ids = [robot.find_bodies(f"{leg}_foot")[0][0] for leg in ("FR", "FL", "RR", "RL")]
    slot_to_controller = torch.tensor([1, 0, 3, 2], device=env.device)   # FR, FL, RR, RL -> FL, FR, RL, RR
    device = env.device

    # which cell each env is in, checked against what was generated where
    assert len(GENERATED) == num_cols, f"{len(GENERATED)} strips generated, expected {num_cols}"
    col_cell = [c // args_cli.strips_per_cell for c in range(num_cols)]
    for c, (asked, got) in enumerate(GENERATED):
        assert asked == cells[col_cell[c]][0], f"strip {c} got void fraction {asked}, expected {cells[col_cell[c]][0]}"
    terrain = env.scene.terrain
    types = terrain.terrain_types.cpu().numpy()
    env_cell = np.array([col_cell[t] for t in types])
    origins_y = env.scene.env_origins[:, 1].cpu().numpy()
    expected_y = (types + 0.5) * 1.0 - num_cols * 0.5
    assert np.allclose(origins_y, expected_y, atol=1e-3), "env origins are not on their strips"
    per_cell_generated = {}
    for c, (asked, got) in enumerate(GENERATED):
        per_cell_generated.setdefault(col_cell[c], []).append(got)
    for i, (d, v) in enumerate(cells):
        logger.info(f"cell {i}: voids {d:.2f} (generated {np.mean(per_cell_generated[i]):.3f}), {v} m/s, "
                    f"{int((env_cell == i).sum())} envs")

    command = torch.tensor([[cells[c][1], 0.0, 0.0] for c in env_cell], device=device)
    term = env.command_manager.get_term("base_velocity")
    term._command = command
    speeds = command[:, 0]
    cell_voids = torch.tensor([d for d, _ in cells], device=device)
    history_start = const.gait_net.robot_state_dim + NUM_OPTIONS * const.gait_net.footstep_option_dim
    adapter = IdmAdapter.load(args_cli.idm_inverse, args_cli.idm_forward, device) if args_cli.idm_inverse else None
    no_op_index = NUM_OPTIONS - 1
    steps = int(round(args_cli.episode_s / env.step_dt))
    fall_terms = [n for n in env.termination_manager.active_terms if not env.termination_manager.get_term_cfg(n).time_out]

    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = args_cli.labels or [f"{Path(p).parent.name}:{Path(p).stem}" for p in args_cli.checkpoints]
    summary = {"cells": cells, "envs_per_cell": args_cli.envs_per_cell, "seed": args_cli.seed,
               "gpu_mpc_variant": args_cli.gpu_mpc_variant, "ctrl": args_cli.ctrl,
               "generated_void_fraction": {str(i): float(np.mean(per_cell_generated[i])) for i in range(len(cells))},
               "checkpoints": {}}

    runs = [(p, l if len(args_cli.inference) == 1 else f"{l}|{r}", r)
            for p, l in zip(args_cli.checkpoints, labels) for r in args_cli.inference]
    for path, label, rule in runs:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state, iteration = checkpoint["model_state_dict"], checkpoint.get("iter", -1)
        actor, obs_dim = build_actor(state, device)
        torch.manual_seed(args_cli.seed)
        obs = env.reset(seed=args_cli.seed)[0]["policy"]
        start_x = env.scene.env_origins[:, 0].clone()   # spawned on the origin, pose range zero
        disp = torch.zeros(num_envs, device=device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        cause = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        ops = torch.zeros(num_envs, device=device)
        ticks = torch.zeros(num_envs, device=device)
        controller = env.cfg.robot_controllers.controller
        pending = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)   # slot order
        target_w = torch.zeros(num_envs, 4, 2, device=device)
        yaw_cmd = torch.zeros(num_envs, 4, device=device)
        count_cmd = torch.zeros(num_envs, 4, dtype=torch.long, device=device)
        land_n = torch.zeros(num_envs, device=device)
        land_along = torch.zeros(num_envs, device=device)
        land_lat = torch.zeros(num_envs, device=device)
        land_sq = torch.zeros(num_envs, device=device)
        envs = torch.arange(num_envs, device=device)
        state_cmd = torch.zeros(num_envs, 4, STATE_DIM, device=device)
        hip_cmd = torch.zeros(num_envs, 4, 2, device=device)
        intent_cmd = torch.zeros(num_envs, 4, 2, device=device)
        sent_cmd = torch.zeros(num_envs, 4, 2, device=device)
        duration_cmd = torch.zeros(num_envs, 4, device=device)
        events: dict[str, list[torch.Tensor]] = {k: [] for k in ("state", "leg", "duration", "intent", "command", "landing", "speed", "voids")}
        intent_log: list[torch.Tensor] = []
        noise_idx = [int(kv.split("=")[0]) for kv in args_cli.history_noise]
        noise_std = torch.tensor([float(kv.split("=")[1]) for kv in args_cli.history_noise], device=device)
        history_noise = torch.zeros(num_envs, 25, len(noise_idx), device=device)
        for _ in range(steps):
            if noise_idx:
                # one draw per policy tick, shifted along with the history it belongs to
                history_noise = torch.cat([history_noise[:, 1:], (torch.randn(num_envs, 1, len(noise_idx), device=device) * noise_std)], dim=1)
                obs = obs.clone()
                hist = obs[:, history_start : history_start + 25 * STATE_DIM].view(num_envs, 25, STATE_DIM)
                hist[:, :, noise_idx] += history_noise
            if args_cli.zero_history:
                obs = obs.clone()
                hist = obs[:, history_start : history_start + 25 * STATE_DIM].view(num_envs, 25, STATE_DIM)
                hist[:, :, args_cli.zero_history] = 0.0
            actions = act(actor, obs, obs_dim, rule)
            live = ~finished
            if args_cli.log_intent:
                with torch.no_grad():
                    logits, _ = actor(obs if obs_dim is None else obs[:, :obs_dim])
                masked = logits.masked_fill(candidate_block(obs)[:, :, 0] == 1, float("-inf"))
                best = masked[:, :-1].argmax(dim=-1)
                best_leg = candidate_block(obs)[envs, best, 1:5].argmax(dim=-1)
                chosen = actions[:, 0].long()
                chosen_leg = torch.where(chosen == no_op_index, torch.full_like(best_leg, -1),
                                         candidate_block(obs)[envs, chosen.clamp(max=no_op_index - 1), 1:5].argmax(dim=-1))
                intent_log.append(torch.stack([chosen_leg.float(), best_leg.float(),
                                               masked.gather(1, best[:, None]).squeeze(1) - logits[:, -1], live.float()], dim=-1).detach().cpu())
            stepping = live & (actions[:, 0].long() != no_op_index)
            ops += stepping.float()
            ticks += live.float()
            # where the chosen candidate is on the terrain: hip now + yaw-frame (dx, dy)
            if stepping.any():
                e = envs[stepping]
                cand = candidate_block(obs)[e, actions[e, 0].long()]
                slot = cand[:, 1:5].argmax(dim=-1)
                q = robot.data.root_quat_w[e]
                yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                hip = robot.data.body_pos_w[e][torch.arange(len(e), device=device), torch.tensor(hip_ids, device=device)[slot], :2]
                c, s_ = torch.cos(yaw), torch.sin(yaw)
                dx, dy = cand[:, 5], cand[:, 6]
                target_w[e, slot] = hip + torch.stack([c * dx - s_ * dy, s_ * dx + c * dy], dim=-1)
                yaw_cmd[e, slot] = yaw
                # what is sent: the candidate, through the IDM adapter, plus collection noise
                state = obs[e, history_start : history_start + 25 * STATE_DIM].view(len(e), 25, STATE_DIM)[:, -1]
                intent = torch.stack([dx, dy], dim=-1)
                sent = intent if adapter is None else adapter(state, slot, actions[e, 1], intent)
                if args_cli.perturb_xy > 0:
                    sent = sent + args_cli.perturb_xy * torch.randn_like(sent)
                # the action term reads the chosen option's (x, y) from here (a new tensor every step)
                env.observation_manager.footstep_options[e, actions[e, 0].long(), 1:3] = sent
                state_cmd[e, slot], hip_cmd[e, slot], intent_cmd[e, slot] = state, hip, intent
                sent_cmd[e, slot], duration_cmd[e, slot] = sent, actions[e, 1]
                count_cmd[e, slot] = controller.touchdown_count[e, slot_to_controller[slot]]
                pending[e, slot] = True
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"]
            done = (terminated | truncated) & live
            # touchdowns of commanded steps, before any reset clears the counters
            counts = controller.touchdown_count[:, slot_to_controller]                 # (N, 4) slot order
            landed = pending & (counts > count_cmd) & ~done.unsqueeze(-1) & live.unsqueeze(-1)
            if landed.any():
                e, sl = landed.nonzero(as_tuple=True)
                foot = robot.data.body_pos_w[e, torch.tensor(foot_ids, device=device)[sl], :2]
                err = foot - target_w[e, sl]
                c, s_ = torch.cos(yaw_cmd[e, sl]), torch.sin(yaw_cmd[e, sl])
                along, lat = c * err[:, 0] + s_ * err[:, 1], -s_ * err[:, 0] + c * err[:, 1]
                land_n.index_add_(0, e, torch.ones_like(along))
                land_along.index_add_(0, e, along)
                land_lat.index_add_(0, e, lat)
                land_sq.index_add_(0, e, along ** 2 + lat ** 2)
                if args_cli.dump_events:
                    rel = foot - hip_cmd[e, sl]
                    landing = torch.stack([c * rel[:, 0] + s_ * rel[:, 1], -s_ * rel[:, 0] + c * rel[:, 1]], dim=-1)
                    cell_ids = torch.as_tensor(env_cell, device=device)[e]
                    for k, v in (("state", state_cmd[e, sl]), ("leg", sl), ("duration", duration_cmd[e, sl]),
                                 ("intent", intent_cmd[e, sl]), ("command", sent_cmd[e, sl]), ("landing", landing),
                                 ("speed", speeds[e]), ("voids", cell_voids[cell_ids])):
                        events[k].append(v.detach().cpu())
            pending &= ~landed
            pending &= ~done.unsqueeze(-1)
            # a done env has already been reset: keep its displacement from the step before
            moving = live & ~done
            disp[moving] = env.scene["robot"].data.root_pos_w[moving, 0] - start_x[moving]
            for k, name in enumerate(fall_terms):
                hit = done & env.termination_manager.get_term(name) & (cause < 0)
                cause[hit] = k
            fell |= done & terminated
            finished |= done
            if finished.all():
                break

        rows, cell_stats = [], {}
        disp_c, fell_c, cause_c = disp.cpu().numpy(), fell.cpu().numpy(), cause.cpu().numpy()
        frac = disp_c / (speeds.cpu().numpy() * args_cli.episode_s)
        ops_rate = (ops / ticks.clamp_min(1)).cpu().numpy()
        for i in range(num_envs):
            rows.append([i, env_cell[i], cells[env_cell[i]][0], cells[env_cell[i]][1], f"{disp_c[i]:.3f}", int(fell_c[i]),
                         fall_terms[cause_c[i]] if cause_c[i] >= 0 else "", f"{ops_rate[i]:.3f}"])
        land = {k: t.cpu().numpy() for k, t in (("n", land_n), ("along", land_along), ("lat", land_lat), ("sq", land_sq))}
        for c, (d, v) in enumerate(cells):
            m = env_cell == c
            n = int(m.sum())
            landings = float(land["n"][m].sum())
            survived = int((~fell_c[m]).sum())
            success = int(((~fell_c[m]) & (frac[m] >= 0.5)).sum())
            causes = {name: int((cause_c[m] == k).sum()) for k, name in enumerate(fall_terms)}
            cell_stats[f"{d:.2f}@{v:.2f}"] = {
                "voids": d, "speed": v, "n": n,
                "survival": survived / n, "survival_ci": wilson(survived, n),
                "success": success / n, "success_ci": wilson(success, n),
                "distance_fraction": float(frac[m].mean()),
                "distance_fraction_survivors": float(frac[m & ~fell_c].mean()) if survived else float("nan"),
                "steps_per_tick": float(ops_rate[m].mean()),
                "falls": causes,
                "landings": int(landings),
                "landing_along_mean": float(land["along"][m].sum() / max(landings, 1)),
                "landing_lateral_mean": float(land["lat"][m].sum() / max(landings, 1)),
                "landing_rms": float(math.sqrt(land["sq"][m].sum() / max(landings, 1))),
            }
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        with open(out / f"{safe}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["env", "cell", "voids", "speed", "x_displacement", "fell", "cause", "steps_per_tick"])
            w.writerows(rows)
        summary["checkpoints"][label] = {"path": str(path), "iter": int(iteration), "inference": rule, "cells": cell_stats}
        if args_cli.log_intent and intent_log:
            # (ticks, envs, 4): chosen leg (-1 no-op), best step candidate's leg, its logit minus the no-op's, live
            torch.save({"log": torch.stack(intent_log), "speed": speeds.cpu(), "voids": cell_voids[torch.as_tensor(env_cell, device=device)].cpu()}, out / f"intent_{safe}.pt")
        if args_cli.dump_events and events["state"]:
            torch.save({k: torch.cat(v) for k, v in events.items()}, out / f"{safe}.events.pt")
            logger.info(f"{sum(len(v) for v in events['leg'])} footstep events -> {out / f'{safe}.events.pt'}")
        logger.info(f"== {label}")
        for key, s in cell_stats.items():
            logger.info(f"  {key}: survival {s['survival']:.2f} [{s['survival_ci'][0]:.2f},{s['survival_ci'][1]:.2f}] "
                        f"success {s['success']:.2f} distance {s['distance_fraction']:.2f} steps/tick {s['steps_per_tick']:.3f} "
                        f"falls {s['falls']} landing along {s['landing_along_mean']*100:+.1f} cm "
                        f"lateral {s['landing_lateral_mean']*100:+.1f} cm rms {s['landing_rms']*100:.1f} cm ({s['landings']})")
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=1)
    print("EVAL-DONE", out / "summary.json")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
