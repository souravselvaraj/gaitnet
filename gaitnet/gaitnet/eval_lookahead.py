"""Evaluate lookahead footstep selection (lookahead.py) against the plain argmax of the same GaitNet
checkpoints, with eval_policy.py's protocol: fixed void fractions x speeds, one block of 1 m x
`--strip_length` m strips per cell, seeded terrain and spawn so every (checkpoint, mode) run sees
the same voids, the training controller ("gaitnet" GPU MPC variant) by default.

eval_policy.py launches the app at import, so the pieces needed here are copied rather than imported
(terrain, actor loading, the masked distribution, the per-cell statistics).

Each `--modes` entry is a `LookaheadCfg.parse` spec: "argmax" (depth 0, the deployed rule), "d1"
(predicted-landing edge margins), "d2" (+ the other legs' dead ends), with overrides such as
"d2:gate=free,w_dead=4". The planner's terrain is an edge map of the true terrain: one vertical ray
per 1 cm over the whole terrain against the ground mesh the scanners cast against, void where the
hit is below -0.1 m, then an exact distance transform (privileged).

Per cell it reports eval_policy's survival / success / distance / steps per tick / falls, plus:
how often the planner changed the policy's step or vetoed it, the edge margin of the actual
touchdowns, the planner's landing prediction error, and for each fall whether a foot had touched
down within 2 cm of a void in the 0.24 s before it. It times the planner per tick (CUDA
synchronised) and saves each run's landings (<label>.landings.pt) for offline calibration.

    python -m gaitnet.gaitnet.eval_lookahead --headless --checkpoints a.pt --labels A --modes argmax d1 d2 --out <dir>
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Evaluate lookahead footstep selection at fixed void densities.")
parser.add_argument("--checkpoints", nargs="+", required=True, help="model_<it>.pt files (rsl_rl runner checkpoints).")
parser.add_argument("--labels", nargs="*", default=None, help="A name per checkpoint (default: run dir + iteration).")
parser.add_argument("--modes", nargs="+", default=["argmax", "d1", "d2"], help="LookaheadCfg.parse specs.")
parser.add_argument("--difficulties", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4], help="Void fractions.")
parser.add_argument("--velocities", nargs="+", type=float, default=[0.1, 0.2], help="Forward commands (m/s).")
parser.add_argument("--envs_per_cell", type=int, default=150)
parser.add_argument("--strips_per_cell", type=int, default=30)
parser.add_argument("--strip_length", type=float, default=10.0)
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu_mpc_variant", choices=["gaitnet", "gaitnet_core"], default="gaitnet")
parser.add_argument("--landing_model", type=str, default=None,
                    help="Forward footstep model (idm.py, direction forward) for modes with landing=model.")
parser.add_argument("--map_resolution", type=float, default=0.01, help="Edge map grid (m).")
parser.add_argument("--out", type=str, required=True, help="Directory for summary.json, per-env CSVs and landings.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import dataclasses
import json
import math
import random
import re
import time
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
from gaitnet.gaitnet.idm import STATE_DIM, load_model
from gaitnet.gaitnet.lookahead import EdgeMap, LookaheadCfg, TickState, candidate_legs, plan, predicted_landing, rotate
from gaitnet.gaitnet.teacher import PrivilegedActor, is_gaitnet_t_state_dict, is_privileged_state_dict
from gaitnet.sim2real.robot_controllers import GpuMpcCfg
from gaitnet.simulation.cfg.terrain import HfVoidTerrainCfg, hole_terrain
from gaitnet.util import log_exceptions

logger = get_logger()

GENERATED: list[tuple[float, float]] = []
"""(void fraction asked for, void fraction generated outside the platform), one per strip."""
VOID_PIXELS: list[np.ndarray] = []
"""Per strip, in the same order, hole_terrain's height field as a bool void mask, 2.5 cm pixels."""

SLOT_TO_CONTROLLER = [1, 0, 3, 2]
"""GaitNet slot FR, FL, RR, RL -> controller leg FL, FR, RL, RR (its own inverse)."""
VOID_BELOW = -0.1
"""(m) terrain lower than this is a void for the edge map (ground at 0, voids at -0.5)."""
RECENT_TICKS = 6
"""A fall counts as following an edge touchdown if one happened in the last 6 ticks (0.24 s)."""


def _fixed_holes(difficulty: float, cfg: "FixedVoidTerrainCfg") -> np.ndarray:
    """hole_terrain at the strip's own void fraction (eval_policy.py)."""
    heights = hole_terrain.__wrapped__(cfg.void_fraction, cfg)
    cells = heights[: heights.shape[0] // 5 * 5 : 5, : heights.shape[1] // 5 * 5 : 5]
    nx, ny = cells.shape
    platform = int(cfg.platform_size / cfg.horizontal_scale / 5)
    mask = np.ones_like(cells, dtype=bool)
    mask[nx // 2 - platform // 2 : nx // 2 + platform // 2, :] = False
    GENERATED.append((cfg.void_fraction, float((cells[mask] < 0).mean()) if mask.any() else 0.0))
    # what the mesher sees: the int16 height field, in which hole_terrain's -0.5 edge padding rounds to 0
    VOID_PIXELS.append(np.asarray(heights).astype(np.int16) < 0)
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
def build_edge_map(env, resolution: float) -> tuple[EdgeMap, torch.Tensor]:
    """The true terrain as an edge map, and the void mask it was made from (privileged)."""
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh

    mesh = RayCaster.meshes["/World/ground"]
    gen = env.cfg.scene.terrain.terrain_generator
    half_x, half_y = gen.size[0] * gen.num_rows / 2, gen.size[1] * gen.num_cols / 2
    device = env.device
    xs = torch.arange(-half_x, half_x + 0.5 * resolution, resolution, device=device)
    ys = torch.arange(-half_y, half_y + 0.5 * resolution, resolution, device=device)
    heights = torch.empty(len(xs), len(ys), device=device)
    rows = max(1, int(2e6 // len(ys)))
    for i in range(0, len(xs), rows):
        x = xs[i : i + rows]
        starts = torch.stack(torch.meshgrid(x, ys, indexing="ij"), dim=-1)
        starts = torch.cat([starts, torch.full_like(starts[..., :1], 20.0)], dim=-1).view(-1, 3)
        dirs = torch.zeros_like(starts)
        dirs[:, 2] = -1.0
        hits = raycast_mesh(starts, dirs, mesh, max_dist=100.0)[0]
        heights[i : i + rows] = hits[:, 2].view(len(x), len(ys))
    invalid = ~(heights > VOID_BELOW)          # NaN / inf (no hit) is a void too
    t0 = time.perf_counter()
    edges = EdgeMap.from_invalid(invalid, (float(xs[0]), float(ys[0])), resolution)
    logger.info(f"edge map {tuple(invalid.shape)} at {resolution} m, {invalid.float().mean().item():.3f} void, "
                f"distance transform {time.perf_counter() - t0:.1f} s")
    return edges, invalid


@torch.no_grad()
def policy_outputs(actor: torch.nn.Module, obs: torch.Tensor, obs_dim: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits (N, M), durations (N, M)) of the actor on its slice of the observation."""
    return actor(obs if obs_dim is None else obs[:, :obs_dim])


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    q = quat_wxyz
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


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
    modes = [(spec, LookaheadCfg.parse(spec)) for spec in args_cli.modes]
    assert args_cli.landing_model or all(cfg.landing != "model" for _, cfg in modes), "landing=model needs --landing_model"

    cfg = get_env_cfg(num_envs, args_cli.device, recipe="step", history_len=25, privileged=True)
    generator: TerrainGeneratorCfg = cfg.scene.terrain.terrain_generator  # type: ignore
    generator.size = (args_cli.strip_length, 1.0)
    generator.num_rows = 1
    generator.num_cols = num_cols
    generator.curriculum = True
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
    update_controllers(cfg, num_envs, gpu_mpc=GpuMpcCfg(variant=args_cli.gpu_mpc_variant, seed=args_cli.seed),
                       joint_names=env.scene["robot"].joint_names)
    robot = env.scene["robot"]
    device = env.device
    hip_ids = torch.tensor([robot.find_bodies(f"{leg}_hip")[0][0] for leg in ("FR", "FL", "RR", "RL")], device=device)
    foot_ids = torch.tensor([robot.find_bodies(f"{leg}_foot")[0][0] for leg in ("FR", "FL", "RR", "RL")], device=device)
    slot_to_controller = torch.tensor(SLOT_TO_CONTROLLER, device=device)

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
    per_cell_generated: dict[int, list[float]] = {}
    for c, (asked, got) in enumerate(GENERATED):
        per_cell_generated.setdefault(col_cell[c], []).append(got)

    # the planner's terrain: checked against what was generated, and every spawn point is solid
    edges, invalid = build_edge_map(env, args_cli.map_resolution)
    res = args_cli.map_resolution
    xs = edges.origin[0] + torch.arange(invalid.shape[0], device=device) * res
    ys = edges.origin[1] + torch.arange(invalid.shape[1], device=device) * res
    away = (xs.abs() > 0.6) & (xs.abs() < args_cli.strip_length / 2 - 0.3)
    # The mesh's voids are wider than hole_terrain's pixels say (the height-field mesher's vertical-wall
    # correction moves each wall out by a 2.5 cm pixel), so void fractions differ; the alignment check is
    # on pixels whose 5 x 5 neighbourhood is all void or all solid, which must agree with the map.
    # Height-field pixel (i, j) of strip c is mesh vertex (1 + i, 1 + j) (one border vertex), 2.5 cm apart,
    # and the strip's mesh is centred on x = 0, y = c + 0.5 - num_cols / 2.
    from scipy.ndimage import maximum_filter, minimum_filter

    hs = 0.025
    map_void, agree = {}, {}
    for i, (d, v) in enumerate(cells):
        cols = [c for c in range(num_cols) if col_cell[c] == i]
        fr, hits, total = [], 0, 0
        for c in cols:
            yc = (c + 0.5) - num_cols * 0.5
            band = (ys > yc - 0.3) & (ys < yc + 0.3)
            fr.append(invalid[away][:, band].float().mean().item())
            pix = VOID_PIXELS[c]
            uniform = minimum_filter(pix, size=5, mode="nearest") == maximum_filter(pix, size=5, mode="nearest")
            ii, jj = np.nonzero(uniform)
            pts = torch.tensor(np.stack([(1 + ii) * hs - args_cli.strip_length / 2, (1 + jj) * hs - 0.5 + yc], -1),
                               device=device, dtype=torch.float32)
            hits += int(((edges(pts) == 0).cpu().numpy() == pix[ii, jj]).sum())
            total += len(ii)
        map_void[i], agree[i] = float(np.mean(fr)), hits / max(total, 1)
        logger.info(f"cell {i}: voids {d:.2f} (generated {np.mean(per_cell_generated[i]):.3f}, edge map "
                    f"{map_void[i]:.3f}, height-field pixels agree {agree[i]:.4f}), {v} m/s, {int((env_cell == i).sum())} envs")
        assert agree[i] > 0.995, "edge map is not aligned with the generated voids"
    spawn_margin = edges(env.scene.env_origins[:, :2])
    assert spawn_margin.min().item() > 0.1, f"a spawn point is {spawn_margin.min().item():.3f} m from a void"

    command = torch.tensor([[cells[c][1], 0.0, 0.0] for c in env_cell], device=device)
    term = env.command_manager.get_term("base_velocity")
    term._command = command
    speeds = command[:, 0]
    history_start = const.gait_net.robot_state_dim + NUM_OPTIONS * const.gait_net.footstep_option_dim
    landing_model = load_model(args_cli.landing_model, device) if args_cli.landing_model else None
    no_op_index = NUM_OPTIONS - 1
    steps = int(round(args_cli.episode_s / env.step_dt))
    fall_terms = [n for n in env.termination_manager.active_terms if not env.termination_manager.get_term_cfg(n).time_out]
    controllers = env.cfg.robot_controllers
    controller = controllers.controller
    envs = torch.arange(num_envs, device=device)
    reference = LookaheadCfg()   # landing prediction logged for every mode with the analytic model

    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = args_cli.labels or [f"{Path(p).parent.name}:{Path(p).stem}" for p in args_cli.checkpoints]
    summary = {"cells": cells, "envs_per_cell": args_cli.envs_per_cell, "seed": args_cli.seed,
               "gpu_mpc_variant": args_cli.gpu_mpc_variant, "landing_model": args_cli.landing_model,
               "map_resolution": res, "map_void_below": VOID_BELOW,
               "generated_void_fraction": {str(i): float(np.mean(per_cell_generated[i])) for i in range(len(cells))},
               "map_void_fraction": {str(i): map_void[i] for i in range(len(cells))},
               "map_cell_agreement": {str(i): agree[i] for i in range(len(cells))},
               "checkpoints": {}}

    for path, ckpt_label in zip(args_cli.checkpoints, labels):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict, iteration = checkpoint["model_state_dict"], checkpoint.get("iter", -1)
        actor, obs_dim = build_actor(state_dict, device)
        for spec, la_cfg in modes:
            label = f"{ckpt_label}|{spec}"
            torch.manual_seed(args_cli.seed)
            obs = env.reset(seed=args_cli.seed)[0]["policy"]
            start_x = env.scene.env_origins[:, 0].clone()
            disp = torch.zeros(num_envs, device=device)
            finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
            fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
            cause = torch.full((num_envs,), -1, dtype=torch.long, device=device)
            ops = torch.zeros(num_envs, device=device)
            ticks = torch.zeros(num_envs, device=device)
            changed = torch.zeros(num_envs, device=device)       # stepping ticks where the planner moved the step
            leg_changed = torch.zeros(num_envs, device=device)   # ... to another leg
            vetoed = torch.zeros(num_envs, device=device)        # the policy stepped, the planner held (all vetoed / no-op won)
            added = torch.zeros(num_envs, device=device)         # the policy held, the planner stepped (gate free)
            by_depth2 = torch.zeros(num_envs, device=device)     # ticks where the depth-2 term changed the choice
            pending = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
            count_cmd = torch.zeros(num_envs, 4, dtype=torch.long, device=device)
            pred_cmd = torch.zeros(num_envs, 4, 2, device=device)       # predicted landing (reference model)
            target_cmd = torch.zeros(num_envs, 4, 2, device=device)     # the chosen candidate on the terrain
            yaw_cmd = torch.zeros(num_envs, 4, device=device)
            travel_cmd = torch.zeros(num_envs, 4, device=device)        # v_along * T at command
            trail_cmd = torch.zeros(num_envs, 4, device=device)         # how far the foot trailed its hip at command
            last_land_margin = torch.full((num_envs, 4), 1.0, device=device)
            last_land_tick = torch.full((num_envs, 4), -100, dtype=torch.long, device=device)
            fall_recent = torch.full((num_envs,), float("nan"), device=device)  # min margin of recent touchdowns at a fall
            fall_pending = torch.zeros(num_envs, device=device)
            fall_pending_pred = torch.full((num_envs,), float("nan"), device=device)  # min predicted margin of steps in flight
            landings: dict[str, list[torch.Tensor]] = {k: [] for k in (
                "env", "leg", "speed", "voids", "pred_margin", "actual_margin", "target_margin", "along", "lateral",
                "pred_along", "pred_lateral", "travel", "trail", "tick")}
            plan_s, act_s, n_plan = 0.0, 0.0, 0
            cell_ids = torch.as_tensor(env_cell, device=device)
            cell_voids = torch.tensor([d for d, _ in cells], device=device)
            for k in range(steps):
                live = ~finished
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                logits, duration = policy_outputs(actor, obs, obs_dim)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                cands = candidate_block(obs)
                data = robot.data
                yaw = yaw_of(data.root_quat_w)
                hip_w = data.body_pos_w[:, hip_ids, :2]
                foot_now = data.body_pos_w[:, foot_ids, :2]
                vel_w = data.root_lin_vel_w[:, :2]
                command_w = rotate(yaw, command[:, :2])
                timing = controller._gait.timing()[:, slot_to_controller]           # (N, 4, 3) slot order
                swing_left = timing[..., 1].to(hip_w.dtype)
                swinging = swing_left > 0
                aimed = controller.applied_foothold[:, slot_to_controller].to(hip_w.dtype)
                swing_land = hip_w + vel_w.unsqueeze(1) * swing_left.unsqueeze(-1) + rotate(yaw.unsqueeze(-1), aimed)
                model_state = obs[:, history_start + 24 * STATE_DIM : history_start + 25 * STATE_DIM]
                state = TickState(
                    logits=logits, duration=duration, cand_leg=candidate_legs(cands), cand_xy=cands[..., 5:7],
                    hip_w=hip_w, yaw=yaw, velocity_w=vel_w, command_w=command_w,
                    foot_w=torch.where(swinging.unsqueeze(-1), swing_land, foot_now), swing_left=swing_left,
                    model_state=model_state,
                )
                p = plan(state, edges, la_cfg, landing_model)
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                act_s += t1 - t0
                plan_s += t2 - t1
                n_plan += 1
                index = p.index
                actions = torch.stack([index.float(), duration.gather(1, index[:, None]).squeeze(1)], dim=-1)

                stepping = live & (index != no_op_index)
                policy_steps = live & (p.policy_index != no_op_index)
                ops += stepping.float()
                ticks += live.float()
                changed += (stepping & policy_steps & (index != p.policy_index)).float()
                other_leg = state.cand_leg.gather(1, index[:, None]) != state.cand_leg.gather(1, p.policy_index[:, None])
                leg_changed += (stepping & policy_steps & other_leg.squeeze(1)).float()
                vetoed += (policy_steps & ~stepping).float()
                added += (stepping & ~policy_steps).float()
                if p.index_d1 is not None:
                    by_depth2 += (live & (index != p.index_d1)).float()
                if stepping.any():
                    e = envs[stepping]
                    idx = index[e]
                    slot = state.cand_leg[e, idx]
                    landing_ref = predicted_landing(state, reference)
                    pred_cmd[e, slot] = landing_ref[e, idx]
                    target_cmd[e, slot] = hip_w[e, slot] + rotate(yaw[e], state.cand_xy[e, idx])
                    yaw_cmd[e, slot] = yaw[e]
                    heading = torch.stack([torch.cos(yaw[e]), torch.sin(yaw[e])], dim=-1)
                    travel_cmd[e, slot] = (vel_w[e] * heading).sum(-1) * duration[e, idx]
                    trail_cmd[e, slot] = ((hip_w[e, slot] - foot_now[e, slot]) * heading).sum(-1)
                    count_cmd[e, slot] = controller.touchdown_count[e, slot_to_controller[slot]]
                    pending[e, slot] = True
                obs_dict, _, terminated, truncated, _ = env.step(actions)
                obs = obs_dict["policy"]
                done = (terminated | truncated) & live
                counts = controller.touchdown_count[:, slot_to_controller]
                landed = pending & (counts > count_cmd) & ~done.unsqueeze(-1) & live.unsqueeze(-1)
                if landed.any():
                    e, sl = landed.nonzero(as_tuple=True)
                    foot = robot.data.body_pos_w[e, foot_ids[sl], :2]
                    c, s_ = torch.cos(yaw_cmd[e, sl]), torch.sin(yaw_cmd[e, sl])
                    err = foot - target_cmd[e, sl]
                    perr = foot - pred_cmd[e, sl]
                    actual = edges(foot)
                    last_land_margin[e, sl] = actual
                    last_land_tick[e, sl] = k
                    for key, val in (("env", e), ("leg", sl), ("speed", speeds[e]), ("voids", cell_voids[cell_ids[e]]),
                            ("pred_margin", edges(pred_cmd[e, sl])), ("actual_margin", actual),
                            ("target_margin", edges(target_cmd[e, sl])),
                            ("along", c * err[:, 0] + s_ * err[:, 1]), ("lateral", -s_ * err[:, 0] + c * err[:, 1]),
                            ("pred_along", c * perr[:, 0] + s_ * perr[:, 1]), ("pred_lateral", -s_ * perr[:, 0] + c * perr[:, 1]),
                            ("travel", travel_cmd[e, sl]), ("trail", trail_cmd[e, sl]), ("tick", torch.full_like(e, k))):
                        landings[key].append(val.detach().float().cpu() if key not in ("env", "leg", "tick") else val.cpu())
                pending &= ~landed
                newly_fell = done & terminated
                if newly_fell.any():
                    recent = (k - last_land_tick) <= RECENT_TICKS
                    m = torch.where(recent, last_land_margin, torch.full_like(last_land_margin, float("inf"))).amin(-1)
                    fall_recent[newly_fell] = m[newly_fell]
                    fall_pending[newly_fell] = pending[newly_fell].float().sum(-1)
                    flight = torch.where(pending, edges(pred_cmd), torch.full_like(last_land_margin, float("inf"))).amin(-1)
                    fall_pending_pred[newly_fell] = flight[newly_fell]
                pending &= ~done.unsqueeze(-1)
                moving = live & ~done
                disp[moving] = robot.data.root_pos_w[moving, 0] - start_x[moving]
                for j, name in enumerate(fall_terms):
                    hit = done & env.termination_manager.get_term(name) & (cause < 0)
                    cause[hit] = j
                fell |= newly_fell
                finished |= done
                if finished.all():
                    break

            land = {key: torch.cat(v) if v else torch.zeros(0) for key, v in landings.items()}
            rows, cell_stats = [], {}
            disp_c, fell_c, cause_c = disp.cpu().numpy(), fell.cpu().numpy(), cause.cpu().numpy()
            frac = disp_c / (speeds.cpu().numpy() * args_cli.episode_s)
            ops_rate = (ops / ticks.clamp_min(1)).cpu().numpy()
            changed_c, vetoed_c, added_c = changed.cpu().numpy(), vetoed.cpu().numpy(), added.cpu().numpy()
            depth2_c = by_depth2.cpu().numpy()
            leg_changed_c = leg_changed.cpu().numpy()
            ticks_c = ticks.cpu().numpy()
            recent_c, pend_c = fall_recent.cpu().numpy(), fall_pending.cpu().numpy()
            flight_c = fall_pending_pred.cpu().numpy()
            for i in range(num_envs):
                rows.append([i, env_cell[i], cells[env_cell[i]][0], cells[env_cell[i]][1], f"{disp_c[i]:.3f}", int(fell_c[i]),
                             fall_terms[cause_c[i]] if cause_c[i] >= 0 else "", f"{ops_rate[i]:.3f}",
                             f"{recent_c[i]:.4f}" if fell_c[i] else "", int(pend_c[i]) if fell_c[i] else "",
                             f"{flight_c[i]:.4f}" if fell_c[i] else ""])
            land_env = land["env"].numpy() if len(land["env"]) else np.zeros(0, dtype=int)
            land_cell = env_cell[land_env] if len(land_env) else np.zeros(0, dtype=int)
            for c, (d, v) in enumerate(cells):
                msk = env_cell == c
                n = int(msk.sum())
                survived = int((~fell_c[msk]).sum())
                success = int(((~fell_c[msk]) & (frac[msk] >= 0.5)).sum())
                causes = {name: int((cause_c[msk] == j).sum()) for j, name in enumerate(fall_terms)}
                falls = msk & fell_c
                lm = torch.as_tensor(land_cell == c)
                nl = int(lm.sum())

                def mean(key: str) -> float:
                    return float(land[key][lm].mean()) if nl else float("nan")

                def frac_below(key: str, x: float) -> float:
                    return float((land[key][lm] < x).float().mean()) if nl else float("nan")

                cell_stats[f"{d:.2f}@{v:.2f}"] = {
                    "voids": d, "speed": v, "n": n,
                    "survival": survived / n, "survival_ci": wilson(survived, n),
                    "success": success / n, "success_ci": wilson(success, n),
                    "distance_fraction": float(frac[msk].mean()),
                    "steps_per_tick": float(ops_rate[msk].mean()),
                    "falls": causes,
                    "falls_after_edge_touchdown": int((falls & (recent_c < 0.02)).sum()),
                    "falls_with_leg_in_swing": int((falls & (pend_c > 0)).sum()),
                    "falls_after_edge_step": int((falls & ((recent_c < 0.02) | (flight_c < 0.02))).sum()),
                    "changed_per_step": float(changed_c[msk].sum() / max(ops_rate[msk].dot(ticks_c[msk]), 1)),
                    "leg_changed_per_step": float(leg_changed_c[msk].sum() / max(ops_rate[msk].dot(ticks_c[msk]), 1)),
                    "held_per_tick": float(vetoed_c[msk].sum() / max(ticks_c[msk].sum(), 1)),
                    "added_per_tick": float(added_c[msk].sum() / max(ticks_c[msk].sum(), 1)),
                    "depth2_changed_per_tick": float(depth2_c[msk].sum() / max(ticks_c[msk].sum(), 1)),
                    "landings": nl,
                    "landing_along_mean": mean("along"),
                    "landing_pred_along_mean": mean("pred_along"),
                    "landing_pred_rms": float((land["pred_along"][lm] ** 2 + land["pred_lateral"][lm] ** 2).mean().sqrt()) if nl else float("nan"),
                    "touchdown_margin_lt_2cm": frac_below("actual_margin", 0.02),
                    "touchdown_margin_lt_5cm": frac_below("actual_margin", 0.05),
                    "predicted_margin_lt_2cm": frac_below("pred_margin", 0.02),
                    "trail_p95": float(np.quantile(land["trail"][lm].numpy(), 0.95)) if nl else float("nan"),
                }
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            with open(out / f"{safe}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["env", "cell", "voids", "speed", "x_displacement", "fell", "cause", "steps_per_tick",
                            "fall_recent_touchdown_margin", "fall_legs_in_swing", "fall_in_flight_predicted_margin"])
                w.writerows(rows)
            torch.save(land, out / f"{safe}.landings.pt")
            timing_ms = {"policy_ms_per_tick": 1e3 * act_s / max(n_plan, 1), "plan_ms_per_tick": 1e3 * plan_s / max(n_plan, 1),
                         "num_envs": num_envs}
            summary["checkpoints"][label] = {"path": str(path), "iter": int(iteration), "mode": spec,
                                             "lookahead": dataclasses.asdict(la_cfg), "timing": timing_ms, "cells": cell_stats}
            logger.info(f"== {label}: planner {timing_ms['plan_ms_per_tick']:.2f} ms/tick, policy "
                        f"{timing_ms['policy_ms_per_tick']:.2f} ms/tick for {num_envs} envs")
            for key, s in cell_stats.items():
                logger.info(f"  {key}: survival {s['survival']:.2f} [{s['survival_ci'][0]:.2f},{s['survival_ci'][1]:.2f}] "
                            f"success {s['success']:.2f} distance {s['distance_fraction']:.2f} steps/tick {s['steps_per_tick']:.3f} "
                            f"falls {sum(s['falls'].values())} (edge td {s['falls_after_edge_touchdown']}, edge step {s['falls_after_edge_step']}, swing {s['falls_with_leg_in_swing']}) "
                            f"changed {s['changed_per_step']:.2f} (leg {s['leg_changed_per_step']:.2f}) held/tick {s['held_per_tick']:.3f} "
                            f"added/tick {s['added_per_tick']:.3f} "
                            f"d2/tick {s['depth2_changed_per_tick']:.3f} "
                            f"td<2cm {s['touchdown_margin_lt_2cm']:.3f} td<5cm {s['touchdown_margin_lt_5cm']:.3f} "
                            f"pred<2cm {s['predicted_margin_lt_2cm']:.3f} along {s['landing_along_mean']*100:+.1f} cm "
                            f"pred err along {s['landing_pred_along_mean']*100:+.1f} rms {s['landing_pred_rms']*100:.1f} cm "
                            f"trail p95 {s['trail_p95']:.3f}")
            with open(out / "summary.json", "w") as f:
                json.dump(summary, f, indent=1)
    print("EVAL-DONE", out / "summary.json")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
