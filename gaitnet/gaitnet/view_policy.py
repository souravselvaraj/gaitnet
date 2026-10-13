"""Watch a GaitNet / GaitNet-T policy walk, live in your browser, through Viser.

Isaac Lab 2.3.2 (what GaitNet trains on) has no browser viewer of its own (the isaac-turing
README's `--viz viser` is Isaac Lab 3.0 only), so this draws the scene itself: the terrain tiles the
robots stand on, each robot as the Go1 it is (the Go1 URDF's link meshes, decimated so a dozen robots
stream to a browser, posed from the simulated bodies), its feet (green in stance, orange in swing)
and the foothold it chose for each step (red), streamed with Viser while the policy plays in real
time on the training controller. Your browser renders it, so turning the camera never lags.

With --run it follows a training run: it plays that run's newest checkpoint and picks up newer
ones as training saves them (--reload_min). Robots are lined up on the curriculum's difficulty
levels (--levels, 0-11: about 2% to 48% voids), all walking forward at --speed.

    sbatch cluster/view.sbatch --run SC-idm --idm_inverse <training/gaitnet/idm_models/gaitnet_inverse.pt>
    sbatch cluster/view.sbatch --checkpoint <model.pt>
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Watch a GaitNet policy in the browser (Viser).")
parser.add_argument("--run", type=str, default=None, help="Training run name; plays its newest checkpoint and follows new ones.")
parser.add_argument("--checkpoint", type=str, default=None, help="A fixed model_<it>.pt instead of --run.")
parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    help="Terrain levels to put robots on, one robot per entry (level L: about (L + 0.5) x 4.2%% voids).")
parser.add_argument("--speed", type=float, default=0.15, help="Forward command (m/s).")
parser.add_argument("--idm_inverse", type=str, default=None, help="Send footholds through this inverse footstep model (idm.py).")
parser.add_argument("--reload_min", type=float, default=5.0, help="How often to look for a newer checkpoint of --run.")
parser.add_argument("--port", type=int, default=None, help="Viser port (default: $_VISER_PORT_OVERRIDE, else 8080).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
import viser

import gaitnet.constants as const
from gaitnet import get_logger
from gaitnet.eval.components.fixed_velocity_command import FixedVelocityCommandCfg
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
from gaitnet.gaitnet.components.policy_history import HISTORY_DIM
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env_cfg, update_controllers
from gaitnet.gaitnet.gaitnet import NUM_OPTIONS, GaitnetActor, candidate_block
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor
from gaitnet.gaitnet.teacher import PrivilegedActor, is_gaitnet_t_state_dict, is_privileged_state_dict
from gaitnet.sim2real.robot_controllers import GpuMpcCfg
from gaitnet.util import log_exceptions

logger = get_logger()
RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "training", "gaitnet", "runs")
LEGS = ("FR", "FL", "RR", "RL")                                  # GaitNet slot order
SLOT_TO_CONTROLLER = [1, 0, 3, 2]                               # -> FL, FR, RL, RR
GO1_URDF = os.path.join(os.path.dirname(__file__), "..", "control", "assets", "go1_description", "urdf", "go1.urdf")
GO1_LINKS = ["trunk"] + [f"{leg}_{part}" for leg in LEGS for part in ("hip", "thigh", "calf")]
# faces kept per mesh (the trunk alone is 734k in the URDF's mesh) and a colour per part
MESH_FACES = {"trunk": 4000, "hip": 1000, "thigh": 1000, "calf": 800}
PART_COLOURS = {"trunk": (62, 64, 70), "hip": (150, 150, 156), "thigh": (48, 48, 54), "calf": (32, 32, 36)}


def newest_checkpoint() -> str:
    if args_cli.checkpoint:
        return args_cli.checkpoint
    dirs = [d for d in glob.glob(f"{RUNS}/gaitnet_2026*_{args_cli.run}") if os.path.basename(d).split("_", 3)[-1] == args_cli.run]
    if not dirs:
        raise FileNotFoundError(f"no run directory for --run {args_cli.run}")
    cks = glob.glob(f"{sorted(dirs)[-1]}/model_*.pt")
    return max(cks, key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]))


def build_actor(state: dict, device: str) -> tuple[torch.nn.Module, int | None]:
    """As eval_policy.build_actor: the checkpoint's actor and how many observation values it reads."""
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
def act(actor, obs: torch.Tensor, obs_dim: int | None) -> torch.Tensor:
    """Masked argmax, as deployed (eval_policy.act)."""
    logits, duration = actor(obs if obs_dim is None else obs[:, :obs_dim])
    padding = candidate_block(obs)[:, :, 0] == 1
    padding[:, -1] = False
    index = logits.masked_fill(padding, float("-inf")).argmax(dim=-1)
    return torch.stack([index.float(), duration.gather(1, index[:, None]).squeeze(1)], dim=-1)


def go1_link_meshes() -> dict[str, trimesh.Trimesh]:
    """Each of GO1_LINKS' visual mesh from the Go1 URDF, decimated, in that link's own frame (the
    visual origin baked in), so a link's mesh is posed by its body's pose in the simulation."""
    meshes_dir = os.path.join(os.path.dirname(GO1_URDF), "..", "meshes")
    loaded: dict[tuple[str, str], trimesh.Trimesh] = {}
    out = {}
    for link in ET.parse(GO1_URDF).getroot().findall("link"):
        name = link.get("name")
        mesh = link.find("visual/geometry/mesh")
        if name not in GO1_LINKS or mesh is None:
            continue
        part = name.split("_")[-1]
        file = os.path.basename(mesh.get("filename"))
        if (file, part) not in loaded:
            full = trimesh.load(os.path.join(meshes_dir, file), force="mesh")
            loaded[(file, part)] = full.simplify_quadric_decimation(face_count=MESH_FACES[part])
        m = loaded[(file, part)].copy()
        origin = link.find("visual/origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        m.apply_transform(trimesh.transformations.compose_matrix(angles=rpy, translate=xyz))  # URDF rpy = static xyz
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=PART_COLOURS[part])
        out[name] = m
    missing = set(GO1_LINKS) - set(out)
    if missing:
        raise RuntimeError(f"{GO1_URDF} has no visual mesh for {sorted(missing)}")
    return out


def terrain_tiles(env, tiles: list[tuple[int, int]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(vertices, faces, vertex colours) of each curriculum tile (level, column) from the ground mesh."""
    sensor = next(s for s in env.scene.sensors.values() if hasattr(s, "meshes") and s.meshes)
    mesh = next(iter(sensor.meshes.values()))
    points, faces = mesh.points.numpy(), mesh.indices.numpy().reshape(-1, 3)
    origins = env.scene.terrain.terrain_origins.cpu().numpy()
    size = env.cfg.scene.terrain.terrain_generator.size
    out = []
    for level, col in tiles:
        cx, cy = origins[level, col, :2]
        inside = (np.abs(points[:, 0] - cx) <= size[0] / 2 + 1e-3) & (np.abs(points[:, 1] - cy) <= size[1] / 2 + 1e-3)
        keep = inside[faces].all(axis=1)
        f = faces[keep]
        used, inv = np.unique(f, return_inverse=True)
        v = points[used]
        colours = np.where(v[:, 2:3] < -0.05, np.array([[40, 40, 48]]), np.array([[196, 190, 178]])).astype(np.uint8)
        out.append((v, inv.reshape(-1, 3), colours))
    return out


def main():
    device = args_cli.device
    num_envs = len(args_cli.levels)
    cfg = get_env_cfg(num_envs, device, recipe="step", history_len=25, privileged=True, commands="forward",
                      idm_inverse=args_cli.idm_inverse)
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity = FixedVelocityCommandCfg(command=(args_cli.speed, 0.0, 0.0))  # type: ignore
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env = GaitNetEnv(cfg=cfg)
    update_controllers(cfg, num_envs, gpu_mpc=GpuMpcCfg(), joint_names=env.scene["robot"].joint_names)

    # robots in a row on one column of tiles, one per requested difficulty level
    terrain = env.scene.terrain
    levels = torch.tensor(args_cli.levels, device=device).clamp(0, terrain.max_terrain_level - 1)
    column = torch.zeros_like(levels)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = column
    terrain.env_origins[:] = terrain.terrain_origins[levels, column]
    obs = env.reset()[0]["policy"]

    robot = env.scene["robot"]
    body = {n: i for i, n in enumerate(robot.body_names)}
    chain = [[body[f"{leg}_{part}"] for part in ("hip", "thigh", "calf", "foot")] for leg in LEGS]
    feet = [body[f"{leg}_foot"] for leg in LEGS]
    controllers = cfg.robot_controllers

    port = args_cli.port or int(os.environ.get("_VISER_PORT_OVERRIDE", 8080))
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+z")
    tiles = sorted({(int(l), 0) for l in levels.tolist()} | {(min(int(l) + 1, terrain.max_terrain_level - 1), 0) for l in levels.tolist()})
    for (level, col), (v, f, c) in zip(tiles, terrain_tiles(env, tiles)):
        server.scene.add_mesh_trimesh(f"/terrain/level_{level}", trimesh.Trimesh(v, f, vertex_colors=c, process=False))
    for i, level in enumerate(levels.tolist()):
        o = terrain.env_origins[i].cpu().numpy()
        server.scene.add_label(f"/labels/{i}", f"level {level} (~{(level + 0.5) / 12 * 50:.0f}% voids)",
                               position=(float(o[0]), float(o[1]) - 1.2, 0.6))
    info = server.gui.add_markdown("starting...")
    speed_slider = server.gui.add_slider("visual updates / s", min=2, max=30, step=1, initial_value=15)

    # every scene node is created once; the loop only moves them (a pose is a few bytes, while
    # re-sending geometry every tick is what made the first version too heavy to orbit the camera)
    meshes = go1_link_meshes()
    link_bodies = [body[name] for name in GO1_LINKS]
    # the meshes are posed by the simulated bodies, which assumes those bodies use the URDF's link
    # frames: there the foot sits 0.213 m down the calf frame's -z (URDF FR_foot_fixed)
    calf, foot_ = robot.data.body_pos_w[:, [body[f"{leg}_calf"] for leg in LEGS]], robot.data.body_pos_w[:, feet]
    q = robot.data.body_quat_w[:, [body[f"{leg}_calf"] for leg in LEGS]]
    w, x, y, z = q.unbind(-1)
    down = -0.213 * torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    logger.info(f"URDF frame check: foot vs calf + 0.213 m along the calf's -z, max error "
                f"{1000 * (calf + down - foot_).norm(dim=-1).max():.1f} mm")
    links, feet_on, feet_off, marks = [], [], [], []      # links[i][k]: robot i, GO1_LINKS[k]
    for i in range(num_envs):
        links.append([server.scene.add_mesh_trimesh(f"/robots/{i}/{name}", meshes[name]) for name in GO1_LINKS])
        for leg in LEGS:
            feet_on.append(server.scene.add_icosphere(f"/robots/{i}/{leg}_foot_stance", radius=0.025, color=(40, 170, 60)))
            feet_off.append(server.scene.add_icosphere(f"/robots/{i}/{leg}_foot_swing", radius=0.025, color=(240, 140, 20), visible=False))
            marks.append(server.scene.add_icosphere(f"/robots/{i}/{leg}_chosen", radius=0.02, color=(220, 30, 30), visible=False))

    row = terrain.env_origins.cpu().numpy()
    centre = row.mean(axis=0)

    @server.on_client_connect
    def _(client) -> None:  # look at the row of robots from the side
        client.camera.position = (float(centre[0]), float(centre[1]) - 9.0, 5.0)
        client.camera.look_at = (float(centre[0]), float(centre[1]), 0.0)

    actor, obs_dim, loaded, loaded_at = None, None, None, 0.0
    targets = torch.full((num_envs, 4, 3), float("nan"), device=device)
    t_sim, falls, last_draw, wall0, sim0 = 0.0, 0, 0.0, time.time(), 0.0
    while simulation_app.is_running():
        if time.time() - loaded_at > args_cli.reload_min * 60 or actor is None:
            path = newest_checkpoint()
            if path != loaded:
                ck = torch.load(path, map_location=device, weights_only=False)
                actor, obs_dim = build_actor(ck["model_state_dict"], device)
                loaded, it = path, ck.get("iter", -1)
                logger.info(f"playing {path} (iteration {it})")
            loaded_at = time.time()
        tick = time.perf_counter()
        actions = act(actor, obs, obs_dim)
        stepping = actions[:, 0].long() != NUM_OPTIONS - 1
        if stepping.any():
            e = stepping.nonzero().squeeze(1)
            cand = candidate_block(obs)[e, actions[e, 0].long()]
            slot = cand[:, 1:5].argmax(dim=-1)
            q = robot.data.root_quat_w[e]
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            hip = robot.data.body_pos_w[e, torch.tensor([chain[s][0] for s in range(4)], device=device)[slot]]
            c, s_ = torch.cos(yaw), torch.sin(yaw)
            dx, dy = cand[:, 5], cand[:, 6]
            targets[e, slot] = torch.stack([hip[:, 0] + c * dx - s_ * dy, hip[:, 1] + s_ * dx + c * dy, hip[:, 2] * 0], dim=-1)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        done = terminated | truncated
        falls += int(terminated.sum())
        targets[done] = float("nan")
        t_sim += env.step_dt

        now = time.time()
        if server.get_clients() and now - last_draw >= 1.0 / speed_slider.value:
            last_draw = now
            pos = robot.data.body_pos_w.cpu().numpy()
            link_pos = pos[:, link_bodies]
            link_quat = robot.data.body_quat_w[:, link_bodies].cpu().numpy()            # wxyz, as viser
            contact = controllers.contact_state()[:, SLOT_TO_CONTROLLER].cpu().numpy().reshape(-1)   # slot order
            tg = targets.cpu().numpy().reshape(-1, 3)
            foot = pos[:, feet].reshape(-1, 3)
            with server.atomic():
                for i, robot_links in enumerate(links):
                    for k, link in enumerate(robot_links):
                        link.position, link.wxyz = link_pos[i, k], link_quat[i, k]
                for k in range(len(feet_on)):
                    feet_on[k].visible, feet_off[k].visible = bool(contact[k]), not bool(contact[k])
                    (feet_on[k] if contact[k] else feet_off[k]).position = foot[k]
                    ok = not np.isnan(tg[k]).any()
                    marks[k].visible = ok
                    if ok:
                        marks[k].position = tg[k]
                rtf = (t_sim - sim0) / max(now - wall0, 1e-6)
                info.content = (f"**{args_cli.run or os.path.basename(args_cli.checkpoint)}**, iteration {it}  \n"
                                f"{os.path.basename(loaded)}{' + IDM' if args_cli.idm_inverse else ''}, {args_cli.speed} m/s forward  \n"
                                f"sim time {t_sim:6.0f} s ({rtf:.2f}x real time), falls {falls}  \n"
                                f"feet: green stance, orange swing; red: chosen footholds")
                if now - wall0 > 10:
                    wall0, sim0 = now, t_sim
        # real time: one policy tick per env.step_dt of wall clock
        time.sleep(max(0.0, env.step_dt - (time.perf_counter() - tick)))
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
