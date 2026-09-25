"""Sim2real hardening: the observation noise model, and training's randomization being on by
default and off in play mode. Without the simulator."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("isaaclab")

from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

from gaitnet_core.grid import FootholdGrid  # noqa: E402
from gaitnet_core.state import Observation, RobotState, TerrainPatch  # noqa: E402
from gaitnet_sim.env.noise import ObservationNoiseCfg, corrupt  # noqa: E402
from gaitnet_sim.tasks import register  # noqa: E402

GRID = FootholdGrid(resolution=0.015, size=(9, 9), border=2)


def observation(n: int = 64) -> Observation:
    torch.manual_seed(0)
    state = RobotState(
        foot_pos=torch.randn(n, 4, 3),
        foot_vel=torch.randn(n, 4, 3),
        base_lin_vel=torch.randn(n, 3),
        base_ang_vel=torch.randn(n, 3),
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]).expand(n, 3).clone(),
        contact=torch.rand(n, 4) < 0.5,
        gait_timing=torch.rand(n, 4, 3),
        command=torch.randn(n, 3),
        base_command=torch.randn(n, 3),
    )
    heights = torch.full((n, 4, *GRID.patch_size), -0.26)
    heights[:, :, :3] = float("-inf")
    return Observation(state, TerrainPatch(heights, GRID))


def test_noise_is_bounded_and_leaves_exact_fields_alone():
    cfg = ObservationNoiseCfg()
    clean = observation()
    noisy = corrupt(clean, cfg)
    for name, bound in [
        ("foot_pos", cfg.foot_pos),
        ("foot_vel", cfg.foot_vel),
        ("base_lin_vel", cfg.base_lin_vel),
        ("base_ang_vel", cfg.base_ang_vel),
        ("projected_gravity", cfg.projected_gravity),
    ]:
        delta = getattr(noisy.state, name) - getattr(clean.state, name)
        assert delta.abs().max() <= bound + 1e-6, name
        assert delta.abs().max() > 0.5 * bound, name  # actually noisy
    for name in ("contact", "gait_timing", "command", "base_command"):
        assert torch.equal(getattr(noisy.state, name), getattr(clean.state, name)), name
    # the input is untouched
    assert torch.equal(clean.state.foot_pos, observation().state.foot_pos)


def test_terrain_noise_is_a_patch_offset_plus_small_cell_noise():
    cfg = ObservationNoiseCfg()
    clean = observation()
    noisy = corrupt(clean, cfg).terrain.heights
    unknown = torch.isinf(clean.terrain.heights)
    assert torch.isinf(noisy[unknown]).all() and torch.isfinite(noisy[~unknown]).all()
    delta = (noisy - clean.terrain.heights)[:, :, 3:]  # known cells
    offset = delta.mean(dim=(-2, -1), keepdim=True)
    assert offset.abs().max() <= cfg.terrain_offset + cfg.terrain_cell
    assert (delta - offset).abs().max() <= 2 * cfg.terrain_cell + 1e-6
    # neighbouring cells never differ by the foothold rules' edge threshold
    assert (delta.diff(dim=-1).abs().max() < 0.02) and (delta.diff(dim=-2).abs().max() < 0.02)


def resolve(task: str = "GaitNet-Holes"):
    register()
    return resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])


def test_training_randomizes_and_play_mode_does_not():
    env, _ = resolve()
    assert env.actions.footstep.observation_noise is not None
    assert env.actions.footstep.front_camera is not None
    assert env.events.add_base_mass is not None and env.events.push_robot is not None
    low, high = env.events.physics_material.params["static_friction_range"]
    assert low < high

    env.play_mode()
    assert env.actions.footstep.observation_noise is None
    assert env.actions.footstep.front_camera is None
    assert env.events.add_base_mass is None and env.events.push_robot is None
    assert env.events.physics_material.params["static_friction_range"] == (1.0, 1.0)
    assert env.events.physics_material.params["dynamic_friction_range"] == (1.0, 1.0)


def test_action_term_clamps_executed_durations_but_not_the_stored_action():
    from types import SimpleNamespace

    from gaitnet_core import action_layout
    from gaitnet_core.action_layout import NO_STEP_LEG
    from gaitnet_core.interfaces import FootstepCommand
    from gaitnet_core.robot_spec import GO1
    from gaitnet_sim.env.actions import FootstepControlAction

    sent = []
    term = FootstepControlAction.__new__(FootstepControlAction)
    term.cfg = SimpleNamespace(clamp_duration=True, apply_nudge=True, step_quality=False)
    term.spec = GO1
    term.controller = SimpleNamespace(
        command_footsteps=lambda footsteps: sent.append(footsteps.duration.clone()), close=lambda: None
    )
    term._raw_actions = torch.zeros(4, action_layout.DIM)
    term._nudge = torch.zeros(4, 3)
    term._footsteps = [FootstepCommand.none(4, device="cpu")]
    term._planner_observation = None
    term._candidates = {}
    term._track_progress = lambda: None

    actions = torch.zeros(4, action_layout.DIM)
    actions[:, 1] = torch.tensor([-0.05, 0.02, 0.2, 0.45])  # duration
    actions[:, 2] = torch.tensor([0.0, 1.0, 2.0, NO_STEP_LEG])  # leg; the last robot holds
    term.process_actions(actions)

    low, high = GO1.swing_duration_range
    assert sent[0][:3].tolist() == pytest.approx([low, low, 0.2])
    assert term.raw_actions[:, 1].tolist() == pytest.approx(actions[:, 1].tolist())


def test_step_quality_measures_edge_clearance_and_stance_time():
    """A foothold's distance to the nearest hole edge on the true terrain, and how long its
    leg had been down, recorded when the step starts."""
    from types import SimpleNamespace

    from gaitnet_core.interfaces import FootstepCommand
    from gaitnet_core.planner import FootholdRules
    from gaitnet_core.robot_spec import GO1
    from gaitnet_core.state import TerrainPatch
    from gaitnet_sim.env.actions import FootstepControlAction
    from gaitnet_sim.env.contract import GaitNetCfg

    grid = GaitNetCfg().foothold_grid()
    n = 3
    term = FootstepControlAction.__new__(FootstepControlAction)
    term.cfg = SimpleNamespace(step_quality=True, step_quality_margin=6)
    term.spec, term.grid, term.rounds = GO1, grid, 1
    heights = torch.full((n, 4, *grid.patch_size), -0.27)
    b = grid.border
    heights[:, :, b + 12, b + 20:] = -0.8   # a hole on the right, from inner column 20
    term.terrain = lambda: TerrainPatch(heights=heights, grid=grid)
    timing = torch.zeros(n, 4, 3)
    timing[:, :, 2] = torch.tensor([0.5, 0.05, 0.3, 0.2])
    term.controller = SimpleNamespace(gait_timing=lambda: timing, close=lambda: None)
    term._env = SimpleNamespace(
        cfg=SimpleNamespace(gaitnet=SimpleNamespace(foothold_rules=lambda: FootholdRules())), num_envs=n, device="cpu"
    )
    steps = FootstepCommand.none(n)
    steps.active[:] = torch.tensor([True, True, False])
    steps.leg[:] = torch.tensor([0, 1, 0])
    # robot 0 steps at the hip (12 cells from the hole's edge cells), robot 1 right beside the hole
    steps.target[0, :2] = grid.cell_to_xy(torch.tensor([12, 12]))
    steps.target[1, :2] = grid.cell_to_xy(torch.tensor([12, 18]))
    term._footsteps = [steps]
    term.step_clearance = torch.full((n, 1), float("inf"))
    term.step_stance_time = torch.full((n, 1), float("inf"))
    term._assess_steps()
    # edge cells straddle the drop: inner column 19 (and 20) are edges, so column 18 is 1 cell away
    assert term.step_clearance[0, 0] == 7.0         # beyond the 6-cell margin
    assert term.step_clearance[1, 0] == 1.0
    assert torch.isinf(term.step_clearance[2, 0])   # no step
    assert term.step_stance_time[:, 0].tolist() == pytest.approx([0.5, 0.05, float("inf")])
