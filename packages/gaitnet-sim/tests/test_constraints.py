"""Constraints as terminations (gaitnet_sim.env.constraints) on a stand-in env. No simulator."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from isaaclab.managers import TerminationTermCfg  # noqa: E402

from gaitnet_sim.env.constraints import ConstraintTermination, constraint_terms, strip_constraints  # noqa: E402

N = 4000


def env_with(tilt=0.0, vz=0.0, rates=0.0, slip=0.0, stance=float("inf"), step_quality=True):
    t = lambda x: SimpleNamespace(torch=x)  # noqa: E731
    gravity = torch.zeros(N, 3)
    gravity[:, 0] = -math.sin(tilt)
    gravity[:, 2] = -math.cos(tilt)
    lin = torch.zeros(N, 3)
    lin[:, 2] = vz
    ang = torch.zeros(N, 3)
    ang[:, 0] = rates
    feet_vel = torch.zeros(N, 4, 3)
    feet_vel[:, 0, 0] = slip
    forces = torch.zeros(N, 4, 3)
    forces[..., 2] = 50.0
    timing = torch.zeros(N, 4, 3)
    timing[..., 2] = 0.5  # every leg planted for 0.5 s
    data = SimpleNamespace(projected_gravity_b=t(gravity), root_link_lin_vel_b=t(lin), root_link_ang_vel_b=t(ang),
                           body_link_lin_vel_w=t(feet_vel))
    io = SimpleNamespace(robot=SimpleNamespace(data=data), contact_ids=[0, 1, 2, 3], foot_ids=[0, 1, 2, 3],
                         contact_sensor=SimpleNamespace(data=SimpleNamespace(net_normal_forces_w=t(forces))))
    term = SimpleNamespace(io=io, controller=SimpleNamespace(gait_timing=lambda: timing), cfg=SimpleNamespace(step_quality=step_quality),
                           step_stance_time=torch.full((N, 2), stance))
    return SimpleNamespace(num_envs=N, device="cpu", step_dt=0.04, action_manager=SimpleNamespace(get_term=lambda name: term))


def constraint(env, **params):
    cfg = TerminationTermCfg(func=ConstraintTermination, params=params)
    return ConstraintTermination(cfg, env), cfg.params


@pytest.mark.parametrize(
    "params, inside, outside",
    [
        ({"quantity": "tilt", "limit": 0.2}, {"tilt": 0.1}, {"tilt": 0.3}),
        ({"quantity": "vertical_speed", "limit": 0.3}, {"vz": -0.2}, {"vz": -0.6}),
        ({"quantity": "body_rates", "limit": 1.5}, {"rates": 1.0}, {"rates": 3.0}),
        ({"quantity": "foot_slip", "limit": 0.1}, {"slip": 0.05}, {"slip": 0.4}),
        ({"quantity": "short_stance", "limit": 0.1}, {"stance": 0.3}, {"stance": 0.02}),
    ],
)
def test_inside_the_limit_never_ends_an_episode_and_beyond_it_does_at_most_p_max(params, inside, outside):
    torch.manual_seed(0)
    term, p = constraint(env_with(**inside), p_max=0.2, **params)
    assert not term(env_with(**inside), **p).any()
    term, p = constraint(env_with(**outside), p_max=0.2, **params)
    ended = term(env_with(**outside), **p).float().mean().item()
    # every robot at the largest violation seen: p = p_max
    assert ended == pytest.approx(0.2, abs=0.03)
    assert term.violating == 1.0


def test_small_violations_end_episodes_less_often_than_large_ones():
    torch.manual_seed(0)
    env = env_with(tilt=0.3)
    term, p = constraint(env, quantity="tilt", limit=0.2, p_max=0.5, tau=1.0)
    term.running_max = torch.tensor(0.4)  # the largest violation seen, kept (tau = 1)
    rate = term(env, **p).float().mean().item()  # a 0.1 rad violation: a quarter of it
    assert rate == pytest.approx(0.5 * 0.25, abs=0.03)


def test_short_stance_needs_step_quality():
    env = env_with(step_quality=False)
    term, p = constraint(env, quantity="short_stance", limit=0.1)
    with pytest.raises(ValueError, match="step_quality"):
        term(env, **p)


def test_the_cat_preset_turns_limits_into_constraints_and_evaluation_strips_them():
    from isaaclab_tasks.utils.hydra import resolve_task_config

    from gaitnet_sim.tasks import register

    register()
    env_cfg, _ = resolve_task_config("GaitNet-Holes", "rsl_rl_cfg_entry_point", overrides=["presets=cat"])
    assert sorted(constraint_terms(env_cfg)) == sorted(
        ["slip_constraint", "tilt_constraint", "bounce_constraint", "rates_constraint", "heading_constraint", "stance_constraint"]
    )
    rewards = env_cfg.rewards
    assert rewards.foot_slip is None and rewards.flat_orientation_l2 is None and rewards.lin_vel_z_l2 is None
    assert rewards.terminating.params["term_keys"] == ["bad_orientation", "bad_height", "foot_below_ground"]
    assert env_cfg.actions.footstep.step_quality is True
    default_cfg, _ = resolve_task_config("GaitNet-Holes", "rsl_rl_cfg_entry_point", overrides=[])
    assert constraint_terms(default_cfg) == [] and default_cfg.rewards.foot_slip is not None
    strip_constraints(env_cfg)
    assert constraint_terms(env_cfg) == []
