"""Reward terms on a stand-in env. No simulator."""

from __future__ import annotations

from types import SimpleNamespace

import math

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_sim.env.rewards import foot_slip  # noqa: E402


def fake_env(forces, speeds, timing):
    data = lambda t: SimpleNamespace(torch=t)  # noqa: E731
    io = SimpleNamespace(
        contact_sensor=SimpleNamespace(data=SimpleNamespace(net_normal_forces_w=data(forces))),
        contact_ids=[0, 1, 2, 3],
        foot_ids=[0, 1, 2, 3],
        robot=SimpleNamespace(data=SimpleNamespace(body_link_lin_vel_w=data(speeds))),
    )
    term = SimpleNamespace(io=io, controller=SimpleNamespace(gait_timing=lambda: timing))
    return SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))


def test_slip_counts_only_settled_stance_feet():
    forces = torch.zeros(1, 4, 3)
    forces[..., 2] = 50.0  # every foot pressing down
    speeds = torch.zeros(1, 4, 3)
    speeds[..., 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])  # every foot moving
    timing = torch.zeros(1, 4, 3)
    timing[0, 0] = torch.tensor([0.0, 0.0, 0.5])  # FL: stance for 0.5 s, slipping
    timing[0, 1] = torch.tensor([0.0, 0.0, 0.01])  # FR: just landed, still settling
    timing[0, 2] = torch.tensor([0.1, 0.18, 0.0])  # RL: swinging (lift-off, still in contact)
    timing[0, 3] = torch.tensor([0.0, 0.0, 0.3])  # RR: stance, slipping
    env = fake_env(forces, speeds, timing)
    assert foot_slip(env).tolist() == pytest.approx([0.1 + 0.4])
    assert foot_slip(env, stance_only=False).tolist() == pytest.approx([1.0])


# --- long horizon ---

DT = 0.04


class Robot:
    """A base that moves as the test tells it, with an operator command, behind a footstep term."""

    def __init__(self, n=2):
        self.pos = torch.zeros(n, 3)
        self.quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(n, 4).clone()  # xyzw
        self.command = torch.zeros(n, 3)
        data = SimpleNamespace()
        self.data = data
        self.refresh()
        io = SimpleNamespace(robot=SimpleNamespace(data=data))
        term = SimpleNamespace(io=io, base_command=lambda: self.command)
        self.env = SimpleNamespace(step_dt=DT, num_envs=n, device="cpu",
                                   action_manager=SimpleNamespace(get_term=lambda name: term))

    def refresh(self):
        self.data.root_link_pos_w = SimpleNamespace(torch=self.pos.clone())
        self.data.root_link_quat_w = SimpleNamespace(torch=self.quat.clone())

    def set_yaw(self, yaw):
        yaw = torch.as_tensor(yaw, dtype=torch.float32)
        self.quat = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)


def window_term(robot, **params):
    from isaaclab.managers import RewardTermCfg

    from gaitnet_sim.env.rewards import WindowTracking

    cfg = RewardTermCfg(func=WindowTracking, weight=1.0, params={"window_s": 1.0, **params})
    return WindowTracking(cfg, robot.env), cfg.params


def test_window_tracking_rewards_keeping_up_and_charges_falling_behind():
    robot = Robot(2)
    term, params = window_term(robot, std=0.1, quantity="xy")
    robot.command[:, 0] = 0.2
    values = []
    for _ in range(40):
        robot.pos[0, 0] += 0.2 * DT          # robot 0 keeps up
        robot.pos[1, 0] += 0.12 * DT         # robot 1 walks at 60% of the command
        robot.refresh()
        values.append(term(robot.env, **params))
    assert values[0].tolist() == [0.0, 0.0]  # no history yet
    assert values[-1][0] == pytest.approx(1.0, abs=1e-4)
    # 0.08 m/s behind over the window: exp(-(0.08/0.1)^2)
    assert values[-1][1] == pytest.approx(torch.exp(torch.tensor(-0.64)).item(), rel=1e-3)


def test_window_tracking_follows_a_turning_command_in_world_frame():
    robot = Robot(1)
    term, params = window_term(robot, std=0.1, quantity="xy")
    robot.command[0] = torch.tensor([0.0, 0.2, 0.0])  # sideways in the base frame
    robot.set_yaw(torch.tensor([math.pi / 2]))       # base facing +y, so "left" is -x in the world
    for _ in range(30):
        robot.pos[0, 0] -= 0.2 * DT
        robot.refresh()
        value = term(robot.env, **params)
    assert value[0] == pytest.approx(1.0, abs=1e-4)


def test_heading_drift_charges_only_uncommanded_turning_and_resets():
    robot = Robot(2)
    term, params = window_term(robot, quantity="heading")
    yaw = torch.zeros(2)
    for _ in range(30):
        yaw[1] += 0.2 * DT                   # robot 1 turns without being asked, through +-pi fine
        robot.set_yaw(yaw)
        robot.refresh()
        value = term(robot.env, **params)
    assert value[0] == pytest.approx(0.0, abs=1e-8)
    assert value[1] == pytest.approx(0.2**2, rel=1e-3)  # 0.2 rad over the 1 s window
    # a commanded turn is no drift
    robot2 = Robot(1)
    term2, params2 = window_term(robot2, quantity="heading")
    robot2.command[0, 2] = 0.3
    yaw2 = torch.zeros(1)
    for _ in range(30):
        yaw2 += 0.3 * DT
        robot2.set_yaw(yaw2)
        robot2.refresh()
        value2 = term2(robot2.env, **params2)
    assert value2[0] == pytest.approx(0.0, abs=1e-6)
    # an episode reset forgets the history
    term.reset(torch.tensor([1]))
    robot.refresh()
    assert term(robot.env, **params)[1] == 0.0


def test_step_quality_rewards_read_the_footstep_term():
    from gaitnet_sim.env.rewards import foothold_edge, short_stance

    term = SimpleNamespace(
        cfg=SimpleNamespace(step_quality=True),
        step_clearance=torch.tensor([[0.0, float("inf")], [3.0, 6.0], [float("inf"), float("inf")]]),
        step_stance_time=torch.tensor([[0.05, float("inf")], [0.5, 0.02], [float("inf"), float("inf")]]),
    )
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    assert foothold_edge(env, margin_cells=6).tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert short_stance(env, min_stance_s=0.1).tolist() == [1.0, 1.0, 0.0]
    term.cfg.step_quality = False
    with pytest.raises(ValueError, match="step_quality"):
        foothold_edge(env)
