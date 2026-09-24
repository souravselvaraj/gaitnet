"""Reward terms on a stand-in env. No simulator."""

from __future__ import annotations

from types import SimpleNamespace

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
