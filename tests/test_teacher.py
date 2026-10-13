"""The privileged teacher (teacher.py) and its terrain features (components/privileged.py).

    python -m pytest -q tests/test_teacher.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import gaitnet.constants as const
from gaitnet.gaitnet.components import privileged
from gaitnet.gaitnet.components.policy_history import HISTORY_DIM
from gaitnet.gaitnet.gaitnet import NUM_OPTIONS
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor
from gaitnet.gaitnet.teacher import PrivilegedActor
from tests.test_gaitnet_t import observation, old_actor

T = 25
TAIL = privileged.privileged_dim(NUM_OPTIONS)


def teacher_obs(n: int = 16) -> torch.Tensor:
    base = observation(n, padded=5, history=T * HISTORY_DIM)
    return torch.cat([base, torch.randn(n, TAIL)], dim=1)


def test_widened_teacher_starts_identical():
    plain = GaitnetTActor.from_gaitnet_actor(old_actor(True), history_len=T, history_dim=HISTORY_DIM).eval()
    teacher = PrivilegedActor.widen(plain).eval()
    obs = teacher_obs()
    with torch.no_grad():
        a, b = plain(obs[:, :-TAIL]), teacher(obs)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=1e-6)


def test_privileged_inputs_reach_the_new_columns():
    teacher = PrivilegedActor.widen(GaitnetTActor.from_gaitnet_actor(old_actor(False), history_len=T, history_dim=HISTORY_DIM))
    obs = teacher_obs()
    teacher(obs)[0].logsumexp(-1).sum().backward()
    s = const.gait_net.robot_state_dim
    assert teacher.inner.shared_encoder[0].weight.grad[:, s:].abs().sum() > 0      # knobs
    assert teacher.inner.unique_encoder[0].weight.grad[:, 8:].abs().sum() > 0      # terrain
    # the rearranged layout puts knob j at state column 25 + j and terrain into token columns 8, 9
    x = teacher.rearrange(obs)
    tail = obs[:, -TAIL:]
    assert torch.equal(x[:, s : s + privileged.KNOB_DIM], tail[:, : privileged.KNOB_DIM])
    tokens = x[:, s + privileged.KNOB_DIM : s + privileged.KNOB_DIM + NUM_OPTIONS * 10].view(-1, NUM_OPTIONS, 10)
    assert torch.equal(tokens[..., 8:], tail[:, privileged.KNOB_DIM :].view(-1, NUM_OPTIONS, 2))


def fake_env(void_x: float) -> SimpleNamespace:
    """One env, four scanners at their hips (hip at z = 0.3), flat ground at z = 0 with a void
    (z = -0.5) for world x > void_x. Rays are laid out the way the observation reshapes them:
    ray (i, j) of the 41 x 41 grid at hip + idx_to_xy(i - 2, j - 2)."""
    g = int(const.footstep_scanner.sensor_grid_size[0])
    res = const.footstep_scanner.grid_resolution
    coords = (torch.arange(g) - (g - 1) / 2) * res
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    scene = {}
    for k, name in enumerate(privileged.SCANNERS):
        hip = torch.tensor([0.2 * (k % 2), 0.1 * (k // 2), 0.3])
        starts = torch.stack([x.flatten() + hip[0], y.flatten() + hip[1], torch.full((g * g,), 20.3)], -1)
        z = torch.where(starts[:, 0] > void_x, -0.5, 0.0)
        hits = torch.stack([starts[:, 0], starts[:, 1], z], -1)
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(ray_hits_w=hits.unsqueeze(0), pos_w=hip.unsqueeze(0)),
            _ray_starts_w=starts.unsqueeze(0),
        )
    return SimpleNamespace(scene=scene)


def test_candidate_terrain_and_foot_edge_distance():
    env = fake_env(void_x=0.105)  # between rays, so no float ties
    k = const.gait_net.num_footstep_options
    options = torch.zeros(1, NUM_OPTIONS, 4)
    options[0, :, 0] = torch.arange(NUM_OPTIONS) // k
    options[0, -1, 0] = -1                                   # the no-op
    options[0, :-1, 1] = 0.0                                 # dx: 0 for every candidate
    options[0, 1, 1] = 0.06                                  # FR candidate 1 at x = 0.06
    features = privileged.candidate_terrain(env, options)
    # FR (hip x = 0): x = 0 is 0.10 m from the first void ray (x = 0.11 -> 0.11), height -0.3
    torch.testing.assert_close(features[0, 0], torch.tensor([0.11, -0.3]), atol=1e-5, rtol=0)
    torch.testing.assert_close(features[0, 1, 0], torch.tensor(0.05), atol=1e-5, rtol=0)
    # FL (hip x = 0.2) stands on the void: distance 0, height clipped at -0.6
    torch.testing.assert_close(features[0, k], torch.tensor([0.0, -0.6]), atol=1e-5, rtol=0)
    assert features[0, -1].abs().sum() == 0                  # no-op slot
    feet = torch.tensor([[[0.0, 0.0], [0.2, 0.0], [0.0, 0.1], [0.08, 0.1]]])
    d = privileged.foot_edge_distance(env, feet)
    torch.testing.assert_close(d, torch.tensor([[0.11, 0.0, 0.11, 0.03]]), atol=1e-5, rtol=0)
