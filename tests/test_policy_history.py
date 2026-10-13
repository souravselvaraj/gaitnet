"""The policy history (components/policy_history.py), the touchdown error it reads, and the
explicit candidate slicing that keeps the history tail out of the actor, critic and masking.

    python -m pytest -q tests/test_policy_history.py
"""

from __future__ import annotations

import torch

import gaitnet.constants as const
from gaitnet.gaitnet.components.policy_history import HISTORY_DIM, PolicyHistory, tick_features
from gaitnet.gaitnet.gaitnet import NUM_OPTIONS, GaitnetActor, GaitnetActorCritic, GaitnetCritic
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor
from gaitnet.sim2real.robot_controllers import GpuMpcCfg, GpuRobotControllers
from tests.test_gaitnet_t import observation
from tests.test_gpu_mpc import isaac_state

S = const.gait_net.robot_state_dim
T = 25


def test_tick_features_are_in_slot_order():
    n = 2
    lin, ang = torch.randn(n, 3), torch.randn(n, 3)
    feet = torch.arange(n * 12, dtype=torch.float32).view(n, 4, 3)   # FL, FR, RL, RR
    contact = torch.tensor([[True, False, True, True]] * n)
    error = torch.arange(n * 8, dtype=torch.float32).view(n, 4, 2)
    f = tick_features(lin, ang, feet, contact, error)
    assert f.shape == (n, HISTORY_DIM)
    slot = [1, 0, 3, 2]                                               # FR, FL, RR, RL
    assert torch.equal(f[:, 6:14], feet[:, slot, :2].reshape(n, 8))
    assert torch.equal(f[:, 14:18], contact[:, slot].float())
    assert torch.equal(f[:, 18:26], error[:, slot].reshape(n, 8))


def test_history_rolls_once_per_tick_and_resets_per_env():
    h = PolicyHistory(3, 4, "cpu", dim=2)
    for tick in range(6):
        h.push(torch.full((3, 2), float(tick)), tick)
        h.push(torch.full((3, 2), 99.0), tick)                         # same tick: ignored
    assert torch.equal(h.buffer[0, :, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))   # oldest first
    h.reset(torch.tensor([1]))
    assert h.buffer[1].abs().max() == 0 and h.buffer[0].abs().max() > 0
    assert h.flat().shape == (3, 8)


def critic() -> GaitnetCritic:
    return GaitnetCritic(S, [64, 64, 64], NUM_OPTIONS, const.gait_net.footstep_option_dim, [32, 32], [64, 64, 64])


def test_candidates_are_sliced_explicitly():
    old = GaitnetActor(S, [128, 128, 128], const.gait_net.footstep_option_dim, [64, 64], [128, 128, 128])
    new = GaitnetTActor.from_gaitnet_actor(old, history_len=T, history_dim=HISTORY_DIM)
    value = critic()
    obs = observation(16, padded=5, history=T * HISTORY_DIM)
    base = obs[:, : S + NUM_OPTIONS * 8]
    with torch.no_grad():
        # the old actor and the critic ignore anything after the candidates
        assert torch.allclose(old(obs)[0], old(base)[0])
        assert torch.allclose(value(obs), value(base))
        policy = GaitnetActorCritic(0, 0, 2, actor=new, critic=value)
        policy.update_distribution({"policy": obs})
    probs = policy.discrete_distribution.probs
    assert (probs[:, :5] == 0).all()                                   # padded no-ops masked
    assert (probs[:, 5:] > 0).all()


def test_touchdown_error_is_held_from_touchdown_to_touchdown():
    controllers = GpuRobotControllers(2, "cpu", 0.004, 5, GpuMpcCfg())
    controller = controllers.controller
    assert controller.touchdown_error.abs().max() == 0                 # nothing has landed
    command = lambda xy: controllers.initiate_footsteps(                # noqa: E731
        torch.tensor([True, True]), torch.tensor([0, 0]), torch.tensor([xy, xy]), torch.tensor([0.1013] * 2)
    )
    command([0.12, 0.09])
    for step in range(30):                                             # 0.12 s > the swing
        controllers.compute_torques(*[x[:2] for x in isaac_state(step)])
    assert (controller.touchdown_count[:, 0] == 1).all()
    expected = controller.touchdown_position_hip[:, 0, :2] - torch.tensor([0.12, 0.09])
    torch.testing.assert_close(controller.touchdown_error[:, 0], expected)
    command([0.05, 0.12])                                              # next swing of the same leg
    torch.testing.assert_close(controller.touchdown_error[:, 0], expected)
    controllers.reset(torch.tensor([1]))
    assert controller.touchdown_error[1].abs().max() == 0
    torch.testing.assert_close(controller.touchdown_error[0, 0], expected[0])


def test_chunked_checkpointed_actor_matches_and_backpropagates():
    old = GaitnetActor(S, [128, 128, 128], const.gait_net.footstep_option_dim, [64, 64], [128, 128, 128])
    new = GaitnetTActor.from_gaitnet_actor(old, history_len=T, history_dim=HISTORY_DIM)
    obs = observation(40, padded=5, history=T * HISTORY_DIM)
    whole = GaitnetActorCritic(0, 0, 2, actor=new, critic=critic())
    chunked = GaitnetActorCritic(0, 0, 2, actor=new, critic=critic(), actor_chunk_size=16)
    a, b = whole._actor_forward(obs), chunked._actor_forward(obs)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    grads = []
    for logits in (a[0], b[0]):
        new.zero_grad()
        logits.logsumexp(-1).sum().backward()
        grads.append(new.token_layer[0].weight.grad.clone())
    torch.testing.assert_close(grads[0], grads[1])
