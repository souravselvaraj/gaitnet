"""GaitnetCritic's asymmetric tail and no-op column (gaitnet.py), and loading a checkpoint saved without the tail."""
import pytest
import torch

import gaitnet.constants as const
from gaitnet.gaitnet.gaitnet import NUM_OPTIONS, GaitnetActor, GaitnetActorCritic, GaitnetCritic

S = const.gait_net.robot_state_dim
D = const.gait_net.footstep_option_dim
TAIL = 40


def critic() -> GaitnetCritic:
    torch.manual_seed(0)
    return GaitnetCritic(S, [64, 64, 64], NUM_OPTIONS, D, [32, 32], [64, 64, 64], [32, 32])


def observation(n: int, tail: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    state = torch.randn(n, S, generator=g)
    legs = torch.randint(0, 5, (n, NUM_OPTIONS), generator=g)
    legs[:, -1] = 0
    one_hot = torch.nn.functional.one_hot(legs, 5).float()
    xy = torch.where(legs.unsqueeze(-1) == 0, 0.0, torch.rand(n, NUM_OPTIONS, 2, generator=g) - 0.5)
    candidates = torch.cat([one_hot, xy, torch.zeros(n, NUM_OPTIONS, 1)], -1)
    return torch.cat([state, candidates.flatten(1), torch.randn(n, tail, generator=g)], 1)


def test_a_new_tail_leaves_the_value_unchanged_until_trained():
    plain, tailed = critic(), critic()
    tailed.add_tail(TAIL)
    obs = observation(32, TAIL)
    with torch.no_grad():
        assert torch.equal(plain(obs[:, : S + NUM_OPTIONS * D]), tailed(obs))
        tailed.tail_projection.weight.normal_(0, 0.1)
        assert not torch.allclose(plain(obs[:, : S + NUM_OPTIONS * D]), tailed(obs))


def test_the_tail_learns():
    tailed = critic()
    tailed.add_tail(TAIL)
    tailed(observation(16, TAIL)).sum().backward()
    assert tailed.tail_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("noop_index, sees_rl", [(4, False), (0, True)])
def test_noop_column_decides_whether_rl_candidates_are_seen(noop_index, sees_rl):
    """Column 4 (legacy) turns every RL-leg candidate into the no-op embedding; column 0 encodes it."""
    c = critic()
    c.noop_index = noop_index
    obs = observation(8)
    candidates = obs[:, S : S + NUM_OPTIONS * D].view(8, NUM_OPTIONS, D)
    rl = candidates[:, :, 4] == 1
    assert rl.any()
    moved = obs.clone()
    moved_candidates = moved[:, S : S + NUM_OPTIONS * D].view(8, NUM_OPTIONS, D)
    moved_candidates[:, :, 5] += torch.where(rl, 0.2, 0.0)       # shift the RL candidates' dx only
    with torch.no_grad():
        assert torch.allclose(c(obs), c(moved)) != sees_rl


def test_checkpoint_without_tail_loads_into_a_tailed_critic():
    actor = GaitnetActor(S, [128, 128, 128], D, [64, 64], [128, 128, 128])
    saved = GaitnetActorCritic(0, 0, 0, actor=actor, critic=critic())
    tailed_critic = critic()
    tailed_critic.add_tail(TAIL)
    loaded = GaitnetActorCritic(0, 0, 0, actor=GaitnetActor(S, [128, 128, 128], D, [64, 64], [128, 128, 128]), critic=tailed_critic)
    with torch.no_grad():
        for p in saved.critic.parameters():
            p.add_(0.01)
    assert loaded.load_state_dict(saved.state_dict()) is False
    assert loaded.loaded_without_critic_tail
    obs = observation(8, TAIL)
    with torch.no_grad():
        assert torch.equal(saved.critic(obs[:, : S + NUM_OPTIONS * D]), loaded.critic(obs))
    # a checkpoint with the tail resumes normally
    again = GaitnetActorCritic(0, 0, 0, actor=GaitnetActor(S, [128, 128, 128], D, [64, 64], [128, 128, 128]), critic=critic())
    again.critic.add_tail(TAIL)
    assert again.load_state_dict(loaded.state_dict()) is True
