"""GaitNet-T (gaitnet_t.py) against the GaitnetActor it extends."""
import os
import re

import pytest
import torch
import torch.nn.functional as F

import gaitnet.constants as const
from gaitnet.gaitnet.gaitnet import GaitnetActor
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor

K = const.gait_net.num_footstep_options
M = K * const.robot.num_legs + 1
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..", "gaitnet-ros", "src", "gaitnet_ros", "gaitnet-models",
                          "gaitnet_20260112_201614-fe1b10f4-random-sampling_1035.pt")


def old_actor(trained: bool) -> GaitnetActor:
    actor = GaitnetActor(const.gait_net.robot_state_dim, [128, 128, 128], const.gait_net.footstep_option_dim, [64, 64], [128, 128, 128])
    if trained and os.path.exists(CHECKPOINT):
        state = torch.load(CHECKPOINT, map_location="cpu")["model_state_dict"]
        actor.load_state_dict({re.sub(r"^actor\.", "", k): v for k, v in state.items() if k.startswith("actor.")})
    return actor.eval()


def observation(n: int, padded: int = 5, history: int = 0, seed: int = 0) -> torch.Tensor:
    """n random observations: state, K candidates per leg in slot order with `padded` of them turned
    into no-op padding, the real no-op last, then `history` extra values."""
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(n, const.gait_net.robot_state_dim, generator=g)
    legs = torch.arange(4).repeat_interleave(K).expand(n, -1)
    one_hot = F.one_hot(legs + 1, 5).float()
    xy = (torch.rand(n, 4 * K, 2, generator=g) - 0.5) * 0.36
    options = torch.cat([one_hot, xy, torch.zeros(n, 4 * K, 1)], -1)
    options[:, :padded] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    no_op = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]).expand(n, 1, 8)
    candidates = torch.cat([options, no_op], 1)
    return torch.cat([state, candidates.flatten(1), torch.randn(n, history, generator=g)], 1)


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("history", [(0, 0), (25, 12)])
def test_warm_start_reproduces_the_old_actor(trained, history):
    old = old_actor(trained)
    new = GaitnetTActor.from_gaitnet_actor(old, history_len=history[0], history_dim=history[1]).eval()
    obs = observation(64, history=history[0] * history[1])
    with torch.no_grad():
        logits_old, dur_old = old(obs[:, : const.gait_net.robot_state_dim + M * 8])
        logits_new, dur_new = new(obs)
    assert torch.allclose(logits_old, logits_new, atol=1e-5)
    assert torch.allclose(dur_old, dur_new, atol=1e-6)


def perturbed(actor: GaitnetTActor) -> GaitnetTActor:
    """Give the zero-initialised mixer and history projection some weight, as training would."""
    with torch.no_grad():
        for p in actor.mixer.parameters():
            p.add_(0.05 * torch.randn_like(p))
        if actor.history_projection is not None:
            actor.history_projection.weight.normal_(0, 0.05)
    return actor.eval()


def test_mixed_scores_depend_on_other_candidates():
    new = perturbed(GaitnetTActor.from_gaitnet_actor(old_actor(False)))
    obs = observation(8, padded=0)
    with torch.no_grad():
        base = new(obs)[0][:, K]                    # first FL candidate
        moved = obs.clone()
        start = const.gait_net.robot_state_dim
        moved[:, start + 5 : start + 7] += 0.1      # move the first FR candidate (dx, dy) only
        assert not torch.allclose(base, new(moved)[0][:, K])


def test_permuting_candidates_permutes_scores():
    new = perturbed(GaitnetTActor.from_gaitnet_actor(old_actor(False)))
    obs = observation(8, padded=3)
    perm = torch.cat([torch.randperm(M - 1), torch.tensor([M - 1])])   # the real no-op stays last
    s = const.gait_net.robot_state_dim
    permuted = obs.clone()
    permuted[:, s : s + M * 8] = obs[:, s : s + M * 8].view(-1, M, 8)[:, perm].flatten(1)
    with torch.no_grad():
        logits, dur = new(obs)
        logits_p, dur_p = new(permuted)
    assert torch.allclose(logits[:, perm], logits_p, atol=1e-5)
    assert torch.allclose(dur[:, perm], dur_p, atol=1e-6)


def test_padding_is_invisible_to_real_candidates():
    """Scores of the real candidates are the same as if the padding slots were not there at all."""
    import copy
    new = perturbed(GaitnetTActor.from_gaitnet_actor(old_actor(False)))
    obs = observation(4, padded=4)
    s = const.gait_net.robot_state_dim
    unpadded = torch.cat([obs[:, :s], obs[:, s + 4 * 8 :]], 1)
    small = copy.deepcopy(new)
    small.num_options = M - 4
    with torch.no_grad():
        assert torch.allclose(new(obs)[0][:, 4:], small(unpadded)[0], atol=1e-5)


def test_history_reaches_the_scores_and_gradients_flow():
    new = perturbed(GaitnetTActor.from_gaitnet_actor(old_actor(False), history_len=25, history_dim=12))
    new.train()
    obs = observation(16, history=25 * 12).requires_grad_(True)
    logits, dur = new(obs)
    (logits.sum() + dur.sum()).backward()
    hist_grad = obs.grad[:, -25 * 12 :]
    assert hist_grad.abs().sum() > 0
    assert all(p.grad is not None for p in new.mixer.parameters())
