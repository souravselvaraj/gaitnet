"""Widening a PPO checkpoint for new state features keeps the policy and critic unchanged."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("rsl_rl")

from tensordict import TensorDict  # noqa: E402

from gaitnet_core import action_layout, lookahead  # noqa: E402
from gaitnet_core.candidates import Candidates  # noqa: E402
from gaitnet_core.features import DEFAULT_FEATURES, LOOKAHEAD_FEATURES, feature_dim  # noqa: E402
from gaitnet_sim.rl.model import GaitNetActor  # noqa: E402
from gaitnet_sim.scripts.widen_checkpoint import insertions, widen, widen_checkpoint  # noqa: E402

NET = {"class_name": "CandidateScorer", "shared_sizes": [32, 32], "candidate_sizes": [16, 16], "trunk_sizes": [32, 32]}
N, K = 5, 8


def actor(features):
    obs = TensorDict({"state": torch.zeros(N, feature_dim(features, 4)),
                      "candidates": Candidates(torch.zeros(N, 4, K, 3), torch.ones(N, 4, K, dtype=torch.bool), torch.zeros(N, 4, K)).pack()},
                     batch_size=[N])
    return GaitNetActor(obs, {"actor": ["state"]}, "actor", action_layout.DIM, network=dict(NET), duration_std_floor=0.01)


def test_the_widened_actor_scores_exactly_as_before_whatever_the_new_inputs():
    torch.manual_seed(0)
    old = actor(DEFAULT_FEATURES)
    critic = torch.nn.Linear(feature_dim(DEFAULT_FEATURES, 4) + 10, 7)  # state then privileged
    saved = {"actor_state_dict": old.state_dict(), "critic_state_dict": {"mlp.0.weight": critic.weight.detach().clone()},
             "optimizer_state_dict": {"x": 1}, "iter": 12700}
    widened = widen_checkpoint(saved, list(DEFAULT_FEATURES), list(LOOKAHEAD_FEATURES))
    assert "optimizer_state_dict" not in widened and widened["iter"] == 0
    assert widened["infos"]["widened_from"]["iter"] == 12700

    new = actor(LOOKAHEAD_FEATURES)
    new.load_state_dict(widened["actor_state_dict"])
    state = torch.randn(N, feature_dim(DEFAULT_FEATURES, 4))
    extra = torch.randn(N, lookahead.FEATURE_DIM)
    candidates = Candidates(torch.randn(N, 4, K, 3) * 0.1, torch.ones(N, 4, K, dtype=torch.bool), torch.zeros(N, 4, K))
    with torch.no_grad():
        before = old.network(state, candidates)
        after = new.network(torch.cat([state, extra], -1), candidates)
    assert torch.allclose(before.step_logits, after.step_logits) and torch.allclose(before.duration, after.duration)

    # the critic's new columns sit after the old state, before the privileged inputs
    privileged = torch.randn(N, 10)
    weight = widened["critic_state_dict"]["mlp.0.weight"]
    assert weight.shape == (7, feature_dim(LOOKAHEAD_FEATURES, 4) + 10)
    old_out = torch.cat([state, privileged], -1) @ critic.weight.detach().T
    new_out = torch.cat([state, extra, privileged], -1) @ weight.T
    assert torch.allclose(old_out, new_out, atol=1e-5)


def test_insertions_need_the_old_features_in_order():
    with pytest.raises(ValueError):
        insertions(list(DEFAULT_FEATURES), list(reversed(LOOKAHEAD_FEATURES)), 4)
    inserts = insertions(list(DEFAULT_FEATURES), list(LOOKAHEAD_FEATURES), 4)
    assert inserts == [(feature_dim(DEFAULT_FEATURES, 4), lookahead.FEATURE_DIM)]
    assert widen(torch.ones(2, 3), [(1, 2)]).tolist() == [[1, 0, 0, 1, 1], [1, 0, 0, 1, 1]]
