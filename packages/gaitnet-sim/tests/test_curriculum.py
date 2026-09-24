"""The terrain curriculum's promotion rule, on a stand-in env. No simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_sim.env.curriculum import terrain_levels_progress  # noqa: E402


def fake_env(time_out, terminated, walked, commanded):
    moves = {}

    def update_env_origins(env_ids, move_up, move_down):
        moves["up"], moves["down"] = move_up.clone(), move_down.clone()

    n = len(time_out)
    terrain = SimpleNamespace(
        update_env_origins=update_env_origins,
        terrain_levels=torch.zeros(n, dtype=torch.long),
        max_terrain_level=12,
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(difficulty_range=(0.0, 0.5))),
    )
    term = SimpleNamespace(episode_progress=torch.tensor(walked), episode_commanded=torch.tensor(commanded))
    env = SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(terrain=terrain),
        termination_manager=SimpleNamespace(time_outs=torch.tensor(time_out), terminated=torch.tensor(terminated)),
        action_manager=SimpleNamespace(get_term=lambda name: term),
    )
    return env, moves


def test_survival_without_progress_does_not_promote():
    # walked 2 m of 4 (promote), crept 0.5 m of 4 (stay), fell (demote), stood still (promote)
    env, moves = fake_env([True, True, False, True], [False, False, True, False], [2.0, 0.5, 1.0, 0.0], [4.0, 4.0, 4.0, 0.1])
    terrain_levels_progress(env, torch.arange(4), p_up_given_success=1.0, p_down_given_failure=1.0, p_random=0.0)
    assert moves["up"].tolist() == [True, False, False, True]
    assert moves["down"].tolist() == [False, False, True, False]
