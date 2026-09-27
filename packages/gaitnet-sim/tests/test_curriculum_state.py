"""Terrain curriculum carried across a resume (gaitnet_sim.rl.curriculum_state and the train script's hooks)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gaitnet_sim.rl import curriculum_state as cs
from gaitnet_sim.scripts import train


def make_env(num_envs=8, rows=12, cols=4):
    origins = torch.arange(rows * cols * 3, dtype=torch.float).reshape(rows, cols, 3)
    terrain = SimpleNamespace(
        terrain_origins=origins,
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        terrain_types=torch.arange(num_envs) % cols,
        env_origins=torch.zeros(num_envs, 3),
        max_terrain_level=rows,
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(terrain=terrain))), terrain


def test_terrain_of_needs_a_level_curriculum():
    env, terrain = make_env()
    assert cs.terrain_of(env) is terrain
    terrain.terrain_origins = None  # grid spawning
    assert cs.terrain_of(env) is None
    assert cs.terrain_of(SimpleNamespace(unwrapped=SimpleNamespace())) is None


def test_resample_keeps_the_distribution_and_clips():
    g = torch.Generator().manual_seed(0)
    saved = torch.tensor([2, 2, 2, 9])
    levels = cs.resample(saved, 4000, 12, g)
    assert set(levels.tolist()) <= {2, 9}
    assert abs((levels == 9).float().mean().item() - 0.25) < 0.03
    assert cs.resample(torch.tensor([30]), 5, 12, g).tolist() == [11] * 5
    with pytest.raises(ValueError):
        cs.resample(torch.tensor([], dtype=torch.long), 3, 12)


def test_initial_levels_range():
    g = torch.Generator().manual_seed(0)
    levels = cs.initial_levels(3, 11, 6000, 12, g)
    assert levels.min().item() == 3 and levels.max().item() == 11
    assert abs(levels.float().mean().item() - 7.0) < 0.1
    assert cs.initial_levels(5, 40, 100, 12, g).max().item() == 11
    with pytest.raises(ValueError):
        cs.initial_levels(9, 3, 10, 12)
    assert cs.parse_range("3:11") == (3, 11)
    with pytest.raises(ValueError):
        cs.parse_range("3-11")


def test_apply_levels_moves_origins():
    env, terrain = make_env()
    levels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    assert cs.apply_levels(env, levels)
    assert torch.equal(terrain.terrain_levels, levels)
    assert torch.equal(terrain.env_origins, terrain.terrain_origins[levels, terrain.terrain_types])
    with pytest.raises(ValueError):
        cs.apply_levels(env, torch.tensor([1, 2]))


@pytest.fixture
def runner_class(monkeypatch):
    """OnPolicyRunner with its own save/load replaced by in-memory fakes, restored afterwards."""
    from rsl_rl.runners import OnPolicyRunner

    store = {}

    def fake_save(self, path, infos=None):
        store[path] = infos

    def fake_load(self, path, *args, **kwargs):
        return store[path]

    monkeypatch.setattr(OnPolicyRunner, "save", fake_save)
    monkeypatch.setattr(OnPolicyRunner, "load", fake_load)
    monkeypatch.delattr(OnPolicyRunner, "_unhooked", raising=False)
    return OnPolicyRunner, store


def test_checkpoint_round_trip_restores_levels(runner_class):
    runner_cls, store = runner_class
    train._carry_curriculum(restore=True, initial=(0, 1))
    env, terrain = make_env(num_envs=8)
    terrain.terrain_levels[:] = 7
    runner = runner_cls.__new__(runner_cls)
    runner.env = env
    runner.save("ckpt", {"other": 1})
    assert store["ckpt"]["other"] == 1 and store["ckpt"][cs.INFOS_KEY].tolist() == [7] * 8

    fresh_env, fresh_terrain = make_env(num_envs=16)
    runner.env = fresh_env
    runner.load("ckpt")
    assert fresh_terrain.terrain_levels.tolist() == [7] * 16  # saved levels win over `initial`


def test_old_checkpoint_uses_initial_range_and_reset_skips(runner_class):
    runner_cls, store = runner_class
    store["old"] = None  # a checkpoint from before levels were saved
    train._carry_curriculum(restore=True, initial=(3, 11))
    env, terrain = make_env(num_envs=500)
    runner = runner_cls.__new__(runner_cls)
    runner.env = env
    runner.load("old")
    assert terrain.terrain_levels.min().item() >= 3 and 6.5 < terrain.terrain_levels.float().mean().item() < 7.5

    store["new"] = {cs.INFOS_KEY: torch.tensor([9, 9])}
    train._carry_curriculum(restore=False, initial=None)
    env2, terrain2 = make_env(num_envs=4)
    runner.env = env2
    runner.load("new")
    assert terrain2.terrain_levels.tolist() == [0, 0, 0, 0]  # --reset_curriculum leaves the importer's levels
