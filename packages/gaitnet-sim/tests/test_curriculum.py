"""The terrain curriculum's promotion rule, on a stand-in env. No simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_sim.env.curriculum import terrain_levels_progress  # noqa: E402


class Terminations:
    """A termination manager with named terms: `time_out`, a fall, and optionally a constraint."""

    def __init__(self, time_out, fell, constraint=None):
        from gaitnet_sim.env.constraints import ConstraintTermination

        self.terms = {"time_out": torch.tensor(time_out), "bad_orientation": torch.tensor(fell)}
        self.cfgs = {"time_out": SimpleNamespace(time_out=True, func=None), "bad_orientation": SimpleNamespace(time_out=False, func=None)}
        if constraint is not None:
            self.terms["slip_constraint"] = torch.tensor(constraint)
            # as the manager holds it once built: an instance of the term class, not the class
            self.cfgs["slip_constraint"] = SimpleNamespace(time_out=False, func=ConstraintTermination.__new__(ConstraintTermination))
        self.time_outs = self.terms["time_out"]
        self.terminated = torch.stack([t for n, t in self.terms.items() if n != "time_out"]).any(0)
        self.active_terms = list(self.terms)

    def get_term(self, name):
        return self.terms[name]

    def get_term_cfg(self, name):
        return self.cfgs[name]


def fake_env(time_out, terminated, walked, commanded, constraint=None):
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
        termination_manager=Terminations(time_out, terminated, constraint),
        action_manager=SimpleNamespace(get_term=lambda name: term),
    )
    return env, moves


def test_survival_without_progress_does_not_promote():
    # walked 2 m of 4 (promote), crept 0.5 m of 4 (stay), fell (demote), stood still (promote)
    env, moves = fake_env([True, True, False, True], [False, False, True, False], [2.0, 0.5, 1.0, 0.0], [4.0, 4.0, 4.0, 0.1])
    terrain_levels_progress(env, torch.arange(4), p_up_given_success=1.0, p_down_given_failure=1.0, p_random=0.0)
    assert moves["up"].tolist() == [True, False, False, True]
    assert moves["down"].tolist() == [False, False, True, False]


def test_constraint_terminations_are_not_falls():
    """A robot ended by a constraint (CaT) is neither promoted nor demoted; only falls demote."""
    from gaitnet_sim.env.curriculum import terrain_levels_survival

    # fell, ended by a constraint, timed out
    env, moves = fake_env([False, False, True], [True, False, False], [1.0, 1.0, 4.0], [4.0, 4.0, 4.0],
                          constraint=[False, True, False])
    terrain_levels_survival(env, torch.arange(3), p_up_given_success=1.0, p_down_given_failure=1.0, p_random=0.0)
    assert moves["down"].tolist() == [True, False, False]
    assert moves["up"].tolist() == [False, False, True]
    # naming the fall terms explicitly gives the same
    terrain_levels_survival(env, torch.arange(3), p_up_given_success=1.0, p_down_given_failure=1.0, p_random=0.0,
                            failure_terms=["bad_orientation"])
    assert moves["down"].tolist() == [True, False, False]
