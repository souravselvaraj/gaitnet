"""`gaitnet_sim.rl.ppo.ScheduledPPO`: learning-rate and entropy schedules, and the policy-change
measure, through RSL-RL's PPO on a fake environment. Needs rsl_rl but not Isaac Lab."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.algorithms import PPO  # noqa: E402

from gaitnet_sim.rl.ppo import Schedule, ScheduledPPO  # noqa: E402
from test_rl_model import N, SCORER, FakeEnv, TwoRoundEnv, fake_obs, rounds_obs  # noqa: E402

LR = {"start": 2, "end": 6, "final": 1e-5}
ENTROPY = {"start": 2, "end": 6, "final": 0.005, "shape": "linear"}


def make(env=None, obs_fn=fake_obs, **algorithm) -> ScheduledPPO:
    cfg = {
        "algorithm": {
            "class_name": "gaitnet_sim.rl.ppo:ScheduledPPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": "fixed",
            "learning_rate": 3e-4,
            "entropy_coef": 0.02,
            **algorithm,
        },
        "actor": {"class_name": "gaitnet_sim.rl.model:GaitNetActor", "network": dict(SCORER), "distribution_cfg": None},
        "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "relu"},
        "obs_groups": {"actor": ["state"], "critic": ["state"]},
        "num_steps_per_env": 6,
        "multi_gpu": None,
    }
    ppo = PPO.construct_algorithm(obs_fn(), env or FakeEnv(), cfg, "cpu")
    assert isinstance(ppo, ScheduledPPO)
    return ppo


def rollout(ppo: ScheduledPPO, obs_fn=fake_obs) -> None:
    obs = obs_fn()
    for _ in range(6):
        with torch.inference_mode():
            ppo.act(obs)
            obs = obs_fn()
            ppo.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    with torch.inference_mode():
        ppo.compute_returns(obs)


def test_schedule_holds_then_moves_to_its_final_value():
    cosine = Schedule(initial=3e-4, start=10, end=20, final=1e-5)
    assert cosine(0) == cosine(10) == 3e-4
    assert cosine(15) == pytest.approx((3e-4 + 1e-5) / 2)
    assert cosine(20) == cosine(1000) == 1e-5
    linear = Schedule(initial=0.02, start=0, end=4, final=0.0, shape="linear")
    assert [linear(i) for i in range(5)] == pytest.approx([0.02, 0.015, 0.01, 0.005, 0.0])
    assert Schedule.from_cfg(None, 1.0) is None and Schedule.from_cfg({}, 1.0) is None
    for bad in ({"start": 5, "end": 5, "final": 0.1}, {"start": 0, "end": 5, "final": 0.1, "shape": "step"},
                {"start": 0, "end": 5, "final": 0.1, "every": 2}):
        with pytest.raises(ValueError):
            Schedule.from_cfg(bad, 1.0)


def test_the_schedules_set_the_optimizer_and_entropy_each_iteration():
    torch.manual_seed(0)
    ppo = make(lr_schedule=LR, entropy_schedule=ENTROPY)
    seen = []
    for _ in range(7):
        rollout(ppo)
        losses = ppo.update()
        seen.append((ppo.optimizer.param_groups[0]["lr"], losses["entropy_coef"]))
        assert all(math.isfinite(v) for v in losses.values()), losses
    # iterations 0-2 at the initial values, 6 at the final ones
    assert seen[0] == pytest.approx((3e-4, 0.02)) and seen[2] == pytest.approx((3e-4, 0.02))
    assert seen[4][0] == pytest.approx((3e-4 + 1e-5) / 2) and seen[4][1] == pytest.approx(0.0125)
    assert seen[6] == pytest.approx((1e-5, 0.005))
    assert ppo.iteration == 7


def test_the_policy_change_is_zero_before_an_update_and_measured_after_one():
    torch.manual_seed(0)
    ppo = make()
    rollout(ppo)
    unchanged = ppo.policy_change()
    assert unchanged["approx_kl"] == pytest.approx(0.0, abs=1e-6) and unchanged["clip_fraction"] == 0.0
    losses = ppo.update()
    assert losses["approx_kl"] >= 0.0 and 0.0 <= losses["clip_fraction"] <= 1.0
    assert math.isfinite(losses["approx_kl"])


def test_the_policy_change_covers_every_footstep_of_a_tick():
    torch.manual_seed(0)
    ppo = make(env=TwoRoundEnv(), obs_fn=rounds_obs)
    assert ppo.actor.rounds == 2
    rollout(ppo, rounds_obs)
    assert ppo.policy_change()["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    losses = ppo.update()
    assert math.isfinite(losses["approx_kl"]) and losses["approx_kl"] >= 0.0


def test_measurement_can_be_turned_off():
    torch.manual_seed(0)
    ppo = make(policy_change_samples=0)
    rollout(ppo)
    assert "approx_kl" not in ppo.update()


def test_a_resumed_run_continues_its_schedules():
    torch.manual_seed(0)
    ppo = make(lr_schedule=LR, entropy_schedule=ENTROPY)
    rollout(ppo)
    ppo.update()
    saved = {**ppo.save(), "iter": 4}  # the runner adds the iteration it saved at
    fresh = make(lr_schedule=LR, entropy_schedule=ENTROPY)
    assert fresh.load(saved, None, strict=True)
    assert fresh.iteration == 4
    assert fresh.learning_rate == pytest.approx((3e-4 + 1e-5) / 2)
    assert fresh.entropy_coef == pytest.approx(0.0125)


def test_rejects_a_learning_rate_schedule_with_the_adaptive_one():
    with pytest.raises(ValueError, match="fixed"):
        make(schedule="adaptive", lr_schedule=LR)
