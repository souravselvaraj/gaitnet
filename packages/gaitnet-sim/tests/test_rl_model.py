"""GaitNetActor through RSL-RL's PPO (construct, act, update) on a fake environment.

Needs rsl_rl but not Isaac Lab.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.algorithms import PPO  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from gaitnet_core import action_layout  # noqa: E402
from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction  # noqa: E402
from gaitnet_core.candidates import Candidates  # noqa: E402
from gaitnet_core.grid import FootholdGrid  # noqa: E402

N, L, K, S = 8, 4, 16, 41
GRID = FootholdGrid(resolution=0.015, size=(9, 9), border=2)
SCORER = {"class_name": "CandidateScorer", "shared_sizes": [32, 32], "candidate_sizes": [16, 16], "trunk_sizes": [32, 32]}
SPATIAL = {"class_name": "DenseSpatialCNN", "grid": GRID.to_dict(), "channels": [4, 4], "state_sizes": [16], "noop_sizes": [8]}


def fake_obs(groups: tuple[str, ...] = ("state", "candidates", "terrain", "privileged", "base_command")) -> TensorDict:
    valid = torch.rand(N, L, K) < 0.7
    xyz = torch.randn(N, L, K, 3) * 0.1
    candidates = Candidates(xyz=xyz, valid=valid, log_q=torch.zeros(N, L, K))
    all_groups = {
        "state": torch.randn(N, S),
        "candidates": candidates.pack(),
        "terrain": -0.26 + 0.05 * torch.randn(N, L, *GRID.patch_size),
        "privileged": torch.randn(N, 10),
        "base_command": torch.rand(N, 3) * 0.2,
    }
    return TensorDict({name: all_groups[name] for name in groups}, batch_size=[N])


class FakeEnv:
    num_envs = N
    num_actions = action_layout.DIM
    cfg: dict = {}

    def get_observations(self) -> TensorDict:
        return fake_obs()


def make_ppo(schedule: str, network: dict = SCORER, critic_groups=("state",), **actor) -> PPO:
    cfg = {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": schedule,
            "desired_kl": 0.01,
            "learning_rate": 3e-4,
        },
        "actor": {
            "class_name": "gaitnet_sim.rl.model:GaitNetActor",
            "network": dict(network),
            "distribution_cfg": None,
            **actor,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "relu"},
        "obs_groups": {"actor": ["state"], "critic": list(critic_groups)},
        "num_steps_per_env": 6,
        "multi_gpu": None,
    }
    return PPO.construct_algorithm(fake_obs(), FakeEnv(), cfg, "cpu")


def rollout_and_update(ppo: PPO) -> dict:
    obs = fake_obs()
    for _ in range(6):
        with torch.inference_mode():
            actions = ppo.act(obs)
            assert actions.shape == (N, action_layout.DIM)
            obs = fake_obs()
            ppo.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    with torch.inference_mode():
        ppo.compute_returns(obs)
    return ppo.update()


@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_ppo_round(schedule):
    torch.manual_seed(0)
    losses = rollout_and_update(make_ppo(schedule))
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses


def test_ppo_round_spatial_cnn_crop_encoder_and_privileged_critic():
    torch.manual_seed(0)
    for network in (SPATIAL, {**SCORER, "candidate_features": "xyz_crop", "grid": GRID.to_dict()}):
        losses = rollout_and_update(make_ppo("adaptive", network=network, critic_groups=("state", "privileged")))
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), (network, losses)


def test_observers_run_only_when_acting():
    torch.manual_seed(0)
    # slows (to a stop) on any tick where some leg may step
    observers = {"step_confidence_slowdown": {"patience": 1, "margin": -1e9, "scale": 0.0}}
    actor = make_ppo("fixed", observers=observers).actor
    (observer,) = actor.observers
    obs = fake_obs()
    can_step = Candidates.unpack(obs["candidates"]).valid.flatten(1).any(dim=1)

    with torch.inference_mode():
        nudge = EnvAction.decode(actor(obs, stochastic_output=True)).nudge
    expected = torch.where(can_step.unsqueeze(-1), -obs["base_command"], 0.0)
    assert torch.allclose(nudge, expected)
    waited = observer._waiting.clone()

    # PPO's update pass: gradients on, the observer neither runs nor advances
    nudge = EnvAction.decode(actor(obs, stochastic_output=True)).nudge
    assert (nudge == 0).all() and torch.equal(observer._waiting, waited)

    # episode ends clear the ended envs' memory
    with torch.inference_mode():
        actor.reset(torch.tensor([1, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert observer._waiting[0] == 0 and observer._waiting[-1] == 0
    assert torch.equal(observer._waiting[1:-1], waited[1:-1])


def test_actor_checks_its_observation_groups():
    from gaitnet_sim.rl.model import GaitNetActor

    def actor(obs, **kwargs):
        return GaitNetActor(obs, {"actor": ["state"]}, "actor", action_layout.DIM, **kwargs)

    with pytest.raises(ValueError, match="terrain"):
        actor(fake_obs(("state", "candidates")), network=SPATIAL)
    with pytest.raises(ValueError, match="patches"):
        actor(fake_obs(), network={**SPATIAL, "grid": FootholdGrid(size=(11, 11), border=2).to_dict()})
    with pytest.raises(ValueError, match="base_command"):
        actor(fake_obs(("state", "candidates")), network=SCORER, observers={"step_confidence_slowdown": {}})
    # a network that doesn't read terrain doesn't need the group
    actor(fake_obs(("state", "candidates")), network=SCORER)


def test_actions_resolve_to_candidates():
    torch.manual_seed(0)
    ppo = make_ppo("fixed")
    obs = fake_obs()
    candidates = Candidates.unpack(obs["candidates"])
    for stochastic in (True, False):
        with torch.inference_mode():
            action = EnvAction.decode(ppo.actor(obs, stochastic_output=stochastic))
        assert action.rounds == 1
        chosen, stepped_leg, stepped_target = action.choice_index[:, 0], action.leg[:, 0], action.target[:, 0]
        is_step = stepped_leg != NO_STEP_LEG
        # a step's target is the chosen candidate, which must be valid
        _, leg, target = candidates.gather(chosen)
        assert torch.equal(stepped_leg[is_step], leg[is_step])
        assert torch.allclose(stepped_target[is_step], target[is_step])
        flat_valid = candidates.valid.flatten(1)[is_step]
        assert flat_valid.gather(1, chosen[is_step].unsqueeze(1)).all()
        assert (action.nudge == 0).all()


def test_log_prob_matches_sampling_distribution():
    torch.manual_seed(0)
    ppo = make_ppo("fixed")
    obs = fake_obs()
    with torch.inference_mode():
        outputs = ppo.actor(obs, stochastic_output=True)
        params = ppo.actor.output_distribution_params
        log_prob = ppo.actor.get_output_log_prob(outputs)
        assert torch.isfinite(log_prob).all()
        # KL of a distribution with itself is zero
        kl = ppo.actor.get_kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_rejects_distribution_cfg():
    obs = fake_obs()
    from gaitnet_sim.rl.model import GaitNetActor

    with pytest.raises(ValueError):
        GaitNetActor(
            obs,
            {"actor": ["state"]},
            "actor",
            action_layout.DIM,
            network={"class_name": "CandidateScorer"},
            distribution_cfg={"class_name": "GaussianDistribution"},
        )


def test_duration_std_starts_at_its_initial_value_and_never_drops_below_the_floor():
    from gaitnet_sim.rl.model import GaitNetActor

    obs = fake_obs(("state", "candidates"))
    actor = GaitNetActor(obs, {"actor": ["state"]}, "actor", action_layout.DIM, network=dict(SCORER),
                         duration_std=0.05, duration_std_floor=0.01)
    assert float(actor.duration_std) == pytest.approx(0.05)
    with torch.no_grad():
        actor.duration_log_std.fill_(-30.0)
    assert float(actor.duration_std) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        GaitNetActor(obs, {"actor": ["state"]}, "actor", action_layout.DIM, network=dict(SCORER),
                     duration_std=0.01, duration_std_floor=0.01)


def rounds_obs() -> TensorDict:
    """Observations with a real state vector (the actor edits its gait timing between rounds):
    every leg in stance, so two legs may lift in one tick."""
    from gaitnet_core.features import DEFAULT_FEATURES, state_vector
    from gaitnet_core.state import RobotState

    state = RobotState(
        foot_pos=torch.randn(N, L, 3) * 0.05,
        foot_vel=torch.zeros(N, L, 3),
        base_lin_vel=torch.zeros(N, 3),
        base_ang_vel=torch.zeros(N, 3),
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]).expand(N, 3).clone(),
        contact=torch.ones(N, L, dtype=torch.bool),
        gait_timing=torch.zeros(N, L, 3),
        command=torch.rand(N, 3) * 0.2,
        base_command=torch.rand(N, 3) * 0.2,
    )
    candidates = Candidates(xyz=torch.randn(N, L, K, 3) * 0.1, valid=torch.ones(N, L, K, dtype=torch.bool), log_q=torch.zeros(N, L, K))
    return TensorDict({"state": state_vector(state, DEFAULT_FEATURES), "candidates": candidates.pack()}, batch_size=[N])


class TwoRoundEnv(FakeEnv):
    num_actions = action_layout.dim(2)

    def get_observations(self) -> TensorDict:
        return rounds_obs()


def test_ppo_round_with_two_footsteps_per_tick():
    torch.manual_seed(0)
    cfg = {
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 2, "num_mini_batches": 2, "schedule": "fixed", "learning_rate": 3e-4},
        "actor": {"class_name": "gaitnet_sim.rl.model:GaitNetActor", "network": dict(SCORER), "distribution_cfg": None},
        "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "relu"},
        "obs_groups": {"actor": ["state"], "critic": ["state"]},
        "num_steps_per_env": 6,
        "multi_gpu": None,
    }
    ppo = PPO.construct_algorithm(rounds_obs(), TwoRoundEnv(), cfg, "cpu")
    assert ppo.actor.rounds == 2
    obs = rounds_obs()
    two_steps = 0
    for _ in range(6):
        with torch.inference_mode():
            actions = ppo.act(obs)
            assert actions.shape == (N, action_layout.dim(2))
            action = EnvAction.decode(actions)
            both = (action.leg != NO_STEP_LEG).all(dim=1)
            assert (action.leg[both, 0] != action.leg[both, 1]).all()
            # a tick that stopped stays stopped
            assert not ((action.leg[:, 0] == NO_STEP_LEG) & (action.leg[:, 1] != NO_STEP_LEG)).any()
            two_steps += int(both.sum())
            # the stored log-probability is what replaying the rounds gives
            stored = ppo.transition.actions_log_prob
            replay = ppo.actor
            replay(obs)  # a fresh distribution, as in the update
            assert torch.allclose(replay.get_output_log_prob(actions), stored, atol=1e-5)
            obs = rounds_obs()
            ppo.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    assert two_steps > 0
    with torch.inference_mode():
        ppo.compute_returns(obs)
    losses = ppo.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses


def test_rounds_need_state_features_that_match_the_state():
    from gaitnet_sim.rl.model import GaitNetActor

    with pytest.raises(ValueError, match="gait timing"):
        GaitNetActor(fake_obs(("state", "candidates")), {"actor": ["state"]}, "actor", action_layout.dim(2), network=dict(SCORER))
