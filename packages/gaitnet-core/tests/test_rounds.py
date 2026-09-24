"""Several footsteps per tick: the action layout, the state edit between rounds, the tick's
distribution (eligibility, stopping, log-probability by replay) and the planner and bundle."""

import pytest
import torch

from conftest import make_observation
from gaitnet_core import action_layout
from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.bundle import BundleError, check_manifest, load_bundle, save_bundle, PolicyBundle
from gaitnet_core.candidates import Candidates
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim, state_vector
from gaitnet_core.interfaces import Nudge
from gaitnet_core.mock_robot import ReplayRobot
from gaitnet_core.networks import CandidateScorer
from gaitnet_core.planner import FootholdRules, FootstepPlanner
from gaitnet_core.robot_spec import GO1
from gaitnet_core.rounds import StateEditor, TickDistribution, TickSelection
from gaitnet_core.runtime import PlannerRuntime
from gaitnet_core.samplers import UniformJitter
from gaitnet_core.selection import FootstepDistribution, Scores

N, L, K = 6, 4, 8
S = feature_dim(DEFAULT_FEATURES, L)


def test_action_layout_dims():
    assert action_layout.DIM == action_layout.dim(1) == 9
    assert action_layout.dim(2) == 15
    assert action_layout.rounds_for_dim(15) == 2
    with pytest.raises(ValueError):
        action_layout.rounds_for_dim(10)


def test_two_round_action_round_trips_and_one_round_keeps_the_old_layout():
    action = EnvAction(
        choice_index=torch.tensor([[3, 9], [32, 32]]),
        duration=torch.tensor([[0.2, 0.15], [0.0, 0.0]]),
        leg=torch.tensor([[0, 1], [NO_STEP_LEG, NO_STEP_LEG]]),
        target=torch.arange(12.0).reshape(2, 2, 3),
        nudge=torch.ones(2, 3),
    )
    decoded = EnvAction.decode(action.encode())
    assert decoded.rounds == 2
    assert torch.equal(decoded.choice_index, action.choice_index)
    assert torch.allclose(decoded.target, action.target)
    commands = decoded.footstep_commands()
    assert [c.active.tolist() for c in commands] == [[True, False], [True, False]]
    assert commands[1].leg.tolist() == [1, 0]

    single = EnvAction(torch.tensor([5]), torch.tensor([0.2]), torch.tensor([2]), torch.tensor([[1.0, 2.0, 3.0]]), torch.zeros(1, 3))
    assert single.encode()[0].tolist() == pytest.approx([5.0, 0.2, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


def test_state_editor_matches_the_feature_of_an_edited_state():
    obs = make_observation(3)
    obs.state.gait_timing[:, :, 2] = 0.3  # standing a while
    state = state_vector(obs.state, DEFAULT_FEATURES)
    editor = StateEditor(DEFAULT_FEATURES, L)
    assert torch.equal(editor.gait_timing(state), obs.state.gait_timing)

    leg, duration, where = torch.tensor([1, 2, 3]), torch.tensor([0.2, 0.25, 0.3]), torch.tensor([True, True, False])
    edited = editor.with_swing(state, leg, duration, where)
    expected = obs.state.gait_timing.clone()
    for i in range(2):
        expected[i, leg[i]] = torch.tensor([0.0, float(duration[i]), 0.0])
    obs.state.gait_timing = expected
    assert torch.allclose(edited, state_vector(obs.state, DEFAULT_FEATURES))


def _tick(rounds, noop=-5.0, min_stance=2, gait_timing=None, seed=0):
    """A tick on made-up scores: every leg's candidates score high, so robots step."""
    gen = torch.Generator().manual_seed(seed)
    candidates = Candidates(torch.randn(N, L, K, 3, generator=gen) * 0.1, torch.ones(N, L, K, dtype=torch.bool), torch.zeros(N, L, K))
    obs = make_observation(N)
    if gait_timing is not None:
        obs.state.gait_timing = gait_timing
    state = state_vector(obs.state, DEFAULT_FEATURES)
    base = torch.randn(N, L, K, generator=gen)
    seen = []
    weights = torch.nn.Parameter(torch.zeros(()))

    def score(s):
        seen.append(s)
        return Scores(base + weights, torch.full((N,), noop) + weights, torch.full((N, L, K), 0.2))

    editor = StateEditor(DEFAULT_FEATURES, L) if rounds > 1 else None
    tick = TickDistribution(score, state, candidates, torch.tensor(0.05), rounds, min_stance, editor)
    return tick, candidates, seen, weights


def test_one_round_is_the_one_step_distribution():
    tick, candidates, _, _ = _tick(1)
    torch.manual_seed(0)
    selection = tick.sample()
    single = FootstepDistribution(tick._first_scores, candidates, torch.tensor(0.05))
    assert torch.allclose(tick.log_prob(selection), single.log_prob(selection.round(0)))


def test_later_rounds_step_other_legs_and_respect_the_stance_count():
    torch.manual_seed(0)
    tick, candidates, seen, _ = _tick(3)
    selection = tick.sample()
    legs = selection.index // K
    stepped = selection.index < candidates.noop_index
    # with four legs in stance and two that must stay, two rounds at most can step
    assert stepped[:, :2].all() and not stepped[:, 2].any()
    assert (legs[:, 0] != legs[:, 1]).all()
    # round 2 saw round 1's leg in swing, with the clamped duration remaining
    editor = StateEditor(DEFAULT_FEATURES, L)
    timing = editor.gait_timing(seen[1])
    rows = torch.arange(N)
    assert torch.allclose(timing[rows, legs[:, 0], 1], selection.duration[:, 0].clamp(0.1, 0.3))


def test_the_no_op_ends_the_tick():
    torch.manual_seed(0)
    tick, candidates, _, _ = _tick(2, noop=50.0)  # waiting always wins
    selection = tick.sample()
    assert (selection.index == candidates.noop_index).all()
    assert torch.allclose(tick.log_prob(selection), torch.zeros(N), atol=1e-4)


def test_log_prob_by_replay_matches_sampling_and_has_gradients():
    torch.manual_seed(0)
    tick, _, _, weights = _tick(2)
    selection = tick.sample()
    sampled = tick.log_prob(selection)
    # a fresh distribution (as in PPO's update) replays the stored choices
    replay, _, _, weights = _tick(2)
    replayed = replay.log_prob(TickSelection(selection.index.clone(), selection.duration.clone()))
    assert torch.allclose(replayed, sampled, atol=1e-5)
    assert torch.isfinite(replay.entropy()).all()
    replayed.sum().backward()
    assert weights.grad is not None and torch.isfinite(weights.grad)


def test_a_robot_with_one_eligible_leg_steps_once():
    timing = torch.zeros(N, L, 3)
    timing[:, 3, 1] = 0.1  # RR already swinging: only one more leg may lift
    torch.manual_seed(0)
    tick, candidates, _, _ = _tick(2, gait_timing=timing)
    selection = tick.sample()
    assert (selection.index[:, 0] < candidates.noop_index).all()
    assert (selection.index[:, 1] == candidates.noop_index).all()


def _planner(rounds):
    torch.manual_seed(0)
    net = CandidateScorer(S, shared_sizes=[32], candidate_sizes=[16], trunk_sizes=[32], use_bf16=False)
    rules = FootholdRules(max_steps_per_tick=rounds)
    return FootstepPlanner(net, GO1, make_observation(1).terrain.grid, DEFAULT_FEATURES, UniformJitter(16), rules)


def test_planner_plans_several_rounds_and_the_runtime_sends_them():
    planner = _planner(2)
    obs = make_observation(4)
    plan = planner.plan(obs, deterministic=False, generator=torch.Generator().manual_seed(0))
    assert len(plan.more) == 1
    commands = plan.footstep_commands()
    both = commands[0].active & commands[1].active
    assert (commands[0].leg[both] != commands[1].leg[both]).all()
    assert plan.env_action(Nudge(torch.zeros(4, 3))).encode().shape == (4, action_layout.dim(2))

    robot = ReplayRobot([obs])
    PlannerRuntime(robot, planner, rate_hz=None).step()
    footsteps, _ = robot.commands[0]
    assert len(footsteps) == 2


def test_bundle_saves_rounds_and_reads_format_2(tmp_path):
    planner = _planner(2)
    bundle = PolicyBundle(
        actor=planner.network, robot=GO1, grid=planner.grid, features=DEFAULT_FEATURES,
        rules=planner.rules, train_sampler={"name": "uniform_jitter", "per_leg": 16}, duration_std=0.05, extra={},
    )
    path = save_bundle(tmp_path / "b.pt", bundle)
    assert load_bundle(path).rules.max_steps_per_tick == 2

    manifest = torch.load(path, weights_only=True)["manifest"]
    old = {**manifest, "format_version": 2, "rules": {k: v for k, v in manifest["rules"].items() if k != "max_steps_per_tick"}}
    check_manifest(old)
    with pytest.raises(BundleError):
        check_manifest({**manifest, "rules": {**manifest["rules"], "made_up": 1}})
    with pytest.raises(BundleError):
        check_manifest({**manifest, "features": [f for f in DEFAULT_FEATURES if f != "gait_timing"]})
