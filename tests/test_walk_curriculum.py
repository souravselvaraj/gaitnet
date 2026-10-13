"""The `walk` terrain curriculum's rule (components/walk_curriculum.py)."""
import torch

from gaitnet.gaitnet.components.walk_curriculum import walk_moves

T, F = True, False


def moves(time_outs, failed, walked, commanded, **kwargs):
    b = lambda xs: torch.tensor(xs, dtype=torch.bool)  # noqa: E731
    f = lambda xs: torch.tensor(xs, dtype=torch.float32)  # noqa: E731
    kwargs = {"required_fraction": 0.5, "p_up_given_success": 1.0, "p_down_given_failure": 1.0, "p_random": 0.0, **kwargs}
    return walk_moves(b(time_outs), b(failed), f(walked), f(commanded), **kwargs)


def test_survivors_move_up_only_when_they_walked():
    up, down, progressed = moves([T, T, T], [F, F, F], [1.6, 1.4, 0.3], [3.0, 3.0, 3.0])
    assert up.tolist() == [T, F, F]
    assert down.tolist() == [F, T, T]           # stalled counts as failing
    assert progressed.tolist() == [T, F, F]


def test_falls_move_down_whatever_was_walked():
    up, down, _ = moves([F, F], [T, T], [3.0, 0.0], [3.0, 3.0])
    assert up.tolist() == [F, F]
    assert down.tolist() == [T, T]


def test_standing_commands_count_survival():
    up, down, progressed = moves([T, F], [F, T], [0.0, 0.0], [0.1, 0.1])
    assert progressed.tolist() == [T, T]
    assert up.tolist() == [T, F]
    assert down.tolist() == [F, T]


def test_probabilities_set_the_rates():
    n = 200_000
    torch.manual_seed(0)
    time_outs = torch.ones(n, dtype=torch.bool)
    up, down, _ = walk_moves(time_outs, ~time_outs, torch.full((n,), 2.0), torch.full((n,), 3.0), 0.5, p_random=0.0)
    assert abs(up.float().mean().item() - 0.1) < 0.005
    assert down.sum() == 0
    up, down, _ = walk_moves(time_outs, ~time_outs, torch.zeros(n), torch.full((n,), 3.0), 0.5, p_random=0.0)
    assert up.sum() == 0
    assert abs(down.float().mean().item() - 0.5) < 0.005


def test_exploration_episodes_do_not_move_the_curriculum():
    from gaitnet.gaitnet.components.walk_curriculum import next_levels
    torch.manual_seed(0)
    level = torch.tensor([3, 3, 3, 11])
    explored = torch.tensor([F, T, F, F])
    up = torch.tensor([T, T, F, T])
    down = torch.tensor([F, F, T, F])
    new, explore, run_on = next_levels(level, explored, up, down, num_levels=12, random_fraction=0.0)
    assert new[:3].tolist() == [4, 3, 2]          # the explored episode leaves its curriculum level alone
    assert 0 <= new[3] < 12                       # climbing past the top goes to a random level
    assert not explore.any() and torch.equal(run_on, new)


def test_random_fraction_sets_the_share_of_random_episodes():
    from gaitnet.gaitnet.components.walk_curriculum import next_levels
    torch.manual_seed(0)
    n = 100_000
    zeros = torch.zeros(n, dtype=torch.long)
    no = torch.zeros(n, dtype=torch.bool)
    new, explore, run_on = next_levels(zeros, no, no, no, num_levels=12, random_fraction=0.25)
    assert abs(explore.float().mean().item() - 0.25) < 0.01
    assert torch.equal(run_on[~explore], new[~explore])
    assert abs(run_on[explore].float().mean().item() - 5.5) < 0.1   # uniform over 0..11
