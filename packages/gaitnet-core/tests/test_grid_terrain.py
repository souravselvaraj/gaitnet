import torch

from conftest import GROUND, make_observation
from gaitnet_core.eligibility import step_eligible
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import GO1
from gaitnet_core.terrain import valid_footholds


def test_cell_round_trip(grid):
    cells = torch.stack(torch.meshgrid(torch.arange(25), torch.arange(25), indexing="ij"), -1)
    xy = grid.cell_to_xy(cells)
    assert torch.allclose(xy, grid.cell_centers(), atol=1e-6)
    back, in_bounds = grid.xy_to_cell(xy + 0.4 * grid.resolution)
    assert torch.equal(back, cells) and in_bounds.all()
    # first index runs along x, centre cell is the hip
    assert torch.allclose(grid.cell_to_xy(torch.tensor([12, 12])), torch.zeros(2), atol=1e-7)
    assert grid.cell_to_xy(torch.tensor([24, 12]))[0] > 0


def test_out_of_bounds(grid):
    half = grid.half_extent[0]
    _, in_bounds = grid.xy_to_cell(torch.tensor([[half + grid.resolution, 0.0], [0.0, 0.0]]))
    assert in_bounds.tolist() == [False, True]


def test_flat_ground_all_valid(grid):
    obs = make_observation(2, grid)
    valid = valid_footholds(obs.terrain.heights, GO1, grid)
    assert valid.shape == (2, 4, *grid.size) and valid.all()


def test_reach_band_and_unknown(grid):
    obs = make_observation(1, grid)
    h = obs.terrain.heights
    h[0, 0] = -0.8  # FL: ground far below reach (a hole)
    h[0, 1] = -0.05  # FR: ground too high
    h[0, 2] = float("-inf")  # RL: no returns
    valid = valid_footholds(h, GO1, grid)
    assert not valid[0, :3].any() and valid[0, 3].all()


def test_edge_margin_around_a_pillar(grid):
    obs = make_observation(1, grid)
    b = grid.border
    h = obs.terrain.heights
    # a 0.1 m raised pillar covering inner cells [10, 15) x [10, 15)
    h[0, 0, b + 10 : b + 15, b + 10 : b + 15] = GROUND + 0.1
    valid = valid_footholds(h, GO1, grid, step_threshold=0.02, edge_margin=2)[0, 0]
    # step cells are rows/cols 9|10 and 14|15; margin 2 extends that to 7..17
    assert not valid[7:18, 7:18].any()
    assert valid[:7].all() and valid[18:].all() and valid[:, :7].all() and valid[:, 18:].all()
    # without a margin only the two cells either side of each step are edges
    valid0 = valid_footholds(h, GO1, grid, edge_margin=0)[0, 0]
    assert valid0[11:14, 11:14].all() and not valid0[9, 12] and not valid0[10, 12]


def test_edge_just_outside_grid_is_seen():
    grid = FootholdGrid(border=3)
    obs = make_observation(1, grid)
    obs.terrain.heights[0, 0, :1] = float("-inf")  # void in the border's outermost row
    valid = valid_footholds(obs.terrain.heights, GO1, grid, edge_margin=2)[0, 0]
    # outer border row 0 -> edge at patch rows 0,1 -> margin reaches patch row 3 = inner row 0
    assert not valid[0].any() and valid[1:].all()


def test_step_eligibility():
    timing = torch.zeros(3, 4, 3)
    timing[1, 0, 1] = 0.1  # one leg in swing: the other three may step
    timing[2, :2, 1] = 0.1  # two in swing: nobody may step
    eligible = step_eligible(timing)
    assert eligible[0].all()
    assert eligible[1].tolist() == [False, True, True, True]
    assert not eligible[2].any()


def test_min_stance_time_keeps_a_just_landed_leg_down():
    from gaitnet_core.eligibility import step_eligible

    timing = torch.zeros(1, 4, 3)
    timing[0, :, 2] = torch.tensor([0.02, 0.2, 0.2, 0.2])  # FL landed 20 ms ago
    assert step_eligible(timing, 2, min_stance_time=0.08).tolist() == [[False, True, True, True]]
    assert step_eligible(timing, 2).all()


def test_kinematic_rules():
    from conftest import make_observation
    from gaitnet_core.planner import FootholdRules
    from gaitnet_core.robot_spec import GO1

    obs = make_observation(1)
    grid = obs.terrain.grid
    centres = grid.cell_centers()
    hips = torch.tensor(GO1.hip_offsets)[:, :2]
    # every foot a little outside its own hip, except FR's, right under FL's hip
    obs.state.foot_pos[0, :, :2] = hips + torch.tensor([0.0, 0.08]) * torch.sign(hips[:, 1:2])
    obs.state.foot_pos[0, 1, :2] = hips[0]
    rules = FootholdRules(min_foot_separation=0.06)
    ok = rules.kinematic(obs, GO1)[0]
    near_fr = centres.norm(dim=-1) < 0.06  # FL's cells within 6 cm of FL's hip = of FR's foot
    assert not ok[0][near_fr].any() and ok[0][~near_fr].all()
    # FR's own foot (under FL's hip) doesn't block FR; FL's foot, which FR's grid reaches, does
    fl_foot = obs.state.foot_pos[0, 0, :2]
    near_fl = (hips[1] + centres - fl_foot).norm(dim=-1) < 0.06
    assert near_fl.any() and torch.equal(ok[1], ~near_fl)

    rules = FootholdRules(midline_margin=0.02)
    ok = rules.kinematic(obs, GO1)[0]
    left_cells_y = hips[0, 1] + centres[..., 1]
    assert torch.equal(ok[0], left_cells_y >= 0.02)
    assert torch.equal(ok[1], -(hips[1, 1] + centres[..., 1]) >= 0.02)

    obs.terrain.heights[:] = -0.38  # a step down: the grid's corners are out of reach
    ok = FootholdRules(max_reach=0.40).kinematic(obs, GO1)[0, 0]
    assert ok[12, 12] and not ok[0, 0] and not ok[24, 24]


def test_median_filter_removes_speckle_and_keeps_steps():
    from conftest import make_observation
    from gaitnet_core.planner import FootholdRules
    from gaitnet_core.robot_spec import GO1
    from gaitnet_core.terrain import median_filter

    torch.manual_seed(0)
    obs = make_observation(1)
    obs.terrain.heights += torch.randn_like(obs.terrain.heights) * 0.01  # 1 cm speckle
    noisy = FootholdRules().valid(obs, GO1).float().mean()
    filtered = FootholdRules(median_window=3).valid(obs, GO1).float().mean()
    assert noisy < 0.3 and filtered > 0.8

    step = torch.full((1, 1, 31, 31), -0.26)
    step[..., 16:, :] = -0.16
    step[0, 0, 5, 5] = float("-inf")  # a lone unknown cell is filled; a big unknown area stays
    step[0, 0, 20:, 20:] = float("-inf")
    out = median_filter(step, 3)
    assert torch.equal(out[0, 0, 10:14, 10], step[0, 0, 10:14, 10]) and out[0, 0, 17, 10] == -0.16
    assert torch.isfinite(out[0, 0, 5, 5]) and torch.isinf(out[0, 0, 25, 25])
