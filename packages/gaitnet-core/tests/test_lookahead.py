"""The terrain ahead (gaitnet_core.lookahead): pooling, holes, unknown ground, and the feature."""

import pytest
import torch

from gaitnet_core import lookahead
from gaitnet_core.features import FEATURES, LOOKAHEAD_FEATURES, feature_dim, feature_slices, state_vector
from gaitnet_core.state import RobotState

FX, FY = lookahead.FINE_SHAPE
CX, CY = lookahead.COARSE_SHAPE


def channels(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, holes, known = features.reshape(features.shape[0], 3, CX, CY).unbind(1)
    return height, holes, known


def test_geometry():
    assert lookahead.FEATURE_DIM == 3 * CX * CY == 144
    points = lookahead.fine_points()
    assert points.shape == (FX, FY, 2)
    # cell centres, 1.6 m x 1.2 m, centred ahead of the base
    assert points[..., 0].min() == pytest.approx(0.6 - 0.775) and points[..., 0].max() == pytest.approx(0.6 + 0.775)
    assert points[..., 1].min() == pytest.approx(-0.575) and points[..., 1].max() == pytest.approx(0.575)


def test_flat_known_ground_reads_level_and_complete():
    ground = torch.tensor([0.3, -1.0])
    heights = ground.view(2, 1, 1).expand(2, FX, FY).clone()
    height, holes, known = channels(lookahead.terrain_ahead(heights, torch.ones(2, FX, FY, dtype=torch.bool), ground))
    assert torch.allclose(height, torch.zeros_like(height)) and (holes == 0).all() and (known == 1).all()


def test_a_hole_and_a_step_land_in_their_cells():
    heights = torch.zeros(1, FX, FY)
    heights[0, 0:4, 0:4] = -0.5           # a whole coarse cell is a hole
    heights[0, 4:6, 0:4] = -0.5           # half of the next one
    heights[0, 8:12, 4:8] = 0.1           # a step up
    height, holes, known = channels(lookahead.terrain_ahead(heights, torch.ones(1, FX, FY, dtype=torch.bool), torch.zeros(1)))
    assert holes[0, 0, 0] == 1.0 and holes[0, 1, 0] == pytest.approx(0.5) and holes[0, 2, 1] == 0.0
    assert height[0, 0, 0] == pytest.approx(-lookahead.HEIGHT_CLIP)  # clipped
    assert height[0, 2, 1] == pytest.approx(0.1)


def test_unknown_ground_is_neither_height_nor_hole():
    heights = torch.full((1, FX, FY), -0.5)
    known = torch.zeros(1, FX, FY, dtype=torch.bool)
    known[0, :, :12] = True               # only the right half of the strip was seen
    heights[0, :, :12] = 0.0
    heights[0, 0, 0] = float("-inf")      # a ray that found nothing
    height, holes, known_fraction = channels(lookahead.terrain_ahead(heights, known, torch.zeros(1)))
    assert (known_fraction[0, :, 3:] == 0).all() and (holes[0, :, 3:] == 0).all() and (height[0, :, 3:] == 0).all()
    assert known_fraction[0, 0, 0] == pytest.approx(15 / 16) and holes[0, 0, 0] == 0.0
    assert torch.isfinite(height).all()


def test_the_feature_reads_the_state_and_asks_for_it():
    n = 3
    state = RobotState(
        foot_pos=torch.zeros(n, 4, 3), foot_vel=torch.zeros(n, 4, 3), base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3), projected_gravity=torch.zeros(n, 3), contact=torch.ones(n, 4, dtype=torch.bool),
        gait_timing=torch.zeros(n, 4, 3), command=torch.zeros(n, 3), base_command=torch.zeros(n, 3),
    )
    with pytest.raises(ValueError, match="terrain_ahead"):
        state_vector(state, LOOKAHEAD_FEATURES)
    ahead = torch.rand(n, lookahead.FEATURE_DIM)
    state.terrain_ahead = ahead
    vector = state_vector(state, LOOKAHEAD_FEATURES)
    assert vector.shape == (n, feature_dim(LOOKAHEAD_FEATURES, 4))
    assert torch.equal(vector[:, feature_slices(LOOKAHEAD_FEATURES, 4)["terrain_ahead"]], ahead)
    # the gait timing sits where it did, so several footsteps per tick still find it
    assert feature_slices(LOOKAHEAD_FEATURES, 4)["gait_timing"] == feature_slices(LOOKAHEAD_FEATURES[:-1], 4)["gait_timing"]
    # robots are selected with it, and a state without it still indexes
    assert torch.equal(state[1:].terrain_ahead, ahead[1:])
    state.terrain_ahead = None
    assert state[1:].terrain_ahead is None
    assert "terrain_ahead" in FEATURES
