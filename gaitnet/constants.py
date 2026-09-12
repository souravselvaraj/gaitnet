from __future__ import annotations
import numpy as np
from gaitnet import get_logger
from dataclasses import dataclass, field

logger = get_logger()


@dataclass(frozen=True)
class _Robot:
    num_legs: int = 4
    """Number of legs on the robot"""


robot = _Robot()
"""Robot constants"""


@dataclass(frozen=True)
class _ContactNet:
    # 0.075 -> 0.074 (2026-09-12). PURE BOOKKEEPING, NO BEHAVIOUR CHANGE:
    # ContactNet is fully ablated in this configuration -- experiments.
    # random_footstep_sampling is True, so CostMapGenerator is never even
    # IMPORTED (footstep_candidate_sampler.py:7-8), and ablate_footstep_cost is
    # True, so the cost feature is zeroed. This constant therefore only feeds the
    # coverage assert below and footstep_scanner.sensor_grid_size.
    # It moved so that 5 * 0.074 = 0.370 matches the footstep scanner's new
    # 37 * 0.010 = 0.370. The alternative was to leave it at 0.075 and accept a
    # 37-cell scanner at 0.0101351, which would have pushed the candidate span to
    # +/-0.18243 -- OUTSIDE the +/-0.18 the policy was trained on.
    # IF YOU EVER SET random_footstep_sampling = False, revisit this: it becomes
    # the real ContactNet cost-map cell size again.
    grid_resolution: float = 0.074
    """Grid resolution used in contact-net training data generation"""
    grid_size: np.ndarray = field(default_factory=lambda: np.asarray((5, 5), dtype=int))
    """Grid size used in contact-net training data generation"""

    def __post_init__(self):
        self.grid_size.setflags(write=False)


contact_net = _ContactNet()
"""ContactNet constants"""


@dataclass(frozen=True)
class _GaitNet:
    num_footstep_options: int = 16
    """Number of footstep options to provide per leg"""
    cspace_dialation: int = 2
    """Number of times to apply max-pooling to the height scan to simulate c-space dialation"""
    upscale_costmap_noise: float = 0.35
    """Amount of noise (+/-) to add to the costmap during upscale"""
    valid_height_range: tuple[float, float] = (-0.5, 0)
    """(min, max) valid height range for footstep options.
    Note that these are negative of the values you would expect."""
    valid_swing_duration_range: tuple[float, float] = (0.1, 0.3)
    """(min, max) valid swing duration range for footstep options."""
    robot_state_dim: int = 25
    """Dimension of the robot state input to GaitNet (shared state)"""
    footstep_option_dim: int = 8
    """Dimension of the footstep option input to GaitNet (unique state)"""


gait_net = _GaitNet()
"""GaitNet constants"""


# The footstep scanner used to be DERIVED from contact_net by an integer scale
# (grid_resolution = 0.075/scale, grid_size = 5*scale). That derivation cannot
# express the 1.5x densification we want: it needs scale 7.5, which makes
# grid_size 37.5 -- not an integer, and multiplying the int grid_size array by a
# float silently turns it into float64, which then breaks every reshape that
# consumes it. The two numbers are literals now. (2026-09-12)
_footstep_scanner_scale: int = 5  # retained only for reference; no longer used


@dataclass(frozen=True)
class _FootstepScanner:
    # 0.015 -> 0.010 (2026-09-12), 1.5x finer, SAME REGION.
    # The candidate half-extent is (grid_size-1)/2 * grid_resolution, so
    # 25 @ 0.015 and 37 @ 0.010 both give exactly +/-0.180 m. That matters: the
    # frozen policy has NO grid dimension (gaitnet.py:108-118 derives the
    # candidate count from the observation length), so densifying is legal, but
    # dx/dy are continuous network inputs and must stay inside the trained range.
    # At 37 the new sample points INTERPOLATE the old ones over an identical span
    # with identical endpoints -- no extrapolation anywhere.
    # MUST MATCH the gait node: LeggedRobotGaitCommandNode.cpp kGrid and the
    # cspace_resolution param, and the fixed sizes in
    # gaitnet_interface/msg/StepCSpace{Leg,Row}.msg. All five move together.
    grid_resolution: float = 0.010
    """Grid resolution used in footstep scanner observations"""
    total_robot_features: int = None  # type: ignore set in __post_init__
    """Number of features in footstep scanner observations. Assumes one scanner per leg."""
    grid_size: np.ndarray = field(
        default_factory=lambda: np.asarray((37, 37), dtype=int)
    )
    """Grid size used in footstep scanner observations"""
    sensor_grid_size: np.ndarray = None  # type: ignore set in __post_init__
    """Grid size of the underlying raycaster sensors used for footstep scanner observations.
    This is larger than `grid_size` to account for c-space dialation."""

    def __post_init__(self):
        object.__setattr__(
            self,
            "total_robot_features",
            robot.num_legs * self.grid_size[0] * self.grid_size[1],
        )
        object.__setattr__(
            self,
            "sensor_grid_size",
            self.grid_size + 2 * gait_net.cspace_dialation,  # account for c-space dialation
        )
        self.grid_size.setflags(write=False)
        self.sensor_grid_size.setflags(write=False)


footstep_scanner = _FootstepScanner()
"""Footstep scanner constants"""

@dataclass(frozen=True)
class _Experiments:
    random_footstep_sampling: bool = True
    """If true, sample footstep options randomly instead of using contactnet's cost estimate for sampling."""
    ablate_footstep_cost: bool = True
    """If true, zero out footstep costs in footstep candidate sampler for ablation study."""
    ablate_swing_duration: bool = False
    """If true, set all swing durations to a constant value for ablation study."""
    constant_swing_duration: float = 0.247
    """Constant swing duration to use if ablate_swing_duration is True."""
    swing_duration_logging: bool = False
    """If true, log swing duration statistics."""
    contact_schedule_logging: bool = False
    """If true, log contact schedule statistics, only for first robot."""
    pga_debug_plots: bool = False
    """If true, generate matplotlib debug plots showing PGA optimization evolution."""


experiments = _Experiments()
"""Experiment constants"""


##### Checks

assert contact_net.grid_size.shape == (2,), "ContactNet grid size must be 2D"
assert footstep_scanner.grid_size.shape == (2,), "Footstep scanner grid size must be 2D"

# allclose, not ==. This compared two FLOATING-POINT PRODUCTS for exact equality
# and only ever passed by luck: 0.074*5 is 0.37000000000000005 while 0.010*37 is
# 0.37, which are not equal as floats even though the grids do cover the same
# area. (2026-09-12)
assert np.allclose(
    contact_net.grid_resolution * contact_net.grid_size,
    footstep_scanner.grid_resolution * footstep_scanner.grid_size,
), "ContactNet grid and footstep scanner grid must cover the same area"
assert footstep_scanner.grid_size.dtype.kind == "i", (
    "Footstep scanner grid size must stay an integer array; a float here silently "
    "breaks every reshape that consumes it"
)

if np.any(np.mod(contact_net.grid_size, footstep_scanner.grid_size) != 0):
    logger.warning(
        "ContactNet and Footstep scanner grid sizes should probably be multiples of each other"
    )

if experiments.random_footstep_sampling:
    assert experiments.ablate_footstep_cost, (
        "If random footstep sampling is enabled, "
        "then ablate_footstep_cost must also be enabled."
    )

NO_STEP = -1  # special value for no step
