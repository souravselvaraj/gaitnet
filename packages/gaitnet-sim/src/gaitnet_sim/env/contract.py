"""The parts of the planner's contract that the environment has to agree on: which robot,
the foothold grid the terrain is scanned on, and the foothold rules. A trained policy's
bundle records the same values, see `gaitnet_core.bundle`.

Terms read these from `env.cfg.gaitnet` at run time rather than through their own
params, so one override changes every term. The foothold scanners' ray patterns are built
from the default grid when the scene cfg is created; a different grid needs scanners to
match (see `gaitnet_sim.env.scene.foothold_scanner_cfg`), and `RobotIO` checks that they do.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.planner import FootholdRules
from gaitnet_core.robot_spec import ROBOTS, RobotSpec


@configclass
class GaitNetCfg:
    robot: str = "go1"
    """Key of `gaitnet_core.robot_spec.ROBOTS`."""

    grid_resolution: float = 0.015
    """Foothold cell size (m)."""
    grid_size: tuple[int, int] = (25, 25)
    """Foothold cells per leg along (x, y)."""
    grid_border: int = 3
    """Extra terrain cells scanned on each side, context for the edge margin."""

    step_threshold: float = 0.02
    """Height difference between neighbouring cells that counts as an edge (m)."""
    edge_margin: int = 2
    """Cells within this many cells of an edge are not valid footholds."""
    min_stance_after_step: int = 2
    """A leg may only lift off if this many legs stay in stance."""
    max_steps_per_tick: int = 1
    """Footsteps a robot may start in one planning tick, chosen one after another
    (`gaitnet_core.rounds`). It sets the action's length, so the actor follows it."""
    min_stance_time: float = 0.08
    """A leg may only lift off after this long in scheduled stance (s). The deployed NMPC
    needs a few solves with a foot loaded; its own gaits keep 0.1 s at every transition."""
    min_foot_separation: float = 0.06
    """Footholds closer than this to another foot are invalid (m). Training runs without
    self-collision; the deployed NMPC keeps feet 5 cm apart."""
    midline_margin: float | None = 0.02
    """A leg's footholds stay this far on its own side of the base's centre line (m)."""
    max_reach: float | None = 0.40
    """Footholds farther than this from the hip are invalid (m); the leg is about 0.43 m."""
    median_window: int = 3
    """The rules read the terrain median filtered over 3 x 3 cells, which removes an elevation
    map's cell-to-cell speckle that would otherwise read as edges everywhere."""

    def robot_spec(self) -> RobotSpec:
        return ROBOTS[self.robot]

    def foothold_grid(self) -> FootholdGrid:
        return FootholdGrid(
            resolution=self.grid_resolution, size=tuple(self.grid_size), border=self.grid_border
        )

    def foothold_rules(self) -> FootholdRules:
        return FootholdRules(
            step_threshold=self.step_threshold,
            edge_margin=self.edge_margin,
            min_stance_after_step=self.min_stance_after_step,
            max_steps_per_tick=self.max_steps_per_tick,
            min_stance_time=self.min_stance_time,
            min_foot_separation=self.min_foot_separation,
            midline_margin=self.midline_margin,
            max_reach=self.max_reach,
            median_window=self.median_window,
        )
