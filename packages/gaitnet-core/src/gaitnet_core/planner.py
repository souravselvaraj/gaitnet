"""One planning tick: valid footholds -> candidates -> scores -> footsteps (or none).

A tick starts at most `FootholdRules.max_steps_per_tick` footsteps, chosen one after another
(`gaitnet_core.rounds`); with the default of one it is a single footstep or none."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.candidates import Candidates
from gaitnet_core.eligibility import step_eligible
from gaitnet_core.features import state_vector
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.networks.candidate_scorer import CandidateScorer
from gaitnet_core.robot_spec import RobotSpec
from gaitnet_core.rounds import StateEditor, TickDistribution
from gaitnet_core.samplers import CandidateSampler
from gaitnet_core.selection import Scores, Selection, leg_marginals
from gaitnet_core.state import Observation
from gaitnet_core.terrain import inner_heights, median_filter, valid_footholds


@dataclass
class FootholdRules:
    """Which cells and legs are allowed. Part of the policy's contract, saved with it."""

    step_threshold: float = 0.02
    """Height difference between neighbouring cells that counts as an edge (m)."""
    edge_margin: int = 2
    """Cells within this many cells of an edge are invalid."""
    min_stance_after_step: int = 2
    """A leg may only lift off if this many legs stay in stance."""
    max_steps_per_tick: int = 1
    """Footsteps a robot may start in one planning tick, chosen one after another."""
    min_stance_time: float = 0.0
    """A leg may only lift off after this long in scheduled stance (s): a controller needs a
    few solves with the foot loaded before it can shift weight off it again."""
    min_foot_separation: float = 0.0
    """Footholds closer than this to another foot, horizontally, are invalid (m); the legs
    would collide."""
    midline_margin: float | None = None
    """If set, a leg's footholds must stay at least this far on its own side of the base's
    centre line (m): legs may not cross under the body. None allows crossing."""
    max_reach: float | None = None
    """If set, footholds farther than this from the hip (m, 3D, at the terrain height) are
    invalid: the grid's corners lie beyond the leg's length."""
    median_window: int = 1
    """The terrain the rules and the candidates' heights read is median filtered over this
    many cells square first (`gaitnet_core.terrain.median_filter`); 1 for none."""

    def heights(self, observation: Observation) -> torch.Tensor:
        """(N, L, *patch_size) the terrain the rules read: the patch, median filtered."""
        return median_filter(observation.terrain.heights, self.median_window)

    def valid(self, observation: Observation, spec: RobotSpec) -> torch.Tensor:
        """(N, L, *grid.size) cells each leg may step to this tick."""
        cells = valid_footholds(
            self.heights(observation),
            spec,
            observation.terrain.grid,
            step_threshold=self.step_threshold,
            edge_margin=self.edge_margin,
        )
        legs = step_eligible(observation.state.gait_timing, self.min_stance_after_step, self.min_stance_time)
        return cells & legs.unsqueeze(-1).unsqueeze(-1) & self.kinematic(observation, spec)

    def kinematic(self, observation: Observation, spec: RobotSpec) -> torch.Tensor:
        """(N, L, *grid.size) cells the leg can reach without meeting another leg: the foot
        separation, midline and reach rules (all True with their defaults)."""
        grid = observation.terrain.grid
        heights = inner_heights(self.heights(observation), grid)
        n, legs = heights.shape[:2]
        ok = torch.ones_like(heights, dtype=torch.bool)
        cells = grid.cell_centers(device=heights.device)  # (*size, 2), hip frame
        hips = torch.tensor(spec.hip_offsets, device=heights.device, dtype=cells.dtype)[:, :2]  # (L, 2)
        # footholds in the base's yaw frame, where foot_pos is: (L, *size, 2)
        footholds = hips.view(legs, 1, 1, 2) + cells
        if self.min_foot_separation > 0:
            feet = observation.state.foot_pos[..., :2]  # (N, L, 2)
            gap = (footholds.unsqueeze(0).unsqueeze(-2) - feet.view(n, 1, 1, 1, legs, 2)).norm(dim=-1)
            others = ~torch.eye(legs, dtype=torch.bool, device=heights.device).view(1, legs, 1, 1, legs)
            ok &= ~((gap < self.min_foot_separation) & others).any(dim=-1)
        if self.midline_margin is not None:
            side = torch.sign(hips[:, 1]).view(legs, 1, 1)
            ok &= (side * footholds[..., 1] >= self.midline_margin).unsqueeze(0)
        if self.max_reach is not None:
            reach = torch.sqrt(cells.square().sum(dim=-1) + heights.square())
            ok &= reach <= self.max_reach
        return ok


@dataclass
class PlanResult:
    candidates: Candidates
    scores: Scores
    selection: Selection
    leg_marginals: torch.Tensor
    """(N, L) log step probability of each leg, comparable to `scores.noop_logit`."""
    is_step: torch.Tensor
    """(N,) bool"""
    leg: torch.Tensor
    """(N,) long, meaningless where not stepping"""
    target: torch.Tensor
    """(N, 3) foothold in the leg's hip yaw frame (m)"""
    more: list["PlanResult"] = field(default_factory=list)
    """The tick's later rounds, in order; empty with one footstep per tick. A round's
    `is_step` is False once the robot has chosen the no-op."""

    def footstep_command(self) -> FootstepCommand:
        """This round's footsteps."""
        return FootstepCommand(
            active=self.is_step, leg=self.leg, target=self.target, duration=self.selection.duration
        )

    def footstep_commands(self) -> list[FootstepCommand]:
        """Every round's footsteps, in order."""
        return [self.footstep_command()] + [plan.footstep_command() for plan in self.more]

    def env_action(self, nudge: Nudge | None = None) -> EnvAction:
        plans = [self] + self.more
        leg = torch.stack([torch.where(p.is_step, p.leg, torch.full_like(p.leg, NO_STEP_LEG)) for p in plans], 1)
        delta = nudge.command_delta if nudge is not None else torch.zeros_like(self.target)
        return EnvAction(
            choice_index=torch.stack([p.selection.index for p in plans], 1),
            duration=torch.stack([p.selection.duration for p in plans], 1),
            leg=leg,
            target=torch.stack([p.target for p in plans], 1),
            nudge=delta,
        )


def plan_from_scores(scores: Scores, candidates: Candidates, selection: Selection) -> PlanResult:
    is_step, leg, target = candidates.gather(selection.index)
    return PlanResult(
        candidates=candidates,
        scores=scores,
        selection=selection,
        leg_marginals=leg_marginals(scores, candidates),
        is_step=is_step,
        leg=leg,
        target=target,
    )


class FootstepPlanner:
    def __init__(
        self,
        network: CandidateScorer,
        spec: RobotSpec,
        grid: FootholdGrid,
        features: tuple[str, ...],
        sampler: CandidateSampler,
        rules: FootholdRules | None = None,
        duration_std: float = 0.05,
        max_rows_per_forward: int = 65536,
    ):
        """
        Args:
            features: robot state features the network was trained on
            duration_std: swing duration noise when planning stochastically
            max_rows_per_forward: robots are scored in chunks so (robots * candidates)
                stays under this, bounding peak activation memory with dense sampling
        """
        self.network = network
        self.spec = spec
        self.grid = grid
        self.features = tuple(features)
        self.sampler = sampler
        self.rules = rules or FootholdRules()
        self.duration_std = duration_std
        self.max_rows_per_forward = max_rows_per_forward

    def sample(self, observation: Observation, generator: torch.Generator | None = None) -> Candidates:
        valid = self.rules.valid(observation, self.spec)
        heights = inner_heights(self.rules.heights(observation), self.grid)
        return self.sampler.sample(valid, self.grid, heights=heights, generator=generator)

    def score(self, observation: Observation, candidates: Candidates) -> Scores:
        state = state_vector(observation.state, self.features)
        return self._score_state(state, candidates, observation.terrain.heights)

    def _score_state(self, state: torch.Tensor, candidates: Candidates, heights: torch.Tensor) -> Scores:
        per_robot = candidates.num_legs * candidates.per_leg + 1
        chunk = max(1, self.max_rows_per_forward // per_robot)
        parts = [
            self.network(state[i : i + chunk], candidates[i : i + chunk], heights[i : i + chunk])
            for i in range(0, state.shape[0], chunk)
        ]
        return Scores(
            step_logits=torch.cat([p.step_logits for p in parts]),
            noop_logit=torch.cat([p.noop_logit for p in parts]),
            duration=torch.cat([p.duration for p in parts]),
        )

    def tick_distribution(self, observation: Observation, candidates: Candidates) -> TickDistribution:
        """The policy over this tick's footsteps (one round or several, see the rules)."""
        state = state_vector(observation.state, self.features)
        heights = observation.terrain.heights
        fixed = getattr(self.network, "fixed_duration", None) is not None
        std = None if fixed else torch.tensor(self.duration_std, device=state.device)
        rounds = self.rules.max_steps_per_tick
        return TickDistribution(
            lambda s: self._score_state(s, candidates, heights),
            state,
            candidates,
            std,
            rounds=rounds,
            min_stance_after_step=self.rules.min_stance_after_step,
            editor=StateEditor(self.features, self.spec.num_legs) if rounds > 1 else None,
            duration_range=self.spec.swing_duration_range,
        )

    @torch.no_grad()
    def plan(
        self,
        observation: Observation,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> PlanResult:
        candidates = self.sample(observation, generator)
        tick = self.tick_distribution(observation, candidates)
        if deterministic:
            tick.deterministic()
        else:
            tick.sample()
        plans = [plan_from_scores(r.scores, r.candidates, r.selection) for r in tick.path]
        first = plans[0]
        first.more = plans[1:]
        return first
