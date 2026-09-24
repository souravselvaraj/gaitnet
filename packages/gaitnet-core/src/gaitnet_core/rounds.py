"""Several footsteps in one planning tick, chosen one after another ("rounds").

Round 1 is the one-step policy of `gaitnet_core.selection`. Every later round sees the state
as it will be once the earlier rounds' legs have lifted off: their scheduled gait timing
reads as a swing of the chosen duration that has just begun. Eligibility is re-applied to
that state, so a round may only step a leg still in scheduled stance, and only while the
foothold rules' minimum number of legs stays in stance. Choosing the no-op ends the tick:
every later round is a no-op too, and not part of the action. All rounds score the same
candidate set, sampled once per tick, with the legs that are no longer eligible masked out.

The tick's policy is the product of the rounds' conditionals, so its log-probability is the
sum over the rounds that were taken. PPO recomputes it by replaying the stored choices: round
r's distribution depends only on the choices of rounds before it, which the stored action
holds. The entropy bonus is the sum of the rounds' categorical entropies along the same path.

This works on flat state vectors, so the simulator's actor (handed the vector) and the
deployment planner (which builds it) share it; `StateEditor` knows where the gait timing sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from gaitnet_core.candidates import Candidates
from gaitnet_core.eligibility import step_eligible
from gaitnet_core.features import feature_slices
from gaitnet_core.selection import FootstepDistribution, Scores, Selection


class StateEditor:
    """Reads and rewrites the gait timing inside a flat state vector (see
    `gaitnet_core.features`, where it is quantity-major: phase, remaining, since touchdown)."""

    def __init__(self, features: tuple[str, ...] | list[str], num_legs: int):
        slices = feature_slices(features, num_legs)
        if "gait_timing" not in slices:
            raise ValueError("choosing several footsteps per tick needs the 'gait_timing' state feature")
        self.slice = slices["gait_timing"]
        self.num_legs = num_legs
        self.state_dim = max(s.stop for s in slices.values())

    def gait_timing(self, state: torch.Tensor) -> torch.Tensor:
        """(N, L, 3) swing phase, remaining swing (s), time since touchdown (s, clipped)."""
        return state[:, self.slice].reshape(state.shape[0], 3, self.num_legs).transpose(1, 2)

    def with_swing(self, state: torch.Tensor, leg: torch.Tensor, duration: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
        """`state` with `leg` (N,) starting a swing of `duration` (N,) s for the robots `where`
        (N,): swing phase 0, the whole duration remaining, no time since touchdown."""
        timing = self.gait_timing(state)
        starting = torch.nn.functional.one_hot(leg.clamp(min=0), self.num_legs).bool() & where.unsqueeze(-1)
        swing = torch.stack(
            [torch.zeros_like(timing[..., 0]), duration.to(state.dtype).unsqueeze(-1).expand_as(timing[..., 1]), torch.zeros_like(timing[..., 2])],
            dim=-1,
        )
        timing = torch.where(starting.unsqueeze(-1), swing, timing)
        block = timing.transpose(1, 2).reshape(state.shape[0], -1)
        return torch.cat([state[:, : self.slice.start], block, state[:, self.slice.stop :]], dim=-1)


@dataclass
class TickSelection:
    """The choices of every round of a tick."""

    index: torch.Tensor
    """(N, R) flat candidate index per round, the no-op index where the robot took no step
    (in the round that chose the no-op and every round after it)."""
    duration: torch.Tensor
    """(N, R) swing duration (s), 0 where no step."""

    @property
    def rounds(self) -> int:
        return self.index.shape[1]

    def round(self, r: int) -> Selection:
        return Selection(index=self.index[:, r], duration=self.duration[:, r])


@dataclass
class Round:
    """One round of a tick, as it was taken."""

    distribution: FootstepDistribution
    candidates: Candidates
    """The tick's candidates with the legs no longer eligible masked out."""
    scores: Scores
    selection: Selection
    """(N,) this round's choice, the no-op where the robot had already stopped."""
    active: torch.Tensor
    """(N,) bool, the robot was still choosing in this round (every earlier round stepped)."""


ScoreFn = Callable[[torch.Tensor], Scores]
"""(N, S) state vector -> the network's scores for this tick's candidates."""


class TickDistribution:
    """The stochastic policy over a tick's rounds, see the module docstring."""

    def __init__(
        self,
        score: ScoreFn,
        state: torch.Tensor,
        candidates: Candidates,
        duration_std: torch.Tensor | None,
        rounds: int = 1,
        min_stance_after_step: int = 2,
        editor: StateEditor | None = None,
        duration_range: tuple[float, float] = (0.1, 0.3),
    ):
        """
        Args:
            score: scores a state vector on this tick's candidates (the network, closed over
                the candidates and terrain)
            duration_std: as for `FootstepDistribution`; None for a fixed-duration network
            rounds: footsteps a robot may start this tick
            min_stance_after_step: the foothold rules' stance count, re-applied every round
            editor: required for more than one round
            duration_range: what the executed duration is clamped to, and so what the next
                round's state shows
        """
        if rounds > 1 and editor is None:
            raise ValueError("more than one round needs a StateEditor to update the state between rounds")
        self.score = score
        self.state = state
        self.candidates = candidates
        self.duration_std = duration_std
        self.rounds = rounds
        self.min_stance_after_step = min_stance_after_step
        self.editor = editor
        self.duration_range = duration_range
        if editor is not None:
            # samplers already leave ineligible legs empty; this makes it hold for any candidates
            eligible = step_eligible(editor.gait_timing(state), min_stance_after_step)
            candidates = Candidates(candidates.xyz, candidates.valid & eligible.unsqueeze(-1), candidates.log_q)
            self.candidates = candidates
        scores = score(state)
        self.first = FootstepDistribution(scores, candidates, duration_std)
        self._first_scores = scores
        self.path: list[Round] = []
        """The rounds of the latest `sample`, `deterministic` or `log_prob` walk."""
        self._log_prob: torch.Tensor | None = None
        self._entropy: torch.Tensor | None = None
        self._sampled: TickSelection | None = None
        self._sampled_log_prob: torch.Tensor | None = None

    def _round(self, r: int, state: torch.Tensor) -> tuple[FootstepDistribution, Candidates, Scores]:
        if r == 0:
            return self.first, self.candidates, self._first_scores
        eligible = step_eligible(self.editor.gait_timing(state), self.min_stance_after_step)
        candidates = Candidates(self.candidates.xyz, self.candidates.valid & eligible.unsqueeze(-1), self.candidates.log_q)
        scores = self.score(state)
        return FootstepDistribution(scores, candidates, self.duration_std), candidates, scores

    def _walk(self, choose: Callable[[int, FootstepDistribution], Selection]) -> TickSelection:
        state = self.state
        n = state.shape[0]
        noop = self.candidates.noop_index
        active = torch.ones(n, dtype=torch.bool, device=state.device)
        log_prob = torch.zeros(n, device=state.device)
        entropy = torch.zeros(n, device=state.device)
        self.path = []
        indices, durations = [], []
        for r in range(self.rounds):
            distribution, candidates, scores = self._round(r, state)
            chosen = choose(r, distribution)
            index = torch.where(active, chosen.index, torch.full_like(chosen.index, noop))
            duration = torch.where(active, chosen.duration, torch.zeros_like(chosen.duration))
            selection = Selection(index=index, duration=duration)
            # a stopped robot's round is the no-op, always valid, so these stay finite
            log_prob = log_prob + torch.where(active, distribution.log_prob(selection), torch.zeros_like(log_prob))
            entropy = entropy + torch.where(active, distribution.entropy(), torch.zeros_like(entropy))
            self.path.append(Round(distribution, candidates, scores, selection, active))
            indices.append(index)
            durations.append(duration)
            stepped = active & (index < noop)
            if r + 1 < self.rounds:
                leg = torch.where(stepped, index, torch.zeros_like(index)) // self.candidates.per_leg
                executed = duration.detach().clamp(*self.duration_range)
                state = self.editor.with_swing(state, leg, executed, stepped)
            active = stepped
        self._log_prob, self._entropy = log_prob, entropy
        return TickSelection(index=torch.stack(indices, dim=1), duration=torch.stack(durations, dim=1))

    def sample(self) -> TickSelection:
        selection = self._walk(lambda r, distribution: distribution.sample())
        self._sampled, self._sampled_log_prob = selection, self._log_prob
        return selection

    def deterministic(self) -> TickSelection:
        return self._walk(lambda r, distribution: distribution.deterministic())

    def log_prob(self, selection: TickSelection) -> torch.Tensor:
        """(N,) log-probability of a tick's choices, replaying them round by round."""
        sampled = self._sampled
        if (
            sampled is not None
            and sampled.index.shape == selection.index.shape
            and torch.equal(sampled.index, selection.index.to(sampled.index.device))
            and torch.equal(sampled.duration, selection.duration.to(sampled.duration.dtype))
        ):
            return self._sampled_log_prob
        self._walk(lambda r, distribution: selection.round(r))
        return self._log_prob

    def entropy(self) -> torch.Tensor:
        """(N,) the rounds' categorical entropies summed along the latest walk (the first
        round's alone before any walk)."""
        return self._entropy if self._entropy is not None else self.first.entropy()
