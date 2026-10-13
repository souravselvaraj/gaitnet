"""Lookahead footstep selection on top of a trained GaitNet policy: instead of taking the policy's
argmax for this tick only, re-score its candidates with where the foot will actually land and
what the step leaves the other legs, and take the first action of the best short sequence.

GaitNet (gaitnet.py, gaitnet_t.py, teacher.py) picks one of M = 4 x 16 + 1 slots every 0.04 s:
a candidate foothold (dx, dy) relative to a leg's hip in the yaw frame, or the no-op (last slot).
Two things it cannot see make that greedy choice fall into voids at 0.2 m/s:

- the training controller (sim2real/gpu_mpc.py, variant "gaitnet") executes a foothold against
  the hip at touchdown, so the foot lands about body velocity x swing time past the chosen spot
  (+1.9 cm at 0.1 m/s, +3.7 cm at 0.2 m/s), while candidates are only kept 2 cm (c-space
  dilation) from a void;
- a step taken now can leave another leg, whose foot is trailing out of reach, with no safe
  foothold in the window where it can still step (two legs swing at most).

Inputs per tick are a `TickState` (all batched over N robots) and an `EdgeMap`, the distance from
any world xy to the nearest void. In simulation the edge map comes from the true terrain
(privileged); on the robot it would come from the elevation map. Nothing else is privileged.

Scoring (`plan`), for each candidate c the policy offers:

    score(c) = prior_weight * log pi(c)                       the policy as a prior
             - w_edge * relu(m_safe - margin(landing(c))) / m_safe   edge margin of the predicted landing
             + w_progress * progress(c)                       (landing ahead of the hip, along the command)
             - w_dead * sum_legs deficit_j(c)                 depth 2: dead ends of the other legs

with landing(c) = hip + R(yaw) (dx, dy) + landing_gain * v * T (or a fitted idm.FootstepModel), a
hard veto below m_veto, and a trust region: only candidates within `trust` nats of the policy's
best candidate are re-ranked (wider only if all of those are vetoed). `gate="policy"` steps exactly
when the policy's argmax steps (falling back to the no-op only if every candidate is vetoed);
`gate="free"` lets the no-op compete on the same score.

Depth 2 (`dead_end_deficit`): the body moves at its measured velocity whatever is chosen, so a
first step c changes the next step only through which leg is in the air until when and where c's
foot ends up. For each leg j after c: the window of ticks in which j may start a step (it is on the
ground, at most one other leg is swinging) before its foot trails `reach_back` behind its hip; the
best edge margin it can reach in that window, over a grid of footholds in its +-`box` reach box
around its future hip (precomputed once per tick, `reachable_margin`); deficit_j = 1 if the window
is empty, else relu(m_safe - best) / m_safe, and 0 for a leg that need not step within `horizon`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import torch

NUM_LEGS = 4


class EdgeMap:
    """Distance (m) from a world xy to the nearest void, on a regular grid: cell (i, j) is at
    x = origin[0] + i * resolution, y = origin[1] + j * resolution. Bilinear between cells; 0
    (a void) outside the map."""

    def __init__(self, distance: torch.Tensor, origin: tuple[float, float], resolution: float):
        self.distance = distance
        """(H, W) float, edge distance (m) at each grid point, H along x, W along y; 0 in a void."""
        self.origin = (float(origin[0]), float(origin[1]))
        self.resolution = float(resolution)

    @classmethod
    def from_invalid(cls, invalid: torch.Tensor, origin: tuple[float, float], resolution: float,
                     cap: float = 0.15) -> "EdgeMap":
        """From an (H, W) bool void mask: Euclidean distance from each solid point to the nearest
        void point, less half a cell (the void's boundary is between the two points), clipped to
        `cap`."""
        mask = invalid.bool()
        try:
            from scipy.ndimage import distance_transform_edt

            cells = torch.from_numpy(distance_transform_edt(~mask.cpu().numpy())).float()
        except ImportError:  # pragma: no cover - small maps only
            cells = _edt_brute_force(mask.cpu())
        distance = (cells * resolution - 0.5 * resolution).clamp(0.0, cap)
        distance[mask.cpu()] = 0.0
        return cls(distance.to(invalid.device), origin, resolution)

    def to(self, device: torch.device | str) -> "EdgeMap":
        return EdgeMap(self.distance.to(device), self.origin, self.resolution)

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """(...) edge distance (m) at world points xy (..., 2)."""
        h, w = self.distance.shape
        gx = (xy[..., 0] - self.origin[0]) / self.resolution
        gy = (xy[..., 1] - self.origin[1]) / self.resolution
        outside = (gx < 0) | (gx > h - 1) | (gy < 0) | (gy > w - 1) | ~torch.isfinite(gx + gy)
        gx = gx.nan_to_num(0.0).clamp(0, h - 1)
        gy = gy.nan_to_num(0.0).clamp(0, w - 1)
        i0 = gx.floor().long().clamp(max=h - 2)
        j0 = gy.floor().long().clamp(max=w - 2)
        fx = (gx - i0).to(self.distance.dtype)
        fy = (gy - j0).to(self.distance.dtype)
        flat = self.distance.view(-1)
        base = i0 * w + j0
        v = (flat[base] * (1 - fx) * (1 - fy) + flat[base + w] * fx * (1 - fy)
             + flat[base + 1] * (1 - fx) * fy + flat[base + w + 1] * fx * fy)
        return v.masked_fill(outside, 0.0)


def _edt_brute_force(mask: torch.Tensor) -> torch.Tensor:
    """(H, W) distance in cells from each cell to the nearest True cell (inf if none)."""
    h, w = mask.shape
    grid = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).float().view(-1, 2)
    voids = grid[mask.view(-1)]
    if voids.shape[0] == 0:
        return torch.full((h, w), float("inf"))
    return torch.cdist(grid, voids).min(dim=1).values.view(h, w)


@dataclass
class TickState:
    """Everything the planner reads on one policy tick, batched over N robots. Legs in GaitNet's
    slot order FR, FL, RR, RL; world frame unless stated."""

    logits: torch.Tensor
    """(N, M) the policy's logits over its M slots (last slot the no-op)."""
    duration: torch.Tensor
    """(N, M) the policy's swing duration (s) for each slot."""
    cand_leg: torch.Tensor
    """(N, M) long, the slot's leg 0-3, -1 for the no-op and for padding slots."""
    cand_xy: torch.Tensor
    """(N, M, 2) candidate foothold (dx, dy) relative to its leg's hip, yaw frame (m)."""
    hip_w: torch.Tensor
    """(N, 4, 2) hip xy (m)."""
    yaw: torch.Tensor
    """(N,) base yaw (rad)."""
    velocity_w: torch.Tensor
    """(N, 2) measured base linear velocity xy (m/s)."""
    command_w: torch.Tensor
    """(N, 2) commanded base linear velocity xy (m/s)."""
    foot_w: torch.Tensor
    """(N, 4, 2) where each foot is (stance) or is predicted to touch down (swing) (m)."""
    swing_left: torch.Tensor
    """(N, 4) time (s) until each swinging leg touches down; 0 for a leg in stance."""
    model_state: torch.Tensor | None = None
    """(N, 26) newest policy-history tick (policy_history.tick_features), for a fitted landing model."""


@dataclass
class LookaheadCfg:
    depth: int = 1
    """0: the policy's argmax; 1: re-score with predicted landings; 2: + the other legs' dead ends."""
    gate: str = "policy"
    """"policy": step on exactly the ticks the policy's argmax steps; "free": the no-op competes."""
    trust: float = 3.0
    """Re-rank only candidates whose log-probability is within this many nats of the best one's."""
    same_leg: bool = False
    """Keep the leg the policy's argmax steps and only re-choose its foothold (gate "policy")."""
    prior_weight: float = 1.0
    landing_gain: float = 1.0
    """Landing = command + gain * v * T (gpu_mpc variant "gaitnet"; 0 for one that compensates)."""
    landing: str = "analytic"
    """"analytic": command + gain * v * T; "model": the fitted forward model passed to `plan` (first step)."""
    m_veto: float = 0.02
    """(m) a candidate whose predicted landing is closer than this to a void is never taken."""
    m_safe: float = 0.05
    """(m) margins below this are penalised, linearly down to w_edge at 0."""
    w_edge: float = 4.0
    w_progress: float = 0.0
    """per m of landing ahead of the hip along the command."""
    w_dead: float = 2.0
    reach_back: float = 0.15
    """(m) a stance foot further than this behind its hip (along the command) must have stepped."""
    horizon: float = 0.4
    tick: float = 0.04
    box: float = 0.18
    """(m) half-width of a leg's reachable foothold box around its hip (the candidate grid's)."""
    box_step: float = 0.02
    nominal_duration: float = 0.2
    """(s) swing duration assumed for a leg's next step (depth 2)."""
    forward_min: float = 0.0
    """(m) a next step only counts if it lands at least this far ahead of the hip (depth 2)."""
    command_margin: float = 0.02
    """(m) edge distance a commanded point needs, the sampler's c-space dilation (depth 2)."""

    @classmethod
    def parse(cls, spec: str) -> "LookaheadCfg":
        """"d2:gate=free,w_dead=4" -> depth 2 with those fields; "argmax" is depth 0."""
        name, _, rest = spec.partition(":")
        depth = {"argmax": 0, "d0": 0, "d1": 1, "d2": 2}[name]
        types = {f.name: f.type for f in fields(cls)}
        kwargs: dict = {"depth": depth}
        for item in filter(None, rest.split(",")):
            key, _, value = item.partition("=")
            kind = types[key] if isinstance(types[key], str) else types[key].__name__
            kwargs[key] = {"str": str, "int": int, "bool": lambda v: v.lower() in ("1", "true", "yes")}.get(kind, float)(value)
        return cls(**kwargs)


@dataclass
class Plan:
    index: torch.Tensor
    """(N,) long, the chosen slot."""
    policy_index: torch.Tensor
    """(N,) long, the policy's own argmax."""
    landing_w: torch.Tensor | None = None
    """(N, M, 2) predicted touchdown of every candidate (m)."""
    margin: torch.Tensor | None = None
    """(N, M) edge distance of each predicted touchdown (m); inf for the no-op and padding."""
    deficit: torch.Tensor | None = None
    """(N, M) depth-2 dead-end deficit summed over legs (0 = every leg keeps a safe step)."""
    vetoed_all: torch.Tensor | None = None
    """(N,) bool, the policy wanted to step but every candidate was vetoed."""
    index_d1: torch.Tensor | None = None
    """(N,) long, what the same scores would choose without the depth-2 term."""


LandingModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""idm.FootstepModel's call: (state (B, 26), leg (B,), duration (B,), command xy (B, 2)) -> landing
xy (B, 2), both relative to the hip in the command-time yaw frame."""


def rotate(yaw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """xy (..., 2) from the yaw frame into the world; yaw broadcasts against xy[..., 0]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], dim=-1)


def masked_log_prior(state: TickState) -> tuple[torch.Tensor, torch.Tensor]:
    """(log-probabilities (N, M) over real candidates and the no-op, real-candidate mask (N, M)),
    the distribution eval_policy.act takes its argmax of."""
    real = state.cand_leg >= 0
    allowed = real.clone()
    allowed[:, -1] = True
    logp = torch.log_softmax(state.logits.masked_fill(~allowed, float("-inf")), dim=-1)
    real[:, -1] = False
    return logp, real


def predicted_landing(state: TickState, cfg: LookaheadCfg, landing_model: LandingModel | None = None) -> torch.Tensor:
    """(N, M, 2) where each candidate's foot is predicted to touch down, world xy (m)."""
    n, m = state.cand_leg.shape
    leg = state.cand_leg.clamp(min=0)
    hip = state.hip_w.gather(1, leg.unsqueeze(-1).expand(n, m, 2))
    if landing_model is None:
        offset = rotate(state.yaw.unsqueeze(-1), state.cand_xy)
        offset = offset + cfg.landing_gain * state.velocity_w.unsqueeze(1) * state.duration.unsqueeze(-1)
    else:
        assert state.model_state is not None, "a fitted landing model needs TickState.model_state"
        rel = landing_model(state.model_state.unsqueeze(1).expand(n, m, -1).reshape(n * m, -1), leg.reshape(-1),
                            state.duration.reshape(-1).clamp_min(1e-3), state.cand_xy.reshape(n * m, 2)).view(n, m, 2)
        offset = rotate(state.yaw.unsqueeze(-1), rel)
    return hip + offset


def _direction(state: TickState) -> torch.Tensor:
    """(N, 2) unit commanded direction in the world (the base's heading where the command is ~0)."""
    heading = torch.stack([torch.cos(state.yaw), torch.sin(state.yaw)], dim=-1)
    speed = state.command_w.norm(dim=-1, keepdim=True)
    return torch.where(speed > 1e-3, state.command_w / speed.clamp_min(1e-6), heading)


def foothold_grid(cfg: LookaheadCfg, device: torch.device | str) -> torch.Tensor:
    """(G, 2) yaw-frame footholds (relative to the hip) on a box_step grid over the +-box reach box."""
    k = int(round(cfg.box / cfg.box_step))
    axis = torch.arange(-k, k + 1, device=device, dtype=torch.float32) * cfg.box_step
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).view(-1, 2)
    return grid


def reachable_margin(state: TickState, edges: EdgeMap, cfg: LookaheadCfg) -> torch.Tensor:
    """(N, 4, K + 1) for each leg and tick t_k = k * tick (k = 0..K, K = horizon / tick): the best
    edge margin (capped at m_safe) of a step that leg could start at t_k - a commanded point with
    command_margin in its reach box around the hip then, landing forward of forward_min."""
    k_max = int(round(cfg.horizon / cfg.tick))
    times = torch.arange(k_max + 1, device=state.hip_w.device, dtype=state.hip_w.dtype) * cfg.tick
    grid = foothold_grid(cfg, state.hip_w.device).to(state.hip_w.dtype)                  # (G, 2)
    forward = _direction(state)                                                          # (N, 2)
    # yaw-frame grid along the command direction: keep footholds whose landing is ahead enough
    grid_w = rotate(state.yaw.view(-1, 1), grid.unsqueeze(0))                            # (N, G, 2)
    shift = cfg.landing_gain * state.velocity_w * cfg.nominal_duration                  # (N, 2)
    ahead = ((grid_w + shift.unsqueeze(1)) * forward.unsqueeze(1)).sum(-1) >= cfg.forward_min   # (N, G)
    hips = state.hip_w.unsqueeze(2) + state.velocity_w.view(-1, 1, 1, 2) * times.view(1, 1, -1, 1)  # (N, 4, K+1, 2)
    command = hips.unsqueeze(3) + grid_w.view(-1, 1, 1, grid.shape[0], 2)                # (N, 4, K+1, G, 2)
    ok = (edges(command) >= cfg.command_margin) & ahead.view(-1, 1, 1, grid.shape[0])
    landing = edges(command + shift.view(-1, 1, 1, 1, 2)).clamp(max=cfg.m_safe)
    return torch.where(ok, landing, torch.zeros_like(landing)).amax(dim=-1)


def dead_end_deficit(state: TickState, cfg: LookaheadCfg, landing_w: torch.Tensor, reachable: torch.Tensor,
                     real: torch.Tensor) -> torch.Tensor:
    """(N, M) summed over legs, for each first action (candidate or no-op): 0 when every leg that has
    to step within the horizon keeps a window in which a step with margin m_safe is reachable."""
    n, m = state.cand_leg.shape
    k_max = reachable.shape[-1] - 1
    dev, dt = state.hip_w.device, state.hip_w.dtype
    times = torch.arange(k_max + 1, device=dev, dtype=dt) * cfg.tick                       # (K+1,)
    legs = torch.arange(NUM_LEGS, device=dev)
    moved = real.unsqueeze(-1) & (state.cand_leg.unsqueeze(-1) == legs)                   # (N, M, 4)
    swing_end = torch.where(moved, state.duration.unsqueeze(-1), state.swing_left.unsqueeze(1))  # (N, M, 4)
    foot = torch.where(moved.unsqueeze(-1), landing_w.unsqueeze(2), state.foot_w.unsqueeze(1))  # (N, M, 4, 2)

    swinging = swing_end.unsqueeze(-1) > times + 1e-6                                     # (N, M, 4, K+1)
    others = swinging.sum(dim=2, keepdim=True) - swinging.long()                         # other legs in the air
    can_start = ~swinging & (others <= 1)
    can_start[..., 0] = False                                                            # tick 0 is this one

    forward = _direction(state)                                                          # (N, 2)
    hips = state.hip_w.unsqueeze(2) + state.velocity_w.view(-1, 1, 1, 2) * times.view(1, 1, -1, 1)  # (N, 4, K+1, 2)
    trail = ((hips.unsqueeze(1) - foot.unsqueeze(3)) * forward.view(-1, 1, 1, 1, 2)).sum(-1)  # (N, M, 4, K+1)
    in_reach = trail <= cfg.reach_back
    window = can_start & in_reach
    best = torch.where(window, reachable.unsqueeze(1), torch.full_like(trail, -1.0)).amax(dim=-1)  # (N, M, 4)
    must_step = ~in_reach[..., -1]
    deficit = torch.where(best < 0, torch.ones_like(best), (cfg.m_safe - best).clamp(min=0) / cfg.m_safe)
    deficit = torch.where(must_step, deficit, torch.zeros_like(deficit))
    return deficit.sum(dim=-1)


@torch.no_grad()
def plan(state: TickState, edges: EdgeMap | None, cfg: LookaheadCfg, landing_model: LandingModel | None = None) -> Plan:
    """The slot to take this tick (see the module docstring)."""
    logp, real = masked_log_prior(state)
    n, m = logp.shape
    noop = m - 1
    policy_index = logp.argmax(dim=-1)
    if cfg.depth <= 0:
        return Plan(index=policy_index, policy_index=policy_index)
    assert edges is not None

    landing_w = predicted_landing(state, cfg, landing_model if cfg.landing == "model" else None)
    margin = torch.where(real, edges(landing_w), torch.full_like(logp, float("inf")))
    score = cfg.prior_weight * logp - cfg.w_edge * (cfg.m_safe - margin).clamp(min=0) / cfg.m_safe
    if cfg.w_progress:
        leg = state.cand_leg.clamp(min=0)
        hip = state.hip_w.gather(1, leg.unsqueeze(-1).expand(n, m, 2))
        progress = ((landing_w - hip) * _direction(state).unsqueeze(1)).sum(-1)
        score = score + cfg.w_progress * torch.where(real, progress, torch.zeros_like(progress))
    deficit = None
    if cfg.depth >= 2:
        reachable = reachable_margin(state, edges, cfg)
        deficit = dead_end_deficit(state, cfg, landing_w, reachable, real)

    vetoed = real & (margin < cfg.m_veto)
    eligible = real
    if cfg.same_leg:
        policy_leg = state.cand_leg.gather(1, policy_index.unsqueeze(-1))       # -1 when the policy holds
        eligible = real & ((state.cand_leg == policy_leg) | (policy_leg < 0))
    best_logp = logp.masked_fill(~eligible, float("-inf")).amax(dim=-1, keepdim=True)
    in_trust = eligible & (logp >= best_logp - cfg.trust) & ~vetoed
    fallback = eligible & ~vetoed
    use = torch.where(in_trust.any(dim=-1, keepdim=True), in_trust, fallback)
    any_ok = use.any(dim=-1)
    wants = policy_index != noop

    def select(score: torch.Tensor) -> torch.Tensor:
        choice = score.masked_fill(~use, float("-inf")).argmax(dim=-1)
        noop_t = torch.full_like(choice, noop)
        if cfg.gate == "policy":
            return torch.where(wants & any_ok, choice, noop_t)
        if cfg.gate == "free":
            chosen = score.gather(1, choice.unsqueeze(-1)).squeeze(-1)
            return torch.where(any_ok & (chosen > score[:, noop]), choice, noop_t)
        raise ValueError(f"unknown gate {cfg.gate!r}")

    index = select(score if deficit is None else score - cfg.w_dead * deficit)
    index_d1 = select(score) if deficit is not None else index
    return Plan(index=index, policy_index=policy_index, landing_w=landing_w, margin=margin, deficit=deficit,
                vetoed_all=wants & ~any_ok, index_d1=index_d1)


def candidate_legs(candidates: torch.Tensor) -> torch.Tensor:
    """(N, M) slot leg 0-3, or -1 for the no-op and padding, from the (N, M, 8) candidate block
    (one-hot no-op/FR/FL/RR/RL, dx, dy, cost)."""
    leg = candidates[..., 1:5].argmax(dim=-1)
    return torch.where(candidates[..., 0] == 1, torch.full_like(leg, -1), leg)


def _synthetic_state(n: int, device: str, generator: torch.Generator) -> TickState:
    """A random tick for benchmarking: 65 slots, robots walking +x at 0.2 m/s around the origin."""
    m = 65
    leg = torch.randint(0, NUM_LEGS, (n, m), generator=generator)
    leg[:, -1] = -1
    hip = torch.tensor([[0.19, -0.05], [0.19, 0.05], [-0.19, -0.05], [-0.19, 0.05]]).expand(n, 4, 2).clone()
    state = TickState(
        logits=torch.randn(n, m, generator=generator), duration=0.1 + 0.2 * torch.rand(n, m, generator=generator),
        cand_leg=leg, cand_xy=(torch.rand(n, m, 2, generator=generator) - 0.5) * 0.36, hip_w=hip,
        yaw=torch.zeros(n), velocity_w=torch.tensor([[0.2, 0.0]]).expand(n, 2).clone(),
        command_w=torch.tensor([[0.2, 0.0]]).expand(n, 2).clone(), foot_w=hip.clone(), swing_left=torch.zeros(n, 4),
    )
    return TickState(**{f.name: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for f in fields(TickState) for v in [getattr(state, f.name)]})


def benchmark(sizes: dict[str, tuple[int, ...]] | None = None, specs: tuple[str, ...] = ("argmax", "d1", "d2"),
              threads: int = 1) -> list[str]:
    """Planner time per tick on random inputs over a 3 m x 3 m, 1 cm edge map with 30 % voids in
    12.5 cm cells (what the robot's elevation map would give)."""
    import time

    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(0)
    cells = torch.rand(24, 24, generator=g) < 0.3
    invalid = cells.repeat_interleave(13, 0).repeat_interleave(13, 1)[:301, :301]
    lines = []
    sizes = sizes or {"cpu": (1,), "cuda": (1, 1500)}
    for device, counts in sizes.items():
        if device == "cuda" and not torch.cuda.is_available():
            continue
        edges = EdgeMap.from_invalid(invalid, (-1.5, -1.5), 0.01).to(device)
        for n in counts:
            state = _synthetic_state(n, device, g)
            for spec in specs:
                cfg = LookaheadCfg.parse(spec)
                for _ in range(3):
                    plan(state, edges, cfg)
                reps = 20
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(reps):
                    plan(state, edges, cfg)
                if device == "cuda":
                    torch.cuda.synchronize()
                ms = 1e3 * (time.perf_counter() - start) / reps
                lines.append(f"{device}{'' if device == 'cuda' else f' ({threads} thread)'} N={n} {spec}: {ms:.2f} ms/tick")
    return lines


if __name__ == "__main__":
    for line in benchmark():
        print(line)

