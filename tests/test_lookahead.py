"""Lookahead footstep selection (gaitnet/gaitnet/lookahead.py) on synthetic terrain, no simulator."""
import math

import torch

from gaitnet.gaitnet.lookahead import (
    EdgeMap,
    LookaheadCfg,
    TickState,
    candidate_legs,
    dead_end_deficit,
    masked_log_prior,
    plan,
    predicted_landing,
    reachable_margin,
)

RES = 0.01
M = 65
NOOP = M - 1


def flat_map(voids: list[tuple[float, float, float, float]] = (), size: float = 2.0) -> EdgeMap:
    """size x size m around the origin at 1 cm, with axis-aligned void boxes (x0, x1, y0, y1)."""
    n = int(round(size / RES)) + 1
    origin = (-size / 2, -size / 2)
    xs = origin[0] + torch.arange(n) * RES
    ys = origin[1] + torch.arange(n) * RES
    x, y = torch.meshgrid(xs, ys, indexing="ij")
    invalid = torch.zeros(n, n, dtype=torch.bool)
    for x0, x1, y0, y1 in voids:
        invalid |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return EdgeMap.from_invalid(invalid, origin, RES)


def tick(slots: list[tuple[int, float, float, float, float]], *, velocity=(0.2, 0.0), noop_logit=0.0,
         foot=None, swing_left=None) -> TickState:
    """One robot at the origin facing +x, hips at (+-0.19, -+0.05); `slots` are (leg, dx, dy, logit,
    duration) for the first slots, the rest padding."""
    hip = torch.tensor([[0.19, -0.05], [0.19, 0.05], [-0.19, -0.05], [-0.19, 0.05]])
    logits = torch.full((1, M), -5.0)
    leg = torch.full((1, M), -1, dtype=torch.long)
    xy = torch.zeros(1, M, 2)
    duration = torch.zeros(1, M)
    for i, (l, dx, dy, logit, t) in enumerate(slots):
        leg[0, i], xy[0, i, 0], xy[0, i, 1], logits[0, i], duration[0, i] = l, dx, dy, logit, t
    logits[0, NOOP] = noop_logit
    return TickState(
        logits=logits, duration=duration, cand_leg=leg, cand_xy=xy, hip_w=hip.unsqueeze(0),
        yaw=torch.zeros(1), velocity_w=torch.tensor([velocity]), command_w=torch.tensor([[0.2, 0.0]]),
        foot_w=(hip if foot is None else torch.tensor(foot)).unsqueeze(0).clone(),
        swing_left=torch.zeros(1, 4) if swing_left is None else torch.tensor([swing_left]),
    )


def test_edge_map_distances():
    edges = flat_map([(0.30, 0.425, -0.5, 0.5)])
    d = edges(torch.tensor([[0.20, 0.0], [0.28, 0.0], [0.35, 0.0], [0.55, 0.0], [5.0, 0.0], [-0.9, 0.0]]))
    assert abs(d[0] - 0.10) < 0.008 and abs(d[1] - 0.02) < 0.008
    assert d[2] == 0.0                       # inside the void
    assert abs(d[3] - 0.125) < 0.008         # beyond it
    assert d[4] == 0.0                       # off the map
    assert abs(d[5] - 0.15) < 1e-6           # clipped at the cap


def test_candidate_legs_from_the_one_hot_block():
    block = torch.zeros(1, 3, 8)
    block[0, 0, 2] = 1   # FL
    block[0, 1, 0] = 1   # padding
    block[0, 2, 0] = 1   # no-op
    assert candidate_legs(block).tolist() == [[1, -1, -1]]


def test_predicted_landing_adds_body_travel():
    s = tick([(0, 0.10, 0.02, 1.0, 0.2)])
    cfg = LookaheadCfg()
    land = predicted_landing(s, cfg)[0, 0]
    assert torch.allclose(land, torch.tensor([0.19 + 0.10 + 0.2 * 0.2, -0.05 + 0.02]), atol=1e-6)
    s.yaw[:] = math.pi / 2
    land = predicted_landing(s, LookaheadCfg(landing_gain=0.0))[0, 0]
    assert torch.allclose(land, torch.tensor([0.19 - 0.02, -0.05 + 0.10]), atol=1e-6)


def test_depth1_avoids_a_landing_shifted_into_a_void():
    # void ahead of FR from x = 0.30: commanding x = 0.27 (3 cm short of it) lands at 0.31 at 0.2 m/s x 0.2 s
    edges = flat_map([(0.30, 0.425, -0.5, 0.5)])
    s = tick([(0, 0.08, 0.0, 2.0, 0.2), (0, 0.0, 0.0, 1.0, 0.2)])
    assert plan(s, edges, LookaheadCfg(depth=0)).index.item() == 0
    tight = dict(m_veto=0.01, m_safe=0.02)
    assert plan(s, edges, LookaheadCfg(depth=1, landing_gain=0.0, **tight)).index.item() == 0  # no shift: looks fine
    p = plan(s, edges, LookaheadCfg(depth=1, **tight))
    assert p.index.item() == 1
    assert p.margin[0, 0] < 0.02 and p.margin[0, 1] > 0.05


def test_padding_and_noop_ticks():
    edges = flat_map()
    s = tick([(0, 0.05, 0.0, -1.0, 0.2)], noop_logit=1.0)
    s.logits[0, 5] = 10.0                     # a padding slot with a large logit is still padding
    logp, real = masked_log_prior(s)
    assert torch.isinf(logp[0, 5]) and not real[0, NOOP]
    for depth in (0, 1, 2):
        assert plan(s, edges, LookaheadCfg(depth=depth)).index.item() == NOOP


def test_all_vetoed_falls_back_to_noop_and_trust_region():
    edges = flat_map([(0.30, 0.425, -0.5, 0.5)])
    s = tick([(0, 0.09, 0.0, 2.0, 0.2), (0, 0.10, 0.0, 1.5, 0.2)])
    p = plan(s, edges, LookaheadCfg(depth=1))
    assert p.index.item() == NOOP and p.vetoed_all.item()
    # a safe candidate far outside the trust region is used only once everything trusted is vetoed
    s = tick([(0, 0.09, 0.0, 2.0, 0.2), (0, 0.0, 0.0, -6.0, 0.2)])
    assert plan(s, edges, LookaheadCfg(depth=1, trust=3.0)).index.item() == 1
    s = tick([(0, 0.0, 0.05, 2.0, 0.2), (0, 0.0, 0.0, -6.0, 0.2)])
    assert plan(s, edges, LookaheadCfg(depth=1, trust=3.0)).index.item() == 0


def test_same_leg_keeps_the_policy_leg():
    edges = flat_map([(0.30, 0.425, -0.5, 0.5)])
    # FR's best foothold lands in the void; another FR foothold and an FL one are safe
    s = tick([(0, 0.08, 0.0, 2.0, 0.2), (1, 0.0, 0.0, 1.9, 0.2), (0, 0.0, 0.0, 1.0, 0.2)])
    assert plan(s, edges, LookaheadCfg(depth=1)).index.item() == 1
    assert plan(s, edges, LookaheadCfg(depth=1, same_leg=True)).index.item() == 2
    # every FR foothold vetoed: hold rather than switch legs
    s = tick([(0, 0.08, 0.0, 2.0, 0.2), (1, 0.0, 0.0, 1.9, 0.2), (0, 0.09, 0.0, 1.0, 0.2)])
    assert plan(s, edges, LookaheadCfg(depth=1, same_leg=True)).index.item() == NOOP


def test_gate_free_lets_the_noop_compete():
    edges = flat_map([(0.30, 0.425, -0.5, 0.5)])
    s = tick([(0, 0.07, 0.0, 2.0, 0.2)], noop_logit=1.8)  # lands on the void's edge (x = 0.30), veto off
    cfg = LookaheadCfg(depth=1, m_veto=0.0, gate="free")
    assert plan(s, edges, cfg).index.item() == NOOP
    cfg = LookaheadCfg(depth=1, m_veto=0.0, gate="policy")
    assert plan(s, edges, cfg).index.item() == 0


def test_depth2_sees_a_trailing_leg_blocked_by_a_long_swing():
    """FL is in the air for 0.2 s; RR trails 0.12 m behind its hip, so at 0.2 m/s it must step within
    0.15 s. A 0.3 s FR swing keeps two legs in the air until 0.2 s: RR has no window, a dead end; a
    0.1 s FR swing frees a slot at 0.12 s, in time."""
    edges = flat_map()
    hip = [[0.19, -0.05], [0.19, 0.05], [-0.19, -0.05], [-0.19, 0.05]]
    foot = [hip[0], [0.30, 0.05], [-0.31, -0.05], hip[3]]
    s = tick([(0, 0.08, 0.0, 2.0, 0.3), (2, 0.10, 0.0, 1.0, 0.2), (0, 0.08, 0.0, 1.2, 0.1)],
             foot=foot, swing_left=[0.0, 0.2, 0.0, 0.0])
    s.logits[0, NOOP] = -10.0
    cfg = LookaheadCfg(depth=2)
    logp, real = masked_log_prior(s)
    reach = reachable_margin(s, edges, cfg)
    assert torch.allclose(reach[0, :, 1:], torch.full_like(reach[0, :, 1:], cfg.m_safe))  # flat: all safe
    deficit = dead_end_deficit(s, cfg, predicted_landing(s, cfg), reach, real)[0]
    assert deficit[0] == 1.0          # long FR swing: RR blocked past its reach
    assert deficit[1] == 0.0          # stepping RR itself
    assert deficit[2] == 0.0          # short FR swing
    assert plan(s, edges, LookaheadCfg(depth=1)).index.item() == 0
    assert plan(s, edges, cfg).index.item() == 2


def test_depth2_reachable_margin_sees_voids_in_the_reach_box():
    # the RR hip's whole reach box is void
    edges = flat_map([(-0.40, 0.0, -0.40, 0.40)])
    s = tick([(0, 0.05, 0.0, 1.0, 0.2)])
    reach = reachable_margin(s, edges, LookaheadCfg(depth=2, horizon=0.08))
    assert reach[0, 2].max() == 0.0          # RR (hip at -0.19, -0.05)
    assert reach[0, 0].min() >= 0.05 - 1e-6  # FR is clear


def test_cfg_parse():
    cfg = LookaheadCfg.parse("d2:gate=free,w_dead=4,trust=1.5,same_leg=1,landing=model")
    assert cfg.depth == 2 and cfg.gate == "free" and cfg.w_dead == 4.0 and cfg.trust == 1.5
    assert cfg.same_leg is True and cfg.landing == "model"
    assert LookaheadCfg.parse("argmax").depth == 0


def test_batched_shapes():
    edges = flat_map([(0.30, 0.425, -0.5, 0.5)])
    states = [tick([(l, 0.05 * l, 0.0, 1.0, 0.2)]) for l in range(4)]
    batch = TickState(**{k: torch.cat([getattr(s, k) for s in states]) for k in
                         ("logits", "duration", "cand_leg", "cand_xy", "hip_w", "yaw", "velocity_w", "command_w",
                          "foot_w", "swing_left")})
    p = plan(batch, edges, LookaheadCfg(depth=2))
    assert p.index.shape == (4,) and p.deficit.shape == (4, M) and p.margin.shape == (4, M)
