"""Footstep forward / inverse models (gaitnet/gaitnet/idm.py) on a synthetic controller."""
import torch

from gaitnet.gaitnet.idm import STATE_DIM, FootstepModel, IdmAdapter, fit


def synthetic(n: int, lead: float, seed: int = 0) -> dict[str, torch.Tensor]:
    """A controller that lands each foot `lead` x (forward speed x duration) past the command."""
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(n, STATE_DIM, generator=g) * 0.1
    state[:, 0] = torch.rand(n, generator=g) * 0.3                       # forward speed (m/s)
    duration = 0.1 + 0.2 * torch.rand(n, generator=g)
    command = (torch.rand(n, 2, generator=g) - 0.5) * 0.36
    landing = command.clone()
    landing[:, 0] += lead * state[:, 0] * duration
    leg = torch.randint(0, 4, (n,), generator=g)
    return {"state": state, "leg": leg, "duration": duration, "command": command, "landing": landing,
            "intent": command, "speed": state[:, 0], "voids": torch.zeros(n)}


def test_untrained_model_is_the_identity():
    m = FootstepModel("inverse")
    ev = synthetic(16, 1.0)
    assert torch.equal(m(ev["state"], ev["leg"], ev["duration"], ev["landing"]), ev["landing"])


def test_forward_and_inverse_learn_the_body_travel_offset():
    ev = synthetic(20000, 1.0)
    fwd, s_f = fit(ev, "forward", epochs=40)
    inv, s_i = fit(ev, "inverse", epochs=40)
    assert s_f["holdout_rms"] < 0.25 * s_f["holdout_rms_identity"]
    assert s_i["holdout_rms"] < 0.25 * s_i["holdout_rms_identity"]


def test_adapter_reproduces_the_source_controller_on_the_target():
    """Policy trained on A (lands at u + travel); deployed on B (lands at u). I_B(F_A(u)) sent to B
    must land where A would have put u."""
    a, b = synthetic(20000, 1.0, seed=1), synthetic(20000, 0.0, seed=2)
    fwd_a, _ = fit(a, "forward", epochs=40)
    inv_b, _ = fit(b, "inverse", epochs=40)
    adapter = IdmAdapter(inv_b, fwd_a)
    test = synthetic(2000, 1.0, seed=3)
    sent = adapter(test["state"], test["leg"], test["duration"], test["command"])
    landed_on_b = sent                                                    # B lands where it is sent
    err = (landed_on_b - test["landing"]).norm(dim=-1).mean()             # vs where A lands u
    naive = (test["command"] - test["landing"]).norm(dim=-1).mean()
    assert err < 0.25 * naive
