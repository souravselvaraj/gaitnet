"""Footstep forward / inverse dynamics models, to run a trained GaitNet on a low-level controller it
was not trained with (FADA, Xie et al., arXiv 2606.28476: freeze the planner, adapt only an
inverse dynamics model on minutes of target data, with supervised learning and no reward).

GaitNet's action is a footstep command u = (dx, dy) relative to a leg's hip in the yaw frame, with
a swing duration T. What a controller does with it is the landing o: where the foot actually
touches down, in the yaw frame of the hip *when the step was commanded* (the frame the candidate
was chosen in). Controllers differ in that map; the GPU MPC variant every policy here was trained
on executes u against the hip at touchdown, so o lands about speed x T past u, and OCS2 anchors u
in the world when it is commanded. A policy trained against controller A has learnt to choose u
for what A does with it, so on controller B it should send

    u' = I_B(s, F_A(s, u))

with F_A the forward model of A (state, command -> landing) and I_B the inverse model of B (state,
desired landing -> command). Both are fitted to footstep events logged by eval_policy.py
--dump_events (with --perturb_xy, so commands are not only the policy's own), on the matching
controller.

State s is the newest tick of the policy history (components/policy_history.py, 26 features: base
linear / angular velocity, feet relative to their hips, scheduled contact, last touchdown errors),
nothing privileged. Each model is an MLP predicting a residual on its xy input, so an untrained
model is the identity.

    python -m gaitnet.gaitnet.idm --events <events.pt> [...] --direction forward|inverse --out <model.pt>
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn

STATE_DIM = 26
INPUT_DIM = STATE_DIM + 4 + 1 + 2


class FootstepModel(nn.Module):
    """xy_out = xy_in + mlp(state, leg one-hot, duration, xy_in), all (N, ...) tensors.

    direction "forward": xy_in is the command, xy_out the landing. "inverse": the other way round.
    """

    def __init__(self, direction: str, hidden: int = 256):
        super().__init__()
        assert direction in ("forward", "inverse")
        self.direction = direction
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, leg: torch.Tensor, duration: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """state (N, 26); leg (N,) slot 0-3 (FR, FL, RR, RL); duration (N,) s; xy (N, 2) m."""
        one_hot = torch.nn.functional.one_hot(leg.long(), 4).to(xy.dtype)
        x = torch.cat([state.to(xy.dtype), one_hot, duration.to(xy.dtype).unsqueeze(-1), xy], dim=-1)
        return xy + self.net(x)


class IdmAdapter:
    """u' = I_B(s, F_A(s, u)); without a forward model the intended landing is u itself."""

    def __init__(self, inverse: FootstepModel, forward: FootstepModel | None = None):
        assert inverse.direction == "inverse" and (forward is None or forward.direction == "forward")
        self.inverse = inverse.eval()
        self.forward = forward.eval() if forward is not None else None

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, leg: torch.Tensor, duration: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        intended = self.forward(state, leg, duration, xy) if self.forward is not None else xy
        return self.inverse(state, leg, duration, intended)

    @classmethod
    def load(cls, inverse_path: str, forward_path: str | None, device: str | torch.device) -> "IdmAdapter":
        return cls(load_model(inverse_path, device), load_model(forward_path, device) if forward_path else None)


def load_model(path: str, device: str | torch.device) -> FootstepModel:
    saved = torch.load(path, map_location=device, weights_only=False)
    model = FootstepModel(saved["direction"], saved.get("hidden", 256)).to(device)
    model.load_state_dict(saved["state_dict"])
    return model


def load_events(paths: list[str]) -> dict[str, torch.Tensor]:
    """Concatenate eval_policy.py --dump_events files: state, leg, duration, intent, command, landing, speed, voids."""
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def fit(events: dict[str, torch.Tensor], direction: str, epochs: int = 60, lr: float = 1e-3, batch: int = 4096,
        holdout: float = 0.1, seed: int = 0, device: str = "cpu") -> tuple[FootstepModel, dict[str, float]]:
    """Supervised fit on the events (hindsight relabelling for the inverse: the landing that a command
    produced is the input, the command the target). Returns the model and held-out errors (m)."""
    g = torch.Generator().manual_seed(seed)
    n = events["state"].shape[0]
    order = torch.randperm(n, generator=g)
    n_val = max(1, int(holdout * n))
    val, train = order[:n_val], order[n_val:]
    src, dst = ("command", "landing") if direction == "forward" else ("landing", "command")
    model = FootstepModel(direction).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ev = {k: v.to(device) for k, v in events.items()}

    def rms(idx: torch.Tensor) -> float:
        with torch.no_grad():
            pred = model(ev["state"][idx], ev["leg"][idx], ev["duration"][idx], ev[src][idx])
            return math.sqrt(((pred - ev[dst][idx]) ** 2).sum(-1).mean().item())

    identity = math.sqrt(((ev[src][val] - ev[dst][val]) ** 2).sum(-1).mean().item())
    for _ in range(epochs):
        perm = train[torch.randperm(len(train), generator=g)].to(device)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            pred = model(ev["state"][idx], ev["leg"][idx], ev["duration"][idx], ev[src][idx])
            loss = ((pred - ev[dst][idx]) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, {"holdout_rms": rms(val), "holdout_rms_identity": identity, "train_rms": rms(train),
                   "events": n, "holdout_events": n_val}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--direction", choices=["forward", "inverse"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    events = load_events(args.events)
    model, stats = fit(events, args.direction, epochs=args.epochs, device=args.device)
    torch.save({"direction": args.direction, "hidden": 256, "state_dict": model.state_dict(), "stats": stats,
                "events": args.events}, args.out)
    print(f"{args.direction} model -> {args.out}: " + ", ".join(f"{k} {v:.4g}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
