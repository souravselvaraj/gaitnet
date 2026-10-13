"""The privileged GaitNet-T teacher ("GaitNet-T v2"): GaitNet-T reading the controller knobs as part
of its robot state and two terrain features as part of every candidate token.

The environment appends the privileged features after everything a deployable policy sees
(components/privileged.py), so the observation is

    [state 25 | candidates M x 8 | history T x D | knobs 13 | candidate terrain M x 2]

and `PrivilegedActor` rearranges it into the layout of an inner GaitnetTActor built with a 38-wide
state and 10-wide candidate tokens: [state 25 + knobs 13 | candidates M x (8 + 2) | history].

`PrivilegedActor.widen` starts from a GaitnetTActor (e.g. one converted from a GaitnetActor
checkpoint) and copies it with the new input columns of the shared encoder's and the candidate
encoder's first layers set to zero, so the teacher starts out computing exactly what it did.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import gaitnet.constants as const
from gaitnet.gaitnet.components.privileged import KNOB_DIM, TERRAIN_DIM
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor


def _linear_sizes(module: nn.Module) -> list[int]:
    return [m.out_features for m in module.modules() if isinstance(m, nn.Linear)]


class PrivilegedActor(nn.Module):
    def __init__(self, inner: GaitnetTActor, knob_dim: int = KNOB_DIM, terrain_dim: int = TERRAIN_DIM):
        super().__init__()
        self.inner = inner
        self.knob_dim = knob_dim
        self.terrain_dim = terrain_dim
        self.state_dim = inner.shared_state_dim - knob_dim
        self.option_dim = inner.option_dim - terrain_dim

    @property
    def history_len(self) -> int:
        return self.inner.history_len

    def rearrange(self, obs: torch.Tensor) -> torch.Tensor:
        """The environment's layout -> the inner GaitNet-T's layout."""
        n, s, m, d = obs.shape[0], self.state_dim, self.inner.num_options, self.option_dim
        h = self.inner.history_len * self.inner.history_dim
        state = obs[:, :s]
        candidates = obs[:, s : s + m * d].view(n, m, d)
        history = obs[:, s + m * d : s + m * d + h]
        tail = obs[:, s + m * d + h :]
        assert tail.shape[1] == self.knob_dim + m * self.terrain_dim, (
            f"expected {self.knob_dim + m * self.terrain_dim} privileged values, got {tail.shape[1]}"
        )
        knobs = tail[:, : self.knob_dim]
        terrain = tail[:, self.knob_dim :].view(n, m, self.terrain_dim)
        tokens = torch.cat([candidates, terrain], dim=-1).flatten(1)
        return torch.cat([state, knobs, tokens, history], dim=1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inner(self.rearrange(obs))

    @classmethod
    def widen(cls, actor: GaitnetTActor, knob_dim: int = KNOB_DIM, terrain_dim: int = TERRAIN_DIM) -> "PrivilegedActor":
        """A teacher that computes exactly what `actor` computes until the new inputs get weight."""
        shared = _linear_sizes(actor.shared_encoder)
        unique = _linear_sizes(actor.unique_encoder)
        trunk = [actor.token_layer[0].out_features] + (
            [] if isinstance(actor.trunk_rest, nn.Identity) else _linear_sizes(actor.trunk_rest)
        )
        history_sizes = _linear_sizes(actor.history_encoder) if actor.history_encoder is not None else (128, 64)
        inner = GaitnetTActor(
            actor.shared_state_dim + knob_dim,
            shared,
            actor.option_dim + terrain_dim,
            unique,
            trunk,
            num_options=actor.num_options,
            history_len=actor.history_len,
            history_dim=actor.history_dim,
            history_layer_sizes=history_sizes,
            mixer_layers=len(actor.mixer.attn),
            mixer_heads=actor.mixer.attn[0].num_heads,
        ).to(next(actor.parameters()).device)

        state = {k: v.clone() for k, v in actor.state_dict().items()}
        for key, extra in (("shared_encoder.0.weight", knob_dim), ("unique_encoder.0.weight", terrain_dim)):
            weight = state[key]
            state[key] = torch.cat([weight, weight.new_zeros(weight.shape[0], extra)], dim=1)
        inner.load_state_dict(state)
        return cls(inner, knob_dim, terrain_dim)


def is_privileged_state_dict(state: dict) -> bool:
    return any(k.startswith("actor.inner.") for k in state)


def is_gaitnet_t_state_dict(state: dict) -> bool:
    return any(k.startswith("actor.mixer.") or k.startswith("actor.inner.mixer.") for k in state)


assert const.gait_net.footstep_option_dim == 8
