"""GaitNet-T: the GaitNet actor with attention across its candidate footsteps and a short history
of the robot's own motion. A drop-in for GaitnetActor (gaitnet.py) that keeps the action set and the
acyclic decision unchanged: every tick it scores the same candidate set (K footholds per leg in
FR, FL, RR, RL slot order, plus the no-op) and the policy picks one, or nothing.

What changes is what one candidate's score can depend on:
- GaitnetActor scores each candidate from (robot state, that candidate) alone, so a front-right
  foothold's value cannot depend on which rear-left footholds exist. Here the candidates attend to
  each other (a small transformer encoder over the candidate tokens), so the choice of leg and of
  foothold is made jointly.
- A window of the last `history_len` ticks of `history_dim` features (body velocity, foot
  positions, commanded vs achieved footholds, ...) is encoded and added to the shared context, so
  the policy can anticipate where the body is going instead of reacting once the feet trail it, and
  can tell which low-level controller it is driving.

Observation layout: [robot state (25) | candidates (num_options x 8) | history (history_len x
history_dim)]; with history_len 0 it is GaitnetActor's layout. Candidates are one-hot(no-op, FR, FL,
RR, RL), dx, dy, cost, as before; padding slots carry the no-op one-hot and are masked out of the
attention (only the last slot is the real no-op).

Warm start: `from_gaitnet_actor` copies a trained GaitnetActor's encoders, trunk and heads into the
matching places. The attention blocks' output projections and the history projection start at
zero, so before any fine-tuning GaitNet-T computes exactly what the old actor did.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

import gaitnet.constants as const
from gaitnet.gaitnet.gaitnet import GaitnetActor, make_mlp


class CandidateMixer(nn.Module):
    """Pre-norm transformer encoder layers over the candidate tokens, residual, with the output
    projections zero-initialised so a fresh mixer is the identity."""

    def __init__(self, dim: int, num_layers: int, num_heads: int, ff_mult: int = 2):
        super().__init__()
        self.norms1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.attn = nn.ModuleList(nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_layers))
        self.norms2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim)) for _ in range(num_layers)
        )
        for attn, ff in zip(self.attn, self.ff):
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)
            nn.init.zeros_(ff[-1].weight)
            nn.init.zeros_(ff[-1].bias)

    def forward(self, tokens: torch.Tensor, ignore: torch.Tensor) -> torch.Tensor:
        """tokens (N, M, dim); ignore (N, M) True for slots no other token should attend to."""
        for norm1, attn, norm2, ff in zip(self.norms1, self.attn, self.norms2, self.ff):
            h = norm1(tokens)
            tokens = tokens + attn(h, h, h, key_padding_mask=ignore, need_weights=False)[0]
            tokens = tokens + ff(norm2(tokens))
        return tokens


class GaitnetTActor(nn.Module):
    def __init__(
        self,
        shared_state_dim: int,
        shared_layer_sizes: Sequence[int],
        unique_state_dim: int,
        unique_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
        num_options: int = const.gait_net.num_footstep_options * const.robot.num_legs + 1,
        history_len: int = 0,
        history_dim: int = 0,
        history_layer_sizes: Sequence[int] = (128, 64),
        mixer_layers: int = 2,
        mixer_heads: int = 4,
    ):
        super().__init__()
        self.shared_state_dim = shared_state_dim
        self.option_dim = unique_state_dim
        self.num_options = num_options
        self.history_len = history_len
        self.history_dim = history_dim

        self.shared_encoder = make_mlp(shared_state_dim, shared_layer_sizes[:-1], shared_layer_sizes[-1], output_activation=nn.ReLU)
        self.unique_encoder = make_mlp(unique_state_dim, unique_layer_sizes[:-1], unique_layer_sizes[-1], output_activation=nn.ReLU)
        self.unique_embedding_size = unique_layer_sizes[-1]
        self.no_op_embedding = nn.Parameter(torch.randn(unique_layer_sizes[-1]))

        if history_len > 0:
            self.history_encoder = make_mlp(history_len * history_dim, history_layer_sizes[:-1], history_layer_sizes[-1], output_activation=nn.ReLU)
            self.history_projection = nn.Linear(history_layer_sizes[-1], shared_layer_sizes[-1])
            nn.init.zeros_(self.history_projection.weight)
            nn.init.zeros_(self.history_projection.bias)
        else:
            self.history_encoder = None
            self.history_projection = None

        # the trunk's first layer turns (context, candidate) into a token; the mixer works on those
        # tokens; the rest of the trunk and the heads read each token out, as in GaitnetActor
        token_dim = trunk_layer_sizes[0]
        self.token_layer = nn.Sequential(nn.Linear(shared_layer_sizes[-1] + unique_layer_sizes[-1], token_dim), nn.ReLU())
        self.mixer = CandidateMixer(token_dim, mixer_layers, mixer_heads)
        self.trunk_rest = make_mlp(token_dim, trunk_layer_sizes[1:-1], trunk_layer_sizes[-1], output_activation=nn.ReLU) \
            if len(trunk_layer_sizes) > 1 else nn.Identity()
        self.value_head = make_mlp(trunk_layer_sizes[-1], [], 1)
        self.duration_head = make_mlp(trunk_layer_sizes[-1], [], 1)

    @property
    def obs_dim(self) -> int:
        return self.shared_state_dim + self.num_options * self.option_dim + self.history_len * self.history_dim

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(state (N, 25), candidates (N, M, 8), history (N, T, D) or None) from the flat layout."""
        assert obs.shape[1] == self.obs_dim, f"expected {self.obs_dim} observation values, got {obs.shape[1]}"
        n, s, m, d = obs.shape[0], self.shared_state_dim, self.num_options, self.option_dim
        state = obs[:, :s]
        candidates = obs[:, s : s + m * d].view(n, m, d)
        history = obs[:, s + m * d :].view(n, self.history_len, self.history_dim) if self.history_len else None
        return state, candidates, history

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logits (N, M) over the candidates and their swing durations (N, M; 0 for no-op slots)."""
        state, candidates, history = self.split(obs)
        n, m = candidates.shape[:2]
        context = self.shared_encoder(state)
        if self.history_encoder is not None:
            context = context + self.history_projection(self.history_encoder(history.reshape(n, -1)))

        no_op = candidates[:, :, 0] == 1
        emb = torch.where(no_op.unsqueeze(-1), self.no_op_embedding.expand(n, m, -1), self.unique_encoder(candidates))
        tokens = self.token_layer(torch.cat([context.unsqueeze(1).expand(-1, m, -1), emb], dim=-1))
        # padding slots (no-op one-hot anywhere but the last slot) take no part in the attention
        ignore = no_op.clone()
        ignore[:, -1] = False
        tokens = self.mixer(tokens, ignore)
        out = self.trunk_rest(tokens)

        logits = self.value_head(out).squeeze(-1)
        min_dur, max_dur = const.gait_net.valid_swing_duration_range
        duration = torch.sigmoid(self.duration_head(out).squeeze(-1)) * (max_dur - min_dur) + min_dur
        duration = torch.where(no_op, torch.zeros_like(duration), duration)
        return logits, duration

    @classmethod
    def from_gaitnet_actor(cls, old: GaitnetActor, **kwargs) -> "GaitnetTActor":
        """A GaitNet-T that starts out computing exactly what `old` computes."""
        shared = [m.out_features for m in old.shared_encoder if isinstance(m, nn.Linear)]
        unique = [m.out_features for m in old.unique_encoder if isinstance(m, nn.Linear)]
        trunk = [m.out_features for m in old.trunk if isinstance(m, nn.Linear)]
        state_dim = next(m for m in old.shared_encoder if isinstance(m, nn.Linear)).in_features
        option_dim = next(m for m in old.unique_encoder if isinstance(m, nn.Linear)).in_features
        new = cls(state_dim, shared, option_dim, unique, trunk, **kwargs)
        new.shared_encoder.load_state_dict(old.shared_encoder.state_dict())
        new.unique_encoder.load_state_dict(old.unique_encoder.state_dict())
        new.no_op_embedding.data.copy_(old.no_op_embedding.data)
        old_trunk = [m for m in old.trunk if isinstance(m, nn.Linear)]
        new.token_layer[0].load_state_dict(old_trunk[0].state_dict())
        rest = [m for m in new.trunk_rest if isinstance(m, nn.Linear)] if not isinstance(new.trunk_rest, nn.Identity) else []
        for dst, src in zip(rest, old_trunk[1:]):
            dst.load_state_dict(src.state_dict())
        new.value_head.load_state_dict(old.value_head.state_dict())
        new.duration_head.load_state_dict(old.duration_head.state_dict())
        return new
