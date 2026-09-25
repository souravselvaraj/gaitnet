"""Warm start a trained PPO checkpoint with more state features: new inputs get zero weights,
so the widened policy (and critic) compute exactly what the checkpoint did until training
teaches them to use the new inputs.

    python -m gaitnet_sim.scripts.widen_checkpoint --checkpoint <run>/model_<i>.pt --out widened.pt \\
        --old_features foot_pos base_lin_vel ... gait_timing --new_features ... gait_timing terrain_ahead

The new feature list must be the old one with features inserted (none removed or reordered).
Widened: the actor's first layer on the state (`network.shared_encoder.0.weight` of a
CandidateScorer) and the critic's first layer (`mlp.0.weight`), whose input starts with the
state group. The optimizer state is dropped (its moments no longer fit) and the iteration
reset to 0, so a run started with `--checkpoint widened.pt` trains from iteration 0 with a
fresh optimizer. Needs no simulator.
"""

from __future__ import annotations

import argparse
import sys

import torch

from gaitnet_core.features import feature_slices
from gaitnet_core.robot_spec import GO1

LAYERS = (("actor_state_dict", "network.shared_encoder.0.weight"), ("critic_state_dict", "mlp.0.weight"))


def insertions(old: list[str], new: list[str], num_legs: int) -> list[tuple[int, int]]:
    """(column, width) of each inserted feature in the new state vector, in order."""
    if [name for name in new if name in old] != list(old):
        raise ValueError(f"the new features {new} must be the old ones {old} with features inserted, in order")
    slices = feature_slices(new, num_legs)
    return [(slices[name].start, slices[name].stop - slices[name].start) for name in new if name not in old]


def widen(weight: torch.Tensor, inserts: list[tuple[int, int]]) -> torch.Tensor:
    """`weight` (out, in) with zero columns inserted at the given new-vector positions."""
    for column, width in inserts:  # in increasing order, so earlier inserts shift later ones already
        zeros = torch.zeros(weight.shape[0], width, dtype=weight.dtype)
        weight = torch.cat([weight[:, :column], zeros, weight[:, column:]], dim=1)
    return weight


def widen_checkpoint(saved: dict, old: list[str], new: list[str], num_legs: int = GO1.num_legs) -> dict:
    inserts = insertions(old, new, num_legs)
    out = {key: value for key, value in saved.items() if key != "optimizer_state_dict"}
    for group, name in LAYERS:
        if group in out and name in out[group]:
            state = dict(out[group])
            state[name] = widen(state[name], inserts)
            out[group] = state
    out["iter"] = 0
    out["infos"] = {"widened_from": {"iter": saved.get("iter"), "old_features": list(old), "new_features": list(new)}}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--old_features", nargs="+", required=True)
    parser.add_argument("--new_features", nargs="+", required=True)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    widened = widen_checkpoint(saved, args.old_features, args.new_features)
    torch.save(widened, args.out)
    shapes = {f"{group}/{name}": tuple(widened[group][name].shape) for group, name in LAYERS if name in widened.get(group, {})}
    print(f"wrote {args.out} from iteration {saved.get('iter')}: {shapes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
