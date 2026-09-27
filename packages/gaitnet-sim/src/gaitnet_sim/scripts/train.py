"""Train a GaitNet task with RSL-RL, through Isaac Lab's training entry point.

    docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.train \\
        --task GaitNet-Holes --num_envs 1024

All of Isaac Lab's train arguments work (`--max_iterations`, `--seed`, `--checkpoint`, ...),
plus `--tf32`, which lets float32 matrix products run on the tensor cores (TF32: 10-bit
mantissa, several times the fp32 rate on Ampere and later); worth it for wide scorers, whose
update is most of an iteration.

Checkpoints also carry the terrain curriculum (`gaitnet_sim.rl.curriculum_state`): a resume starts
its robots on the saved level distribution instead of re-climbing from the importer's initial
spread. `--reset_curriculum` turns that off; `--init_terrain_levels LO:HI` spreads the robots
uniformly over rows LO..HI when the checkpoint has no saved levels (one written before this).

Presets and Hydra overrides of the env and agent cfgs work too, e.g.
`presets=spatial,privileged agent.algorithm.entropy_coef=0.01`; see
packages/gaitnet-sim/README.md. Runs are written to
`logs/rsl_rl/<experiment_name>/<timestamp>` and tracked in MLflow.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _diff_this_repo_only() -> None:
    """Have RSL-RL store this checkout's git diff instead of its own install's.

    It logs the diff of rsl_rl and Isaac Lab, neither of which is a git checkout in the image
    (hence "Could not find git repository ... Skipping"), and never our code.
    """
    from rsl_rl.utils.logger import Logger

    repo = str(Path(__file__).resolve())
    store = Logger._store_code_state

    def patched(self) -> list[str]:
        # the runner appends Isaac Lab's train script after the Logger exists, so replace at use
        self.git_status_repos = [repo]
        return store(self)

    Logger._store_code_state = patched


def _allow_tf32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _carry_curriculum(restore: bool, initial: tuple[int, int] | None) -> None:
    """Store the terrain levels in every checkpoint, and restore them (or `initial`) on load."""
    from rsl_rl.runners import OnPolicyRunner

    from gaitnet_sim.rl import curriculum_state as cs

    # wrap the originals, so calling this again replaces the hooks instead of stacking them
    save, load = getattr(OnPolicyRunner, "_unhooked", (OnPolicyRunner.save, OnPolicyRunner.load))
    OnPolicyRunner._unhooked = (save, load)

    def save_with_levels(self, path, infos=None):
        levels = cs.saved_levels(self.env)
        if levels is not None:
            infos = {**(infos or {}), cs.INFOS_KEY: levels}
        return save(self, path, infos)

    def load_with_levels(self, path, *args, **kwargs):
        infos = load(self, path, *args, **kwargs)
        terrain = cs.terrain_of(self.env)
        if terrain is None:
            return infos
        num_envs, rows = terrain.terrain_levels.shape[0], terrain.max_terrain_level
        saved = infos.get(cs.INFOS_KEY) if isinstance(infos, dict) else None
        if restore and saved is not None:
            levels, source = cs.resample(saved, num_envs, rows), f"the checkpoint's {saved.numel()} saved levels"
        elif initial is not None:
            levels, source = cs.initial_levels(*initial, num_envs, rows), f"rows {initial[0]}..{initial[1]}"
        else:
            return infos
        cs.apply_levels(self.env, levels)
        print(f"[INFO]: terrain curriculum restored from {source}: mean level {levels.float().mean():.2f} of {rows}")
        return infos

    OnPolicyRunner.save = save_with_levels
    OnPolicyRunner.load = load_with_levels


def _pop_option(argv: list[str], name: str) -> str | None:
    """Remove `name VALUE` from argv and return VALUE (None if absent)."""
    if name not in argv:
        return None
    i = argv.index(name)
    if i + 1 >= len(argv):
        raise SystemExit(f"{name} needs a value")
    value = argv[i + 1]
    del argv[i : i + 2]
    return value


def main(argv: list[str] | None = None) -> None:
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

    from gaitnet_sim.rl.curriculum_state import parse_range

    _diff_this_repo_only()

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--tf32" in argv:
        argv.remove("--tf32")
        _allow_tf32()
    restore = "--reset_curriculum" not in argv
    if not restore:
        argv.remove("--reset_curriculum")
    initial = _pop_option(argv, "--init_terrain_levels")
    _carry_curriculum(restore, parse_range(initial) if initial is not None else None)

    run(["--external_callback", "gaitnet_sim.tasks.register", *argv])


if __name__ == "__main__":
    main()
