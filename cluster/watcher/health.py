"""Health check of the watched training runs (runs.txt: "<job id> <run name>" per line).

Prints one line per run and a final "FLAGS:" line; the flags that should wake Claude are listed in
RULES.md. Cheap: squeue, the Slurm log's tail and a TensorBoard read.

    python health.py [--window 20]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGS = os.path.join(ROOT, "cluster", "logs")
RUNS = os.path.join(ROOT, "training", "gaitnet", "runs")

TAGS = {
    "tip": "Episode_Termination/bad_orientation",
    "void": "Episode_Termination/foot_below_ground",
    "walk": "Curriculum/walked_fraction",
    "lvl": "Curriculum/terrain_levels",
    "rew": "Train/mean_reward",
    "ops": "Custom_Metrics/ops_per_step",
    "vloss": "Loss/value_function",
    "surr": "Loss/surrogate",
}


def job_states() -> dict[str, str]:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %T"],
                         capture_output=True, text=True).stdout
    return dict(line.split() for line in out.splitlines() if line.strip())


def run_dir(name: str) -> str | None:
    paths = [p for p in glob.glob(f"{RUNS}/gaitnet_2026*_{name}") if os.path.basename(p).split("_", 3)[-1] == name]
    return sorted(paths)[-1] if paths else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stall_min", type=float, default=20.0)
    parser.add_argument("--tip", type=float, default=0.30)
    args = parser.parse_args()

    states = job_states()
    flags = []
    runs = [line.split() for line in open(os.path.join(HERE, "runs.txt")) if line.strip() and not line.startswith("#")]
    now = time.time()
    for job, name in runs:
        logs = glob.glob(f"{LOGS}/*-{job}.out")
        log = logs[0] if logs else None
        text = open(log, errors="replace").read() if log else ""
        state = states.get(job, "GONE")
        done = "GAITNET-TRAIN-DONE" in text
        crashed = "Traceback" in text
        stale_min = (now - os.path.getmtime(log)) / 60 if log else float("nan")
        line = f"{name:28s} job {job} {state:8s}"
        if state == "GONE":
            flags.append(f"{'DONE' if done else 'CRASH'}:{name}")
        elif crashed:
            flags.append(f"TRACEBACK:{name}")
        elif state == "RUNNING" and stale_min > args.stall_min:
            flags.append(f"STALL:{name}({stale_min:.0f}min)")
        d = run_dir(name)
        if d:
            ea = EventAccumulator(d, size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags()["scalars"]
            means = {}
            for short, tag in TAGS.items():
                if tag in tags:
                    ev = ea.Scalars(tag)
                    vals = [e.value for e in ev[-args.window:]]
                    means[short] = sum(vals) / len(vals)
                    if any(math.isnan(v) or math.isinf(v) for v in vals):
                        flags.append(f"NAN:{name}:{short}")
            it = ea.Scalars(TAGS["rew"])[-1].step if TAGS["rew"] in tags else -1
            minutes = (ea.Scalars(TAGS["rew"])[-1].wall_time - ea.Scalars(TAGS["rew"])[0].wall_time) / 60 if TAGS["rew"] in tags else 0
            line += f" it {it} ({minutes:.0f} min) " + " ".join(f"{k} {v:.3f}" for k, v in means.items())
            # a fresh policy falls over constantly at first; only flag runs past their first hour (4 h from scratch)
            old_enough = minutes >= (240 if "scratch" in name else 60)
            if old_enough and means.get("tip", 0.0) > args.tip:
                flags.append(f"TIPPING:{name}({means['tip']:.2f})")
            # RULES.md rule 4: the curriculum has kept the mean level under 0.5 for the last 2 h
            # not for controller-randomised runs: their deliberately degraded executors demote robots by design
            # (decisions.md 2026-10-12 ~20:10)
            if TAGS["lvl"] in tags and minutes >= (360 if "scratch" in name else 150) and "agnostic" not in name:
                ev = ea.Scalars(TAGS["lvl"])
                recent = [e.value for e in ev if e.wall_time >= ev[-1].wall_time - 120 * 60]
                if recent and max(recent) < 0.5:
                    flags.append(f"CURRICULUM_LOW:{name}({max(recent):.2f})")
        print(line)
    print("FLAGS:", " ".join(flags) if flags else "none")


if __name__ == "__main__":
    main()
