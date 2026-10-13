"""Mirror GaitNet training runs (TensorBoard scalars) and the watcher's evals to Weights & Biases.

The runs log to TensorBoard only (train.py); this copies every scalar into a W&B run of the same
name, one W&B run per training run (stable id, resumed on every pass), plus each eval of that run
(evals/*/summary.json, *-idm for the same checkpoints run through the training controller's IDM, and
*-emu on the OCS2-like executor) as eval/*, eval_idm/* and eval_emu/* metrics on their own x axis, eval/iteration.

    python wandb_sync.py --all-today --once      # backfill every run started today
    python wandb_sync.py --interval 300          # keep the runs in runs.txt in sync
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(ROOT, "training", "gaitnet", "runs")
EVALS = os.path.join(ROOT, "training", "gaitnet", "evals")
STATE = os.path.join(HERE, "wandb_state.json")
PROJECT = os.environ.get("WANDB_PROJECT", "gaitnet-training")


def family(name: str) -> str:
    """WB500-walk-track-r3 -> WB500-walk-track: the restarts of one run share a W&B group."""
    return re.sub(r"-r\d+$", "", name)


def run_dirs(all_today: bool) -> dict[str, str]:
    if all_today:
        dirs = [d for d in glob.glob(f"{RUNS}/gaitnet_20261012_*_*") if "CANCELLED" not in d and "smoke" not in d]
        dirs = [d for d in dirs if re.search(r"[A-Za-z]", os.path.basename(d).split("_", 3)[-1])]
    else:
        names = [line.split()[1] for line in open(os.path.join(HERE, "runs.txt")) if line.strip() and not line.startswith("#")]
        dirs = [d for n in names for d in glob.glob(f"{RUNS}/gaitnet_2026*_{n}") if os.path.basename(d).split("_", 3)[-1] == n]
    return {os.path.basename(d).split("_", 3)[-1]: d for d in sorted(dirs)}


def command_of(name: str) -> str | None:
    for log in sorted(glob.glob(os.path.join(ROOT, "cluster", "logs", "gaitnet-*.out")), key=os.path.getmtime, reverse=True):
        with open(log, errors="replace") as f:
            for line in f:
                if line.startswith("+ python -m gaitnet.gaitnet") and f"--run_name {name}" in line and line.rstrip().endswith(name) or (
                        line.startswith("+ python -m gaitnet.gaitnet") and f"--run_name {name} " in line):
                    return line[2:].strip()
    return None


def eval_rows(name: str) -> list[tuple[str, int, dict[str, float]]]:
    """(eval dir, iteration, metrics) for every eval of this run, plain and -idm."""
    rows = []
    for path in sorted(glob.glob(f"{EVALS}/2026*/summary.json")):
        d = os.path.basename(os.path.dirname(path))
        prefix = "eval_idm" if d.endswith("-idm") else "eval_emu" if d.endswith("-emu") else "eval"
        for label, ck in json.load(open(path))["checkpoints"].items():
            base, _, rule = label.partition("|")
            if base != name or (rule and rule != "argmax"):
                continue
            cells = ck["cells"]
            m: dict[str, float] = {}
            for speed in (0.1, 0.2):
                for band, lo, hi in (("10-30", 0.05, 0.35), ("40", 0.35, 0.45), ("0", -1, 0.05)):
                    xs = [c for c in cells.values() if lo < c["voids"] < hi and abs(c["speed"] - speed) < 1e-6]
                    if not xs:
                        continue
                    for key in ("success", "survival", "distance_fraction"):
                        m[f"{prefix}/{key}_{band}pct_v{speed}"] = sum(c[key] * c["n"] for c in xs) / sum(c["n"] for c in xs)
            xs = [c for c in cells.values() if 0.05 < c["voids"] < 0.35]
            m[f"{prefix}/success_10-30pct"] = sum(c["success"] for c in xs) / len(xs)
            rows.append((d, int(ck["iter"]), m))
    return rows


def sync(name: str, path: str, state: dict) -> None:
    import wandb

    st = state.setdefault(name, {"id": hashlib.md5(f"gaitnet:{name}".encode()).hexdigest()[:12], "last_step": -1, "evals": []})
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    new: dict[int, dict[str, float]] = {}
    for tag in ea.Tags()["scalars"]:
        if tag.endswith("/time"):  # rsl_rl logs these against seconds, not iterations
            continue
        for e in ea.Scalars(tag):
            if e.step > st["last_step"]:
                new.setdefault(e.step, {})[tag] = e.value
    evals = [r for r in eval_rows(name) if f"{r[0]}:{r[1]}" not in st["evals"]]
    if not new and not evals:
        return
    run = wandb.init(project=PROJECT, id=st["id"], name=name, group=family(name), resume="allow",
                     config={"run_dir": path, "command": command_of(name)}, settings=wandb.Settings(silent=True))
    run.define_metric("eval/iteration")
    run.define_metric("eval/*", step_metric="eval/iteration")
    run.define_metric("eval_idm/*", step_metric="eval/iteration")
    run.define_metric("eval_emu/*", step_metric="eval/iteration")
    for step in sorted(new):
        run.log(new[step], step=step)
    last = max(new) if new else st["last_step"]
    for d, it, m in sorted(evals, key=lambda r: r[1]):
        run.log({"eval/iteration": it, **m}, step=max(last, 0))
        st["evals"].append(f"{d}:{it}")
    st["last_step"] = last
    run.finish()
    print(f"{time.strftime('%H:%M:%S')} {name}: {len(new)} new steps, {len(evals)} evals -> {run.url}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all-today", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--one", nargs=2, metavar=("NAME", "DIR"), help="(internal) sync one run in this process")
    a = p.parse_args()
    if a.one:
        # one W&B run per process: reusing a process across runs loses the W&B service connection
        state = json.load(open(STATE)) if os.path.exists(STATE) else {}
        sync(a.one[0], a.one[1], state)       # raises on failure, so the state is only saved on success
        json.dump(state, open(STATE, "w"), indent=1)
        return
    while True:
        for name, path in run_dirs(a.all_today).items():
            r = subprocess.run([sys.executable, __file__, "--one", name, path], capture_output=True, text=True, timeout=900)
            lines = [l for l in (r.stdout + r.stderr).splitlines() if "->" in l or "Error" in l or "error" in l]
            print(f"{name}: {'ok' if r.returncode == 0 else 'FAILED'} {lines[-1] if lines else ''}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
