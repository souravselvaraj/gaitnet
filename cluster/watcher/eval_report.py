"""Compare every eval so far (training/gaitnet/evals/*/summary.json, from eval_policy.py): per run and
eval, success / survival / distance fraction averaged over voids 0.1-0.3 and both speeds, and on flat
ground, next to the 1035 baseline.

    python eval_report.py [--detail]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.abspath(os.path.join(HERE, "..", "..", "training", "gaitnet", "evals"))


def band(cells: dict, lo: float, hi: float, key: str) -> tuple[float, int]:
    picked = [c for c in cells.values() if lo - 1e-9 <= c["voids"] <= hi + 1e-9]
    n = sum(c["n"] for c in picked)
    return (sum(c[key] * c["n"] for c in picked) / n if n else float("nan")), n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", action="store_true", help="Also print every cell of the latest eval.")
    args = parser.parse_args()
    summaries = []
    for path in sorted(glob.glob(f"{EVALS}/*/summary.json")):
        with open(path) as f:
            summaries.append((os.path.basename(os.path.dirname(path)), json.load(f)))
    if not summaries:
        print("no evals yet")
        return
    print(f"{'run':28s} {'eval':14s} {'iter':>5s}  {'succ.1-.3':>9s} {'surv.1-.3':>9s} {'dist.1-.3':>9s}  {'succ0':>6s} {'surv0':>6s} {'dist0':>6s}")
    rows = {}
    for when, s in summaries:
        for label, ck in s["checkpoints"].items():
            rows.setdefault(label, []).append((when, ck))
    for label in sorted(rows, key=lambda l: (l != "base1035", l)):
        for when, ck in rows[label]:
            c = ck["cells"]
            succ, _ = band(c, 0.1, 0.3, "success")
            surv, _ = band(c, 0.1, 0.3, "survival")
            dist, _ = band(c, 0.1, 0.3, "distance_fraction")
            s0, _ = band(c, 0.0, 0.0, "success")
            v0, _ = band(c, 0.0, 0.0, "survival")
            d0, _ = band(c, 0.0, 0.0, "distance_fraction")
            print(f"{label:28s} {when:14s} {ck['iter']:5d}  {succ:9.3f} {surv:9.3f} {dist:9.3f}  {s0:6.3f} {v0:6.3f} {d0:6.3f}")
    if args.detail:
        when, s = summaries[-1]
        print(f"\nlatest eval {when}, per cell (success / survival / distance, steps per tick):")
        for label, ck in s["checkpoints"].items():
            print(f"  {label}")
            for key, c in ck["cells"].items():
                print(f"    {key:10s} succ {c['success']:.2f} [{c['success_ci'][0]:.2f},{c['success_ci'][1]:.2f}] "
                      f"surv {c['survival']:.2f} dist {c['distance_fraction']:.2f} ops {c['steps_per_tick']:.3f} falls {c['falls']}")


if __name__ == "__main__":
    main()
