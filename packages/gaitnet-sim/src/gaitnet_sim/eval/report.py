"""Summary numbers, a plot and the MLflow record of an `eval_sweep` run.

Sits after `Evaluator` (which produces the per-robot rows) and before nothing: it turns those
rows into what is compared between policies. A robot succeeded if its episode reached the time
limit. Walking off the side of its 1 m wide row also ends an episode as a time-out
(`truncated`), but it is not success: it is counted apart, as `exited`. The headline
`survival_mean` is the mean of the per-(difficulty, velocity) success rates, so every cell of the
sweep weighs the same however many robots it held. Needs no simulator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

Rows = list[dict]
"""One dict per robot and trial, with the CSV columns of `eval_sweep`: `difficulty`, `velocity`,
`trial`, `env`, `distance` (m), `steps`, `truncated` (0/1) and `terminated_by`."""


EXIT_TERMS = ("terrain_out_of_bounds",)
"""Terminations that end an episode as a time-out without the robot having succeeded: in the
evaluation scene, leaving the robot's own sub-terrain."""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _exited(row: dict) -> bool:
    return row["terminated_by"] in EXIT_TERMS


def _survived(row: dict) -> bool:
    return bool(int(row["truncated"])) and not _exited(row)


def cells(rows: Rows, step_dt: float) -> dict[tuple[float, float], dict[str, float]]:
    """Per (difficulty, velocity): `survival` (fraction that reached the time limit), `exited`
    (fraction that walked off their row) and, for a nonzero command, `distance_ratio`: the mean
    of distance walked over distance commanded (velocity times the time the robot's episode
    lasted, `steps * step_dt` seconds)."""
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["difficulty"], row["velocity"])].append(row)
    per_cell = {}
    for key, group in sorted(grouped.items()):
        cell = {
            "survival": _mean([float(_survived(row)) for row in group]),
            "exited": _mean([float(_exited(row)) for row in group]),
        }
        if key[1] > 0:
            cell["distance_ratio"] = _mean(
                [row["distance"] / (key[1] * row["steps"] * step_dt) for row in group if row["steps"] > 0]
            )
        per_cell[key] = cell
    return per_cell


def summarize(rows: Rows, step_dt: float) -> dict[str, float]:
    """The metrics logged to MLflow, keyed by name.

    `survival_mean`, `exited_mean` and `distance_ratio_mean` average the cells (see module
    docstring; zero-velocity cells have no distance ratio), `survival/d<difficulty>_v<velocity>`
    and so on are the cells, and `terminated/<term>` is the fraction of all robots ended by
    each termination (`truncated` for rows that name none)."""
    per_cell = cells(rows, step_dt)
    metrics = {
        "survival_mean": _mean([cell["survival"] for cell in per_cell.values()]),
        "exited_mean": _mean([cell["exited"] for cell in per_cell.values()]),
        "distance_ratio_mean": _mean([cell["distance_ratio"] for cell in per_cell.values() if "distance_ratio" in cell]),
    }
    for (difficulty, velocity), cell in per_cell.items():
        for name, value in cell.items():
            metrics[f"{name}/d{difficulty:g}_v{velocity:g}"] = value
    for reason, count in Counter(row["terminated_by"] or "truncated" for row in rows).items():
        metrics[f"terminated/{reason}"] = count / len(rows)
    # e.g. no distance ratio when every command was zero
    return {name: value for name, value in metrics.items() if value == value}


def plot(rows: Rows, step_dt: float, title: str = ""):
    """Survival and distance ratio against terrain difficulty, a line per commanded velocity.

    Returns a matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    per_cell = cells(rows, step_dt)
    difficulties = sorted({difficulty for difficulty, _ in per_cell})
    velocities = sorted({velocity for _, velocity in per_cell})
    colors = plt.cm.plasma([i / max(len(velocities) - 1, 1) for i in range(len(velocities))])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name, label in zip(axes, ("survival", "distance_ratio"), ("Success rate", "Distance / commanded")):
        for velocity, color in zip(velocities, colors):
            xs = [d for d in difficulties if name in per_cell.get((d, velocity), {})]
            ys = [per_cell[(d, velocity)][name] for d in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=6, color=color, label=f"{velocity:g} m/s")
            if name == "survival":
                ax.fill_between(xs, ys, alpha=0.2, color=color)
        ax.set_xlabel("Terrain difficulty")
        ax.set_ylabel(label)
        ax.set_xticks(difficulties)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].legend(title="Command velocity", loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def log_to_mlflow(
    parent_run_id: str,
    rows: Rows,
    step_dt: float,
    params: dict[str, object],
    tags: dict[str, str],
    csv_path: Path,
    run_name: str,
) -> str:
    """Record an evaluation as a run nested under the training run `parent_run_id`, and return
    its id. Settings go in as params and the checkpoint in `tags`, so evaluations of one run
    under different settings stay apart; the CSV and the plot go up as artifacts."""
    import mlflow

    metrics = summarize(rows, step_dt)
    figure = plot(rows, step_dt, title=run_name)
    # a nested run lands in the process's default experiment unless told otherwise, and the UI
    # only shows children under a parent in the same experiment
    experiment_id = mlflow.MlflowClient().get_run(parent_run_id).info.experiment_id
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(
            run_name=run_name, nested=True, experiment_id=experiment_id, tags={"kind": "eval", **tags}
        ) as run:
            mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(csv_path))
            mlflow.log_figure(figure, "sweep.png")
    return run.info.run_id
