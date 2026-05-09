import csv
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.collector import MetricsCollector
from src.model.disaster_model import DisasterModel

STRATEGIES = ["pheromone", "heterogeneous"]
SWARM_SIZE = 6
HAZARD_RATE = "medium"
SEEDS = range(30)

RESULTS_DIR = Path(__file__).parent.parent / "results"
HETERO_CSV = RESULTS_DIR / "hetero_runs.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
ERRORS_LOG = RESULTS_DIR / "errors.log"

STRATEGY_COLORS = {
    "pheromone": "#2ca02c",
    "heterogeneous": "#17becf",
}

STRATEGY_LABELS = {
    "pheromone": "Pheromone",
    "heterogeneous": "Heterogeneous",
}

_CSV_FIELDS = [
    "strategy",
    "swarm_size",
    "hazard_rate",
    "seed",
    "survivors_found",
    "agents_lost",
    "coverage_pct",
    "duplicate_visits",
    "timesteps_run",
]

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
    }
)


def _run_single(strategy: str, seed: int) -> dict:
    """
    Runs one simulation for the given strategy and seed.

    Args:
        - strategy: Search strategy name.
        - seed: Random seed.

    Returns:
        - Metrics summary dict.
    """
    model = DisasterModel(
        strategy=strategy,
        swarm_size=SWARM_SIZE,
        hazard_rate=HAZARD_RATE,
        seed=seed,
    )
    while not model.is_done:
        model.step()
    return MetricsCollector(model).get_summary()


def run_hetero_experiment() -> pd.DataFrame:
    """
    Executes all 60 hetero-experiment runs and writes results/hetero_runs.csv.

    Returns:
        - DataFrame with one row per run.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(STRATEGIES) * len(SEEDS)
    i = 0

    with open(HETERO_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()

        for strategy in STRATEGIES:
            for seed in SEEDS:
                i += 1
                run_name = f"{strategy}_r{seed:02d}"
                if i % 10 == 0:
                    print(f"[{i}/{total}] {run_name}")

                try:
                    row = _run_single(strategy, seed)
                    writer.writerow(row)
                except Exception:
                    msg = f"FAILED: {run_name}\n" f"{traceback.format_exc()}\n"
                    print(msg, file=sys.stderr)
                    with open(ERRORS_LOG, "a") as ef:
                        ef.write(msg)

    print(f"[{total}/{total}] Done. Results written to {HETERO_CSV}")
    return pd.read_csv(HETERO_CSV)


def plot_hetero_comparison(
    df: pd.DataFrame, figures_dir: Path = FIGURES_DIR
) -> None:
    """
    Generates hetero_comparison.png: grouped bar chart of survivors_found and
    agents_lost for pheromone vs heterogeneous.

    Args:
        - df: DataFrame produced by run_hetero_experiment().
        - figures_dir: Directory to write the figure into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["survivors_found", "agents_lost"]
    metric_labels = ["Survivors Found", "Agents Lost"]

    n_metrics = len(metrics)
    n_strategies = len(STRATEGIES)
    bar_width = 0.35
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, strategy in enumerate(STRATEGIES):
        sdf = df[df["strategy"] == strategy]
        means = [sdf[m].mean() for m in metrics]
        stds = [sdf[m].std() for m in metrics]

        offset = (i - (n_strategies - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
            error_kw={"elinewidth": 1.2, "alpha": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Mean value (+/-1 SD, 30 seeds)")
    ax.set_title("Heterogeneous vs Pheromone Swarm (swarm=6, hazard=medium)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, None)

    out_path = figures_dir / "hetero_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main() -> None:
    """
    Runs the heterogeneous experiment and generates the comparison chart.
    """
    df = run_hetero_experiment()
    plot_hetero_comparison(df)


if __name__ == "__main__":
    main()
