import csv
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.collector import MetricsCollector
from src.model.disaster_model import DisasterModel

STRATEGIES = ["random", "astar", "pheromone"]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
SWARM_SIZE = 6
HAZARD_RATE = "medium"
SEEDS = range(30)

RESULTS_DIR = Path(__file__).parent.parent / "results"
NOISE_CSV = RESULTS_DIR / "noise_runs.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
ERRORS_LOG = RESULTS_DIR / "errors.log"

STRATEGY_COLORS = {
    "random": "#1f77b4",
    "astar": "#ff7f0e",
    "pheromone": "#2ca02c",
}

_CSV_FIELDS = [
    "strategy",
    "swarm_size",
    "hazard_rate",
    "noise_level",
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


def _run_single(strategy: str, noise_level: float, seed: int) -> dict:
    """
    Runs one simulation with both noise types set to noise_level.

    Args:
        - strategy: Search strategy name.
        - noise_level: Applied to both survivor_detection_noise and hazard_detection_noise.
        - seed: Random seed.

    Returns:
        - Metrics dict with noise_level column added.
    """
    model = DisasterModel(
        strategy=strategy,
        swarm_size=SWARM_SIZE,
        hazard_rate=HAZARD_RATE,
        seed=seed,
        survivor_detection_noise=noise_level,
        hazard_detection_noise=noise_level,
    )
    while not model.is_done:
        model.step()
    summary = MetricsCollector(model).get_summary()
    summary["noise_level"] = noise_level
    return summary


def run_noise_experiment() -> pd.DataFrame:
    """
    Executes all 360 noise-experiment runs and writes results/noise_runs.csv.

    Returns:
        - DataFrame with one row per run.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(STRATEGIES) * len(NOISE_LEVELS) * len(SEEDS)
    i = 0

    with open(NOISE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()

        for strategy in STRATEGIES:
            for noise_level in NOISE_LEVELS:
                for seed in SEEDS:
                    i += 1
                    run_name = f"{strategy}_n{noise_level:.2f}_r{seed:02d}"
                    if i % 10 == 0:
                        print(f"[{i}/{total}] {run_name}")

                    try:
                        row = _run_single(strategy, noise_level, seed)
                        writer.writerow(row)
                    except Exception:
                        msg = (
                            f"FAILED: {run_name}\n"
                            f"{traceback.format_exc()}\n"
                        )
                        print(msg, file=sys.stderr)
                        with open(ERRORS_LOG, "a") as ef:
                            ef.write(msg)

    print(f"[{total}/{total}] Done. Results written to {NOISE_CSV}")
    return pd.read_csv(NOISE_CSV)


def plot_survivors_vs_noise(
    df: pd.DataFrame, figures_dir: Path = FIGURES_DIR
) -> None:
    """
    Generates survivors_vs_noise.png from the noise experiment DataFrame.

    Args:
        - df: DataFrame produced by run_noise_experiment().
        - figures_dir: Directory to write the figure into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy in STRATEGIES:
        sdf = df[df["strategy"] == strategy]
        means = sdf.groupby("noise_level")["survivors_found"].mean()
        stds = sdf.groupby("noise_level")["survivors_found"].std()

        ax.plot(
            means.index,
            means.values,
            marker="o",
            color=STRATEGY_COLORS[strategy],
            label=strategy.capitalize(),
        )
        ax.fill_between(
            means.index,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.15,
            color=STRATEGY_COLORS[strategy],
        )

    ax.set_xlabel("Sensor noise level")
    ax.set_ylabel("Mean survivors found")
    ax.set_title("Survivors Found vs Sensor Noise (swarm=6, hazard=medium)")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_ylim(0, 10.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = figures_dir / "survivors_vs_noise.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def plot_agents_lost_vs_noise(
    df: pd.DataFrame, figures_dir: Path = FIGURES_DIR
) -> None:
    """
    Generates agents_lost_vs_noise.png from the noise experiment DataFrame.

    Args:
        - df: DataFrame produced by run_noise_experiment().
        - figures_dir: Directory to write the figure into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy in STRATEGIES:
        sdf = df[df["strategy"] == strategy]
        means = sdf.groupby("noise_level")["agents_lost"].mean()
        stds = sdf.groupby("noise_level")["agents_lost"].std()

        ax.plot(
            means.index,
            means.values,
            marker="o",
            color=STRATEGY_COLORS[strategy],
            label=strategy.capitalize(),
        )
        ax.fill_between(
            means.index,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.15,
            color=STRATEGY_COLORS[strategy],
        )

    ax.set_xlabel("Sensor noise level")
    ax.set_ylabel("Mean agents lost")
    ax.set_title("Agent Losses vs Sensor Noise (swarm=6, hazard=medium)")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_ylim(0, 6.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = figures_dir / "agents_lost_vs_noise.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main() -> None:
    """
    Runs the noise experiment and generates both charts.
    """
    df = run_noise_experiment()
    plot_survivors_vs_noise(df)
    plot_agents_lost_vs_noise(df)


if __name__ == "__main__":
    main()
