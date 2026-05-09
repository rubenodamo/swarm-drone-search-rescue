import csv
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.collector import MetricsCollector
from src.model.disaster_model import DisasterModel

STRATEGIES = ["random", "astar", "pheromone", "rl"]
SWARM_SIZE = 1
HAZARD_RATES = ["slow", "medium", "fast"]
SEEDS = range(30)

_MODEL_PATH = (
    Path(__file__).parent.parent / "results" / "rl_model" / "ppo_drone_search.zip"
)
RESULTS_DIR = Path(__file__).parent.parent / "results"
RL_BASELINE_CSV = RESULTS_DIR / "rl_baseline_runs.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
ERRORS_LOG = RESULTS_DIR / "errors.log"

STRATEGY_COLORS = {
    "random": "#1f77b4",
    "astar": "#ff7f0e",
    "pheromone": "#2ca02c",
    "rl": "#d62728",
}

STRATEGY_LABELS = {
    "random": "Random",
    "astar": "A*",
    "pheromone": "Pheromone",
    "rl": "RL (PPO)",
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


def _run_single(
    strategy: str,
    hazard_rate: str,
    seed: int,
    policy: PPO | None,
) -> dict:
    """
    Runs one single-agent simulation to completion.

    Args:
        - strategy: Search strategy name.
        - hazard_rate: Fire spread rate label.
        - seed: Random seed.
        - policy: Loaded PPO model; only used when strategy == 'rl'.

    Returns:
        - Metrics summary dict.
    """
    model = DisasterModel(
        strategy=strategy,
        swarm_size=SWARM_SIZE,
        hazard_rate=hazard_rate,
        seed=seed,
    )

    if strategy == "rl":
        drone = next(iter(model.agents))
        while not model.is_done:
            obs = _build_obs(model, drone)
            action, _ = policy.predict(obs, deterministic=True)
            drone.pending_action = int(action)
            model.step()
    else:
        while not model.is_done:
            model.step()

    return MetricsCollector(model).get_summary()


def _build_obs(model: DisasterModel, drone) -> np.ndarray:
    """
    Constructs the 100-dim PPO observation, mirroring DroneSearchEnv._get_obs() so inference matches training.

    Args:
        - model: The active DisasterModel.
        - drone: The RLDrone agent.

    Returns:
        - Flattened np.ndarray of shape (100,).
    """
    _PHEROMONE_MAX = 20.0
    cx, cy = drone.pos
    grid_state = model.disaster_grid.grid_state
    pheromone = model.pheromone_grid
    width = model.disaster_grid.width
    height = model.disaster_grid.height
    visited = drone.visited_cells
    survivor_positions = {
        s.pos for s in model.disaster_grid.survivors if not s.found
    }
    obs = np.zeros((4, 5, 5), dtype=np.float32)
    for di, dx in enumerate(range(-2, 3)):
        for dj, dy in enumerate(range(-2, 3)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                obs[0, di, dj] = grid_state[nx, ny] / 2.0
                obs[1, di, dj] = min(pheromone[nx, ny] / _PHEROMONE_MAX, 1.0)
                obs[2, di, dj] = 1.0 if (nx, ny) in survivor_positions else 0.0
                obs[3, di, dj] = 1.0 if (nx, ny) in visited else 0.0
            else:
                obs[0, di, dj] = 0.5
    return obs.flatten()


def run_rl_baseline() -> pd.DataFrame:
    """
    Executes all 360 single-agent baseline runs and writes results/rl_baseline_runs.csv.

    Returns:
        - DataFrame with one row per run.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPO model from {_MODEL_PATH}")
    policy = PPO.load(str(_MODEL_PATH))

    total = len(STRATEGIES) * len(HAZARD_RATES) * len(SEEDS)
    i = 0

    with open(RL_BASELINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()

        for strategy in STRATEGIES:
            for hazard_rate in HAZARD_RATES:
                for seed in SEEDS:
                    i += 1
                    run_name = f"{strategy}_s1_h{hazard_rate}_r{seed:02d}"
                    if i % 30 == 0:
                        print(f"[{i}/{total}] {run_name}")

                    try:
                        row = _run_single(strategy, hazard_rate, seed, policy)
                        writer.writerow(row)
                    except Exception:
                        msg = (
                            f"FAILED: {run_name}\n"
                            f"{traceback.format_exc()}\n"
                        )
                        print(msg, file=sys.stderr)
                        with open(ERRORS_LOG, "a") as ef:
                            ef.write(msg)

    print(f"[{total}/{total}] Done. Results written to {RL_BASELINE_CSV}")
    return pd.read_csv(RL_BASELINE_CSV)


def plot_rl_baseline(
    df: pd.DataFrame, figures_dir: Path = FIGURES_DIR
) -> None:
    """
    Generates rl_baseline_comparison.png: survivors_found by strategy and hazard rate.

    Args:
        - df: DataFrame produced by run_rl_baseline().
        - figures_dir: Directory to write the figure into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    hazard_rates = HAZARD_RATES
    x = np.arange(len(hazard_rates))
    bar_width = 0.2
    n_strategies = len(STRATEGIES)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, strategy in enumerate(STRATEGIES):
        sdf = df[df["strategy"] == strategy]
        means = [
            sdf[sdf["hazard_rate"] == h]["survivors_found"].mean()
            for h in hazard_rates
        ]
        stds = [
            sdf[sdf["hazard_rate"] == h]["survivors_found"].std()
            for h in hazard_rates
        ]
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
    ax.set_xticklabels(["Slow hazard", "Medium hazard", "Fast hazard"])
    ax.set_ylabel("Mean survivors found (±1 SD, 30 seeds)")
    ax.set_title("Single-agent strategy comparison (swarm_size=1)")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, None)

    out_path = figures_dir / "rl_baseline_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main() -> None:
    """
    Runs the RL single-agent baseline experiment and generates the comparison figure.
    """
    df = run_rl_baseline()
    plot_rl_baseline(df)


if __name__ == "__main__":
    main()
