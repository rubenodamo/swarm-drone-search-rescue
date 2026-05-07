import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.grid import CellType
from src.model.disaster_model import DisasterModel

STRATEGIES = ["random", "astar", "pheromone"]
SWARM_SIZE = 6
HAZARD_RATE = "medium"
SEEDS = range(30)

RESULTS_DIR = Path(__file__).parent.parent / "results"
COVERAGE_MEAN_PATH = RESULTS_DIR / "coverage_mean.csv"
FIRE_SEEDS_PATH = RESULTS_DIR / "fire_seed_positions.csv"
ERRORS_LOG = RESULTS_DIR / "errors.log"


def _run_single(strategy: str, seed: int) -> np.ndarray:
    """
    Runs one simulation to completion and returns its coverage_grid.

    Args:
        - strategy: Search strategy name.
        - seed: Random seed.

    Returns:
        - Copy of the model's coverage_grid after the run.
    """
    model = DisasterModel(
        strategy=strategy,
        swarm_size=SWARM_SIZE,
        hazard_rate=HAZARD_RATE,
        seed=seed,
    )
    while not model.is_done:
        model.step()
    return model.coverage_grid.copy()


def _get_fire_seed_positions() -> pd.DataFrame:
    """
    Returns fire seed positions from seed=0 as a DataFrame.

    Returns:
        - DataFrame with columns x, y (one row per fire seed).
    """
    model = DisasterModel(
        strategy="random",
        swarm_size=SWARM_SIZE,
        hazard_rate=HAZARD_RATE,
        seed=0,
    )
    coords = np.argwhere(model.disaster_grid.grid_state == CellType.FIRE)
    return pd.DataFrame(coords, columns=["x", "y"])


def main() -> None:
    """
    Runs 90 simulations (3 strategies x 30 seeds at swarm_size=6, hazard_rate=medium),
    averages coverage_grids per strategy, and writes results/coverage_mean.csv and
    results/fire_seed_positions.csv.
    """
    rows = []
    total = len(list(SEEDS))

    for strategy in STRATEGIES:
        grids = []
        for i, seed in enumerate(SEEDS, start=1):
            if i % 10 == 0 or i == 1:
                print(f"[{strategy}] [{i}/{total}] seed={seed:02d}")
            try:
                grids.append(_run_single(strategy, seed))
            except Exception:
                msg = f"FAILED: {strategy} seed={seed}\n{traceback.format_exc()}\n"
                print(msg, file=sys.stderr)
                with open(ERRORS_LOG, "a") as f:
                    f.write(msg)

        mean_grid = np.mean(grids, axis=0)
        for x in range(mean_grid.shape[0]):
            for y in range(mean_grid.shape[1]):
                rows.append(
                    {
                        "strategy": strategy,
                        "x": x,
                        "y": y,
                        "mean_visits": mean_grid[x, y],
                    }
                )

    pd.DataFrame(rows).to_csv(COVERAGE_MEAN_PATH, index=False)
    print(f"Coverage means written to {COVERAGE_MEAN_PATH}")

    fire_seeds = _get_fire_seed_positions()
    fire_seeds.to_csv(FIRE_SEEDS_PATH, index=False)
    print(f"Fire seed positions written to {FIRE_SEEDS_PATH}")


if __name__ == "__main__":
    main()
