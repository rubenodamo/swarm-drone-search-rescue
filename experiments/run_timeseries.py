import csv
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.disaster_model import DisasterModel

STRATEGIES = ["random", "astar", "pheromone"]
SWARM_SIZE = 6
HAZARD_RATE = "medium"
SEEDS = range(30)
MAX_STEPS = 200

RESULTS_DIR = Path(__file__).parent.parent / "results"
TIMESERIES_DIR = RESULTS_DIR / "raw" / "timeseries"
TIMESERIES_ALL_PATH = RESULTS_DIR / "timeseries_all.csv"
ERRORS_LOG = RESULTS_DIR / "errors.log"


def _run_single(strategy: str, seed: int) -> list[int]:
    """
    Runs one simulation and returns per-step survivor counts, padded to MAX_STEPS.

    Args:
        - strategy: Search strategy name.
        - seed: Random seed.

    Returns:
        - List of length MAX_STEPS with cumulative survivors_found at each step.
    """
    model = DisasterModel(
        strategy=strategy,
        swarm_size=SWARM_SIZE,
        hazard_rate=HAZARD_RATE,
        seed=seed,
    )
    history: list[int] = []
    while not model.is_done:
        model.step()
        history.append(model.survivors_found_count)

    final = history[-1] if history else 0
    while len(history) < MAX_STEPS:
        history.append(final)

    return history[:MAX_STEPS]


def _write_timeseries(strategy: str, seed: int, history: list[int]) -> None:
    """
    Writes per-step survivor counts to a CSV file.

    Args:
        - strategy: Search strategy name.
        - seed: Random seed used for this run.
        - history: List of per-step survivors_found values.
    """
    fname = f"{strategy}_s{SWARM_SIZE}_h{HAZARD_RATE}_r{seed:02d}_ts.csv"
    path = TIMESERIES_DIR / fname
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "survivors_found"])
        for t, count in enumerate(history, start=1):
            writer.writerow([t, count])


def _flatten_timeseries() -> None:
    """
    Concatenates all per-run timeseries CSVs into results/timeseries_all.csv.
    """
    frames = []
    for f in sorted(TIMESERIES_DIR.glob("*_ts.csv")):
        # filename: {strategy}_s6_hmedium_r{seed:02d}_ts.csv
        parts = f.stem.split("_")
        strategy = parts[0]
        seed = int(parts[3][1:])
        df = pd.read_csv(f)
        df.insert(0, "seed", seed)
        df.insert(0, "strategy", strategy)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(
        TIMESERIES_ALL_PATH, index=False
    )
    print(f"Timeseries flattened to {TIMESERIES_ALL_PATH}")


def main() -> None:
    """
    Runs 90 simulations (3 strategies x 30 seeds at swarm_size=6, hazard_rate=medium),
    writes one per-step CSV per run, then flattens to results/timeseries_all.csv.
    """
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    total = len(STRATEGIES) * len(list(SEEDS))
    i = 0

    for strategy in STRATEGIES:
        for seed in SEEDS:
            i += 1
            run_name = f"{strategy}_s{SWARM_SIZE}_h{HAZARD_RATE}_r{seed:02d}"
            if i % 10 == 0 or i == 1:
                print(f"[{i}/{total}] {run_name}")
            try:
                history = _run_single(strategy, seed)
                _write_timeseries(strategy, seed, history)
            except Exception:
                msg = f"FAILED: {run_name}\n{traceback.format_exc()}\n"
                print(msg, file=sys.stderr)
                with open(ERRORS_LOG, "a") as f:
                    f.write(msg)

    print(f"[{total}/{total}] Done.")
    _flatten_timeseries()


if __name__ == "__main__":
    main()
