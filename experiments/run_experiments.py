import itertools
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.collector import MetricsCollector, write_csv
from src.model.disaster_model import DisasterModel

STRATEGIES = ["random", "astar", "pheromone"]
SWARM_SIZES = [3, 6, 12]
HAZARD_RATES = ["slow", "medium", "fast"]
SEEDS = range(30)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
ERRORS_LOG = Path(__file__).parent.parent / "results" / "errors.log"


def _run_single(
    strategy: str, swarm_size: int, hazard_rate: str, seed: int
) -> None:
    """
    Runs one simulation to completion and writes its CSV.

    Args:
        - strategy: Search strategy name.
        - swarm_size: Number of drones.
        - hazard_rate: Fire spread rate label.
        - seed: Random seed.
    """
    model = DisasterModel(
        strategy=strategy,
        swarm_size=swarm_size,
        hazard_rate=hazard_rate,
        seed=seed,
    )
    while not model.is_done:
        model.step()

    summary = MetricsCollector(model).get_summary()
    filename = f"{strategy}_s{swarm_size}_h{hazard_rate}_r{seed:02d}.csv"
    write_csv(summary, RESULTS_DIR / filename)


def main() -> None:
    """
    Runs all 810 experiment conditions and writes one CSV per run.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conditions = list(itertools.product(STRATEGIES, SWARM_SIZES, HAZARD_RATES))
    total = len(conditions) * len(SEEDS)
    i = 0

    for strategy, swarm_size, hazard_rate in conditions:
        for seed in SEEDS:
            i += 1
            run_name = f"{strategy}_s{swarm_size}_h{hazard_rate}_r{seed:02d}"

            if i % 10 == 0:
                print(f"[{i}/{total}] {run_name}")

            try:
                _run_single(strategy, swarm_size, hazard_rate, seed)
            except Exception:
                msg = f"FAILED: {run_name}\n{traceback.format_exc()}\n"
                print(msg, file=sys.stderr)
                with open(ERRORS_LOG, "a") as f:
                    f.write(msg)

    print(f"[{total}/{total}] Done.")


if __name__ == "__main__":
    main()
