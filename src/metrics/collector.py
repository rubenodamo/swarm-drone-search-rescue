import csv
from pathlib import Path

import numpy as np

from src.environment.grid import CellType


class MetricsCollector:
    """
    Gathers per-run metrics and produces a final summary dictionary.

    Attributes:
        - model: Reference to the DisasterModel being tracked.
    """

    def __init__(self, model) -> None:
        """
        Initialises the collector with a model reference.

        Args:
            - model: The DisasterModel instance to track.
        """
        self._model = model

    def get_summary(self) -> dict:
        """
        Returns a summary dictionary of all run metrics.

        Returns:
            - Dict with keys: strategy, swarm_size, hazard_rate, seed,
              survivors_found, agents_lost, coverage_pct, duplicate_visits,
              timesteps_run.
        """
        model = self._model
        grid_state = model.disaster_grid.grid_state
        passable_count = int((grid_state != CellType.OBSTACLE).sum())

        if passable_count == 0:
            coverage_pct = 0.0
        else:
            coverage_pct = float(
                (model.coverage_grid > 0).sum() / passable_count * 100
            )

        duplicate_visits = int(np.maximum(model.coverage_grid - 1, 0).sum())

        return {
            "strategy": model.strategy,
            "swarm_size": model.swarm_size,
            "hazard_rate": model.hazard_rate,
            "seed": model.seed,
            "survivors_found": model.survivors_found_count,
            "agents_lost": model.agents_lost,
            "coverage_pct": coverage_pct,
            "duplicate_visits": duplicate_visits,
            "timesteps_run": model.timestep,
        }


def write_csv(summary: dict, path: Path) -> None:
    """
    Writes a summary dict as a single CSV row to path.

    Args:
        - summary: Dict produced by MetricsCollector.get_summary().
        - path: Destination file path; created with headers if absent,
                otherwise the row is appended.
    """
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)
