import csv

import pytest

from src.metrics.collector import MetricsCollector, write_csv
from src.model.disaster_model import DisasterModel

EXPECTED_KEYS = {
    "strategy",
    "swarm_size",
    "hazard_rate",
    "seed",
    "survivors_found",
    "agents_lost",
    "coverage_pct",
    "duplicate_visits",
    "timesteps_run",
}


def _run_model(strategy="random", swarm_size=3, seed=0, steps=10):
    model = DisasterModel(
        strategy=strategy,
        swarm_size=swarm_size,
        hazard_rate="medium",
        seed=seed,
    )
    for _ in range(steps):
        model.step()
    return model


class TestMetricsCollector:
    """Tests for MetricsCollector.get_summary()."""

    def test_all_fields_present_after_run(self):
        model = _run_model()
        summary = MetricsCollector(model).get_summary()
        assert set(summary.keys()) == EXPECTED_KEYS

    def test_coverage_pct_in_valid_range(self):
        model = _run_model()
        summary = MetricsCollector(model).get_summary()
        assert 0 <= summary["coverage_pct"] <= 100

    def test_duplicate_visits_non_negative(self):
        model = _run_model()
        summary = MetricsCollector(model).get_summary()
        assert summary["duplicate_visits"] >= 0

    def test_summary_values_match_model_state(self):
        model = _run_model(steps=10)
        summary = MetricsCollector(model).get_summary()
        assert summary["strategy"] == "random"
        assert summary["swarm_size"] == 3
        assert summary["hazard_rate"] == "medium"
        assert summary["seed"] == 0
        assert summary["survivors_found"] == model.survivors_found_count
        assert summary["agents_lost"] == model.agents_lost
        assert summary["timesteps_run"] == model.timestep

    def test_coverage_pct_increases_with_more_steps(self):
        model_short = _run_model(steps=5)
        model_long = _run_model(steps=50)
        pct_short = MetricsCollector(model_short).get_summary()["coverage_pct"]
        pct_long = MetricsCollector(model_long).get_summary()["coverage_pct"]
        assert pct_long >= pct_short

    def test_coverage_pct_nonzero_after_steps(self):
        model = _run_model(steps=10)
        summary = MetricsCollector(model).get_summary()
        assert summary["coverage_pct"] > 0


class TestWriteCsv:
    """Tests for write_csv()."""

    def test_csv_written_with_correct_values(self, tmp_path):
        path = tmp_path / "out.csv"
        model = _run_model()
        summary = MetricsCollector(model).get_summary()
        write_csv(summary, path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["strategy"] == summary["strategy"]
        assert float(rows[0]["coverage_pct"]) == pytest.approx(
            summary["coverage_pct"]
        )

    def test_headers_present_exactly_once_after_two_writes(self, tmp_path):
        path = tmp_path / "out.csv"
        model = _run_model()
        summary = MetricsCollector(model).get_summary()
        write_csv(summary, path)
        write_csv(summary, path)
        with open(path, newline="") as f:
            lines = f.readlines()
        header_lines = [l for l in lines if l.startswith("strategy")]
        assert len(header_lines) == 1

    def test_second_write_appends_row(self, tmp_path):
        path = tmp_path / "out.csv"
        model_a = _run_model(seed=0)
        model_b = _run_model(seed=1)
        write_csv(MetricsCollector(model_a).get_summary(), path)
        write_csv(MetricsCollector(model_b).get_summary(), path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["seed"] == "0"
        assert rows[1]["seed"] == "1"
