import pytest

from src.metrics.collector import MetricsCollector
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
        strategy=strategy, swarm_size=swarm_size, hazard_rate="medium", seed=seed
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
