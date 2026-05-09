import pytest
import pandas as pd

from experiments.run_hetero_experiment import (
    HETERO_CSV,
    STRATEGIES,
    plot_hetero_comparison,
)


@pytest.fixture()
def hetero_df() -> pd.DataFrame:
    return pd.read_csv(HETERO_CSV)


class TestHeteroCSV:
    """Tests for results/hetero_runs.csv structure."""

    def test_csv_has_60_rows(self, hetero_df: pd.DataFrame):
        assert len(hetero_df) == len(STRATEGIES) * 30

    def test_csv_has_both_strategies(self, hetero_df: pd.DataFrame):
        assert set(hetero_df["strategy"].unique()) == set(STRATEGIES)

    def test_csv_has_correct_swarm_size(self, hetero_df: pd.DataFrame):
        assert (hetero_df["swarm_size"] == 6).all()

    def test_csv_has_correct_hazard_rate(self, hetero_df: pd.DataFrame):
        assert (hetero_df["hazard_rate"] == "medium").all()


class TestHeteroChart:
    """Tests for plot_hetero_comparison()."""

    def test_figure_exists_and_is_large_enough(
        self, hetero_df: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ):
        plot_hetero_comparison(hetero_df, figures_dir=tmp_path)
        out = tmp_path / "hetero_comparison.png"
        assert out.exists()
        assert out.stat().st_size > 10_000
