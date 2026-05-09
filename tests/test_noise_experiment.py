from pathlib import Path

import pandas as pd
import pytest

from experiments.run_noise_experiment import (
    FIGURES_DIR,
    NOISE_CSV,
    NOISE_LEVELS,
    STRATEGIES,
    plot_agents_lost_vs_noise,
    plot_survivors_vs_noise,
)


@pytest.fixture()
def noise_df() -> pd.DataFrame:
    return pd.read_csv(NOISE_CSV)


class TestNoiseCSV:
    """
    Tests for results/noise_runs.csv structure.
    """

    def test_csv_has_expected_row_count(self, noise_df: pd.DataFrame):
        assert len(noise_df) == len(STRATEGIES) * len(NOISE_LEVELS) * 30

    def test_csv_has_all_strategies(self, noise_df: pd.DataFrame):
        assert set(noise_df["strategy"].unique()) == set(STRATEGIES)

    def test_csv_has_all_noise_levels(self, noise_df: pd.DataFrame):
        assert set(noise_df["noise_level"].unique()) == set(NOISE_LEVELS)


class TestNoiseCharts:
    """
    Tests for plot_survivors_vs_noise() and plot_agents_lost_vs_noise().
    """

    def test_survivors_figure_exists_and_is_large_enough(
        self, noise_df: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ):
        plot_survivors_vs_noise(noise_df, figures_dir=tmp_path)
        out = tmp_path / "survivors_vs_noise.png"
        assert out.exists()
        assert out.stat().st_size > 10_000

    def test_agents_lost_figure_exists_and_is_large_enough(
        self, noise_df: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ):
        plot_agents_lost_vs_noise(noise_df, figures_dir=tmp_path)
        out = tmp_path / "agents_lost_vs_noise.png"
        assert out.exists()
        assert out.stat().st_size > 10_000
