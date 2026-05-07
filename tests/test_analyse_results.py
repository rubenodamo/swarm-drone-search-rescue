from pathlib import Path

import pandas as pd
import pytest

from experiments.analyse_results import (
    FIGURES_DIR,
    SUMMARY_PATH,
    plot_coverage_vs_hazard_rate,
    plot_survivors_by_strategy,
)


@pytest.fixture()
def summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)


class TestPlotSurvivorsByStrategy:
    """Tests for plot_survivors_by_strategy()."""

    def test_figure_file_exists_and_is_large_enough(
        self, summary: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ) -> None:
        import experiments.analyse_results as ar

        original = ar.FIGURES_DIR
        ar.FIGURES_DIR = tmp_path
        try:
            plot_survivors_by_strategy(summary)
        finally:
            ar.FIGURES_DIR = original

        out = tmp_path / "survivors_by_strategy.png"
        assert out.exists()
        assert out.stat().st_size > 10_000


class TestPlotCoverageVsHazardRate:
    """Tests for plot_coverage_vs_hazard_rate()."""

    def test_figure_file_exists_and_is_large_enough(
        self, summary: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ) -> None:
        import experiments.analyse_results as ar

        original = ar.FIGURES_DIR
        ar.FIGURES_DIR = tmp_path
        try:
            plot_coverage_vs_hazard_rate(summary)
        finally:
            ar.FIGURES_DIR = original

        out = tmp_path / "coverage_vs_hazard_rate.png"
        assert out.exists()
        assert out.stat().st_size > 10_000
