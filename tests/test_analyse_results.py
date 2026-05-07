from pathlib import Path

import pandas as pd
import pytest

from experiments.analyse_results import (
    ALL_RUNS_PATH,
    COVERAGE_MEAN_PATH,
    FIGURES_DIR,
    FIRE_SEEDS_PATH,
    SUMMARY_PATH,
    TIMESERIES_ALL_PATH,
    plot_agent_losses_boxplot,
    plot_coverage_heatmaps,
    plot_coverage_vs_hazard_rate,
    plot_survivors_by_strategy,
    plot_survivors_over_time,
)


@pytest.fixture()
def summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)


@pytest.fixture()
def all_runs() -> pd.DataFrame:
    return pd.read_csv(ALL_RUNS_PATH)


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


class TestPlotAgentLossesBoxplot:
    """Tests for plot_agent_losses_boxplot()."""

    def test_figure_file_exists_and_is_large_enough(
        self, all_runs: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ) -> None:
        import experiments.analyse_results as ar

        original = ar.FIGURES_DIR
        ar.FIGURES_DIR = tmp_path
        try:
            plot_agent_losses_boxplot(all_runs)
        finally:
            ar.FIGURES_DIR = original

        out = tmp_path / "agent_losses_boxplot.png"
        assert out.exists()
        assert out.stat().st_size > 10_000


class TestPlotSurvivorsOverTime:
    """Tests for plot_survivors_over_time()."""

    def test_figure_file_exists_and_is_large_enough(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        import experiments.analyse_results as ar

        timeseries_all = pd.read_csv(TIMESERIES_ALL_PATH)
        original = ar.FIGURES_DIR
        ar.FIGURES_DIR = tmp_path
        try:
            plot_survivors_over_time(timeseries_all)
        finally:
            ar.FIGURES_DIR = original

        out = tmp_path / "survivors_over_time.png"
        assert out.exists()
        assert out.stat().st_size > 10_000


class TestPlotCoverageHeatmaps:
    """Tests for plot_coverage_heatmaps()."""

    def test_all_three_figures_exist_and_are_large_enough(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        import experiments.analyse_results as ar

        coverage_mean = pd.read_csv(COVERAGE_MEAN_PATH)
        fire_seeds = pd.read_csv(FIRE_SEEDS_PATH)
        original = ar.FIGURES_DIR
        ar.FIGURES_DIR = tmp_path
        try:
            plot_coverage_heatmaps(coverage_mean, fire_seeds)
        finally:
            ar.FIGURES_DIR = original

        for strategy in ["random", "astar", "pheromone"]:
            out = tmp_path / f"coverage_heatmap_{strategy}.png"
            assert out.exists(), f"Missing: coverage_heatmap_{strategy}.png"
            assert out.stat().st_size > 10_000
