import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
ALL_RUNS_PATH = RESULTS_DIR / "all_runs.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
SIGNIFICANCE_PATH = RESULTS_DIR / "significance_tests.txt"
FIGURES_DIR = RESULTS_DIR / "figures"
TIMESERIES_ALL_PATH = RESULTS_DIR / "timeseries_all.csv"
COVERAGE_MEAN_PATH = RESULTS_DIR / "coverage_mean.csv"
FIRE_SEEDS_PATH = RESULTS_DIR / "fire_seed_positions.csv"

STRATEGY_COLORS = {
    "random": "#1f77b4",
    "astar": "#ff7f0e",
    "pheromone": "#2ca02c",
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
    }
)

METRICS = ["survivors_found", "agents_lost", "coverage_pct", "duplicate_visits"]
STRATEGY_PAIRS = [
    ("random", "astar"),
    ("random", "pheromone"),
    ("astar", "pheromone"),
]
BONFERRONI_ALPHA = 0.05 / 3


def load_results(raw_dir: Path) -> pd.DataFrame:
    """
    Loads all per-run CSVs from raw_dir into a single DataFrame.

    Args:
        - raw_dir: Directory containing per-run CSV files.

    Returns:
        - Combined DataFrame with one row per run.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df.to_csv(ALL_RUNS_PATH, index=False)
    print(f"All runs written to {ALL_RUNS_PATH}")
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by (strategy, swarm_size, hazard_rate) and aggregates metrics.

    Args:
        - df: DataFrame from load_results().

    Returns:
        - Flattened summary DataFrame written to results/summary.csv.
    """
    grouped = df.groupby(["strategy", "swarm_size", "hazard_rate"])[
        METRICS
    ].agg(["mean", "std", "median"])
    grouped.columns = ["_".join(col) for col in grouped.columns]
    summary = grouped.reset_index()
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary written to {SUMMARY_PATH}")
    return summary


def run_significance_tests(df: pd.DataFrame) -> None:
    """
    Runs Kruskal-Wallis group-level tests then pairwise Mann-Whitney U post-hoc
    tests (Bonferroni-corrected) for each metric, and writes results to file.

    Args:
        - df: DataFrame from load_results().
    """
    lines = [
        "Statistical Significance Tests",
        "=" * 60,
        "",
        "Step 1: Kruskal-Wallis H-test (group-level test - are any strategies different?)",
        "Step 2: Pairwise Mann-Whitney U post-hoc tests (only if Step 1 significant)",
        f"Bonferroni-corrected threshold for post-hoc: alpha = {BONFERRONI_ALPHA:.4f}",
        "",
    ]

    groups = {
        s: df.loc[df["strategy"] == s] for s in ["random", "astar", "pheromone"]
    }

    summary_lines = []

    for metric in METRICS:
        lines.append(f"Metric: {metric}")
        lines.append("-" * 40)

        g = [
            groups[s][metric].dropna() for s in ["random", "astar", "pheromone"]
        ]
        kw_stat, kw_p = stats.kruskal(*g)
        kw_sig = kw_p < 0.05
        kw_verdict = "SIGNIFICANT" if kw_sig else "not significant"
        lines.append(
            f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f} - {kw_verdict}"
        )

        if not kw_sig:
            lines.append(
                "  Post-hoc tests skipped (overall test not significant)."
            )
            summary_lines.append(
                f"{metric}: No significant differences between strategies."
            )
            lines.append("")
            continue

        lines.append(
            f"  Post-hoc Mann-Whitney U (Bonferroni alpha={BONFERRONI_ALPHA:.4f}):"
        )
        sig_pairs = []
        for s1, s2 in STRATEGY_PAIRS:
            group1 = groups[s1][metric].dropna()
            group2 = groups[s2][metric].dropna()
            stat, p = stats.mannwhitneyu(
                group1, group2, alternative="two-sided"
            )
            significant = p < BONFERRONI_ALPHA
            verdict = "SIGNIFICANT" if significant else "not significant"
            direction = ""
            if significant:
                m1, m2 = group1.mean(), group2.mean()
                if m1 != m2:
                    higher = s1 if m1 > m2 else s2
                    lower = s2 if m1 > m2 else s1
                    direction = f" ({higher} is higher)"
                    sig_pairs.append(f"{higher} > {lower}")
            lines.append(
                f"    {s1} vs {s2}: U={stat:.1f}, p={p:.4f} - {verdict}{direction}"
            )

        if sig_pairs:
            summary_lines.append(
                f"{metric}: Significant differences - {'; '.join(sig_pairs)}."
            )
        else:
            summary_lines.append(
                f"{metric}: Kruskal-Wallis significant but no pairwise pairs survive "
                "Bonferroni correction."
            )
        lines.append("")

    lines.append("Plain-English Summary")
    lines.append("=" * 60)
    lines.append(
        "810 runs total (270 per strategy) across all conditions. "
        "Kruskal-Wallis tests for any difference among 3 strategies; "
        "pairwise Mann-Whitney U with Bonferroni correction identifies which pairs differ."
    )
    lines.append("")
    lines.extend(summary_lines)

    SIGNIFICANCE_PATH.write_text("\n".join(lines))
    print(f"Significance tests written to {SIGNIFICANCE_PATH}")


def plot_survivors_by_strategy(summary: pd.DataFrame) -> None:
    """
    Saves a grouped bar chart of mean survivors found by strategy and hazard rate.

    Args:
        - summary: DataFrame from compute_summary(), filtered to swarm_size=6 internally.
    """
    df = summary[summary["swarm_size"] == 6].copy()

    strategies = ["random", "astar", "pheromone"]
    hazard_rates = ["slow", "medium", "fast"]
    hazard_colors = {"slow": "#aec7e8", "medium": "#6baed6", "fast": "#08519c"}

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, hr in enumerate(hazard_rates):
        means, stds = [], []
        for s in strategies:
            row = df[(df["strategy"] == s) & (df["hazard_rate"] == hr)]
            means.append(row["survivors_found_mean"].values[0])
            stds.append(row["survivors_found_std"].values[0])
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=hr.capitalize(),
            color=hazard_colors[hr],
            yerr=stds,
            capsize=4,
            ecolor="black",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean Survivors Found")
    ax.set_title(
        "Mean Survivors Found by Strategy and Hazard Rate (swarm size = 6)"
    )
    ax.set_ylim(0, 10.5)
    ax.legend(title="Hazard Rate")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "survivors_by_strategy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_coverage_vs_hazard_rate(summary: pd.DataFrame) -> None:
    """
    Saves a line chart of mean coverage% vs hazard rate, one line per strategy.

    Args:
        - summary: DataFrame from compute_summary(), filtered to swarm_size=6 internally.
    """
    df = summary[summary["swarm_size"] == 6].copy()

    hazard_order = ["slow", "medium", "fast"]
    strategies = ["random", "astar", "pheromone"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for s in strategies:
        sdf = (
            df[df["strategy"] == s]
            .set_index("hazard_rate")
            .reindex(hazard_order)
        )
        means = sdf["coverage_pct_mean"].values
        stds = sdf["coverage_pct_std"].values
        color = STRATEGY_COLORS[s]
        ax.plot(
            hazard_order, means, marker="o", label=s.capitalize(), color=color
        )
        ax.fill_between(
            hazard_order,
            means - stds,
            means + stds,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Hazard Rate")
    ax.set_ylabel("Mean Coverage (%)")
    ax.set_title("Coverage vs Hazard Rate by Strategy (swarm size = 6)")
    ax.set_ylim(0, 80)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "coverage_vs_hazard_rate.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_agent_losses_boxplot(all_runs: pd.DataFrame) -> None:
    """
    Saves a grouped boxplot of agents_lost by strategy and swarm_size with scatter overlay.

    Args:
        - all_runs: Full per-run DataFrame from load_results().
    """
    strategies = ["random", "astar", "pheromone"]
    swarm_sizes = [3, 6, 12]
    size_palette = {3: "#c7e9b4", 6: "#41b6c4", 12: "#225ea8"}

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=all_runs,
        x="strategy",
        y="agents_lost",
        hue="swarm_size",
        order=strategies,
        hue_order=swarm_sizes,
        palette=size_palette,
        width=0.6,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=all_runs,
        x="strategy",
        y="agents_lost",
        hue="swarm_size",
        order=strategies,
        hue_order=swarm_sizes,
        palette=size_palette,
        dodge=True,
        size=3,
        alpha=0.4,
        jitter=True,
        legend=False,
        ax=ax,
    )

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Agents Lost")
    ax.set_title("Agent Losses by Strategy and Swarm Size")
    ax.grid(axis="y", alpha=0.3)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles[: len(swarm_sizes)],
        [f"Swarm size {sz}" for sz in swarm_sizes],
        title="Swarm Size",
    )

    fig.tight_layout()
    out = FIGURES_DIR / "agent_losses_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_survivors_scaling(summary: pd.DataFrame) -> None:
    """
    Saves a faceted line chart of mean survivors found vs swarm size, one panel per hazard rate.

    Args:
        - summary: DataFrame from compute_summary().
    """
    strategies = ["random", "astar", "pheromone"]
    hazard_rates = ["slow", "medium", "fast"]
    swarm_sizes = [3, 6, 12]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, hr in zip(axes, hazard_rates):
        df = summary[summary["hazard_rate"] == hr]
        for s in strategies:
            sdf = (
                df[df["strategy"] == s]
                .set_index("swarm_size")
                .reindex(swarm_sizes)
            )
            means = sdf["survivors_found_mean"].values
            stds = sdf["survivors_found_std"].values
            color = STRATEGY_COLORS[s]
            ax.plot(
                swarm_sizes,
                means,
                marker="o",
                label=s.capitalize(),
                color=color,
            )
            ax.fill_between(
                swarm_sizes,
                means - stds,
                means + stds,
                alpha=0.15,
                color=color,
            )
        ax.set_title(f"Hazard: {hr.capitalize()}")
        ax.set_xlabel("Swarm Size")
        ax.set_xticks(swarm_sizes)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 10.5)

    axes[0].set_ylabel("Mean Survivors Found")
    axes[2].legend(title="Strategy")
    fig.suptitle(
        "Survivors Found vs Swarm Size by Strategy and Hazard Rate", y=1.02
    )

    fig.tight_layout()
    out = FIGURES_DIR / "survivors_scaling.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_survivors_over_time(timeseries_all: pd.DataFrame) -> None:
    """
    Saves a line chart of mean cumulative survivors found per timestep by strategy.

    Args:
        - timeseries_all: Flat DataFrame from results/timeseries_all.csv with
          columns: strategy, seed, timestep, survivors_found.
    """
    strategies = ["random", "astar", "pheromone"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for s in strategies:
        sdf = timeseries_all[timeseries_all["strategy"] == s]
        grouped = sdf.groupby("timestep")["survivors_found"]
        means = grouped.mean()
        stds = grouped.std()
        color = STRATEGY_COLORS[s]
        ax.plot(means.index, means.values, label=s.capitalize(), color=color)
        ax.fill_between(
            means.index,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Cumulative Survivors Found")
    ax.set_title(
        "Survivors Found Over Time by Strategy (swarm size=6, hazard rate=medium)"
    )
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 10.5)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "survivors_over_time.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_coverage_heatmaps(
    coverage_mean: pd.DataFrame, fire_seeds: pd.DataFrame
) -> None:
    """
    Saves one 20x20 coverage heatmap PNG per strategy.

    Args:
        - coverage_mean: DataFrame with columns: strategy, x, y, mean_visits.
        - fire_seeds: DataFrame with columns: x, y (seed=0 representative positions).
    """
    strategies = ["random", "astar", "pheromone"]

    for s in strategies:
        sdf = coverage_mean[coverage_mean["strategy"] == s]
        grid = sdf.pivot(index="y", columns="x", values="mean_visits").values

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            grid, origin="lower", cmap="YlOrRd", interpolation="nearest"
        )
        plt.colorbar(im, ax=ax, label="Mean Visit Count", shrink=0.8)

        for i, (_, row) in enumerate(fire_seeds.iterrows()):
            label = "Fire seed (seed=0)" if i == 0 else ""
            ax.plot(row["x"], row["y"], "b*", markersize=12, label=label)

        fig.suptitle(
            f"Coverage Heatmap - {s.capitalize()}"
            "\n(swarm size=6, hazard rate=medium)",
            fontsize=12,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        out = FIGURES_DIR / f"coverage_heatmap_{s}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    """
    Loads raw results, computes summary statistics, and runs significance tests.
    """
    print("Loading raw results...")
    df = load_results(RAW_DIR)
    print(f"Loaded {len(df)} runs.")

    print("Computing summary statistics...")
    compute_summary(df)

    print("Running significance tests...")
    run_significance_tests(df)

    print("Generating figures...")
    summary = pd.read_csv(SUMMARY_PATH)
    all_runs = pd.read_csv(ALL_RUNS_PATH)
    plot_survivors_by_strategy(summary)
    plot_coverage_vs_hazard_rate(summary)
    plot_agent_losses_boxplot(all_runs)
    plot_survivors_scaling(summary)
    timeseries_all = pd.read_csv(TIMESERIES_ALL_PATH)
    plot_survivors_over_time(timeseries_all)
    coverage_mean = pd.read_csv(COVERAGE_MEAN_PATH)
    fire_seeds = pd.read_csv(FIRE_SEEDS_PATH)
    plot_coverage_heatmaps(coverage_mean, fire_seeds)

    print("Done.")


if __name__ == "__main__":
    main()
