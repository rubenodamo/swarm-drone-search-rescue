import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
ALL_RUNS_PATH = RESULTS_DIR / "all_runs.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
SIGNIFICANCE_PATH = RESULTS_DIR / "significance_tests.txt"
FIGURES_DIR = RESULTS_DIR / "figures"

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
    Runs Kruskal-Wallis omnibus tests then pairwise Mann-Whitney U post-hoc
    tests (Bonferroni-corrected) for each metric, and writes results to file.

    Args:
        - df: DataFrame from load_results().
    """
    lines = [
        "Statistical Significance Tests",
        "=" * 60,
        "",
        "Step 1: Kruskal-Wallis H-test (omnibus — are any strategies different?)",
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
            f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f} — {kw_verdict}"
        )

        if not kw_sig:
            lines.append(
                "  Post-hoc tests skipped (omnibus test not significant)."
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
                f"    {s1} vs {s2}: U={stat:.1f}, p={p:.4f} — {verdict}{direction}"
            )

        if sig_pairs:
            summary_lines.append(
                f"{metric}: Significant differences — {'; '.join(sig_pairs)}."
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

    print("Done.")


if __name__ == "__main__":
    main()
