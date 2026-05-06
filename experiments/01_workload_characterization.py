"""
Workload characterization plots (M1 visualization).

Generates a multi-panel figure showing one representative tenant from each
archetype, demonstrating visually distinguishable signatures across the
eight workload categories.

Usage:
    uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42
    uv run python experiments/01_workload_characterization.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────

# Map archetypes to representative tenant indices (first 100 tenants are
# stratified across archetypes due to the distribution).
ARCHETYPE_REPRESENTATIVES = {
    "low-traffic-blog": "tenant_0006",
    "ecommerce-retail": "tenant_0012",
    "news-publisher": "tenant_0025",
    "b2b-saas": "tenant_0038",
    "wp-cron-heavy": "tenant_0003",
    "cache-driven": "tenant_0000",
    "compute-heavy": "tenant_0002",
    "idle-ish": "tenant_0017",
}

# Display name overrides for plot labels
ARCHETYPE_LABELS = {
    "low-traffic-blog": "Low-Traffic Blog",
    "ecommerce-retail": "Ecommerce Retail",
    "news-publisher": "News Publisher",
    "b2b-saas": "B2B SaaS",
    "wp-cron-heavy": "WP-Cron Heavy",
    "cache-driven": "Cache-Driven",
    "compute-heavy": "Compute-Heavy",
    "idle-ish": "Idle-ish",
}

# Colors for each archetype (colorblind-friendly palette)
ARCHETYPE_COLORS = {
    "low-traffic-blog": "#0173B2",
    "ecommerce-retail": "#DE8F05",
    "news-publisher": "#029E73",
    "b2b-saas": "#D55E00",
    "wp-cron-heavy": "#CC78BC",
    "cache-driven": "#CA9161",
    "compute-heavy": "#FBAFE4",
    "idle-ish": "#949494",
}


def plot_archetype_samples(
    data_dir: Path,
    output_path: Path | None = None,
    hours_to_show: int = 72,
    figsize: tuple[int, int] = (24, 18),
) -> plt.Figure:
    """
    Generate a multi-panel plot showing CPU utilization for one
    representative tenant per archetype.

    Each panel shows `hours_to_show` of 1-minute data to reveal
    the distinctive temporal signatures.
    """
    sns.set_theme(style="darkgrid", context="talk")
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    n_points = hours_to_show * 60  # 1-minute resolution
    x_hours = range(hours_to_show)

    for idx, (archetype, tenant_id) in enumerate(ARCHETYPE_REPRESENTATIVES.items()):
        ax = axes[idx]
        color = ARCHETYPE_COLORS[archetype]

        parquet_path = data_dir / f"{tenant_id}.parquet"
        if not parquet_path.exists():
            ax.text(0.5, 0.5, f"Missing: {tenant_id}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(f"{idx+1}. {ARCHETYPE_LABELS[archetype]}", fontweight="bold")
            continue

        df = pl.read_parquet(parquet_path)

        # Sample evenly to get exactly hours_to_show hours worth of 1-min data
        # Start at a random offset to avoid always showing the same window
        total = len(df)
        start = 1440  # Skip first day (warmup)
        if start + n_points > total:
            start = total - n_points - 1

        cpu = df["cpu_util"][start : start + n_points].to_numpy()
        # Downsample for display: take min/max/mean per 5-min window for ribbon
        window = 5
        n_windows = n_points // window

        cpu_reshaped = cpu[: n_windows * window].reshape(n_windows, window)
        cpu_min = cpu_reshaped.min(axis=1)
        cpu_max = cpu_reshaped.max(axis=1)
        cpu_mean = cpu_reshaped.mean(axis=1)
        x_display = [i * 5 / 60 for i in range(n_windows)]  # convert to hours

        # Plot ribbon (min/max range) + mean line
        ax.fill_between(x_display, cpu_min, cpu_max, alpha=0.25, color=color)
        ax.plot(x_display, cpu_mean, color=color, linewidth=1.2)

        # Add spike indicator markers
        # Mark points where CPU exceeds 2σ above the mean
        threshold = cpu_mean + 2 * cpu_reshaped.std()
        spike_mask = cpu_max > threshold

        ax.set_title(f"{idx+1}. {ARCHETYPE_LABELS[archetype]}",
                     fontweight="bold", fontsize=13)
        ax.set_ylabel("CPU Utilization")
        ax.set_ylim(0, None)

        # Add a faint horizontal line at the mean
        overall_mean = cpu_mean.mean()
        ax.axhline(overall_mean, color=color, linestyle="--", alpha=0.3, linewidth=0.8)

    # Shared x-axis label
    fig.text(0.5, 0.02, "Time (hours)", ha="center", fontsize=14)
    fig.suptitle(
        "Synthetic Workload Archetypes — CPU Utilization Signatures\n"
        f"Showing {hours_to_show}h window at 1-minute resolution (representative tenants, seed=42)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    return fig


def plot_resource_heatmap(
    data_dir: Path,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Generate a heatmap showing mean resource utilization per archetype across
    all 4 dimensions, normalized for comparison.

    This validates that the eight archetypes occupy distinct regions
    of the resource-usage feature space.
    """
    import numpy as np

    # Load manifest
    import csv
    manifest_path = data_dir / "manifest.csv"
    archetype_stats: dict[str, dict[str, list[float]]] = {}
    metrics = ["cpu_baseline", "mem_baseline", "net_baseline", "disk_baseline", "spike_rate", "diurnal_amplitude"]

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            arch = row["archetype"]
            if arch not in archetype_stats:
                archetype_stats[arch] = {m: [] for m in metrics}
            for m in metrics:
                archetype_stats[arch][m].append(float(row[m]))

    # Compute means
    archetypes = list(ARCHETYPE_LABELS.keys())
    metric_labels = ["CPU\nBaseline", "Mem\nBaseline", "Net\nBaseline", "Disk\nBaseline", "Spike\nRate", "Diurnal\nAmp"]
    data_matrix = np.zeros((len(archetypes), len(metrics)))

    for i, arch in enumerate(archetypes):
        for j, m in enumerate(metrics):
            data_matrix[i, j] = np.mean(archetype_stats[arch][m])

    # Normalize columns for comparison
    data_norm = (data_matrix - data_matrix.min(axis=0)) / (
        data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        data_norm,
        annot=data_matrix.round(2),
        fmt=".2f",
        xticklabels=metric_labels,
        yticklabels=[ARCHETYPE_LABELS[a] for a in archetypes],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Normalized Value"},
    )
    ax.set_title("Archetype Resource Profile Heatmap\n(Mean Values, Normalized per Metric)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate workload characterization plots"
    )
    parser.add_argument(
        "--data", type=str, default="data/synthetic", help="Synthetic data directory"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory for plots"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)

    print("Generating archetype sample plots...")
    plot_archetype_samples(
        data_dir,
        output_path=output_dir / "01_archetype_cpu_signatures.png",
    )

    print("Generating resource profile heatmap...")
    plot_resource_heatmap(
        data_dir,
        output_path=output_dir / "01_archetype_resource_heatmap.png",
    )

    print(f"\n✓ Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
