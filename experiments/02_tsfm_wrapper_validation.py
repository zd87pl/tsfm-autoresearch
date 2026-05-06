"""
TimesFM Wrapper Validation (M2 visualization).

Generates a forecast plot for a single synthetic tenant to validate the
TSFMClient wrapper and demonstrate quantile forecasting.

Usage:
    uv run python experiments/02_tsfm_wrapper_validation.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from tsfm_autoresearch.tsfm_client import ForecastConfig, TSFMClient


def plot_quantile_forecast(
    client: TSFMClient,
    data_dir: Path,
    tenant_id: str = "tenant_0000",
    context_len: int = 256,
    horizon: int = 60,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Produce a multi-panel plot showing:

    1. Historical data + point forecast continuation for all 4 resources
    2. Zoom on one resource with quantile bands

    This validates that:
    - The wrapper correctly interfaces with TimesFM
    - Quantile forecasts are non-degenerate (q10 < q50 < q90)
    - Forecasts look reasonable for the given archetype
    """
    # Load tenant data
    parquet_path = data_dir / f"{tenant_id}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Tenant data not found: {parquet_path}")

    df = pl.read_parquet(parquet_path)
    cpu = df["cpu_util"].to_numpy()
    mem = df["mem_util"].to_numpy()
    net = df["net_bytes"].to_numpy()
    disk = df["disk_iops"].to_numpy()

    # Build history as (T, 4)
    # Use a slice from the middle of the series for a good visual
    total = len(cpu)
    start = total // 2  # Start from middle
    end = start + context_len

    if end > total:
        end = total
        start = end - context_len

    history = np.column_stack([
        cpu[start:end],
        mem[start:end],
        net[start:end],
        disk[start:end],
    ])

    # Also get the ground truth for the next `horizon` steps
    gt_start = end
    gt_end = min(gt_start + horizon, total)
    ground_truth = np.column_stack([
        cpu[gt_start:gt_end],
        mem[gt_start:gt_end],
        net[gt_start:gt_end],
        disk[gt_start:gt_end],
    ])
    actual_horizon = ground_truth.shape[0]

    # Forecast
    config = ForecastConfig(context_len=context_len, quantiles=[0.1, 0.5, 0.9])
    forecast = client.forecast(history, config, horizon=actual_horizon)

    print(f"Forecast latency: {forecast.latency_ms:.1f}ms")
    print(f"Point shape: {forecast.point.shape}")
    print(f"Quantile shape: {forecast.quantiles.shape}")

    # ── Plotting ────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    resource_names = ["CPU Utilization", "Memory Utilization",
                      "Network (bytes/s)", "Disk IOPS"]
    resource_colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00"]
    resource_keys = ["cpu_util", "mem_util", "net_bytes", "disk_iops"]

    x_hist = np.arange(context_len)  # minutes
    x_forecast = np.arange(context_len, context_len + actual_horizon)

    for d in range(4):
        ax = axes[d]
        color = resource_colors[d]

        # History
        ax.plot(x_hist, history[:, d], color=color, alpha=0.6, linewidth=0.8,
                label="History")

        # Ground truth (if available)
        if actual_horizon > 0:
            ax.plot(x_forecast, ground_truth[:, d], color="black", linewidth=1.5,
                    linestyle="--", alpha=0.7, label="Actual")

        # Point forecast
        ax.plot(x_forecast, forecast.point[:, d], color=color, linewidth=2.0,
                label="Point Forecast")

        # Quantile bands (q10–q90)
        ax.fill_between(
            x_forecast,
            forecast.quantiles[:, d, 0],  # q10
            forecast.quantiles[:, d, 2],  # q90
            alpha=0.2, color=color,
            label="80% CI (q10–q90)"
        )

        # Median forecast as dashed line
        ax.plot(x_forecast, forecast.quantiles[:, d, 1], color=color,
                linewidth=1.0, linestyle=":", alpha=0.8, label="Median (q50)")

        ax.set_title(resource_names[d], fontweight="bold")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(resource_keys[d])
        ax.legend(fontsize=9, loc="upper left")

        # Add vertical line at forecast boundary
        ax.axvline(x=context_len, color="gray", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"TimesFM 2.5 Forecast — {tenant_id}\n"
        f"Context: {context_len} min | Horizon: {actual_horizon} min | "
        f"Latency: {forecast.latency_ms:.1f}ms",
        fontsize=16, fontweight="bold", y=1.01,
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate TimesFM wrapper with a forecast plot"
    )
    parser.add_argument("--data", default="data/synthetic", help="Synthetic data dir")
    parser.add_argument("--tenant", default="tenant_0000", help="Tenant ID to forecast")
    parser.add_argument("--context", type=int, default=256, help="Context length")
    parser.add_argument("--horizon", type=int, default=60, help="Forecast horizon")
    parser.add_argument("--output", default="results/02_tsfm_forecast_validation.png",
                       help="Output plot path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "auto"
    client = TSFMClient(device=device, max_context=512, max_horizon=128)

    data_dir = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_quantile_forecast(
        client,
        data_dir,
        tenant_id=args.tenant,
        context_len=args.context,
        horizon=args.horizon,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
