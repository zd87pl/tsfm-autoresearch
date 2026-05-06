"""
Fixed-configuration TimesFM baseline.

Uses TimesFM with a single globally-optimal configuration found via
offline grid search over the synthetic tenant fleet. This is the
STRONGEST baseline — it represents the best a single fixed config
can do, which is exactly what the autoresearch thesis claims to beat.

The grid search evaluates context_len ∈ {64, 128, 256, 384, 512}
across a stratified sample of tenants and selects the configuration
that minimizes average cost-asymmetric loss.

WHY THIS MATTERS:
If autoresearch can beat this baseline on cost-asymmetric loss, it
proves that per-request configuration adaptation provides value beyond
even the best possible single configuration. If it CAN'T beat it, the
thesis is falsified and we need to know.

PATENT NOTE: The fixed-config baseline is what ALL current systems use
— a single model deployment with one configuration. Proving we can beat
it with per-request adaptation is the central empirical claim.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss
from tsfm_autoresearch.tsfm_client import ForecastConfig, TSFMClient

logger = logging.getLogger(__name__)

# Context length candidates for grid search
_CONTEXT_CANDIDATES = [64, 128, 256, 384, 512]

# Default best config path
_DEFAULT_CONFIG_PATH = Path("data/best_fixed_config.json")


def find_best_fixed_config(
    client: TSFMClient,
    data_dir: str | Path = "data/synthetic",
    n_tenants: int = 20,
    horizon: int = 12,
    sla_tier: SLATier = SLATier.STANDARD,
    seed: int = 42,
) -> ForecastConfig:
    """
    Offline grid search to find the single best fixed configuration.

    Evaluates each context_len candidate across a stratified sample
    of tenants and returns the config with the lowest average
    cost-asymmetric loss.

    This is computationally expensive on CPU (~N_tenants × N_configs
    × TimesFM latency). On GPU it's fast. Results are cached to disk.

    Args:
        client: Initialized TSFMClient.
        data_dir: Directory containing tenant Parquet files.
        n_tenants: Number of tenants to sample for grid search.
        horizon: Forecast horizon for evaluation.
        sla_tier: SLA tier for cost-asymmetric scoring.
        seed: RNG seed for tenant sampling.

    Returns:
        Best ForecastConfig found by grid search.
    """
    import csv

    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"

    # Load manifest
    manifest: dict[str, str] = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            manifest[row["tenant_id"]] = row["archetype"]

    # Stratified sample
    rng = np.random.default_rng(seed)
    archetype_tenants: dict[str, list[str]] = {}
    for tid, arch in manifest.items():
        archetype_tenants.setdefault(arch, []).append(tid)

    sampled: list[str] = []
    n_per_arch = max(1, n_tenants // len(archetype_tenants))
    for arch, tids in archetype_tenants.items():
        chosen = rng.choice(tids, size=min(n_per_arch, len(tids)), replace=False)
        sampled.extend(chosen.tolist())

    sampled = sampled[:n_tenants]

    # Grid search
    loss_fn = CostAsymmetricLoss(alpha=sla_tier.alpha, per_resource=True)
    config_scores: dict[int, list[float]] = {ctx: [] for ctx in _CONTEXT_CANDIDATES}

    eval_points_per_tenant = 2  # Keep small for CPU

    for tid in sampled:
        df = pl.read_parquet(data_dir / f"{tid}.parquet")
        history = np.column_stack([
            df["cpu_util"].to_numpy(), df["mem_util"].to_numpy(),
            df["net_bytes"].to_numpy(), df["disk_iops"].to_numpy(),
        ])
        T = history.shape[0]

        # Evaluate at multiple points
        for ep in range(eval_points_per_tenant):
            t = 200 + ep * 500  # Skip early points
            if t + horizon >= T:
                continue

            hist_slice = history[:t, :]
            actuals = history[t : t + horizon, :]

            for ctx in _CONTEXT_CANDIDATES:
                config = ForecastConfig(context_len=ctx)
                try:
                    forecast = client.forecast(hist_slice, config, horizon)
                    score = loss_fn.point_loss(actuals, forecast.point)
                    config_scores[ctx].append(score)
                except Exception as e:
                    logger.warning("Grid search error on %s ctx=%d: %s", tid, ctx, e)
                    continue

    # Compute mean scores
    mean_scores = {}
    for ctx, scores in config_scores.items():
        if scores:
            mean_scores[ctx] = float(np.mean(scores))
        else:
            mean_scores[ctx] = float("inf")

    best_ctx = min(mean_scores, key=mean_scores.get)  # type: ignore[arg-type]
    best_config = ForecastConfig(context_len=best_ctx)

    logger.info(
        "Grid search complete: best context_len=%d (score=%.4f)",
        best_ctx, mean_scores[best_ctx],
    )
    for ctx in sorted(mean_scores):
        logger.info("  ctx=%d: %.4f", ctx, mean_scores[ctx])

    # Cache to disk
    _DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULT_CONFIG_PATH, "w") as f:
        json.dump({
            "context_len": best_ctx,
            "mean_score": mean_scores[best_ctx],
            "all_scores": mean_scores,
            "n_tenants": n_tenants,
            "sla_tier": sla_tier.value,
        }, f, indent=2)

    return best_config


class FixedConfigTSFM:
    """
    TimesFM with a single globally-optimal fixed configuration.

    This baseline loads the best config found by grid search (or uses
    a specified config) and applies it to ALL tenants and ALL timestamps.
    No per-request adaptation — this is the status quo.

    Usage:
        # Option 1: run grid search
        config = find_best_fixed_config(client)
        forecaster = FixedConfigTSFM(client, config)

        # Option 2: load cached config
        forecaster = FixedConfigTSFM.from_cached(client)

        forecast = forecaster.forecast(history, horizon=60)
    """

    def __init__(self, client: TSFMClient, config: ForecastConfig):
        self._client = client
        self._config = config

    @classmethod
    def from_cached(
        cls,
        client: TSFMClient,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
    ) -> FixedConfigTSFM:
        """Load best config from a cached grid search result."""
        with open(config_path) as f:
            data = json.load(f)
        config = ForecastConfig(context_len=data["context_len"])
        return cls(client, config)

    @classmethod
    def from_grid_search(
        cls,
        client: TSFMClient,
        data_dir: str | Path = "data/synthetic",
        n_tenants: int = 20,
        **kwargs,
    ) -> FixedConfigTSFM:
        """Run grid search and create forecaster with best config."""
        config = find_best_fixed_config(client, data_dir, n_tenants, **kwargs)
        return cls(client, config)

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """
        Forecast using the fixed TimesFM configuration.

        Args:
            history: Shape (T, D).
            horizon: Number of steps to forecast.

        Returns:
            Point forecast, shape (horizon, D).
        """
        result = self._client.forecast(history, self._config, horizon)
        return result.point

    @property
    def config(self) -> ForecastConfig:
        return self._config
