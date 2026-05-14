"""
Experiment runner for the TimesFM autoresearch outer loop.

This is the MODIFIABLE file per the karpathy/autoresearch pattern.
The autonomous agent modifies this file to try different experiment
configurations, baselines, and hyperparameters.

Usage: uv run python autoresearch_agent/run_experiment.py

The script:
  1. Loads synthetic tenant data via infra.py
  2. Initializes the frozen TimesFM client
  3. Runs the configured experiment(s)
  4. Computes headline metrics (cost-asymmetric loss, MAE, latency)
  5. Logs results to results.tsv

MODIFY THIS FILE to explore:
  - Which experiment to run
  - How many tenants to evaluate
  - What baselines to compare against
  - Inner-loop K value
  - SLA tier distributions
  - Context length search ranges
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from autoresearch_agent.infra import (
    DATA_DIR,
    DEFAULT_HORIZON,
    DEFAULT_K,
    DEFAULT_N_TENANTS,
    RESULTS_PATH,
    SLA_TIERS,
    evaluate_cost_asymmetric,
    evaluate_mae,
    load_tenant_data,
    log_result,
    sample_tenants,
)
from tsfm_autoresearch.autoresearch import AutoresearchHarness
from tsfm_autoresearch.losses import SLATier
from tsfm_autoresearch.tsfm_client import TSFMClient

# ── Experiment Configuration ───────────────────────────────────────────
# MODIFY THESE to change what the experiment evaluates.

# Which experiment to run
CURRENT_EXPERIMENT = "02_fixed_vs_autoresearch"

# Number of tenants to evaluate
N_TENANTS = DEFAULT_N_TENANTS  # 200 for headline experiment

# Forecast horizon (minutes)
HORIZON = DEFAULT_HORIZON

# Number of configs per autoresearch request
K = DEFAULT_K

# SLA tier for this experiment
SLA_TIER = SLATier.STANDARD

# Sample timestamps per tenant: how many forecast points to evaluate
TIMESTAMPS_PER_TENANT = 50  # M6 default

# Description for results.tsv
DESCRIPTION = "M6 headline: fixed-config (ctx=256) vs autoresearch (K=8) on 200 tenants, standard SLA"


# ── Experiment Runner ──────────────────────────────────────────────────


def run_smoke_test(harness: AutoresearchHarness) -> dict:
    """
    Smoke test: run autoresearch on a few tenants, compare against
    a naive last-value baseline (M5 will formalize this).
    """
    tenant_ids = sample_tenants(N_TENANTS, seed=42)
    print(f"Evaluating {len(tenant_ids)} tenants, "
          f"{TIMESTAMPS_PER_TENANT} timestamps each, K={K}")

    all_autoresearch_losses: list[float] = []
    all_baseline_losses: list[float] = []
    all_latencies: list[float] = []

    for tid in tenant_ids:
        history = load_tenant_data(tid)
        T = history.shape[0]

        # Evaluate at multiple timestamps
        eval_points = np.linspace(
            HORIZON + 100,  # Skip early points (need enough history)
            T - HORIZON - 1,
            TIMESTAMPS_PER_TENANT,
            dtype=int,
        )

        for t in eval_points:
            # History up to time t
            hist_slice = history[:t, :]

            # Autoresearch forecast
            try:
                response = harness.forecast(
                    tenant_id=tid,
                    history=hist_slice,
                    horizon=HORIZON,
                    sla_tier=SLA_TIER,
                    K=K,
                )
                all_latencies.append(response.total_latency_ms)

                # Get actuals for comparison
                actuals = history[t : t + HORIZON, :]
                if actuals.shape[0] < HORIZON:
                    continue

                # Score autoresearch
                ar_loss = evaluate_cost_asymmetric(
                    actuals, response.final_forecast.point, SLA_TIER
                )
                all_autoresearch_losses.append(ar_loss)

                # Naive last-value baseline: predict last observed value
                last_value = hist_slice[-1, :]  # (D,)
                baseline_forecast = np.tile(last_value, (HORIZON, 1))  # (H, D)
                bl_loss = evaluate_cost_asymmetric(
                    actuals, baseline_forecast, SLA_TIER
                )
                all_baseline_losses.append(bl_loss)

            except Exception as e:
                print(f"  Error on {tid} @ t={t}: {e}")
                continue

    # Compute summary metrics
    ar_mean = float(np.mean(all_autoresearch_losses)) if all_autoresearch_losses else float("inf")
    bl_mean = float(np.mean(all_baseline_losses)) if all_baseline_losses else float("inf")
    mae_value = float(np.mean([abs(a - b) for a, b in zip(all_autoresearch_losses, all_baseline_losses)])) if all_autoresearch_losses else 0.0
    p50_lat = float(np.median(all_latencies)) if all_latencies else 0.0

    return {
        "cost_asym_loss": ar_mean,
        "mae": mae_value,
        "p50_latency_ms": p50_lat,
        "baseline_loss": bl_mean,
        "n_tenants": N_TENANTS,
        "n_evaluations": len(all_autoresearch_losses),
    }


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    print(f"=== autoresearch experiment: {CURRENT_EXPERIMENT} ===")
    print(f"Tenants: {N_TENANTS}, Horizon: {HORIZON}min, K: {K}, "
          f"SLA: {SLA_TIER.value}")
    print()

    # Load TimesFM (this is the slow part — one-time cost)
    print("Loading TimesFM 2.5 (frozen)...")
    t0 = time.perf_counter()
    client = TSFMClient(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=1,
        torch_compile=False,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Create harness
    harness = AutoresearchHarness(client, default_K=K, seed=42)
    print(f"  Harness ready: K={K}, val_split=15%")

    # Run experiment
    print(f"\nRunning: {CURRENT_EXPERIMENT}")
    t0 = time.perf_counter()

    if CURRENT_EXPERIMENT == "m3_smoke_test":
        results = run_smoke_test(harness)
    elif CURRENT_EXPERIMENT == "02_fixed_vs_autoresearch":
        # M6 headline experiment
        from baselines.fixed_config import FixedConfigTSFM, find_best_fixed_config
        from experiments.m6_headline import run_headline_experiment

        results = run_headline_experiment(
            client=client,
            n_tenants=N_TENANTS,
            horizon=HORIZON,
            timestamps_per_tenant=TIMESTAMPS_PER_TENANT,
            sla_tier=SLA_TIER,
            K=K,
            seed=42,
        )
        # run_headline_experiment handles its own logging
        print(f"\nExperiment completed in {time.perf_counter() - t0:.1f}s")
        print(f"\n---")
        for key, value in results.items():
            if isinstance(value, (float, int, str)):
                print(f"{str(key):30s}: {value}")
        return
    elif CURRENT_EXPERIMENT == "03_latency_budget_sweep":
        # M7 latency sweep
        from experiments.m7_latency_sweep import run_latency_sweep

        results = run_latency_sweep(
            client=client,
            n_tenants=N_TENANTS,
            horizon=HORIZON,
            timestamps_per_tenant=TIMESTAMPS_PER_TENANT,
            sla_tier=SLA_TIER,
            seed=42,
        )
        print(f"\nExperiment completed in {time.perf_counter() - t0:.1f}s")
        return
    elif CURRENT_EXPERIMENT == "08_cold_start_archetype":
        # M8 cold-start experiment
        from experiments.m8_cold_start import run_cold_start_experiment

        results = run_cold_start_experiment(
            client=client,
            n_tenants=N_TENANTS,
            horizon=HORIZON,
            history_lengths=[30, 60, 120, 240, 480],
            sla_tier=SLA_TIER,
            seed=42,
        )
        print(f"\nExperiment completed in {time.perf_counter() - t0:.1f}s")
        return
    elif CURRENT_EXPERIMENT == "04_retrieval_accuracy":
        # M4 archetype-retrieval accuracy (no TimesFM required)
        from experiments.m4_retrieval_accuracy import run_retrieval_accuracy

        results = run_retrieval_accuracy(centroid_frac=0.5, seed=42)
        print(f"\nExperiment completed in {time.perf_counter() - t0:.1f}s")
        return
    elif CURRENT_EXPERIMENT == "09_sla_tier_asymmetry":
        # M9 SLA asymmetry
        from experiments.m9_sla_asymmetry import run_sla_asymmetry_experiment

        results = run_sla_asymmetry_experiment(
            client=client,
            n_tenants=N_TENANTS,
            horizon=HORIZON,
            seed=42,
        )
        print(f"\nExperiment completed in {time.perf_counter() - t0:.1f}s")
        return
    else:
        print(f"Unknown experiment: {CURRENT_EXPERIMENT}")
        return

    elapsed = time.perf_counter() - t0
    print(f"\nExperiment completed in {elapsed:.1f}s")

    # Print summary
    print("\n---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
        else:
            print(f"{key:20s}: {value}")
    print(f"elapsed_seconds:    {elapsed:.1f}")

    # Log to results.tsv
    log_result(
        experiment=CURRENT_EXPERIMENT,
        cost_asym_loss=results["cost_asym_loss"],
        mae=results.get("mae", 0.0),
        p50_latency_ms=results.get("p50_latency_ms", 0.0),
        status="keep" if results["cost_asym_loss"] < results.get("baseline_loss", float("inf")) else "discard",
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    main()
