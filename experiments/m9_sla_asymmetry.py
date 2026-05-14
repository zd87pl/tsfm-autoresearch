"""
M9 SLA Tier Asymmetry: validate that α controls forecast bias.

The thesis claims that different SLA tiers produce measurably different
allocation behavior:
  - Premium (α=0.90): 90% weight on under-prediction → protective (higher)
  - Standard (α=0.75): balanced
  - Basic (α=0.65): 65% weight on under-prediction → cost-efficient (lower)

Validates that:
  1. Premium forecasts > Standard forecasts > Basic forecasts (systematic)
  2. Premium selects longer context_len configs (more history = safer)
  3. Config distributions shift by SLA tier
  4. The asymmetry is measurable and monotonic in α

Usage:
    uv run python experiments/m9_sla_asymmetry.py --tenants 100 --horizon 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np

from tsfm_autoresearch.autoresearch import AutoresearchHarness
from tsfm_autoresearch.losses import CostAsymmetricLoss, SLATier
from tsfm_autoresearch.tsfm_client import TSFMClient

from autoresearch_agent.prepare import (
    DEFAULT_HORIZON,
    DEFAULT_K,
    load_tenant_data,
    load_manifest,
    log_result,
    sample_tenants,
)

EXPERIMENT_NAME = "09_sla_tier_asymmetry"
MIN_HISTORY = 200
DEFAULT_TIMESTAMPS = 5


def run_sla_asymmetry_experiment(
    client: TSFMClient,
    n_tenants: int = 100,
    horizon: int = DEFAULT_HORIZON,
    timestamps_per_tenant: int = DEFAULT_TIMESTAMPS,
    K: int = DEFAULT_K,
    seed: int = 42,
) -> dict:
    """Run SLA tier comparison across premium/standard/basic."""

    rng = np.random.default_rng(seed)
    tenant_ids = sample_tenants(n_tenants, seed=seed)
    manifest = load_manifest()

    tiers = [SLATier.PREMIUM, SLATier.STANDARD, SLATier.BASIC]

    print(f"{'='*70}")
    print(f"  M9 SLA TIER ASYMMETRY")
    print(f"{'='*70}")
    print(f"  Tenants: {n_tenants}")
    print(f"  Horizon: {horizon} min, K: {K}")
    print(f"  Tiers: {[t.value for t in tiers]}")
    print(f"{'='*70}\n")

    # Per-tier accumulators (parallel-indexed across paired rows).
    per_tier: dict[str, dict] = {
        t.value: {
            "forecast_means": [],
            "config_context_lens": [],
            "latencies": [],
            # Direct loss-asymmetry validation: each tier's forecast scored
            # under EVERY tier's α. If a tier's forecast is the right
            # response to a particular α, then forecast_premium should
            # minimize the loss when scored at α=0.90, etc.
            "loss_under_premium_alpha": [],
            "loss_under_basic_alpha": [],
            # Decomposed under/over-prediction error (paired).
            "under_pred_error": [],
            "over_pred_error": [],
        }
        for t in tiers
    }

    premium_loss_fn = CostAsymmetricLoss(alpha=SLATier.PREMIUM.alpha, per_resource=True)
    basic_loss_fn = CostAsymmetricLoss(alpha=SLATier.BASIC.alpha, per_resource=True)

    total_evals = 0
    paired_rows = 0

    for tidx, tid in enumerate(tenant_ids):
        history = load_tenant_data(tid)
        T = history.shape[0]

        if T <= MIN_HISTORY + horizon:
            continue

        eval_candidates = np.arange(MIN_HISTORY, T - horizon)
        if len(eval_candidates) == 0:
            continue

        n_pts = min(timestamps_per_tenant, len(eval_candidates))
        eval_points = rng.choice(eval_candidates, size=n_pts, replace=False)
        eval_points.sort()

        for t in eval_points:
            hist_slice = history[:t, :]
            actuals = history[t : t + horizon, :]
            if actuals.shape[0] < horizon:
                continue

            # Run all tiers; require all three to succeed before recording
            # the row, so per-tier lists stay paired.
            row: dict[str, dict] = {}
            ok = True
            for tier in tiers:
                try:
                    harness = AutoresearchHarness(client, default_K=K, seed=seed)
                    response = harness.forecast(
                        tenant_id=tid, history=hist_slice, horizon=horizon,
                        sla_tier=tier, K=K,
                    )
                    row[tier.value] = {
                        "forecast": response.final_forecast.point,
                        "ctx": response.winning_config.context_len,
                        "latency": response.total_latency_ms,
                    }
                except Exception:
                    ok = False
                    break

            if not ok:
                continue

            paired_rows += 1
            for tier in tiers:
                tn = tier.value
                forecast = row[tn]["forecast"]
                err = actuals - forecast
                under = float(np.mean(np.maximum(0, err)))
                over = float(np.mean(np.maximum(0, -err)))
                per_tier[tn]["forecast_means"].append(float(np.mean(forecast)))
                per_tier[tn]["config_context_lens"].append(row[tn]["ctx"])
                per_tier[tn]["latencies"].append(row[tn]["latency"])
                per_tier[tn]["under_pred_error"].append(under)
                per_tier[tn]["over_pred_error"].append(over)
                per_tier[tn]["loss_under_premium_alpha"].append(
                    premium_loss_fn.point_loss(actuals, forecast)
                )
                per_tier[tn]["loss_under_basic_alpha"].append(
                    basic_loss_fn.point_loss(actuals, forecast)
                )
                total_evals += 1

        if (tidx + 1) % 20 == 0:
            print(f"  Progress: {tidx + 1}/{len(tenant_ids)} tenants "
                  f"({paired_rows} paired rows, {total_evals} forecasts)")

    print(f"\n  Total evaluations: {total_evals}")

    # ── Compute statistics ──────────────────────────────────────────
    results: dict = {
        "experiment": EXPERIMENT_NAME,
        "n_tenants": n_tenants,
        "horizon": horizon,
        "k": K,
        "n_evaluations": total_evals,
        "tiers": {},
    }

    print(f"\n{'─'*76}")
    print(f"  {'Tier':<12s}  {'Mean Forecast':>15s}  "
          f"{'Median Ctx':>12s}  {'p50 Lat(ms)':>12s}  "
          f"{'Config Profile':>20s}")
    print(f"  {'─'*12}  {'─'*15}  {'─'*12}  {'─'*12}  {'─'*20}")

    tiers_ordered = ["premium", "standard", "basic"]
    means = {}

    for tier_name in tiers_ordered:
        data = per_tier[tier_name]
        if not data["forecast_means"]:
            continue

        fmeans = np.array(data["forecast_means"])
        ctxs = np.array(data["config_context_lens"])
        lats = np.array(data["latencies"])

        mean_f = float(np.mean(fmeans))
        median_ctx = float(np.median(ctxs))
        p50_lat = float(np.median(lats)) if len(lats) > 0 else 0.0

        # Context length distribution
        ctx_counts = Counter(int(c) for c in ctxs)
        ctx_profile = ", ".join(
            f"{c}={ctx_counts[c]}" for c in sorted(ctx_counts)
        )

        means[tier_name] = mean_f

        print(f"  {tier_name:<12s}  {mean_f:>15.6f}  "
              f"{median_ctx:>12.0f}  {p50_lat:>12.0f}  "
              f"{ctx_profile:>20s}")

        results["tiers"][tier_name] = {
            "mean_forecast": mean_f,
            "median_context_len": median_ctx,
            "p50_latency_ms": p50_lat,
            "context_len_distribution": dict(ctx_counts),
            "n_evaluations": len(fmeans),
        }

    # ── Monotonicity check ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MONOTONICITY CHECK")
    print(f"{'='*70}")

    checks = []
    if means.get("premium", 0) > means.get("standard", 0):
        print(f"  ✓ Premium > Standard: {means['premium']:.6f} > "
              f"{means['standard']:.6f}")
        checks.append("premium_gt_standard")
    else:
        print(f"  ✗ Premium > Standard: FAILED")

    if means.get("standard", 0) > means.get("basic", 0):
        print(f"  ✓ Standard > Basic:   {means['standard']:.6f} > "
              f"{means['basic']:.6f}")
        checks.append("standard_gt_basic")
    else:
        print(f"  ✗ Standard > Basic:   FAILED")

    if means.get("premium", 0) > means.get("basic", 0):
        print(f"  ✓ Premium > Basic:    {means['premium']:.6f} > "
              f"{means['basic']:.6f}")
        checks.append("premium_gt_basic")
    else:
        print(f"  ✗ Premium > Basic:    FAILED")

    # Config shift check: premium should prefer longer contexts
    prem_ctx = results["tiers"].get("premium", {}).get("median_context_len", 0)
    basic_ctx = results["tiers"].get("basic", {}).get("median_context_len", 0)
    if prem_ctx >= basic_ctx:
        print(f"  ✓ Premium ctx >= Basic ctx: {prem_ctx:.0f} >= "
              f"{basic_ctx:.0f}")
        checks.append("premium_ctx_ge_basic")
    else:
        print(f"  ✗ Premium ctx >= Basic ctx: FAILED")

    # ── Direct loss-asymmetry validation ─────────────────────────────
    # The thesis claim is about COST-ASYMMETRIC LOSS, not forecast level.
    # Validate directly: scored under α=0.90 (premium), the premium
    # forecast should beat the basic forecast (and vice-versa under α=0.65).
    print(f"\n  Direct loss-asymmetry checks (paired):")
    direct = {}
    if (per_tier["premium"]["loss_under_premium_alpha"]
            and per_tier["basic"]["loss_under_premium_alpha"]):
        prem_at_prem_alpha = float(
            np.mean(per_tier["premium"]["loss_under_premium_alpha"])
        )
        basic_at_prem_alpha = float(
            np.mean(per_tier["basic"]["loss_under_premium_alpha"])
        )
        prem_at_basic_alpha = float(
            np.mean(per_tier["premium"]["loss_under_basic_alpha"])
        )
        basic_at_basic_alpha = float(
            np.mean(per_tier["basic"]["loss_under_basic_alpha"])
        )
        direct = {
            "premium_forecast_at_premium_alpha": prem_at_prem_alpha,
            "basic_forecast_at_premium_alpha": basic_at_prem_alpha,
            "premium_forecast_at_basic_alpha": prem_at_basic_alpha,
            "basic_forecast_at_basic_alpha": basic_at_basic_alpha,
        }
        if prem_at_prem_alpha < basic_at_prem_alpha:
            print(f"  ✓ Premium beats basic under α=0.90: "
                  f"{prem_at_prem_alpha:.6f} < {basic_at_prem_alpha:.6f}")
            checks.append("premium_wins_at_premium_alpha")
        else:
            print(f"  ✗ Premium beats basic under α=0.90: FAILED "
                  f"({prem_at_prem_alpha:.6f} vs {basic_at_prem_alpha:.6f})")
        if basic_at_basic_alpha < prem_at_basic_alpha:
            print(f"  ✓ Basic beats premium under α=0.65: "
                  f"{basic_at_basic_alpha:.6f} < {prem_at_basic_alpha:.6f}")
            checks.append("basic_wins_at_basic_alpha")
        else:
            print(f"  ✗ Basic beats premium under α=0.65: FAILED "
                  f"({basic_at_basic_alpha:.6f} vs {prem_at_basic_alpha:.6f})")

    results["direct_loss_asymmetry"] = direct
    results["monotonicity_checks"] = checks
    expected_total = 6
    results["all_checks_pass"] = len(checks) == expected_total

    print(f"  {len(checks)}/{expected_total} checks passed")
    print(f"{'='*70}")

    # ── Log ─────────────────────────────────────────────────────────
    spread = means.get("premium", 0) - means.get("basic", 0)
    description = (
        f"M9: SLA asymmetry on {n_tenants} tenants. "
        f"Premium={means.get('premium', 0):.6f} > "
        f"Standard={means.get('standard', 0):.6f} > "
        f"Basic={means.get('basic', 0):.6f}. "
        f"Spread: {spread:.6f}. "
        f"{len(checks)}/6 monotonicity + direct-loss checks pass."
    )
    log_result(
        experiment=EXPERIMENT_NAME,
        cost_asym_loss=0.0,  # not a loss experiment
        mae=0.0,
        p50_latency_ms=results["tiers"].get("standard", {}).get("p50_latency_ms", 0),
        status="keep" if results["all_checks_pass"] else "discard",
        description=description,
    )

    return results


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M9 SLA Tier Asymmetry experiment",
    )
    parser.add_argument("--tenants", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--timestamps", type=int, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading TimesFM 2.5 (frozen)...")
    t0 = time.perf_counter()
    client = TSFMClient(
        max_context=512, max_horizon=128,
        per_core_batch_size=1, torch_compile=False,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    results = run_sla_asymmetry_experiment(
        client=client,
        n_tenants=args.tenants,
        horizon=args.horizon,
        timestamps_per_tenant=args.timestamps,
    )

    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_convert(v) for v in obj]
        return obj

    output_path = Path("results/09_sla_asymmetry.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
