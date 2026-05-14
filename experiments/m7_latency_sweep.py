"""
M7 Latency Budget Sweep: K-config scaling experiment.

Measures how latency and forecast quality scale with the number of
configurations per autoresearch request. Validates the central claim
that K=8 achieves competitive loss within the 200ms latency budget.

Sweeps K ∈ {1, 2, 4, 8, 16, 32} across the synthetic tenant fleet
and reports:
  - Cost-asymmetric loss per K
  - p50/p95/p99 latency per K
  - Diminishing returns (Δloss / ΔK)
  - Latency budget feasibility (p95 under 200ms?)

Usage:
    uv run python experiments/03_latency_budget_sweep.py \
        --tenants 50 --horizon 60 --timestamps 5

Design note: This experiment runs authors without external baselines
since it's testing the internal scaling properties of the algorithm,
not comparing against alternatives (that's M6).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tsfm_autoresearch.autoresearch import AutoresearchHarness
from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss
from tsfm_autoresearch.tsfm_client import TSFMClient

from autoresearch_agent.infra import (
    DEFAULT_HORIZON,
    DEFAULT_K,
    DEFAULT_N_TENANTS,
    evaluate_cost_asymmetric,
    evaluate_mae,
    load_tenant_data,
    log_result,
    sample_tenants,
)

# ── Experiment Configuration ───────────────────────────────────────────

EXPERIMENT_NAME = "03_latency_budget_sweep"

# K values to sweep
K_SWEEP = [1, 2, 4, 8, 16, 32]

# Warmup forecast count before timing begins. The first TimesFM call after
# load (and, with torch_compile=True, the first call at each new batch
# shape) incurs JIT cost that would inflate p95 for the smallest K and
# distort the latency-vs-K curve.
WARMUP_FORECASTS = 3

# Timestamps per tenant (fewer than M6 — each tenant runs 6× K values)
DEFAULT_TIMESTAMPS = 5

# Fewer tenants — K=32 is expensive
DEFAULT_SWEEP_TENANTS = 50

# Minimum history required
MIN_HISTORY = 200


# ── Experiment Runner ──────────────────────────────────────────────────


def run_latency_sweep(
    client: TSFMClient,
    n_tenants: int = DEFAULT_SWEEP_TENANTS,
    horizon: int = DEFAULT_HORIZON,
    timestamps_per_tenant: int = DEFAULT_TIMESTAMPS,
    sla_tier: SLATier = SLATier.STANDARD,
    seed: int = 42,
) -> dict:
    """
    Sweep K values and measure latency + loss trade-off.

    Returns a dict with per-K statistics suitable for JSON export.
    """
    rng = np.random.default_rng(seed)
    tenant_ids = sample_tenants(n_tenants, seed=seed)
    loss_fn = CostAsymmetricLoss(alpha=sla_tier.alpha, per_resource=True)

    print(f"{'='*70}")
    print(f"  M7 LATENCY BUDGET SWEEP")
    print(f"{'='*70}")
    print(f"  Tenants: {n_tenants}")
    print(f"  Timestamps/tenant: {timestamps_per_tenant}")
    print(f"  Horizon: {horizon} min")
    print(f"  K sweep: {K_SWEEP}")
    print(f"  SLA Tier: {sla_tier.value} (α={sla_tier.alpha})")
    print(f"{'='*70}\n")

    # Per-K accumulators
    per_k: dict[int, dict[str, list[float]]] = {
        k: {"losses": [], "maes": [], "latencies": []}
        for k in K_SWEEP
    }

    # Per-stage timing for K=8 (the default, most interesting)
    stage_timing_accum: dict[str, list[float]] = {}

    total_paired_rows = 0

    # ── Warmup ────────────────────────────────────────────────────────
    # Drain JIT / compilation / GPU-context costs before timing. Use the
    # first usable tenant for warmup at every K value so per-K timing is
    # comparable from the first recorded row onward.
    print(f"\nWarming up ({WARMUP_FORECASTS} forecasts per K)...")
    for tid in tenant_ids:
        history = load_tenant_data(tid)
        T = history.shape[0]
        if T <= MIN_HISTORY + horizon:
            continue
        for k in K_SWEEP:
            warmup_harness = AutoresearchHarness(client, default_K=k, seed=seed)
            for _ in range(WARMUP_FORECASTS):
                try:
                    warmup_harness.forecast(
                        tenant_id=tid,
                        history=history[:MIN_HISTORY, :],
                        horizon=horizon,
                        sla_tier=sla_tier,
                        K=k,
                    )
                except Exception:
                    pass
        break
    print("  Done.\n")

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

            # ── Paired sweep across K values ─────────────────────────
            # Run every K, then commit the row only if ALL K succeed.
            # This keeps per-K loss/latency lists index-aligned across K
            # so paired diminishing-returns claims are valid (no drift
            # when K=32 fails on a row that K=1 happens to succeed on).
            row: dict[int, dict] = {}
            ok = True
            for k in K_SWEEP:
                try:
                    harness = AutoresearchHarness(client, default_K=k, seed=seed)
                    response = harness.forecast(
                        tenant_id=tid,
                        history=hist_slice,
                        horizon=horizon,
                        sla_tier=sla_tier,
                        K=k,
                    )
                    row[k] = {
                        "forecast": response.final_forecast.point,
                        "latency_ms": response.total_latency_ms,
                        "timing_ms": dict(response.timing_ms) if response.timing_ms else {},
                    }
                except Exception as e:
                    from logging import getLogger
                    getLogger(__name__).warning(
                        "K=%d failed on %s @ t=%d: %s", k, tid, t, e,
                    )
                    ok = False
                    break

            if not ok:
                continue

            for k in K_SWEEP:
                forecast = row[k]["forecast"]
                loss = loss_fn.point_loss(actuals, forecast)
                mae = evaluate_mae(actuals, forecast)
                per_k[k]["losses"].append(loss)
                per_k[k]["maes"].append(mae)
                per_k[k]["latencies"].append(row[k]["latency_ms"])
                if k == 8 and row[k]["timing_ms"]:
                    for stage, ms in row[k]["timing_ms"].items():
                        stage_timing_accum.setdefault(stage, []).append(ms)
            total_paired_rows += 1

        if (tidx + 1) % 10 == 0:
            print(f"  Progress: {tidx + 1}/{len(tenant_ids)} tenants "
                  f"({total_paired_rows} paired rows)")

    # ── Compute Summary Statistics ─────────────────────────────────────
    print(f"\n  Total paired rows: {total_paired_rows}")

    results: dict = {
        "experiment": EXPERIMENT_NAME,
        "n_tenants": n_tenants,
        "horizon": horizon,
        "sla_tier": sla_tier.value,
        "n_paired_rows": total_paired_rows,
        "warmup_forecasts": WARMUP_FORECASTS,
        "k_sweep": [],
    }

    print(f"\n{'─'*70}")
    print(f"  {'K':>4s}  {'Loss':>10s}  {'MAE':>10s}  "
          f"{'p50(ms)':>9s}  {'p95(ms)':>9s}  {'p99(ms)':>9s}  "
          f"{'ΔLoss':>8s}  {'ΔLoss/K':>10s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  "
          f"{'─'*9}  {'─'*9}  {'─'*9}  "
          f"{'─'*8}  {'─'*10}")

    prev_loss: float | None = None
    k_results = []

    for k in K_SWEEP:
        data = per_k[k]
        if not data["losses"]:
            continue

        losses = np.array(data["losses"])
        maes = np.array(data["maes"])
        lats = np.array(data["latencies"])

        mean_loss = float(np.mean(losses))
        mean_mae = float(np.mean(maes))
        p50 = float(np.median(lats))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))

        # Diminishing returns: improvement from previous K
        delta_loss = ""
        delta_per_k = ""
        if prev_loss is not None:
            delta = prev_loss - mean_loss
            delta_loss = f"{delta:+.6f}"
            # Δloss per additional config
            prev_k = k_results[-1]["k"] if k_results else 1
            dk = k - prev_k
            if dk > 0 and delta != 0:
                delta_per_k = f"{delta/dk:+.6f}"
            else:
                delta_per_k = "—"

        prev_loss = mean_loss

        k_entry = {
            "k": k,
            "mean_loss": mean_loss,
            "mean_mae": mean_mae,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "n_evaluations": len(losses),
            "budget_200ms_ok": p95 <= 200.0,
        }
        k_results.append(k_entry)

        budget_icon = "✓" if p95 <= 200 else "✗"
        print(f"  {k:>4d}  {mean_loss:>10.6f}  {mean_mae:>10.6f}  "
              f"{p50:>9.1f}  {p95:>9.1f}  {p99:>9.1f}  "
              f"{delta_loss:>8s}  {delta_per_k:>10s}  {budget_icon}")

    results["k_sweep"] = k_results

    # ── Stage Timing Breakdown (K=8) ────────────────────────────────────
    if stage_timing_accum:
        print(f"\n{'─'*70}")
        print(f"  Stage Timing Breakdown (K=8)")
        print(f"  {'Stage':<25s}  {'Mean(ms)':>10s}  {'p95(ms)':>10s}  {'%Total':>8s}")
        print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*8}")

        total_mean = sum(np.mean(v) for v in stage_timing_accum.values())
        stage_breakdown = {}
        for stage, values in sorted(stage_timing_accum.items()):
            mean_ms = float(np.mean(values))
            p95_ms = float(np.percentile(values, 95))
            pct = mean_ms / total_mean * 100 if total_mean > 0 else 0
            print(f"  {stage:<25s}  {mean_ms:>10.1f}  {p95_ms:>10.1f}  {pct:>7.1f}%")
            stage_breakdown[stage] = {
                "mean_ms": mean_ms,
                "p95_ms": p95_ms,
                "pct_total": round(pct, 1),
            }
        results["stage_timing_k8"] = stage_breakdown

    # ── Feasibility Assessment ─────────────────────────────────────────
    best_k = None
    for entry in k_results:
        if entry["budget_200ms_ok"]:
            best_k = entry["k"]
    results["max_k_within_200ms"] = best_k

    print(f"\n{'='*70}")
    print(f"  FEASIBILITY")
    print(f"{'='*70}")
    print(f"  Max K within 200ms budget: {best_k if best_k else 'NONE'}")
    if best_k:
        for entry in k_results:
            if entry["k"] == best_k:
                print(f"  Loss at K={best_k}:  {entry['mean_loss']:.6f}")
                print(f"  p95 at K={best_k}:  {entry['p95_latency_ms']:.0f}ms")
    print(f"{'='*70}")

    # ── Log to results.tsv ────────────────────────────────────────────
    description = (
        f"M7: latency sweep K∈{K_SWEEP} on {n_tenants} tenants, "
        f"{sla_tier.value} SLA. "
        f"Max K within 200ms: {best_k}. "
        f"Loss range: {k_results[0]['mean_loss']:.6f} → {k_results[-1]['mean_loss']:.6f}"
    )

    # Use the best K's loss as the headline metric
    headline_loss = k_results[-1]["mean_loss"]
    headline_p50 = k_results[-1]["p50_latency_ms"]

    log_result(
        experiment=EXPERIMENT_NAME,
        cost_asym_loss=headline_loss,
        mae=k_results[-1]["mean_mae"],
        p50_latency_ms=headline_p50,
        status="keep",
        description=description,
    )

    return results


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M7 Latency Budget Sweep: K-config scaling experiment",
    )
    parser.add_argument(
        "--tenants", type=int, default=DEFAULT_SWEEP_TENANTS,
        help=f"Tenants to evaluate (default: {DEFAULT_SWEEP_TENANTS})",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Forecast horizon in minutes (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--timestamps", type=int, default=DEFAULT_TIMESTAMPS,
        help=f"Timestamps per tenant (default: {DEFAULT_TIMESTAMPS})",
    )
    parser.add_argument(
        "--sla", type=str, default="standard",
        choices=["premium", "standard", "basic"],
        help="SLA tier (default: standard)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed",
    )
    args = parser.parse_args()

    sla_tier_map = {
        "premium": SLATier.PREMIUM,
        "standard": SLATier.STANDARD,
        "basic": SLATier.BASIC,
    }
    sla_tier = sla_tier_map[args.sla]

    print("Loading TimesFM 2.5 (frozen)...")
    t0 = time.perf_counter()
    client = TSFMClient(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=1,
        torch_compile=False,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    results = run_latency_sweep(
        client=client,
        n_tenants=args.tenants,
        horizon=args.horizon,
        timestamps_per_tenant=args.timestamps,
        sla_tier=sla_tier,
        seed=args.seed,
    )

    # Save results
    output_path = Path("results/03_latency_budget_sweep.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
