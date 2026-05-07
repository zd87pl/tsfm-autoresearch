"""
M6 Headline Experiment: Fixed-Config vs Autoresearch.

Empirically validates the core thesis:
  "A per-request autoresearch loop over a frozen TimesFM achieves better
   cost-asymmetric performance than any single fixed configuration."

Compares four forecasters on the synthetic tenant fleet:
  1. FixedConfigTSFM (best single context_len via grid search)
  2. AutoresearchHarness (per-request K-config search)
  3. NaiveLast (trivial lower bound)
  4. NaiveSeasonal (exploits diurnal patterns)
  5. PerTenantARIMA (per-tenant stateful upper bound — slow by design)

Reports: cost-asymmetric loss, win rate, per-archetype breakdown, latency.

Usage:
    uv run python experiments/02_fixed_vs_autoresearch.py \
        --tenants 200 --horizon 60 --timestamps 50 --sla standard
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from scipy.stats import binomtest

from baselines.fixed_config import FixedConfigTSFM
from baselines.naive_last import NaiveLast
from baselines.naive_seasonal import NaiveSeasonal
from baselines.per_tenant_arima import PerTenantARIMA
from baselines.protocol import Forecaster
from tsfm_autoresearch.autoresearch import AutoresearchHarness
from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss
from tsfm_autoresearch.tsfm_client import ForecastConfig, TSFMClient

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

EXPERIMENT_NAME = "02_fixed_vs_autoresearch"

# Per-tenant timestamps to evaluate
DEFAULT_TIMESTAMPS = 50

# Minimum history required before forecasting
MIN_HISTORY = 200


# ── Experiment Runner ──────────────────────────────────────────────────


def _run_single_forecaster(
    forecaster, tid: str, hist_slice: np.ndarray, horizon: int,
    sla_tier: SLATier, K: int,
):
    """Dispatch to the right API; return (forecast, latency_ms) or None on failure."""
    try:
        if isinstance(forecaster, AutoresearchHarness):
            response = forecaster.forecast(
                tenant_id=tid, history=hist_slice, horizon=horizon,
                sla_tier=sla_tier, K=K,
            )
            return response.final_forecast.point, response.total_latency_ms
        else:
            t0 = time.perf_counter()
            forecast = forecaster.forecast(hist_slice, horizon)
            return forecast, (time.perf_counter() - t0) * 1000
    except Exception as e:
        from logging import getLogger
        getLogger(__name__).warning(
            "Forecaster %s failed on %s: %s",
            type(forecaster).__name__, tid, e,
        )
        return None


def evaluate_paired(
    forecasters: dict,
    tenant_ids: list[str],
    horizon: int,
    timestamps_per_tenant: int,
    sla_tier: SLATier,
    K: int,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Paired evaluation: every forecaster scored at the same (tenant, t) pairs.
    If any forecaster fails on a pair, the pair is dropped for ALL forecasters,
    so per-index lists stay aligned and downstream paired tests are valid.
    """
    rng = np.random.default_rng(seed)
    loss_fn = CostAsymmetricLoss(alpha=sla_tier.alpha, per_resource=True)

    out: dict[str, dict] = {
        name: {"losses": [], "maes": [], "latencies": []}
        for name in forecasters
    }
    pairs: list[tuple[str, int]] = []

    for tid in tenant_ids:
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

            results = {}
            ok = True
            for name, fc in forecasters.items():
                res = _run_single_forecaster(fc, tid, hist_slice, horizon, sla_tier, K)
                if res is None:
                    ok = False
                    break
                results[name] = res
            if not ok:
                continue

            pairs.append((tid, int(t)))
            for name, (forecast, elapsed) in results.items():
                out[name]["losses"].append(loss_fn.point_loss(actuals, forecast))
                out[name]["maes"].append(evaluate_mae(actuals, forecast))
                out[name]["latencies"].append(elapsed)

    return {"pairs": pairs, **out}


def evaluate_forecaster(
    forecaster: Forecaster,
    tenant_ids: list[str],
    horizon: int,
    timestamps_per_tenant: int,
    sla_tier: SLATier,
    seed: int = 42,
) -> tuple[list[float], list[float], list[float]]:
    """Single-forecaster wrapper retained for naive baseline calls."""
    paired = evaluate_paired(
        {"_": forecaster}, tenant_ids, horizon, timestamps_per_tenant,
        sla_tier, K=1, seed=seed,
    )
    d = paired["_"]
    return d["losses"], d["maes"], d["latencies"]


def evaluate_autoresearch(
    harness: AutoresearchHarness,
    tenant_ids: list[str],
    horizon: int,
    timestamps_per_tenant: int,
    sla_tier: SLATier,
    K: int,
    seed: int = 42,
) -> tuple[list[float], list[float], list[float]]:
    """Single-forecaster wrapper retained for backwards compatibility."""
    paired = evaluate_paired(
        {"_": harness}, tenant_ids, horizon, timestamps_per_tenant,
        sla_tier, K=K, seed=seed,
    )
    d = paired["_"]
    return d["losses"], d["maes"], d["latencies"]


def per_archetype_breakdown(
    forecaster: Forecaster,
    tenant_ids: list[str],
    horizon: int,
    timestamps_per_tenant: int,
    sla_tier: SLATier,
    manifest: dict[str, str],
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute per-archetype cost-asymmetric loss for a forecaster.

    Returns:
        {archetype_name: {"mean_loss": ..., "n_evals": ..., "p50_latency_ms": ...}}
    """
    rng = np.random.default_rng(seed)
    loss_fn = CostAsymmetricLoss(alpha=sla_tier.alpha, per_resource=True)

    archetype_losses: dict[str, list[float]] = {}
    archetype_latencies: dict[str, list[float]] = {}

    for tid in tenant_ids:
        arch = manifest.get(tid, "unknown")
        history = load_tenant_data(tid)
        T = history.shape[0]

        if T <= MIN_HISTORY + horizon:
            continue

        eval_candidates = np.arange(MIN_HISTORY, T - horizon)
        if len(eval_candidates) == 0:
            continue

        n_pts = min(max(5, timestamps_per_tenant // 10), len(eval_candidates))
        eval_points = rng.choice(eval_candidates, size=n_pts, replace=False)
        eval_points.sort()

        for t in eval_points:
            hist_slice = history[:t, :]
            actuals = history[t : t + horizon, :]

            if actuals.shape[0] < horizon:
                continue

            try:
                t0 = time.perf_counter()
                forecast = forecaster.forecast(hist_slice, horizon)
                elapsed = (time.perf_counter() - t0) * 1000

                loss = loss_fn.point_loss(actuals, forecast)
                archetype_losses.setdefault(arch, []).append(loss)
                archetype_latencies.setdefault(arch, []).append(elapsed)
            except Exception:
                continue

    breakdown: dict[str, dict[str, float]] = {}
    for arch in sorted(archetype_losses.keys()):
        arch_losses = archetype_losses[arch]
        arch_lats = archetype_latencies[arch]
        breakdown[arch] = {
            "mean_loss": float(np.mean(arch_losses)),
            "n_evals": len(arch_losses),
            "p50_latency_ms": float(np.median(arch_lats)) if arch_lats else 0.0,
        }

    return breakdown


def compute_win_rate(
    ar_losses: list[float],
    baseline_losses: list[float],
) -> dict:
    """
    Paired win-rate with a two-sided binomial significance test.

    Caller must guarantee `ar_losses[i]` and `baseline_losses[i]` come from
    the SAME (tenant, timestamp) pair (use `evaluate_paired` for this).
    Excludes ties from the binomial denominator (sign-test convention).
    """
    if len(ar_losses) != len(baseline_losses):
        raise ValueError(
            f"Paired losses must have equal length: "
            f"{len(ar_losses)} vs {len(baseline_losses)}"
        )
    n_total = len(ar_losses)
    if n_total == 0:
        return {"win_rate": 0.0, "n": 0, "wins": 0, "losses": 0,
                "ties": 0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 1.0}

    wins = sum(1 for a, b in zip(ar_losses, baseline_losses) if a < b)
    losses_n = sum(1 for a, b in zip(ar_losses, baseline_losses) if a > b)
    ties = n_total - wins - losses_n
    decided = wins + losses_n

    if decided == 0:
        return {"win_rate": 0.5, "n": n_total, "wins": 0, "losses": 0,
                "ties": ties, "p_value": 1.0, "ci_low": 0.0, "ci_high": 1.0}

    test = binomtest(wins, decided, p=0.5, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95)
    return {
        "win_rate": wins / decided,
        "n": n_total,
        "wins": wins,
        "losses": losses_n,
        "ties": ties,
        "p_value": float(test.pvalue),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
    }


def run_headline_experiment(
    client: TSFMClient,
    n_tenants: int = DEFAULT_N_TENANTS,
    horizon: int = DEFAULT_HORIZON,
    timestamps_per_tenant: int = DEFAULT_TIMESTAMPS,
    sla_tier: SLATier = SLATier.STANDARD,
    K: int = DEFAULT_K,
    seed: int = 42,
) -> dict:
    """Run the full headline experiment."""
    from autoresearch_agent.infra import load_manifest

    print(f"{'='*70}")
    print(f"  M6 HEADLINE EXPERIMENT: Fixed-Config vs Autoresearch")
    print(f"{'='*70}")
    print(f"  Tenants: {n_tenants}")
    print(f"  Timestamps/tenant: {timestamps_per_tenant}")
    print(f"  Horizon: {horizon} min")
    print(f"  K: {K}")
    print(f"  SLA Tier: {sla_tier.value} (α={sla_tier.alpha})")
    print(f"{'='*70}\n")

    # ── Setup ────────────────────────────────────────────────────────
    manifest = load_manifest()
    tenant_ids = sample_tenants(n_tenants, seed=seed)

    # Build forecasters. Grid-search the FixedConfig baseline on a tenant
    # set DISJOINT from the evaluation set so the baseline is not tuned on
    # the same data it is later scored on.
    print("Grid search for best fixed config (held-out tenant split)...")
    t0 = time.perf_counter()
    best_config = ForecastConfig(context_len=256)  # Default; grid search requires data

    try:
        from baselines.fixed_config import find_best_fixed_config
        best_config = find_best_fixed_config(
            client, n_tenants=min(20, n_tenants), horizon=horizon,
            sla_tier=sla_tier, seed=seed + 1,
            exclude_tenant_ids=tenant_ids,
        )
    except Exception as e:
        # Grid search may fail if data dir doesn't exist or TimesFM unavailable
        best_config = ForecastConfig(context_len=256)
        print(f"  Grid search skipped ({type(e).__name__}: {e}), using context_len=256")

    fixed_tsfm = FixedConfigTSFM(client, best_config)
    harness = AutoresearchHarness(client, default_K=K, seed=seed)

    print(f"  Fixed config: context_len={best_config.context_len}")
    print(f"  Harness: K={K}, val_split=15%")
    print()

    results: dict = {
        "experiment": EXPERIMENT_NAME,
        "n_tenants": n_tenants,
        "horizon": horizon,
        "K": K,
        "sla_tier": sla_tier.value,
        "config_context_len": best_config.context_len,
    }

    # ── Naive baselines (fast, run first for sanity) ──────────────────
    print("Evaluating NaiveLast...")
    t0 = time.perf_counter()
    naive_losses, naive_maes, naive_lats = evaluate_forecaster(
        NaiveLast(), tenant_ids, horizon, timestamps_per_tenant, sla_tier, seed,
    )
    print(f"  Loss: {np.mean(naive_losses):.6f} ± {np.std(naive_losses):.6f}")
    print(f"  MAE:  {np.mean(naive_maes):.6f}")
    print(f"  N:    {len(naive_losses)} evaluations")
    print(f"  Time: {time.perf_counter() - t0:.1f}s\n")
    results["naive_loss"] = float(np.mean(naive_losses)) if naive_losses else float("inf")

    print("Evaluating NaiveSeasonal...")
    t0 = time.perf_counter()
    seas_losses, seas_maes, seas_lats = evaluate_forecaster(
        NaiveSeasonal(), tenant_ids, horizon, timestamps_per_tenant, sla_tier, seed,
    )
    print(f"  Loss: {np.mean(seas_losses):.6f} ± {np.std(seas_losses):.6f}")
    print(f"  MAE:  {np.mean(seas_maes):.6f}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s\n")
    results["seasonal_loss"] = float(np.mean(seas_losses)) if seas_losses else float("inf")

    # ── Headline pair: FixedConfigTSFM vs AutoresearchHarness ────────
    # Paired so wins/losses are computed on the same (tenant, t) pairs.
    print("Evaluating FixedConfigTSFM and AutoresearchHarness (paired)...")
    t0 = time.perf_counter()
    paired = evaluate_paired(
        {"fixed": fixed_tsfm, "ar": harness},
        tenant_ids, horizon, timestamps_per_tenant, sla_tier, K, seed,
    )
    fixed_losses = paired["fixed"]["losses"]
    fixed_maes = paired["fixed"]["maes"]
    fixed_lats = paired["fixed"]["latencies"]
    ar_losses = paired["ar"]["losses"]
    ar_maes = paired["ar"]["maes"]
    ar_lats = paired["ar"]["latencies"]
    n_paired = len(paired["pairs"])
    print(f"  Paired evaluations: {n_paired}")
    print(f"  FixedConfigTSFM:")
    print(f"    Loss:    {np.mean(fixed_losses):.6f} ± {np.std(fixed_losses):.6f}")
    print(f"    MAE:     {np.mean(fixed_maes):.6f}")
    print(f"    Latency: p50={np.median(fixed_lats):.0f}ms p95={np.percentile(fixed_lats, 95):.0f}ms")
    print(f"  AutoresearchHarness:")
    print(f"    Loss:    {np.mean(ar_losses):.6f} ± {np.std(ar_losses):.6f}")
    print(f"    MAE:     {np.mean(ar_maes):.6f}")
    print(f"    Latency: p50={np.median(ar_lats):.0f}ms p95={np.percentile(ar_lats, 95):.0f}ms")
    print(f"  Time: {time.perf_counter() - t0:.1f}s\n")
    results["n_paired"] = n_paired
    results["fixed_loss"] = float(np.mean(fixed_losses)) if fixed_losses else float("inf")
    results["fixed_p50_latency_ms"] = float(np.median(fixed_lats)) if fixed_lats else 0.0
    results["autoresearch_loss"] = float(np.mean(ar_losses)) if ar_losses else float("inf")
    results["autoresearch_p50_latency_ms"] = float(np.median(ar_lats)) if ar_lats else 0.0

    # ── Win Rate (paired binomial test) ──────────────────────────────
    wr = compute_win_rate(ar_losses, fixed_losses)
    print(
        f"Win rate (autoresearch beats fixed-config): "
        f"{wr['win_rate']:.2%} "
        f"[{wr['ci_low']:.2%}, {wr['ci_high']:.2%}]  "
        f"wins={wr['wins']} losses={wr['losses']} ties={wr['ties']}  "
        f"p={wr['p_value']:.4g}"
    )
    win_rate = wr["win_rate"]
    results["win_rate"] = wr["win_rate"]
    results["win_rate_stats"] = wr

    # ── Per-Archetype Breakdown ──────────────────────────────────────
    print("\nPer-Archetype Breakdown (cost-asymmetric loss, lower is better):")
    print(f"  {'Archetype':<25s} {'Fixed':>10s} {'AR':>10s} {'Δ%':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

    ar_breakdown = per_archetype_breakdown(
        harness, tenant_ids, horizon, timestamps_per_tenant,
        sla_tier, manifest, seed,
    )
    fixed_breakdown = per_archetype_breakdown(
        fixed_tsfm, tenant_ids, horizon, timestamps_per_tenant,
        sla_tier, manifest, seed,
    )

    per_arch: dict[str, dict[str, float]] = {}
    all_arches = sorted(set(ar_breakdown.keys()) | set(fixed_breakdown.keys()))
    for arch in all_arches:
        ar_loss = ar_breakdown.get(arch, {}).get("mean_loss", float("inf"))
        fixed_loss = fixed_breakdown.get(arch, {}).get("mean_loss", float("inf"))
        delta_pct = ((fixed_loss - ar_loss) / fixed_loss * 100) if fixed_loss != float("inf") else 0.0
        sign = "+" if delta_pct > 0 else ""
        print(f"  {arch:<25s} {fixed_loss:>10.6f} {ar_loss:>10.6f} {sign}{delta_pct:>7.1f}%")
        per_arch[arch] = {
            "fixed_loss": fixed_loss,
            "ar_loss": ar_loss,
            "delta_pct": delta_pct,
        }
    results["per_archetype"] = per_arch

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Fixed-Config Loss:     {results['fixed_loss']:.6f}")
    print(f"  Autoresearch Loss:     {results['autoresearch_loss']:.6f}")
    improvement = (
        (results["fixed_loss"] - results["autoresearch_loss"])
        / results["fixed_loss"] * 100
        if results["fixed_loss"] != float("inf")
        else 0.0
    )
    print(f"  Improvement:           {improvement:+.1f}%")
    print(f"  Win Rate:              {win_rate:.2%}  "
          f"95% CI [{wr['ci_low']:.2%}, {wr['ci_high']:.2%}]  "
          f"p={wr['p_value']:.4g}  n={wr['wins']+wr['losses']}")
    print(f"  Fixed p50 Latency:     {results['fixed_p50_latency_ms']:.0f}ms")
    print(f"  AR p50 Latency:        {results['autoresearch_p50_latency_ms']:.0f}ms")
    print(f"{'='*70}")

    # ── Log to results.tsv ───────────────────────────────────────────
    description = (
        f"M6: fixed-config (ctx={best_config.context_len}) vs autoresearch (K={K}) "
        f"on {n_tenants} tenants, {sla_tier.value} SLA. "
        f"Win rate: {win_rate:.2%}, improvement: {improvement:+.1f}%"
    )
    log_result(
        experiment=EXPERIMENT_NAME,
        cost_asym_loss=results["autoresearch_loss"],
        mae=float(np.mean(ar_maes)) if ar_maes else 0.0,
        p50_latency_ms=results["autoresearch_p50_latency_ms"],
        status="keep" if win_rate > 0.5 else "discard",
        description=description,
    )

    return results


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M6 Headline Experiment: Fixed-Config vs Autoresearch",
    )
    parser.add_argument(
        "--tenants", type=int, default=DEFAULT_N_TENANTS,
        help=f"Number of tenants to evaluate (default: {DEFAULT_N_TENANTS})",
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
        help="SLA tier for cost-asymmetric scoring (default: standard)",
    )
    parser.add_argument(
        "--K", type=int, default=DEFAULT_K,
        help=f"Configs per autoresearch request (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU (default: auto-detect)",
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

    results = run_headline_experiment(
        client=client,
        n_tenants=args.tenants,
        horizon=args.horizon,
        timestamps_per_tenant=args.timestamps,
        sla_tier=sla_tier,
        K=args.K,
        seed=args.seed,
    )

    # Save results JSON for programmatic consumption
    output_path = Path("results/02_fixed_vs_autoresearch.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(output_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
