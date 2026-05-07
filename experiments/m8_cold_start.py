"""
M8 Cold-Start Experiment: archetype retrieval for new tenants.

Validates the cold-start thesis:
  "Archetype retrieval from minimal history (30-480 min) closes the gap
   between cold autoresearch and the oracle (full-history autoresearch)."

Compares four strategies at increasing history lengths:
  1. Cold autoresearch (no archetype prior — uniform config sampling)
  2. Archetype-guided autoresearch (FAISS retrieval → bias config sampling)
  3. FixedConfigTSFM (status quo — same config for all tenants)
  4. Oracle (full-history autoresearch — upper bound)

Usage:
    uv run python experiments/m8_cold_start.py \
        --tenants 50 --horizon 60 --lengths 30,60,120,240,480
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tsfm_autoresearch.archetype_store import (
    ArchetypeStore,
    extract_features,
    _ARCHETYPE_CONTEXT_BIAS,
)
from tsfm_autoresearch.autoresearch import AutoresearchHarness
from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss
from tsfm_autoresearch.tsfm_client import ForecastConfig, TSFMClient

from baselines.fixed_config import FixedConfigTSFM
from autoresearch_agent.infra import (
    DEFAULT_HORIZON,
    DEFAULT_K,
    DEFAULT_N_TENANTS,
    evaluate_mae,
    load_tenant_data,
    load_manifest,
    log_result,
    sample_tenants,
)

# ── Experiment Configuration ───────────────────────────────────────────

EXPERIMENT_NAME = "08_cold_start_archetype"

# History lengths to test (minutes at 1-min resolution)
DEFAULT_HISTORY_LENGTHS = [30, 60, 120, 240, 480]

# Timestamps per tenant (fewer since we test multiple lengths × strategies)
DEFAULT_TIMESTAMPS = 3

# K for autoresearch
K = DEFAULT_K


# ── Experiment Runner ──────────────────────────────────────────────────


def run_cold_start_experiment(
    client: TSFMClient,
    n_tenants: int = DEFAULT_N_TENANTS,
    horizon: int = DEFAULT_HORIZON,
    history_lengths: list[int] | None = None,
    timestamps_per_tenant: int = DEFAULT_TIMESTAMPS,
    sla_tier: SLATier = SLATier.STANDARD,
    seed: int = 42,
) -> dict:
    """
    Run the cold-start experiment.

    For each tenant, evaluates forecasters at increasing history lengths
    and compares against the oracle (full-history autoresearch).
    """
    if history_lengths is None:
        history_lengths = DEFAULT_HISTORY_LENGTHS

    rng = np.random.default_rng(seed)
    tenant_ids = sample_tenants(n_tenants, seed=seed)
    manifest = load_manifest()
    loss_fn = CostAsymmetricLoss(alpha=sla_tier.alpha, per_resource=True)

    # ── Build Archetype Store ───────────────────────────────────────
    print(f"{'='*70}")
    print(f"  M8 COLD-START EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Tenants: {n_tenants}")
    print(f"  History lengths: {history_lengths} min")
    print(f"  Horizon: {horizon} min")
    print(f"  K: {K}")
    print(f"  SLA Tier: {sla_tier.value}")
    print(f"{'='*70}\n")

    print("Building archetype store from synthetic fleet...")
    t0 = time.perf_counter()
    store = ArchetypeStore()
    store.build(data_dir="data/synthetic")
    print(f"  Built in {time.perf_counter() - t0:.1f}s "
          f"({len(store._centroids)} archetypes)")

    # Build oracle harness and fixed-config baseline
    oracle_harness = AutoresearchHarness(client, default_K=K, seed=seed)
    fixed_config = ForecastConfig(context_len=256)
    fixed_tsfm = FixedConfigTSFM(client, fixed_config)

    # ── Per-length accumulators ─────────────────────────────────────
    per_length: dict[int, dict[str, list[float]]] = {
        h: {
            "cold_losses": [],
            "archetype_losses": [],
            "fixed_losses": [],
            "cold_latencies": [],
            "archetype_latencies": [],
            "fixed_latencies": [],
            "retrieval_correct": 0,
            "retrieval_total": 0,
        }
        for h in history_lengths
    }

    oracle_losses: list[float] = []
    oracle_latencies: list[float] = []

    total_evals = 0

    for tidx, tid in enumerate(tenant_ids):
        history = load_tenant_data(tid)
        true_archetype = manifest.get(tid, "unknown")
        T = history.shape[0]

        # Need enough history for oracle evaluation
        min_needed = max(history_lengths) + horizon + 100
        if T <= min_needed:
            continue

        # ── Oracle: full-history autoresearch ───────────────────────
        eval_candidates = np.arange(min_needed, T - horizon)
        if len(eval_candidates) == 0:
            continue

        n_pts = min(timestamps_per_tenant, len(eval_candidates))
        eval_points = rng.choice(eval_candidates, size=n_pts, replace=False)
        eval_points.sort()

        for t in eval_points:
            full_hist = history[:t, :]
            actuals = history[t : t + horizon, :]
            if actuals.shape[0] < horizon:
                continue

            try:
                resp = oracle_harness.forecast(
                    tenant_id=tid, history=full_hist,
                    horizon=horizon, sla_tier=sla_tier, K=K,
                )
                oracle_losses.append(
                    loss_fn.point_loss(actuals, resp.final_forecast.point)
                )
                oracle_latencies.append(resp.total_latency_ms)
            except Exception:
                continue

        # ── Cold-start at increasing history lengths ─────────────────
        for hist_len in history_lengths:
            # Use a different eval point: after hist_len, before horizon
            eval_start = max(hist_len + 100, min_needed)
            eval_candidates = np.arange(eval_start, T - horizon)
            if len(eval_candidates) == 0:
                continue

            # Pick one eval point per length per tenant
            eval_t = int(rng.choice(eval_candidates))

            short_hist = history[eval_t - hist_len : eval_t, :]
            actuals = history[eval_t : eval_t + horizon, :]

            if actuals.shape[0] < horizon:
                continue

            # ── 1. Cold autoresearch (no archetype) ─────────────────
            try:
                cold_harness = AutoresearchHarness(client, default_K=K, seed=seed)
                cold_resp = cold_harness.forecast(
                    tenant_id=tid, history=short_hist,
                    horizon=horizon, sla_tier=sla_tier, K=K,
                    # No archetype_embedding → uniform config sampling
                )
                per_length[hist_len]["cold_losses"].append(
                    loss_fn.point_loss(actuals, cold_resp.final_forecast.point)
                )
                per_length[hist_len]["cold_latencies"].append(
                    cold_resp.total_latency_ms
                )
            except Exception:
                pass

            # ── 2. Archetype-guided autoresearch ─────────────────────
            try:
                # Extract features from cold-start history
                features = extract_features(short_hist)

                # Retrieve archetype
                arch_name, similarity = store.query_archetype(features)
                per_length[hist_len]["retrieval_total"] += 1
                if arch_name == true_archetype:
                    per_length[hist_len]["retrieval_correct"] += 1

                # Create harness with archetype bias
                arch_harness = AutoresearchHarness(client, default_K=K, seed=seed)
                arch_resp = arch_harness.forecast(
                    tenant_id=tid, history=short_hist,
                    horizon=horizon, sla_tier=sla_tier, K=K,
                    archetype_embedding=arch_name,  # triggers context bias
                )
                per_length[hist_len]["archetype_losses"].append(
                    loss_fn.point_loss(actuals, arch_resp.final_forecast.point)
                )
                per_length[hist_len]["archetype_latencies"].append(
                    arch_resp.total_latency_ms
                )
            except Exception:
                pass

            # ── 3. FixedConfigTSFM ───────────────────────────────────
            try:
                fc_forecast = fixed_tsfm.forecast(short_hist, horizon)
                per_length[hist_len]["fixed_losses"].append(
                    loss_fn.point_loss(actuals, fc_forecast)
                )
            except Exception:
                pass

            total_evals += 1

        if (tidx + 1) % 10 == 0:
            print(f"  Progress: {tidx + 1}/{len(tenant_ids)} tenants")

    # ── Compute Summary ─────────────────────────────────────────────
    print(f"\n  Total evaluations: {total_evals}")
    oracle_mean = float(np.mean(oracle_losses)) if oracle_losses else float("inf")
    print(f"  Oracle loss (full history): {oracle_mean:.6f}")

    results: dict = {
        "experiment": EXPERIMENT_NAME,
        "n_tenants": n_tenants,
        "horizon": horizon,
        "sla_tier": sla_tier.value,
        "k": K,
        "oracle_loss": oracle_mean,
        "oracle_p50_latency_ms": float(np.median(oracle_latencies)) if oracle_latencies else 0.0,
        "history_lengths": [],
    }

    print(f"\n{'─'*85}")
    print(f"  {'Min':>5s}  {'Cold':>10s}  {'Archetype':>10s}  "
          f"{'Fixed':>10s}  {'Δ Archetype':>12s}  "
          f"{'Gap to Oracle':>14s}  {'Retrieval':>10s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*12}  "
          f"{'─'*14}  {'─'*10}")

    for hist_len in history_lengths:
        data = per_length[hist_len]
        cold_mean = float(np.mean(data["cold_losses"])) if data["cold_losses"] else float("inf")
        arch_mean = float(np.mean(data["archetype_losses"])) if data["archetype_losses"] else float("inf")
        fixed_mean = float(np.mean(data["fixed_losses"])) if data["fixed_losses"] else float("inf")

        # Improvement from archetype vs. cold
        delta = cold_mean - arch_mean
        delta_str = f"{delta:+.6f}" if delta != 0 else "—"

        # Gap to oracle
        gap = arch_mean - oracle_mean
        gap_str = f"{gap:+.6f}" if gap != 0 else "—"

        # Retrieval accuracy
        total_ret = data["retrieval_total"]
        correct_ret = data["retrieval_correct"]
        ret_acc = f"{correct_ret}/{total_ret} ({correct_ret/total_ret:.0%})" if total_ret > 0 else "—"

        cold_p50 = float(np.median(data["cold_latencies"])) if data["cold_latencies"] else 0
        arch_p50 = float(np.median(data["archetype_latencies"])) if data["archetype_latencies"] else 0

        length_entry = {
            "history_minutes": hist_len,
            "cold_loss": cold_mean,
            "archetype_loss": arch_mean,
            "fixed_loss": fixed_mean,
            "delta_archetype_vs_cold": delta,
            "gap_to_oracle": gap,
            "retrieval_accuracy": correct_ret / total_ret if total_ret > 0 else 0,
            "retrieval_correct": correct_ret,
            "retrieval_total": total_ret,
            "cold_p50_latency_ms": cold_p50,
            "archetype_p50_latency_ms": arch_p50,
            "cold_n": len(data["cold_losses"]),
            "archetype_n": len(data["archetype_losses"]),
        }
        results["history_lengths"].append(length_entry)

        print(f"  {hist_len:>5d}  {cold_mean:>10.6f}  {arch_mean:>10.6f}  "
              f"{fixed_mean:>10.6f}  {delta_str:>12s}  "
              f"{gap_str:>14s}  {ret_acc:>10s}")

    # ── Key finding: does archetype reduce the gap? ─────────────────
    if results["history_lengths"]:
        best_len = max(results["history_lengths"],
                       key=lambda e: e.get("delta_archetype_vs_cold", 0))
        print(f"\n{'='*70}")
        print(f"  FINDINGS")
        print(f"{'='*70}")
        print(f"  Best archetype improvement: "
              f"{best_len['delta_archetype_vs_cold']:+.6f} "
              f"at {best_len['history_minutes']} min history")
        if oracle_mean != float("inf"):
            gap_reduction = (
                (results["history_lengths"][0]["gap_to_oracle"] -
                 best_len["gap_to_oracle"])
                / results["history_lengths"][0]["gap_to_oracle"] * 100
                if results["history_lengths"][0]["gap_to_oracle"] != 0
                else 0
            )
            print(f"  Gap to oracle reduced by: {gap_reduction:+.1f}%")
        print(f"{'='*70}")

    # ── Log ─────────────────────────────────────────────────────────
    description = (
        f"M8: cold-start on {n_tenants} tenants, "
        f"lengths={history_lengths}. "
        f"Best archetype Δ: {best_len['delta_archetype_vs_cold']:+.6f} "
        f"at {best_len['history_minutes']}min. "
        f"Oracle loss: {oracle_mean:.6f}"
    )
    log_result(
        experiment=EXPERIMENT_NAME,
        cost_asym_loss=best_len["archetype_loss"],
        mae=0.0,  # not tracked in this experiment
        p50_latency_ms=best_len["archetype_p50_latency_ms"],
        status="keep",
        description=description,
    )

    return results


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M8 Cold-Start Experiment: archetype retrieval",
    )
    parser.add_argument(
        "--tenants", type=int, default=50,
        help="Tenants to evaluate (default: 50)",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Forecast horizon (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--lengths", type=str, default="30,60,120,240,480",
        help="Comma-separated history lengths in minutes",
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

    history_lengths = [int(x.strip()) for x in args.lengths.split(",")]

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

    results = run_cold_start_experiment(
        client=client,
        n_tenants=args.tenants,
        horizon=args.horizon,
        history_lengths=history_lengths,
        sla_tier=sla_tier,
        seed=args.seed,
    )

    # Save results
    output_path = Path("results/08_cold_start_archetype.json")
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
