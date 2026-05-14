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

    # Build the archetype store on tenants DISJOINT from the evaluation
    # set. Otherwise a cold-start retrieval for tenant X is "looking up" a
    # centroid that already saw X's full 30-day history — inflating the
    # gap-to-oracle claim.
    print("Building archetype store from held-out tenants...")
    t0 = time.perf_counter()
    store = ArchetypeStore()
    store.build(
        data_dir="data/synthetic",
        exclude_tenants=set(tenant_ids),
    )
    print(f"  Built in {time.perf_counter() - t0:.1f}s "
          f"({len(store._centroids)} archetypes, "
          f"eval tenants excluded: {len(tenant_ids)})")

    # Build oracle harness and fixed-config baseline
    oracle_harness = AutoresearchHarness(client, default_K=K, seed=seed)
    fixed_config = ForecastConfig(context_len=256)
    fixed_tsfm = FixedConfigTSFM(client, fixed_config)

    # ── Per-length accumulators ─────────────────────────────────────
    # Each strategy stores losses indexed parallel to per_length[h]["pair_keys"]
    # so per-length, per-tenant comparisons are paired across strategies.
    per_length: dict[int, dict[str, list]] = {
        h: {
            "pair_keys": [],
            "oracle_losses": [],
            "cold_losses": [],
            "archetype_losses": [],
            "fixed_losses": [],
            "oracle_latencies": [],
            "cold_latencies": [],
            "archetype_latencies": [],
            "fixed_latencies": [],
            "retrieval_correct": 0,
            "retrieval_total": 0,
        }
        for h in history_lengths
    }

    total_evals = 0

    for tidx, tid in enumerate(tenant_ids):
        history = load_tenant_data(tid)
        true_archetype = manifest.get(tid, "unknown")
        T = history.shape[0]

        # Need enough history for oracle evaluation
        min_needed = max(history_lengths) + horizon + 100
        if T <= min_needed:
            continue

        # Sample shared eval points used by every strategy at this tenant.
        eval_candidates = np.arange(min_needed, T - horizon)
        if len(eval_candidates) == 0:
            continue
        n_pts = min(timestamps_per_tenant, len(eval_candidates))
        shared_eval_points = rng.choice(eval_candidates, size=n_pts, replace=False)
        shared_eval_points.sort()

        for eval_t in shared_eval_points:
            eval_t = int(eval_t)
            actuals = history[eval_t : eval_t + horizon, :]
            if actuals.shape[0] < horizon:
                continue

            # Oracle uses the full prefix at this eval_t.
            full_hist = history[:eval_t, :]

            for hist_len in history_lengths:
                if eval_t - hist_len < 0:
                    continue
                short_hist = history[eval_t - hist_len : eval_t, :]

                # Run all four strategies; only record the row if every
                # one succeeds, so per-length lists remain index-aligned.
                row_results: dict[str, tuple[float, float]] = {}
                ok = True

                try:
                    resp = oracle_harness.forecast(
                        tenant_id=tid, history=full_hist,
                        horizon=horizon, sla_tier=sla_tier, K=K,
                    )
                    row_results["oracle"] = (
                        loss_fn.point_loss(actuals, resp.final_forecast.point),
                        resp.total_latency_ms,
                    )
                except Exception:
                    ok = False

                if ok:
                    try:
                        cold_harness = AutoresearchHarness(client, default_K=K, seed=seed)
                        cold_resp = cold_harness.forecast(
                            tenant_id=tid, history=short_hist,
                            horizon=horizon, sla_tier=sla_tier, K=K,
                        )
                        row_results["cold"] = (
                            loss_fn.point_loss(actuals, cold_resp.final_forecast.point),
                            cold_resp.total_latency_ms,
                        )
                    except Exception:
                        ok = False

                arch_name = None
                if ok:
                    try:
                        features = extract_features(short_hist)
                        arch_name, _ = store.query_archetype(features)
                        arch_harness = AutoresearchHarness(client, default_K=K, seed=seed)
                        arch_resp = arch_harness.forecast(
                            tenant_id=tid, history=short_hist,
                            horizon=horizon, sla_tier=sla_tier, K=K,
                            archetype_embedding=arch_name,
                        )
                        row_results["archetype"] = (
                            loss_fn.point_loss(actuals, arch_resp.final_forecast.point),
                            arch_resp.total_latency_ms,
                        )
                    except Exception:
                        ok = False

                if ok:
                    try:
                        fc_forecast = fixed_tsfm.forecast(short_hist, horizon)
                        row_results["fixed"] = (
                            loss_fn.point_loss(actuals, fc_forecast),
                            0.0,
                        )
                    except Exception:
                        ok = False

                if not ok:
                    continue

                bucket = per_length[hist_len]
                bucket["pair_keys"].append((tid, eval_t))
                bucket["oracle_losses"].append(row_results["oracle"][0])
                bucket["oracle_latencies"].append(row_results["oracle"][1])
                bucket["cold_losses"].append(row_results["cold"][0])
                bucket["cold_latencies"].append(row_results["cold"][1])
                bucket["archetype_losses"].append(row_results["archetype"][0])
                bucket["archetype_latencies"].append(row_results["archetype"][1])
                bucket["fixed_losses"].append(row_results["fixed"][0])
                bucket["fixed_latencies"].append(row_results["fixed"][1])
                bucket["retrieval_total"] += 1
                if arch_name == true_archetype:
                    bucket["retrieval_correct"] += 1
                total_evals += 1

        if (tidx + 1) % 10 == 0:
            print(f"  Progress: {tidx + 1}/{len(tenant_ids)} tenants")

    # ── Compute Summary ─────────────────────────────────────────────
    print(f"\n  Total evaluations: {total_evals}")

    # Pool oracle losses across hist_len buckets for an overall mean (each
    # row-paired oracle measurement appears once per hist_len; report the
    # per-length means below for the apples-to-apples comparison).
    overall_oracle = [
        x for h in history_lengths for x in per_length[h]["oracle_losses"]
    ]
    overall_oracle_lat = [
        x for h in history_lengths for x in per_length[h]["oracle_latencies"]
    ]
    oracle_mean = float(np.mean(overall_oracle)) if overall_oracle else float("inf")
    print(f"  Oracle loss (paired with cold-start, all lengths): {oracle_mean:.6f}")

    results: dict = {
        "experiment": EXPERIMENT_NAME,
        "n_tenants": n_tenants,
        "horizon": horizon,
        "sla_tier": sla_tier.value,
        "k": K,
        "oracle_loss": oracle_mean,
        "oracle_p50_latency_ms": (
            float(np.median(overall_oracle_lat)) if overall_oracle_lat else 0.0
        ),
        "history_lengths": [],
    }

    print(f"\n{'─'*92}")
    print(f"  {'Min':>5s}  {'Oracle':>10s}  {'Cold':>10s}  {'Archetype':>10s}  "
          f"{'Fixed':>10s}  {'Δ Archetype':>12s}  "
          f"{'Gap to Oracle':>14s}  {'Retrieval':>10s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*12}  "
          f"{'─'*14}  {'─'*10}")

    for hist_len in history_lengths:
        data = per_length[hist_len]
        # Per-length oracle is paired with cold/archetype/fixed at SAME
        # (tenant, eval_t) rows — this is the apples-to-apples gap-to-oracle.
        oracle_mean_h = (
            float(np.mean(data["oracle_losses"])) if data["oracle_losses"] else float("inf")
        )
        cold_mean = float(np.mean(data["cold_losses"])) if data["cold_losses"] else float("inf")
        arch_mean = float(np.mean(data["archetype_losses"])) if data["archetype_losses"] else float("inf")
        fixed_mean = float(np.mean(data["fixed_losses"])) if data["fixed_losses"] else float("inf")

        # Improvement from archetype vs. cold (paired diff, same rows)
        delta = cold_mean - arch_mean
        delta_str = f"{delta:+.6f}" if delta != 0 else "—"

        # Gap to oracle (paired: same eval rows used for both)
        gap = arch_mean - oracle_mean_h
        gap_str = f"{gap:+.6f}" if gap != 0 else "—"

        # Retrieval accuracy
        total_ret = data["retrieval_total"]
        correct_ret = data["retrieval_correct"]
        ret_acc = f"{correct_ret}/{total_ret} ({correct_ret/total_ret:.0%})" if total_ret > 0 else "—"

        cold_p50 = float(np.median(data["cold_latencies"])) if data["cold_latencies"] else 0
        arch_p50 = float(np.median(data["archetype_latencies"])) if data["archetype_latencies"] else 0

        length_entry = {
            "history_minutes": hist_len,
            "oracle_loss": oracle_mean_h,
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

        print(f"  {hist_len:>5d}  {oracle_mean_h:>10.6f}  {cold_mean:>10.6f}  {arch_mean:>10.6f}  "
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
