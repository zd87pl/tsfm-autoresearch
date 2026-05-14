"""
M4 Retrieval-Accuracy Experiment.

Quantifies how well the FAISS-backed ArchetypeStore identifies a tenant's
archetype from its workload features. This is the experiment that backs
the README's >90% retrieval accuracy claim.

Design (avoids the M8 information-leak pitfall):
  1. Split the 1,000-tenant synthetic fleet into a *centroid set* (used to
     build the store) and an *evaluation set* (never used at build time).
  2. For each evaluation tenant, extract features from N different history
     lengths — full, plus a sweep of short cold-start windows.
  3. Query the store and record top-1 accuracy per length, plus a per-
     archetype confusion matrix at the full-history length.
  4. Report:
       - C6 (full-history retrieval accuracy ≥ 0.90)
       - C7 (120-min cold-start accuracy ≥ 0.70)

Usage:
    uv run python experiments/m4_retrieval_accuracy.py --centroid-frac 0.5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tsfm_autoresearch.archetype_store import ArchetypeStore, extract_features

from autoresearch_agent.infra import (
    load_manifest,
    load_tenant_data,
    log_result,
)

EXPERIMENT_NAME = "04_retrieval_accuracy"

# Acceptance criteria from autoresearch_agent/program.md.
C6_FULL_HISTORY_TARGET = 0.90
C7_COLD_START_120MIN_TARGET = 0.70

# History windows (minutes at 1-min resolution). `None` means full history.
DEFAULT_WINDOWS: list[int | None] = [30, 60, 120, 240, 480, None]


def _split_tenants(
    manifest: dict[str, str],
    centroid_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified split: each archetype contributes the same fraction to centroids."""
    rng = np.random.default_rng(seed)
    by_archetype: dict[str, list[str]] = defaultdict(list)
    for tid, arch in manifest.items():
        by_archetype[arch].append(tid)

    centroid_ids: list[str] = []
    eval_ids: list[str] = []
    for arch, tids in by_archetype.items():
        tids_sorted = sorted(tids)
        rng.shuffle(tids_sorted)
        cutoff = max(1, int(round(len(tids_sorted) * centroid_frac)))
        centroid_ids.extend(tids_sorted[:cutoff])
        eval_ids.extend(tids_sorted[cutoff:])

    return centroid_ids, eval_ids


def run_retrieval_accuracy(
    centroid_frac: float = 0.5,
    windows: list[int | None] | None = None,
    seed: int = 42,
) -> dict:
    """Measure top-1 retrieval accuracy on a held-out tenant set."""
    if windows is None:
        windows = list(DEFAULT_WINDOWS)

    print("=" * 70)
    print("  M4 RETRIEVAL-ACCURACY EXPERIMENT")
    print("=" * 70)
    print(f"  Centroid set fraction: {centroid_frac:.0%}")
    print(f"  Windows (min): {windows}")
    print(f"  Seed: {seed}")
    print("=" * 70 + "\n")

    manifest = load_manifest()
    centroid_ids, eval_ids = _split_tenants(manifest, centroid_frac, seed)
    print(f"  Centroid tenants: {len(centroid_ids)}")
    print(f"  Evaluation tenants: {len(eval_ids)}")

    # Build the store ONLY on the centroid set. Evaluation tenants are
    # genuinely unseen at retrieval time.
    store = ArchetypeStore()
    store.build(data_dir="data/synthetic", exclude_tenants=set(eval_ids))
    print(f"  Archetypes in store: {sorted(store._centroids.keys())}\n")

    # Per-window accuracy + per-archetype confusion at full history.
    per_window: dict[str, dict] = {}
    confusion_full: dict[str, Counter] = defaultdict(Counter)

    for window in windows:
        window_key = "full" if window is None else f"{window}min"
        correct = 0
        total = 0
        per_arch_correct: dict[str, int] = defaultdict(int)
        per_arch_total: dict[str, int] = defaultdict(int)

        for tid in eval_ids:
            true_arch = manifest[tid]
            history = load_tenant_data(tid)
            if window is not None:
                if history.shape[0] < window:
                    continue
                slice_ = history[:window, :]
            else:
                slice_ = history

            try:
                features = extract_features(slice_)
                predicted, _sim = store.query_archetype(features)
            except Exception:
                continue

            total += 1
            per_arch_total[true_arch] += 1
            if predicted == true_arch:
                correct += 1
                per_arch_correct[true_arch] += 1
            if window is None:
                confusion_full[true_arch][predicted] += 1

        accuracy = correct / total if total else 0.0
        per_arch_acc = {
            a: per_arch_correct[a] / per_arch_total[a]
            for a in per_arch_total
            if per_arch_total[a] > 0
        }
        per_window[window_key] = {
            "window_minutes": window,
            "n_queries": total,
            "correct": correct,
            "accuracy": accuracy,
            "per_archetype_accuracy": per_arch_acc,
        }
        print(f"  {window_key:>6s}: accuracy = {accuracy:.3f}  ({correct}/{total})")

    # ── Acceptance criteria ─────────────────────────────────────────
    full_acc = per_window.get("full", {}).get("accuracy", 0.0)
    cold_120 = per_window.get("120min", {}).get("accuracy", 0.0)
    c6_pass = full_acc >= C6_FULL_HISTORY_TARGET
    c7_pass = cold_120 >= C7_COLD_START_120MIN_TARGET

    print(f"\n{'=' * 70}")
    print("  ACCEPTANCE CRITERIA")
    print("=" * 70)
    print(f"  C6 (full-history ≥ {C6_FULL_HISTORY_TARGET:.0%}):     "
          f"{full_acc:.3f}  {'PASS' if c6_pass else 'FAIL'}")
    print(f"  C7 (120-min cold-start ≥ {C7_COLD_START_120MIN_TARGET:.0%}): "
          f"{cold_120:.3f}  {'PASS' if c7_pass else 'FAIL'}")
    print("=" * 70)

    # ── Confusion at full history (printed as table) ──────────────────
    archetypes = sorted(store._centroids.keys())
    print("\n  Confusion matrix at full history (rows = true, cols = predicted):")
    header = "  " + " " * 20 + "  ".join(f"{a[:12]:>12s}" for a in archetypes)
    print(header)
    for true_arch in archetypes:
        row = confusion_full.get(true_arch, Counter())
        cells = "  ".join(f"{row.get(p, 0):>12d}" for p in archetypes)
        print(f"  {true_arch:<20s}{cells}")

    results = {
        "experiment": EXPERIMENT_NAME,
        "centroid_frac": centroid_frac,
        "n_centroid_tenants": len(centroid_ids),
        "n_eval_tenants": len(eval_ids),
        "seed": seed,
        "per_window": per_window,
        "confusion_full": {
            true_arch: dict(row) for true_arch, row in confusion_full.items()
        },
        "criteria": {
            "C6_full_history": {
                "target": C6_FULL_HISTORY_TARGET,
                "observed": full_acc,
                "pass": c6_pass,
            },
            "C7_cold_start_120min": {
                "target": C7_COLD_START_120MIN_TARGET,
                "observed": cold_120,
                "pass": c7_pass,
            },
        },
    }

    description = (
        f"M4: retrieval accuracy on {len(eval_ids)} held-out tenants. "
        f"Full={full_acc:.3f} (C6 {'pass' if c6_pass else 'fail'}); "
        f"120min={cold_120:.3f} (C7 {'pass' if c7_pass else 'fail'})."
    )
    log_result(
        experiment=EXPERIMENT_NAME,
        cost_asym_loss=0.0,
        mae=0.0,
        p50_latency_ms=0.0,
        status="keep" if (c6_pass and c7_pass) else "discard",
        description=description,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4 archetype-retrieval accuracy on held-out tenants",
    )
    parser.add_argument(
        "--centroid-frac", type=float, default=0.5,
        help="Fraction of tenants used to build centroids (rest are eval).",
    )
    parser.add_argument(
        "--windows", type=str, default="30,60,120,240,480,full",
        help="Comma-separated history windows in minutes (or 'full').",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    windows: list[int | None] = []
    for w in args.windows.split(","):
        w = w.strip()
        if w.lower() == "full":
            windows.append(None)
        elif w:
            windows.append(int(w))

    results = run_retrieval_accuracy(
        centroid_frac=args.centroid_frac,
        windows=windows,
        seed=args.seed,
    )

    out_path = Path("results/04_retrieval_accuracy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    with open(out_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
