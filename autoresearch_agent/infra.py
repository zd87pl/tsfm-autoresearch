"""
Fixed infrastructure for the TimesFM autoresearch outer loop.

This file is READ-ONLY per the karpathy/autoresearch pattern.
It provides data loading, evaluation metrics, and fixed constants
that the experiment runner and autonomous agent rely on.

Do NOT modify this file. The agent modifies run_experiment.py instead.
"""

from __future__ import annotations

import contextlib
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    import fcntl as _fcntl
except ImportError:  # non-POSIX
    _fcntl = None

from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss

# ── Fixed Constants ────────────────────────────────────────────────────
# These are the known, stable parameters of the experimental setup.

# Data
DATA_DIR = Path("data/synthetic")
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Model (frozen)
MODEL_ID = "google/timesfm-2.5-200m-pytorch"

# Experiment defaults
DEFAULT_N_TENANTS = 200
DEFAULT_HORIZON = 60  # minutes (1 hour)
DEFAULT_K = 8  # Configs per autoresearch request
DEFAULT_VAL_SPLIT = 0.15

# SLA tiers for multi-tier experiments
SLA_TIERS = [SLATier.PREMIUM, SLATier.STANDARD, SLATier.BASIC]
SLA_DISTRIBUTION = {"premium": 0.2, "standard": 0.5, "basic": 0.3}

# Results tracking
RESULTS_PATH = Path("autoresearch_agent/results.tsv")
RESULTS_COLUMNS = [
    "timestamp", "commit", "experiment", "cost_asym_loss", "mae",
    "p50_latency_ms", "status", "description",
]

# Time budget (seconds) — None means no hard limit
TIME_BUDGET_SECONDS: int | None = None


# ── Data Loading ───────────────────────────────────────────────────────


def load_tenant_data(tenant_id: str) -> np.ndarray:
    """
    Load a tenant's time series as a (T, 4) numpy array.

    Returns:
        Array with columns [cpu_util, mem_util, net_bytes, disk_iops].
    """
    parquet_path = DATA_DIR / f"{tenant_id}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Tenant data not found: {parquet_path}")

    df = pl.read_parquet(parquet_path)
    return np.column_stack([
        df["cpu_util"].to_numpy(),
        df["mem_util"].to_numpy(),
        df["net_bytes"].to_numpy(),
        df["disk_iops"].to_numpy(),
    ])


def load_manifest() -> dict[str, str]:
    """
    Load manifest mapping tenant_id → archetype.

    Returns:
        Dict of {tenant_id: archetype_name}.
    """
    manifest: dict[str, str] = {}
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["tenant_id"]] = row["archetype"]
    return manifest


def sample_tenants(
    n: int = DEFAULT_N_TENANTS,
    stratify_by_archetype: bool = True,
    seed: int = 42,
) -> list[str]:
    """
    Sample n tenant IDs from the synthetic fleet.

    Args:
        n: Number of tenants to sample.
        stratify_by_archetype: If True, sample proportionally per archetype.
        seed: RNG seed.

    Returns:
        List of tenant_id strings.
    """
    manifest = load_manifest()
    rng = np.random.default_rng(seed)

    if not stratify_by_archetype:
        ids = list(manifest.keys())
        return rng.choice(ids, size=min(n, len(ids)), replace=False).tolist()

    # Stratified sampling: preserve archetype proportions
    archetype_groups: dict[str, list[str]] = {}
    for tid, arch in manifest.items():
        archetype_groups.setdefault(arch, []).append(tid)

    sampled: list[str] = []
    total = sum(len(v) for v in archetype_groups.values())

    for arch, tids in archetype_groups.items():
        n_arch = max(1, int(n * len(tids) / total))
        sampled.extend(
            rng.choice(tids, size=min(n_arch, len(tids)), replace=False).tolist()
        )

    return sampled[:n]


# ── Evaluation Metrics ─────────────────────────────────────────────────


def evaluate_cost_asymmetric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sla_tier: SLATier = SLATier.STANDARD,
) -> float:
    """
    Compute cost-asymmetric point loss for a set of predictions.

    This is the HEADLINE METRIC for the thesis.
    """
    loss_fn = CostAsymmetricLoss(alpha=sla_tier.alpha, per_resource=True)
    return loss_fn.point_loss(y_true, y_pred)


def evaluate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (sanity-check metric).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Results Logging ────────────────────────────────────────────────────


@contextlib.contextmanager
def _exclusive_lock(file_obj):
    """Acquire an OS-level exclusive lock on `file_obj` while the block runs.

    Serializes concurrent writers to results.tsv so two outer-loop runs that
    finish at the same moment don't interleave header creation or row writes.
    Falls back to a no-op on platforms without fcntl (e.g. Windows): callers
    on those platforms must serialize externally.
    """
    if _fcntl is None:
        yield
        return
    _fcntl.flock(file_obj.fileno(), _fcntl.LOCK_EX)
    try:
        yield
    finally:
        _fcntl.flock(file_obj.fileno(), _fcntl.LOCK_UN)


def init_results_tsv(overwrite: bool = False) -> None:
    """Create results.tsv with header if it doesn't exist.

    Uses an exclusive lock so concurrent first-time writers don't both pass
    the existence check and clobber each other's header rows.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Open with "a" so the file is created if missing without truncating
    # an existing one; under the exclusive lock, decide whether to write
    # the header based on the actual file size after acquiring the lock.
    with open(RESULTS_PATH, "a", newline="") as f:
        with _exclusive_lock(f):
            if overwrite or f.tell() == 0:
                if overwrite:
                    f.seek(0)
                    f.truncate()
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(RESULTS_COLUMNS)


def log_result(
    experiment: str,
    cost_asym_loss: float,
    mae: float,
    p50_latency_ms: float,
    status: str = "keep",
    description: str = "",
) -> None:
    """
    Log an experiment result to results.tsv.

    Args:
        experiment: Experiment name (e.g. "m6_headline").
        cost_asym_loss: Headline cost-asymmetric loss.
        mae: Mean absolute error (sanity check).
        p50_latency_ms: Median forecast latency.
        status: "keep", "discard", or "crash".
        description: What this experiment tried.
    """
    import subprocess
    from datetime import datetime, timezone

    # Get current commit hash
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    init_results_tsv()

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(RESULTS_PATH, "a", newline="") as f:
        with _exclusive_lock(f):
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                timestamp,
                commit,
                experiment,
                f"{cost_asym_loss:.6f}",
                f"{mae:.6f}",
                f"{p50_latency_ms:.1f}",
                status,
                description,
            ])
            f.flush()

    print(f"\n✓ Logged to {RESULTS_PATH}: {experiment} | "
          f"loss={cost_asym_loss:.6f} | mae={mae:.6f} | "
          f"latency={p50_latency_ms:.0f}ms | {status}")
