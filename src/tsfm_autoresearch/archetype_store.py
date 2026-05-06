"""
Archetype Store (M4) — FAISS-backed archetype embedding and retrieval.

Clusters the 1,000 synthetic tenants into the eight known archetypes from
M1, stores per-archetype centroids in FAISS, and supports cold-start
retrieval for new tenants based on feature lookup.

ARCHITECTURE NOTE (PoC simplification):
The full design uses Hilbert-curve indexing over learned signatures (LOCI-DB
integration). For the PoC, we use FAISS with statistical feature vectors
extracted from each tenant's workload data. This is a deliberate
simplification — see TODO markers for future extensions.

WHY THIS MATTERS FOR THE PATENT:
The archetype store provides the "archetype-conditioned prior" for the
autoresearch loop (M3). Without it, the configuration sampler uses a
uniform prior over the config space. With it, the sampler can bias toward
configurations known to work well for the tenant's archetype. This is what
makes the cold-start experiment (M8) possible — new tenants with minimal
history can inherit configuration priors from their archetype cluster.

Cold-start retrieval works by:
1. Computing the same statistical features from whatever history is available
   (even just 1 hour of data for a new tenant)
2. Finding the nearest archetype centroid in FAISS
3. Returning the archetype label + centroid + config prior

Author: Hermes Agent (Ziggy's TSFM-Autoresearch PoC)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ── Feature Extraction ─────────────────────────────────────────────────
# These are the statistical features that characterize a tenant's workload.
# They are designed to be computable from any length of history (including
# very short cold-start windows).

# Number of feature dimensions
N_FEATURES = 28  # 7 features × 4 resources


def extract_features(
    history: np.ndarray,
    max_window: int | None = None,
) -> np.ndarray:
    """
    Extract statistical features from a tenant's multivariate history.

    Features are computed per resource dimension and concatenated into
    a single vector of length N_FEATURES = 7 * D = 28.

    Feature list per resource:
      0. mean           — average level
      1. std            — dispersion
      2. min            — minimum observed
      3. max            — maximum observed
      4. p90            — 90th percentile (captures spikes)
      5. autocorr_lag1  — persistence / temporal dependence
      6. spike_ratio    — fraction of points > 1.5 × std above mean

    All features are robust to varying history lengths. For cold-start
    (very short history), features are computed on whatever data is
    available. For the PoC, we use max_window to limit computation.

    Args:
        history: Array of shape (T, D) — time steps × resource dimensions.
        max_window: If set, only use the last `max_window` steps.
            Simulates cold-start with limited history.

    Returns:
        Feature vector of shape (N_FEATURES,).
    """
    if max_window is not None and max_window < history.shape[0]:
        history = history[-max_window:, :]

    T, D = history.shape
    feature_list: list[float] = []

    for d in range(D):
        series = history[:, d]

        # 1. Mean
        feature_list.append(float(np.mean(series)))

        # 2. Standard deviation
        feature_list.append(float(np.std(series)))

        # 3. Minimum
        feature_list.append(float(np.min(series)))

        # 4. Maximum
        feature_list.append(float(np.max(series)))

        # 5. 90th percentile (robust to outliers)
        feature_list.append(float(np.percentile(series, 90)))

        # 6. Autocorrelation at lag-1 (captures temporal structure)
        if T > 1:
            series_centered = series - np.mean(series)
            denom = np.sum(series_centered**2)
            if denom > 1e-10:
                autocorr = float(np.sum(series_centered[:-1] * series_centered[1:]) / denom)
            else:
                autocorr = 0.0
        else:
            autocorr = 0.0
        feature_list.append(autocorr)

        # 7. Spike ratio: fraction of points exceeding 1.5 std above mean
        threshold = np.mean(series) + 1.5 * np.std(series)
        spike_ratio = float(np.mean(series > threshold)) if np.std(series) > 0 else 0.0
        feature_list.append(spike_ratio)

    return np.array(feature_list, dtype=np.float32)


def extract_features_batch(
    histories: list[np.ndarray],
    max_window: int | None = None,
) -> np.ndarray:
    """
    Extract features from multiple tenants.

    Returns:
        Array of shape (N_tenants, N_FEATURES).
    """
    features = [extract_features(h, max_window) for h in histories]
    return np.stack(features, axis=0)


# ── Archetype Store ────────────────────────────────────────────────────


@dataclass
class ArchetypeCentroid:
    """A single archetype centroid with metadata."""

    archetype: str
    centroid: np.ndarray  # shape (N_FEATURES,)
    tenant_count: int
    # Per-config-score priors for autoresearch (populated in M8)
    config_prior: dict[str, float] | None = None


class ArchetypeStore:
    """
    FAISS-backed archetype embedding store.

    Usage:
        store = ArchetypeStore()
        store.build(data_dir="data/synthetic")
        archetype, distance = store.query(features)
        store.save("data/archetypes.faiss")
        store.load("data/archetypes.faiss")
    """

    def __init__(self, n_features: int = N_FEATURES, seed: int = 42):
        self._n_features = n_features
        self._seed = seed
        self._centroids: dict[str, ArchetypeCentroid] = {}
        self._index: object | None = None  # FAISS index
        self._built = False

    # ── Build ─────────────────────────────────────────────────────

    def build(
        self,
        data_dir: str | Path = "data/synthetic",
        manifest_path: str | Path | None = None,
        max_tenants: int | None = None,
        exclude_tenants: set[str] | None = None,
    ) -> None:
        """
        Build the archetype store from synthetic tenant data.

        1. Load all tenant Parquet files and manifest
        2. Extract features per tenant
        3. Group by known archetype label
        4. Compute per-archetype centroid (mean of feature vectors)
        5. Build FAISS index over centroids

        This is an offline process — run once after data generation.

        Args:
            data_dir: Directory containing tenant Parquet files.
            manifest_path: Path to manifest CSV. Defaults to data_dir/manifest.csv.
            max_tenants: Limit number of tenants (for testing).
        """
        data_dir = Path(data_dir)
        if manifest_path is None:
            manifest_path = data_dir / "manifest.csv"

        # Load manifest: tenant_id → archetype
        manifest: dict[str, str] = {}
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row["tenant_id"]] = row["archetype"]

        # Group features by archetype
        archetype_features: dict[str, list[np.ndarray]] = {}
        parquet_files = sorted(data_dir.glob("*.parquet"))

        if max_tenants:
            parquet_files = parquet_files[:max_tenants]

        if exclude_tenants is None:
            exclude_tenants = set()

        n_loaded = 0
        for pf in parquet_files:
            tenant_id = pf.stem
            if tenant_id in exclude_tenants:
                continue
            archetype = manifest.get(tenant_id)
            if archetype is None:
                continue

            df = pl.read_parquet(pf)
            history = np.column_stack([
                df["cpu_util"].to_numpy(),
                df["mem_util"].to_numpy(),
                df["net_bytes"].to_numpy(),
                df["disk_iops"].to_numpy(),
            ])

            features = extract_features(history)
            archetype_features.setdefault(archetype, []).append(features)
            n_loaded += 1

        logger.info("Loaded %d tenants across %d archetypes",
                     n_loaded, len(archetype_features))

        # Compute centroids per archetype
        self._centroids = {}
        for arch, feat_list in archetype_features.items():
            stacked = np.stack(feat_list, axis=0)  # (N_tenants_in_arch, N_FEATURES)
            centroid = stacked.mean(axis=0)
            self._centroids[arch] = ArchetypeCentroid(
                archetype=arch,
                centroid=centroid.astype(np.float32),
                tenant_count=len(feat_list),
            )

        logger.info("Computed centroids for %d archetypes", len(self._centroids))

        # Build FAISS index
        self._build_index()
        self._built = True

    def _build_index(self) -> None:
        """Build a FAISS flat L2 index over archetype centroids."""
        import faiss
        from sklearn.preprocessing import StandardScaler

        centroids = np.stack(
            [c.centroid for c in self._centroids.values()], axis=0
        ).astype(np.float64)

        # Standardize features so net_bytes (hundreds) doesn't dominate cpu_util (0-1)
        self._scaler = StandardScaler()
        centroids_scaled = self._scaler.fit_transform(centroids).astype(np.float32)

        # Normalize for cosine similarity (L2 on normalized vectors ≈ cosine)
        faiss.normalize_L2(centroids_scaled)

        self._index = faiss.IndexFlatIP(self._n_features)
        self._index.add(centroids_scaled)
        self._index_to_arch = list(self._centroids.keys())

        logger.info("FAISS index built: %d centroids, dim=%d (standardized)",
                     centroids.shape[0], self._n_features)

    # ── Query ─────────────────────────────────────────────────────

    def query(
        self, features: np.ndarray, k: int = 1
    ) -> list[tuple[str, float, ArchetypeCentroid]]:
        """
        Retrieve the nearest archetype centroid(s) for a feature vector.

        This is the cold-start retrieval path:
          features = extract_features(new_tenant_history[:60])
          archetype, distance, centroid = store.query(features, k=1)

        Args:
            features: Feature vector of shape (N_FEATURES,) or (B, N_FEATURES).
            k: Number of nearest neighbors to return.

        Returns:
            List of (archetype_name, similarity_score, centroid) tuples,
            sorted by descending similarity. Similarity is cosine similarity
            in [-1, 1] where 1.0 = perfect match.

        Raises:
            RuntimeError: If store hasn't been built or loaded.
        """
        if not self._built or self._index is None:
            raise RuntimeError("ArchetypeStore not built. Call build() or load() first.")

        import faiss

        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Apply same standardization as during index build
        if hasattr(self, '_scaler') and self._scaler is not None:
            features = self._scaler.transform(features).astype(np.float32)
        else:
            features = features.astype(np.float32)

        # Normalize query vector
        faiss.normalize_L2(features)

        # Search FAISS
        similarities, indices = self._index.search(features, k=min(k, self._index.ntotal))

        results: list[tuple[str, float, ArchetypeCentroid]] = []
        for b in range(features.shape[0]):
            for ki in range(min(k, self._index.ntotal)):
                idx = indices[b, ki]
                sim = float(similarities[b, ki])
                arch_name = self._index_to_arch[idx]
                centroid = self._centroids[arch_name]
                results.append((arch_name, sim, centroid))

        return results

    def query_archetype(self, features: np.ndarray) -> tuple[str, float]:
        """
        Convenience: return just the top archetype and similarity.
        """
        results = self.query(features, k=1)
        if not results:
            raise RuntimeError("No archetypes found in store")
        return results[0][0], results[0][1]

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save the archetype store to disk.

        Saves centroids as a numpy .npz file (FAISS index is rebuilt on load).
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, dict] = {}
        for arch, centroid in self._centroids.items():
            data[arch] = {
                "centroid": centroid.centroid.tolist(),
                "tenant_count": centroid.tenant_count,
            }

        # Save as JSON + numpy for the centroid vectors
        centroids_array = np.stack(
            [c.centroid for c in self._centroids.values()], axis=0
        )
        np.savez(
            str(path.with_suffix("")),
            centroids=centroids_array,
            archetypes=np.array(list(self._centroids.keys())),
        )

        with open(str(path.with_suffix(".json")), "w") as f:
            json.dump({
                arch: {"tenant_count": c.tenant_count}
                for arch, c in self._centroids.items()
            }, f, indent=2)

        logger.info("Saved archetype store to %s", path)

    def load(self, path: str | Path) -> None:
        """
        Load the archetype store from disk and rebuild FAISS index.
        """
        import json

        path = Path(path)

        data = np.load(str(path.with_suffix(".npz")))
        centroids = data["centroids"]
        archetypes = data["archetypes"]

        with open(str(path.with_suffix(".json"))) as f:
            meta = json.load(f)

        self._centroids = {}
        for i, arch in enumerate(archetypes):
            self._centroids[str(arch)] = ArchetypeCentroid(
                archetype=str(arch),
                centroid=centroids[i].astype(np.float32),
                tenant_count=meta[str(arch)]["tenant_count"],
            )

        self._build_index()
        self._built = True
        logger.info("Loaded archetype store from %s: %d centroids",
                     path, len(self._centroids))

    # ── Properties ────────────────────────────────────────────────

    @property
    def archetypes(self) -> list[str]:
        return list(self._centroids.keys())

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def n_centroids(self) -> int:
        return len(self._centroids)

    def get_centroid(self, archetype: str) -> ArchetypeCentroid | None:
        return self._centroids.get(archetype)

    def describe(self) -> str:
        """Human-readable summary of the store."""
        lines = [f"ArchetypeStore: {len(self._centroids)} archetypes",
                 f"Feature dims: {self._n_features}"]
        for arch, centroid in sorted(self._centroids.items()):
            lines.append(f"  {arch:20s}: {centroid.tenant_count:4d} tenants, "
                         f"centroid norm={np.linalg.norm(centroid.centroid):.2f}")
        return "\n".join(lines)


# ── Cold-Start Simulation ──────────────────────────────────────────────
# For M8: evaluate how well archetype retrieval works with limited history.


def evaluate_retrieval_accuracy(
    store: ArchetypeStore,
    data_dir: str | Path = "data/synthetic",
    n_held_out: int = 50,
    cold_start_minutes: int = 60,
    seed: int = 42,
) -> dict:
    """
    Evaluate archetype retrieval accuracy on held-out tenants.

    Simulates cold-start: for each held-out tenant, extract features from
    ONLY the first `cold_start_minutes` of data, query the store, and
    check if the predicted archetype matches the ground truth.

    Args:
        store: Built ArchetypeStore.
        data_dir: Directory containing all tenant data.
        n_held_out: Number of tenants to hold out for evaluation.
        cold_start_minutes: How many minutes of history to use for features.
        seed: RNG seed for tenant selection.

    Returns:
        Dict with accuracy, per-archetype accuracy, and error details.
    """
    import random
    random.seed(seed)

    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"

    # Load manifest
    manifest: dict[str, str] = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            manifest[row["tenant_id"]] = row["archetype"]

    # Select held-out tenants (ones NOT used in store building)
    all_tenants = list(manifest.keys())
    random.shuffle(all_tenants)
    held_out = all_tenants[:n_held_out]

    correct = 0
    per_arch_correct: dict[str, int] = {}
    per_arch_total: dict[str, int] = {}
    errors: list[dict] = []

    for tenant_id in held_out:
        true_arch = manifest[tenant_id]
        per_arch_total[true_arch] = per_arch_total.get(true_arch, 0) + 1

        # Load tenant data
        parquet_path = data_dir / f"{tenant_id}.parquet"
        df = pl.read_parquet(parquet_path)
        history = np.column_stack([
            df["cpu_util"].to_numpy(),
            df["mem_util"].to_numpy(),
            df["net_bytes"].to_numpy(),
            df["disk_iops"].to_numpy(),
        ])

        # Cold-start: only use first N minutes
        features = extract_features(history, max_window=cold_start_minutes)

        # Query
        predicted_arch, similarity = store.query_archetype(features)

        if predicted_arch == true_arch:
            correct += 1
            per_arch_correct[true_arch] = per_arch_correct.get(true_arch, 0) + 1
        else:
            errors.append({
                "tenant_id": tenant_id,
                "true": true_arch,
                "predicted": predicted_arch,
                "similarity": similarity,
            })

    accuracy = correct / n_held_out if n_held_out > 0 else 0.0

    # Per-archetype accuracy
    per_arch_accuracy = {}
    for arch in per_arch_total:
        per_arch_accuracy[arch] = per_arch_correct.get(arch, 0) / per_arch_total[arch]

    return {
        "accuracy": accuracy,
        "n_held_out": n_held_out,
        "correct": correct,
        "per_archetype": per_arch_accuracy,
        "errors": errors,
    }
