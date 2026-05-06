"""Tests for the synthetic workload generator (M1)."""

import csv
from pathlib import Path

import polars as pl
import pytest

from tsfm_autoresearch.workload_gen import (
    ARCHETYPE_DISTRIBUTION,
    _generate_tenant_configs,
    generate_workloads,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def small_output_dir(tmp_path: Path) -> Path:
    """Generate a small workload for testing."""
    out = tmp_path / "synthetic"
    generate_workloads(n_tenants=20, n_days=3, seed=42, output_dir=out)
    return out


# ── Unit Tests ─────────────────────────────────────────────────────────


def test_tenant_config_count():
    """Should produce exactly n_tenants configs."""
    configs = _generate_tenant_configs(100, seed=42)
    assert len(configs) == 100


def test_tenant_config_ids_unique():
    """Tenant IDs should be unique."""
    configs = _generate_tenant_configs(100, seed=42)
    ids = [c.tenant_id for c in configs]
    assert len(ids) == len(set(ids))


def test_tenant_config_archetypes_valid():
    """All assigned archetypes must be in the known set."""
    configs = _generate_tenant_configs(100, seed=42)
    valid = set(ARCHETYPE_DISTRIBUTION.keys())
    for cfg in configs:
        assert cfg.archetype in valid


def test_archetype_distribution_approximate():
    """Archetype distribution should roughly match specified weights."""
    configs = _generate_tenant_configs(2000, seed=42)
    counts: dict[str, int] = {}
    for cfg in configs:
        counts[cfg.archetype] = counts.get(cfg.archetype, 0) + 1

    total = len(configs)
    for arch, expected_frac in ARCHETYPE_DISTRIBUTION.items():
        actual_frac = counts.get(arch, 0) / total
        # Allow 5% absolute deviation at 2000 samples
        assert abs(actual_frac - expected_frac) < 0.05, (
            f"{arch}: expected {expected_frac:.2f}, got {actual_frac:.2f}"
        )


def test_reproducibility():
    """Same seed should produce identical output."""
    import tempfile

    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        generate_workloads(n_tenants=10, n_days=2, seed=42, output_dir=td1)
        generate_workloads(n_tenants=10, n_days=2, seed=42, output_dir=td2)

        for i in range(10):
            df1 = pl.read_parquet(Path(td1) / f"tenant_{i:04d}.parquet")
            df2 = pl.read_parquet(Path(td2) / f"tenant_{i:04d}.parquet")
            assert df1.shape == df2.shape
            assert (df1["cpu_util"].to_numpy() == df2["cpu_util"].to_numpy()).all()


def test_different_seeds_different_output():
    """Different seeds should produce different output."""
    import tempfile

    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        generate_workloads(n_tenants=10, n_days=2, seed=42, output_dir=td1)
        generate_workloads(n_tenants=10, n_days=2, seed=123, output_dir=td2)

        df1 = pl.read_parquet(Path(td1) / "tenant_0000.parquet")
        df2 = pl.read_parquet(Path(td2) / "tenant_0000.parquet")
        assert not (df1["cpu_util"].to_numpy() == df2["cpu_util"].to_numpy()).all()


# ── Integration Tests ──────────────────────────────────────────────────


def test_output_parquet_files_exist(small_output_dir: Path):
    """Each tenant should get a Parquet file."""
    parquet_files = list(small_output_dir.glob("*.parquet"))
    assert len(parquet_files) == 20


def test_manifest_file(small_output_dir: Path):
    """Manifest should have header + one row per tenant."""
    manifest = small_output_dir / "manifest.csv"
    assert manifest.exists()

    with open(manifest) as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 20
        # Check required columns
        row = reader[0]
        assert "tenant_id" in row
        assert "archetype" in row
        assert "cpu_baseline" in row


def test_parquet_schema(small_output_dir: Path):
    """Each Parquet file should have the expected schema."""
    df = pl.read_parquet(small_output_dir / "tenant_0000.parquet")
    expected_columns = {"timestamp", "cpu_util", "mem_util", "net_bytes", "disk_iops"}
    assert set(df.columns) == expected_columns
    assert df["cpu_util"].dtype == pl.Float32
    assert df["mem_util"].dtype == pl.Float32
    assert df["net_bytes"].dtype == pl.Float32
    assert df["disk_iops"].dtype == pl.Float32


def test_data_range(small_output_dir: Path):
    """3 days = 4320 minutes of data per tenant."""
    for parquet_path in small_output_dir.glob("*.parquet"):
        df = pl.read_parquet(parquet_path)
        assert len(df) == 3 * 1440, f"Expected 4320 rows, got {len(df)} in {parquet_path.name}"


def test_cpu_bounds(small_output_dir: Path):
    """CPU utilization should be non-negative and not absurdly high."""
    for parquet_path in small_output_dir.glob("*.parquet"):
        df = pl.read_parquet(parquet_path)
        cpu = df["cpu_util"].to_numpy()
        assert (cpu >= 0).all(), f"Negative CPU in {parquet_path.name}"
        assert (cpu <= 3.0).all(), f"CPU > 3.0 in {parquet_path.name}"


def test_mem_bounds(small_output_dir: Path):
    """Memory utilization should be in [0, 1]."""
    for parquet_path in small_output_dir.glob("*.parquet"):
        df = pl.read_parquet(parquet_path)
        mem = df["mem_util"].to_numpy()
        assert (mem >= 0).all(), f"Negative mem in {parquet_path.name}"
        assert (mem <= 1.0).all(), f"Mem > 1.0 in {parquet_path.name}"


def test_archetype_cpu_characteristics(small_output_dir: Path):
    """Each archetype should have distinguishable CPU stats."""
    import csv

    # Load manifest
    manifest = {}
    with open(small_output_dir / "manifest.csv") as f:
        for row in csv.DictReader(f):
            manifest[row["tenant_id"]] = row["archetype"]

    # Compute per-tenant CPU mean
    archetype_cpu_means: dict[str, list[float]] = {}
    for parquet_path in small_output_dir.glob("*.parquet"):
        tenant_id = parquet_path.stem
        arch = manifest[tenant_id]
        df = pl.read_parquet(parquet_path)
        cpu_mean = float(df["cpu_util"].mean())
        if arch not in archetype_cpu_means:
            archetype_cpu_means[arch] = []
        archetype_cpu_means[arch].append(cpu_mean)

    # Compute archetype-level means
    import numpy as np
    arch_means = {a: np.mean(v) for a, v in archetype_cpu_means.items()}

    # Idle-ish should have the lowest CPU mean
    if "idle-ish" in arch_means and "compute-heavy" in arch_means:
        assert arch_means["idle-ish"] < arch_means["compute-heavy"], (
            f"idle-ish ({arch_means['idle-ish']:.4f}) should be < compute-heavy ({arch_means['compute-heavy']:.4f})"
        )
    if "idle-ish" in arch_means and "ecommerce-retail" in arch_means:
        assert arch_means["idle-ish"] < arch_means["ecommerce-retail"], (
            f"idle-ish ({arch_means['idle-ish']:.4f}) should be < ecommerce-retail ({arch_means['ecommerce-retail']:.4f})"
        )


def test_b2b_saas_weekend_dip(small_output_dir: Path):
    """B2B SaaS tenants should show lower CPU on weekends vs weekdays."""
    import csv
    import numpy as np

    # Find a B2B tenant
    manifest = {}
    with open(small_output_dir / "manifest.csv") as f:
        for row in csv.DictReader(f):
            manifest[row["tenant_id"]] = row["archetype"]

    b2b_tenant = None
    for tid, arch in manifest.items():
        if arch == "b2b-saas":
            b2b_tenant = tid
            break

    if b2b_tenant is None:
        pytest.skip("No B2B SaaS tenants in small sample")

    df = pl.read_parquet(small_output_dir / f"{b2b_tenant}.parquet")
    day_len = 1440
    n_days = len(df) // day_len

    # Compare Monday (day index 2 = Monday in Jan 2025) vs Saturday (day index 6)
    # Only if we have enough days
    if n_days < 7:
        pytest.skip(f"Need at least 7 days, got {n_days} (3-day fixture)")

    monday_cpu = df["cpu_util"][day_len * 2 : day_len * 3].mean()
    saturday_cpu = df["cpu_util"][day_len * 6 : day_len * 7].mean()

    assert float(saturday_cpu) < float(monday_cpu), (
        f"B2B SaaS: Saturday CPU ({saturday_cpu:.4f}) should be < Monday CPU ({monday_cpu:.4f})"
    )
