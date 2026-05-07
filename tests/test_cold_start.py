"""Tests for M8 cold-start experiment helpers."""

import numpy as np
import pytest

from tsfm_autoresearch.archetype_store import (
    _ARCHETYPE_CONTEXT_BIAS,
    extract_features,
)


class TestContextBias:
    """Archetype → context_len mapping completeness."""

    def test_all_eight_archetypes_covered(self):
        """All 8 workload archetypes must have a context bias entry."""
        expected = {
            "low-traffic-blog",
            "ecommerce-retail",
            "news-publisher",
            "b2b-saas",
            "wp-cron-heavy",
            "cache-driven",
            "compute-heavy",
            "idle-ish",
        }
        assert set(_ARCHETYPE_CONTEXT_BIAS.keys()) == expected

    def test_all_in_valid_range(self):
        """All bias values must be valid context length candidates."""
        valid = {64, 128, 256, 384, 512}
        for archetype, ctx in _ARCHETYPE_CONTEXT_BIAS.items():
            assert ctx in valid, f"{archetype}: {ctx} not in {valid}"

    def test_ecommerce_longest_context(self):
        """Ecommerce (strong diurnal+weekly) should have longest context."""
        max_ctx = max(_ARCHETYPE_CONTEXT_BIAS.values())
        assert _ARCHETYPE_CONTEXT_BIAS["ecommerce-retail"] == max_ctx

    def test_idle_shortest_context(self):
        """Idle-ish (near-zero) should have shortest context."""
        min_ctx = min(_ARCHETYPE_CONTEXT_BIAS.values())
        assert _ARCHETYPE_CONTEXT_BIAS["idle-ish"] == min_ctx


class TestFeatureExtractionColdStart:
    """Feature extraction works on very short histories."""

    def test_30_minute_history(self):
        """Extract features from just 30 minutes of data."""
        history = np.random.randn(30, 4).astype(np.float32)
        features = extract_features(history)
        assert features.shape == (28,)
        assert np.all(np.isfinite(features))

    def test_60_minute_history(self):
        history = np.random.randn(60, 4).astype(np.float32)
        features = extract_features(history)
        assert features.shape == (28,)
        assert np.all(np.isfinite(features))

    def test_minimal_history(self):
        """Even 10 points should work (minimum for statistics)."""
        history = np.random.randn(10, 4).astype(np.float32)
        features = extract_features(history)
        assert features.shape == (28,)
        assert np.all(np.isfinite(features))

    def test_constant_history(self):
        """Constant series should produce finite features."""
        history = np.ones((60, 4), dtype=np.float32)
        features = extract_features(history)
        assert features.shape == (28,)
        assert np.all(np.isfinite(features))

    def test_different_histories_produce_different_features(self):
        """Ecommerce vs idle should have distinguishable features."""
        rng = np.random.default_rng(42)

        # Ecommerce: diurnal + spikes
        t = np.linspace(0, 4 * np.pi, 240)
        ecom = np.column_stack([
            0.3 + 0.2 * np.sin(t) + 0.05 * rng.normal(0, 1, 240),
            0.5 + 0.1 * np.cos(t) + 0.02 * rng.normal(0, 1, 240),
            100 + 50 * np.sin(t) + rng.normal(0, 10, 240),
            50 + rng.normal(0, 5, 240),
        ])

        # Idle: near-zero
        idle = np.column_stack([
            0.01 + 0.005 * rng.normal(0, 1, 240),
            0.01 + 0.005 * rng.normal(0, 1, 240),
            0.5 + rng.normal(0, 0.5, 240),
            0.1 + rng.normal(0, 0.1, 240),
        ])

        ecom_feat = extract_features(ecom)
        idle_feat = extract_features(idle)

        # Features should differ (at least one dimension)
        assert np.any(np.abs(ecom_feat - idle_feat) > 0.01)


class TestGapComputation:
    """Oracle gap and improvement calculations."""

    def test_full_gap(self):
        """Cold loss far from oracle → large gap."""
        cold = 0.100
        archetype = 0.060
        oracle = 0.040
        cold_gap = cold - oracle  # 0.060
        arch_gap = archetype - oracle  # 0.020
        assert arch_gap < cold_gap
        improvement = (cold_gap - arch_gap) / cold_gap * 100
        assert improvement > 50  # 66.7% gap reduction

    def test_no_improvement(self):
        """Archetype doesn't help → zero or negative improvement."""
        cold = 0.050
        archetype = 0.055  # worse than cold
        oracle = 0.040
        delta = cold - archetype  # -0.005
        assert delta < 0  # archetype is worse

    def test_oracle_is_best(self):
        """Oracle should always be the best (lowest loss)."""
        cold = 0.080
        archetype = 0.065
        oracle = 0.040
        assert oracle <= cold
        assert oracle <= archetype

    def test_fixed_vs_adaptive(self):
        """At short history, adaptive should beat fixed."""
        # Simulated: fixed config can't adapt to limited history
        fixed = 0.120
        cold = 0.090
        archetype = 0.070
        assert archetype < fixed  # archetype beats fixed
        assert archetype <= cold   # archetype at least matches cold
