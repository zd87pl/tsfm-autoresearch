"""Tests for M7 latency budget sweep helpers."""

import numpy as np
import pytest


class TestLatencyBudget:
    """Budget feasibility: p95 <= 200ms."""

    def test_all_under_budget(self):
        lats = np.array([10, 20, 30, 40, 50] * 20)  # all under 200
        assert np.percentile(lats, 95) <= 200

    def test_exceeds_budget(self):
        lats = np.array([10, 20, 30, 250, 300] * 20)
        assert not (np.percentile(lats, 95) <= 200)

    def test_boundary(self):
        # Exactly 200ms at p95
        lats = np.array([0] * 95 + [200] * 5)
        assert np.percentile(lats, 95) <= 200

    def test_boundary_exceeded(self):
        # Need >5% of values above 200 for p95 to exceed
        lats = np.array([0] * 90 + [201] * 10)
        assert not (np.percentile(lats, 95) <= 200)

    def test_empty(self):
        """Empty array should throw on percentile — guard in experiment code."""
        lats = np.array([])
        with pytest.raises(IndexError):
            np.percentile(lats, 95)


class TestKSweepAggregation:
    """Statistical aggregation across K sweep."""

    @pytest.fixture
    def mock_per_k_data(self):
        """Simulate per-K accumulated data."""
        rng = np.random.default_rng(42)
        K_SWEEP = [1, 2, 4, 8, 16, 32]
        data = {}
        base_loss = 0.050
        base_latency = 100
        for k in K_SWEEP:
            n = 100  # evaluations per K
            # Loss decreases with K (diminishing returns)
            loss = base_loss / np.sqrt(k) + rng.normal(0, 0.001, n)
            # Latency increases with K
            latency = base_latency * np.sqrt(k) + rng.normal(0, 5, n)
            data[k] = {"losses": loss.tolist(), "latencies": latency.tolist()}
        return data

    def test_loss_decreases_with_k(self, mock_per_k_data):
        """Higher K should give lower mean loss."""
        means = {
            k: np.mean(data["losses"])
            for k, data in mock_per_k_data.items()
        }
        # K=32 should have lower loss than K=1
        assert means[32] < means[1]
        # Diminishing returns: improvement slows
        improvement_1_8 = means[1] - means[8]
        improvement_8_32 = means[8] - means[32]
        assert improvement_1_8 > improvement_8_32

    def test_latency_increases_with_k(self, mock_per_k_data):
        """Higher K should have higher latency."""
        means = {
            k: np.mean(data["latencies"])
            for k, data in mock_per_k_data.items()
        }
        assert means[32] > means[1]

    def test_p95_greater_than_p50(self, mock_per_k_data):
        """p95 should always be >= p50."""
        for k, data in mock_per_k_data.items():
            lats = np.array(data["latencies"])
            p50 = np.median(lats)
            p95 = np.percentile(lats, 95)
            assert p95 >= p50

    def test_diminishing_returns(self, mock_per_k_data):
        """Δloss/ΔK should decrease (diminishing returns per config)."""
        means = {
            k: np.mean(data["losses"])
            for k, data in mock_per_k_data.items()
        }
        K_SWEEP = sorted(means.keys())
        deltas = []
        for i in range(1, len(K_SWEEP)):
            prev_k = K_SWEEP[i - 1]
            curr_k = K_SWEEP[i]
            delta = means[prev_k] - means[curr_k]
            dk = curr_k - prev_k
            deltas.append(delta / dk)
        # Later deltas should be smaller than early ones
        # (at least one comparison should hold)
        assert deltas[-1] <= deltas[0]
