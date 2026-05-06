"""Tests for forecast baselines (M5)."""

import numpy as np
import pytest

from baselines.naive_last import NaiveLast
from baselines.naive_seasonal import NaiveSeasonal
from baselines.per_tenant_arima import PerTenantARIMA
from baselines.protocol import Forecaster


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_history() -> np.ndarray:
    """4 resources, 500 time steps with diurnal pattern."""
    rng = np.random.default_rng(42)
    T, D = 500, 4
    t = np.linspace(0, 5 * np.pi, T)
    return np.column_stack([
        0.3 + 0.1 * np.sin(t) + 0.02 * rng.normal(0, 1, T),
        0.5 + 0.05 * np.cos(t) + 0.01 * rng.normal(0, 1, T),
        100 + 20 * np.sin(t / 2) + rng.normal(0, 3, T),
        10 + 5 * np.cos(t / 3) + rng.normal(0, 1, T),
    ])


# ── Protocol Compliance ────────────────────────────────────────────────


class TestForecasterProtocol:
    def test_naive_last_is_forecaster(self):
        assert isinstance(NaiveLast(), Forecaster)

    def test_naive_seasonal_is_forecaster(self):
        assert isinstance(NaiveSeasonal(), Forecaster)

    def test_per_tenant_arima_is_forecaster(self):
        assert isinstance(PerTenantARIMA(), Forecaster)

    def test_forecaster_signature(self, sample_history):
        """All forecasters should accept (history, horizon) → (horizon, D)."""
        for fc in [NaiveLast(), NaiveSeasonal(), PerTenantARIMA()]:
            result = fc.forecast(sample_history, horizon=12)
            assert result.shape == (12, 4)
            assert result.dtype in (np.float32, np.float64)
            assert np.all(np.isfinite(result))


# ── NaiveLast Tests ────────────────────────────────────────────────────


class TestNaiveLast:
    def test_constant_forecast(self):
        """Should return the last value repeated."""
        history = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        fc = NaiveLast()
        result = fc.forecast(history, horizon=3)
        expected = np.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_single_timestep(self):
        fc = NaiveLast()
        result = fc.forecast(np.array([[1.0, 2.0, 3.0, 4.0]]), horizon=5)
        assert result.shape == (5, 4)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0, 4.0])

    def test_empty_history_raises(self):
        with pytest.raises(ValueError):
            NaiveLast().forecast(np.array([]).reshape(0, 4), horizon=4)

    def test_large_horizon(self, sample_history):
        result = NaiveLast().forecast(sample_history, horizon=1000)
        assert result.shape == (1000, 4)


# ── NaiveSeasonal Tests ────────────────────────────────────────────────


class TestNaiveSeasonal:
    def test_seasonal_pattern(self):
        """Should use lag=2 seasonal pattern."""
        history = np.array([
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0],
        ])
        fc = NaiveSeasonal(seasonal_lag=2)
        result = fc.forecast(history, horizon=4)

        # forecast[0] uses history[-2] = 5.0
        # forecast[1] uses history[-1] = 6.0
        # forecast[2] uses history[-2] = 5.0 (wraps)
        # forecast[3] uses history[-1] = 6.0
        expected = np.array([[5.0], [6.0], [5.0], [6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_default_24h_lag(self):
        """Default seasonal lag should be 1440 (24 hours at 1-min resolution)."""
        fc = NaiveSeasonal()
        assert fc._lag == 1440

    def test_short_history(self):
        """When history is shorter than lag, should use first available."""
        history = np.array([[1.0], [2.0], [3.0]])
        fc = NaiveSeasonal(seasonal_lag=100)
        result = fc.forecast(history, horizon=5)
        assert result.shape == (5, 1)
        assert np.all(np.isfinite(result))

    def test_multi_dimension(self, sample_history):
        fc = NaiveSeasonal(seasonal_lag=10)
        result = fc.forecast(sample_history, horizon=20)
        assert result.shape == (20, 4)


# ── PerTenantARIMA Tests ───────────────────────────────────────────────


class TestPerTenantARIMA:
    def test_forecast_shape(self, sample_history):
        fc = PerTenantARIMA()
        result = fc.forecast(sample_history, horizon=8)
        assert result.shape == (8, 4)
        assert np.all(np.isfinite(result))

    def test_constant_series(self):
        """ARIMA should handle constant series gracefully."""
        history = np.ones((100, 2), dtype=np.float64)
        fc = PerTenantARIMA()
        result = fc.forecast(history, horizon=5)
        assert result.shape == (5, 2)
        # Constant series → forecast should be near the constant
        assert np.allclose(result, 1.0, atol=0.01)

    def test_linear_trend(self):
        """ARIMA with d=1 should capture a linear trend."""
        t = np.arange(100, dtype=np.float64)
        history = np.column_stack([t, 2 * t])
        fc = PerTenantARIMA(max_p=1, max_d=1, max_q=0)
        result = fc.forecast(history, horizon=10)
        assert result.shape == (10, 2)
        # With d=1, ARIMA should extrapolate the trend
        # Last history value is 99, next should be ~100 for dim 0
        assert abs(result[0, 0] - 100.0) < 5.0, f"Expected ~100, got {result[0, 0]}"
        assert abs(result[0, 1] - 200.0) < 10.0, f"Expected ~200, got {result[0, 1]}"

    def test_single_value_series(self):
        """Single-value series should fall back gracefully."""
        history = np.array([[5.0, 10.0]])
        fc = PerTenantARIMA()
        result = fc.forecast(history, horizon=3)
        assert result.shape == (3, 2)
        assert np.all(np.isfinite(result))

    def test_different_orders(self):
        """Different ARIMA orders should all work."""
        history = np.random.randn(50, 1).astype(np.float64) + 10.0
        for order in [(1, 0, 0), (2, 0, 1), (1, 1, 1)]:
            fc = PerTenantARIMA(max_p=order[0], max_q=order[2], max_d=order[1])
            result = fc.forecast(history, horizon=3)
            assert result.shape == (3, 1)
            assert np.all(np.isfinite(result))
