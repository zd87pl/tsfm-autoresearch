"""
Naive seasonal baseline.

Predicts that future values equal the values from 24 hours ago.
For 1-minute resolution data, the seasonal lag is 24 × 60 = 1440 steps.

This baseline exploits the strong diurnal patterns in ecommerce, news,
and B2B SaaS tenants. It fails for tenants without daily seasonality
(wp-cron-heavy, idle-ish) or with trend components.
"""

from __future__ import annotations

import numpy as np

# Seasonal lag: 24 hours at 1-minute resolution
SEASONAL_LAG = 24 * 60  # 1440


class NaiveSeasonal:
    """
    Predict using the seasonal naive method: forecast[t] = history[t - lag].

    For any forecast step t, uses the value from `lag` steps ago in the
    history. If the history is shorter than `lag`, falls back to the
    oldest available value. If the forecast horizon extends beyond the
    available seasonal pattern, repeats from the beginning.
    """

    def __init__(self, seasonal_lag: int = SEASONAL_LAG):
        self._lag = seasonal_lag

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        T = history.shape[0]
        if T == 0:
            raise ValueError("Cannot forecast from empty history")

        lag = min(self._lag, T)

        forecast = np.zeros((horizon, history.shape[1]), dtype=history.dtype)

        for t in range(horizon):
            # Index in history: T - lag + (t % lag)
            idx = T - lag + (t % lag)
            if idx >= T:
                idx = T - 1
            forecast[t, :] = history[idx, :]

        return forecast
