"""
Naive last-value baseline.

Predicts that all future values equal the last observed value.
This is the simplest possible forecaster and serves as a sanity-check
lower bound — any sophisticated method must beat this.
"""

from __future__ import annotations

import numpy as np


class NaiveLast:
    """
    Predict the last observed value for all future steps.

    forecast[t] = history[-1] for all t in [0, horizon).

    This is surprisingly competitive for near-zero-variance tenants
    (idle-ish, compute-heavy) but fails badly for tenants with trends
    or diurnal patterns.
    """

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if history.shape[0] == 0:
            raise ValueError("Cannot forecast from empty history")

        last = history[-1, :]  # (D,)
        return np.tile(last, (horizon, 1))  # (horizon, D)
