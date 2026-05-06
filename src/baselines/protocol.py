"""
Forecaster protocol — common interface for all forecasters (baselines + autoresearch).

All forecasters conform to this protocol so they can be swapped in experiments.
The protocol defines the minimal surface area: given a multivariate history,
produce a point forecast for a specified horizon.

This enables head-to-head comparison in experiment scripts without
if/else chains for different forecaster types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """
    Protocol for time-series forecasters.

    Any forecaster (baseline or autoresearch) must implement:
      forecast(history, horizon) -> np.ndarray

    where history has shape (T, D) and the output has shape (horizon, D).
    """

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """
        Produce a point forecast.

        Args:
            history: Past observations, shape (T, D).
            horizon: Number of future steps to predict.

        Returns:
            Point forecast, shape (horizon, D).
        """
        ...
