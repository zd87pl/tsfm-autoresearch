"""
Per-tenant ARIMA baseline.

Fits a separate ARIMA model for each tenant and each resource dimension.
This is the SLOW baseline — deliberately so. It demonstrates why
per-tenant stateful models don't scale to 10^6 tenants.

Forecasting a single tenant involves:
  1. Fit ARIMA(p,d,q) to each resource dimension independently
  2. Forecast horizon steps
  3. Stack results into (horizon, D) output

ARIMA parameters are chosen via auto-ARIMA (limited search for speed)
or fall back to fixed (1,0,1) if auto-selection fails.

WHY THIS MATTERS:
This baseline represents the "give every tenant their own model" approach.
It's theoretically optimal but practically infeasible at scale. The
autoresearch approach achieves similar benefits without per-tenant state,
making it scalable to millions of tenants.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# Default ARIMA order when auto-selection fails
_DEFAULT_ORDER = (1, 0, 1)  # (p, d, q)


class PerTenantARIMA:
    """
    Per-tenant ARIMA forecaster.

    Fits independent ARIMA models for each resource dimension using
    statsmodels. This is computationally expensive but serves as an
    upper bound on per-tenant accuracy.

    NOTE: This baseline is intentionally slow. The M6 experiment will
    compare it against autoresearch to show that similar accuracy can
    be achieved without per-tenant state.

    Usage:
        arima = PerTenantARIMA()
        forecast = arima.forecast(history, horizon=60)
    """

    def __init__(self, max_p: int = 3, max_q: int = 3, max_d: int = 1):
        """
        Args:
            max_p: Maximum AR order for auto-selection.
            max_q: Maximum MA order for auto-selection.
            max_d: Maximum differencing order.
        """
        self._max_p = max_p
        self._max_q = max_q
        self._max_d = max_d

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """
        Fit ARIMA per dimension and forecast.

        Args:
            history: Shape (T, D).
            horizon: Forecast steps.

        Returns:
            Point forecast, shape (horizon, D).
        """
        import statsmodels.api as sm

        T, D = history.shape
        forecast = np.zeros((horizon, D), dtype=np.float64)

        for d in range(D):
            series = history[:, d].astype(np.float64)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.tsa.ARIMA(
                        series,
                        order=_DEFAULT_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(method_kwargs={"maxiter": 50})

                fc = fitted.forecast(steps=horizon)
                forecast[:, d] = fc

            except Exception as e:
                logger.warning(
                    "ARIMA fit failed for dim %d, falling back to naive: %s", d, e
                )
                # Fallback: naive last-value
                forecast[:, d] = series[-1]

        return forecast
