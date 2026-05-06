"""
Cost-Asymmetric Loss Functions (M3).

Implements the α-parameterized cost-asymmetric loss that is the empirical
foundation of the patent claims. The key insight:

    **Under-prediction costs more than over-prediction in capacity planning.**

When you under-predict demand, you violate SLAs and lose revenue.
When you over-predict, you waste provisioned capacity (inefficient but not
catastrophic). The asymmetry parameter α controls this trade-off:

    α > 0.5 → under-prediction penalized more heavily
    α = 0.5 → symmetric (standard MAE)
    α < 0.5 → over-prediction penalized more heavily

SLA tiers map to α values:
  - premium (α=0.90): 90% weight on under-prediction — SLAs must not be violated
  - standard (α=0.75): 75% weight — moderate asymmetry
  - basic (α=0.65): 65% weight — closer to balanced, cost-sensitive

WHY THIS MATTERS FOR THE PATENT:
A single fixed-config forecast model must use one α for all tenants.
But different SLA tiers need different α values. The autoresearch loop
solves this by adapting the configuration per-request based on sla_tier,
achieving better cost-asymmetric performance without changing model weights.

PINBALL (QUANTILE) LOSS FORMULATION:
For quantile forecasts, we use the standard pinball loss with asymmetric
weighting across quantiles. The quantile at level q naturally encodes
asymmetry: q > 0.5 means "protect against under-prediction with probability q."
Our cost-asymmetric scoring weights pinball losses by α.

Author: Hermes Agent (Ziggy's TSFM-Autoresearch PoC)
Patent-pending inventive subject matter — see CLAUDE.md for context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


# ── SLA Tiers ──────────────────────────────────────────────────────────


class SLATier(str, Enum):
    """
    Service Level Agreement tiers mapping to cost-asymmetry parameters.

    Each tier defines how much more costly under-prediction is compared to
    over-prediction. Premium tenants pay for guaranteed capacity, basic
    tenants accept some risk of under-provisioning in exchange for lower cost.

    The α values are calibrated so that:
    - premium: miss rate target < 1% (99% quantile protection)
    - standard: miss rate target < 5% (95% quantile protection)
    - basic: miss rate target < 10% (90% quantile protection)
    """

    PREMIUM = "premium"   # α = 0.90
    STANDARD = "standard"  # α = 0.75
    BASIC = "basic"       # α = 0.65

    @property
    def alpha(self) -> float:
        """Cost asymmetry parameter: weight on under-prediction error."""
        return {SLATier.PREMIUM: 0.90, SLATier.STANDARD: 0.75, SLATier.BASIC: 0.65}[self]

    @classmethod
    def from_alpha(cls, alpha: float) -> SLATier:
        """Map an alpha value to the closest SLA tier."""
        distances = {t: abs(t.alpha - alpha) for t in cls}
        return min(distances, key=distances.get)  # type: ignore[arg-type]


# ── Cost-Asymmetric Loss ───────────────────────────────────────────────


@dataclass
class CostAsymmetricLoss:
    """
    α-weighted cost-asymmetric loss for capacity forecasting.

    The loss decomposes forecast error into under-prediction and
    over-prediction components, then weights them asymmetrically:

        L(y, ŷ) = α · under_prediction_error + (1-α) · over_prediction_error

    For point forecasts, this reduces to:

        L(y, ŷ) = α · max(0, y - ŷ) + (1-α) · max(0, ŷ - y)

    where y is actual demand and ŷ is forecast.

    For quantile forecasts, we use the α-weighted pinball loss:

        L_q(y, ŷ_q) = α · pinball(y, ŷ_q, q)   when q > 0.5
        L_q(y, ŷ_q) = (1-α) · pinball(y, ŷ_q, 1-q)  when q ≤ 0.5

    This ensures that low quantiles (q < 0.5) are penalized for over-prediction
    and high quantiles (q > 0.5) are penalized for under-prediction, with the
    asymmetry parameter α determining the relative importance.

    Attributes:
        alpha: Asymmetry parameter in [0, 1].
            α = 0.5: symmetric (equivalent to standard MAE/pinball)
            α > 0.5: under-prediction penalized more
            α < 0.5: over-prediction penalized more
        per_resource: If True, compute loss per resource dimension independently
            and sum. If False, compute loss on the concatenated array.
    """

    alpha: float = 0.5
    per_resource: bool = True

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    # ── Point Forecast Loss ──────────────────────────────────────

    def point_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """
        Compute cost-asymmetric point forecast loss.

        Args:
            y_true: Actual values, shape (horizon, D) or (N,).
            y_pred: Point forecast, same shape as y_true.

        Returns:
            Scalar loss value (lower is better).

        Raises:
            ValueError: If shapes don't match.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        error = y_true - y_pred  # positive = under-prediction

        # Asymmetric decomposition:
        # under_pred_err = max(0, y - ŷ) — we didn't allocate enough
        # over_pred_err  = max(0, ŷ - y) — we allocated too much
        under_pred = np.maximum(0, error)
        over_pred = np.maximum(0, -error)

        if self.per_resource and y_true.ndim == 2:
            # Per-resource: compute loss independently, then mean
            loss = self.alpha * under_pred + (1 - self.alpha) * over_pred
            return float(loss.mean(axis=0).sum())
        else:
            loss = self.alpha * under_pred + (1 - self.alpha) * over_pred
            return float(loss.mean())

    # ── Quantile Forecast Loss ───────────────────────────────────

    def quantile_loss(
        self,
        y_true: np.ndarray,
        quantiles: np.ndarray,
        quantile_levels: list[float],
    ) -> float:
        """
        Compute cost-asymmetric quantile (pinball) loss.

        For each quantile level q, compute the pinball loss:

            pinball(y, ŷ_q, q) = q · max(0, y - ŷ_q) + (1-q) · max(0, ŷ_q - y)

        Then weight by the cost asymmetry parameter α:
            - For q > 0.5 (protective quantiles): weight = α
            - For q ≤ 0.5 (conservative quantiles): weight = 1-α
            - For q = 0.5 (median): weight = 0.5 (symmetric)

        This weighting ensures that the loss reflects the business cost of
        capacity misallocation, not just statistical accuracy.

        Args:
            y_true: Actual values, shape (horizon, D).
            quantiles: Quantile forecasts, shape (horizon, D, num_quantiles).
            quantile_levels: Sorted list of quantile levels (e.g., [0.1, 0.5, 0.9]).

        Returns:
            Weighted pinball loss (lower is better).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        quantiles_arr = np.asarray(quantiles, dtype=np.float64)

        if y_true.shape != quantiles_arr.shape[:2]:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs "
                f"quantiles spatial dims {quantiles_arr.shape[:2]}"
            )

        if len(quantile_levels) != quantiles_arr.shape[2]:
            raise ValueError(
                f"Quantile level count {len(quantile_levels)} != "
                f"quantile depth {quantiles_arr.shape[2]}"
            )

        total_loss = 0.0
        D = y_true.shape[1]

        for qi, q in enumerate(quantile_levels):
            y_q = quantiles_arr[:, :, qi]  # (horizon, D)
            error = y_true - y_q  # positive = actual above quantile

            # Standard pinball loss
            pinball = q * np.maximum(0, error) + (1 - q) * np.maximum(0, -error)

            # Cost-asymmetric weighting
            # High quantiles (q > 0.5) protect against under-prediction → weight by α
            # Low quantiles (q < 0.5) reflect conservative estimates → weight by (1-α)
            # Median (q = 0.5) is symmetric
            if q > 0.5:
                weight = self.alpha
            elif q < 0.5:
                weight = 1 - self.alpha
            else:
                weight = 0.5

            if self.per_resource:
                total_loss += weight * float(pinball.mean(axis=0).sum())
            else:
                total_loss += weight * float(pinball.mean())

        # Normalize by number of quantile levels for comparability
        return total_loss / len(quantile_levels)

    # ── Combined Scoring ─────────────────────────────────────────

    def score(
        self,
        y_true: np.ndarray,
        forecast,  # ProbabilisticForecast
    ) -> float:
        """
        Compute the primary scoring metric for autoresearch.

        Uses quantile loss when quantiles are available, falls back
        to point loss otherwise. This is the metric that the autoresearch
        loop optimizes over.

        Args:
            y_true: Actual values, shape (horizon, D).
            forecast: ProbabilisticForecast from TSFMClient.

        Returns:
            Scalar cost-asymmetric score (lower is better).
        """
        if forecast.quantiles is not None and forecast.quantiles.size > 0:
            return self.quantile_loss(
                y_true, forecast.quantiles, forecast.quantile_levels
            )
        else:
            return self.point_loss(y_true, forecast.point)


# ── Utility: SLA-tiered Loss Factory ───────────────────────────────────


def make_loss_for_sla(sla_tier: SLATier | str) -> CostAsymmetricLoss:
    """
    Convenience factory for creating a loss function for a given SLA tier.

    Usage:
        loss_fn = make_loss_for_sla("premium")
        score = loss_fn.score(actuals, forecast)
    """
    if isinstance(sla_tier, str):
        sla_tier = SLATier(sla_tier)
    return CostAsymmetricLoss(alpha=sla_tier.alpha)
