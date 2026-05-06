"""Property-based tests for cost-asymmetric loss functions (M3).

Uses Hypothesis to verify mathematical invariants that MUST hold for
the loss functions to be valid. These are the "airtight" tests referenced
in the CLAUDE.md spec — the patent attorney and arXiv reviewer need to
see that the loss functions are mathematically sound.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tsfm_autoresearch.losses import (
    SLATier,
    CostAsymmetricLoss,
    make_loss_for_sla,
)
from tsfm_autoresearch.tsfm_client import ForecastConfig, ProbabilisticForecast

# ── Strategies ─────────────────────────────────────────────────────────

# Generate float arrays with reasonable ranges (like normalized resource metrics)
# Shared shape strategy so y_true and y_pred always have the same dims
_shape_st = st.tuples(
    st.integers(min_value=1, max_value=24),  # horizon
    st.integers(min_value=1, max_value=4),   # resources
)

def matching_arrays(shape, elements=st.floats(min_value=0.0, max_value=2.0, allow_nan=False)):
    """Generate two arrays with identical shape."""
    return st.tuples(
        arrays(dtype=np.float64, shape=st.just(shape), elements=elements),
        arrays(dtype=np.float64, shape=st.just(shape), elements=elements),
    )

# Strategy: (shape, (y_true, y_pred))
matched_pair_arrays = _shape_st.flatmap(
    lambda s: st.tuples(st.just(s), matching_arrays(s))
).map(lambda t: t[1])  # Extract just (y_true, y_pred)

# Valid alpha values
valid_alphas = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Quantile levels (sorted, in (0,1))
quantile_levels_strategy = st.lists(
    st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=5, unique=True,
).map(sorted)


# ── Unit Tests: SLATier ────────────────────────────────────────────────


class TestSLATier:
    def test_premium_alpha(self):
        assert SLATier.PREMIUM.alpha == 0.90

    def test_standard_alpha(self):
        assert SLATier.STANDARD.alpha == 0.75

    def test_basic_alpha(self):
        assert SLATier.BASIC.alpha == 0.65

    def test_from_alpha_exact(self):
        assert SLATier.from_alpha(0.90) == SLATier.PREMIUM
        assert SLATier.from_alpha(0.75) == SLATier.STANDARD
        assert SLATier.from_alpha(0.65) == SLATier.BASIC

    def test_from_alpha_approximate(self):
        # Closest match
        assert SLATier.from_alpha(0.95) == SLATier.PREMIUM
        assert SLATier.from_alpha(0.73) == SLATier.STANDARD   # 0.73 closer to 0.75 than 0.65
        assert SLATier.from_alpha(0.62) == SLATier.BASIC      # 0.62 closer to 0.65 than 0.75


# ── Unit Tests: CostAsymmetricLoss ─────────────────────────────────────


class TestCostAsymmetricLoss:
    def test_alpha_validation(self):
        """Alpha must be in [0, 1]."""
        with pytest.raises(ValueError):
            CostAsymmetricLoss(alpha=-0.1)
        with pytest.raises(ValueError):
            CostAsymmetricLoss(alpha=1.1)

    def test_symmetric_point_loss_is_half_mae(self):
        """
        With α=0.5, point loss = 0.5 × MAE.

        The cost-asymmetric loss with α=0.5 weights both error directions
        equally at 0.5 each: L = 0.5·max(0, y-ŷ) + 0.5·max(0, ŷ-y) = 0.5·|y-ŷ|.
        This is intentional — α controls both asymmetry AND scale.
        """
        loss = CostAsymmetricLoss(alpha=0.5, per_resource=False)
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.5, 1.5], [2.5, 4.5]])

        result = loss.point_loss(y_true, y_pred)

        # Half MAE: 0.5 × mean(|errors|) = 0.5 × (0.5+0.5+0.5+0.5)/4 = 0.5 × 0.5 = 0.25
        expected = 0.25
        assert abs(result - expected) < 1e-10

    def test_alpha_1_0_only_underprediction(self):
        """With α=1.0, over-prediction should have zero cost."""
        loss = CostAsymmetricLoss(alpha=1.0, per_resource=False)

        # Over-prediction (forecast > actual) → cost should be 0
        y_true = np.array([1.0])
        y_pred = np.array([2.0])
        assert loss.point_loss(y_true, y_pred) == 0.0

        # Under-prediction (forecast < actual) → cost should be positive
        assert loss.point_loss(np.array([2.0]), np.array([1.0])) > 0.0

    def test_alpha_0_0_only_overprediction(self):
        """With α=0.0, under-prediction should have zero cost."""
        loss = CostAsymmetricLoss(alpha=0.0, per_resource=False)

        # Under-prediction (forecast < actual) → cost should be 0
        y_true = np.array([2.0])
        y_pred = np.array([1.0])
        assert loss.point_loss(y_true, y_pred) == 0.0

        # Over-prediction (forecast > actual) → cost should be positive
        assert loss.point_loss(np.array([1.0]), np.array([2.0])) > 0.0

    def test_shape_mismatch_raises(self):
        """Point loss should raise on shape mismatch."""
        loss = CostAsymmetricLoss()
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss.point_loss(np.array([1.0, 2.0]), np.array([1.0]))

    def test_quantile_shape_mismatch_raises(self):
        """Quantile loss should raise on shape mismatch."""
        loss = CostAsymmetricLoss()
        y_true = np.ones((10, 4))
        quantiles = np.ones((5, 4, 3))  # Wrong horizon
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss.quantile_loss(y_true, quantiles, [0.1, 0.5, 0.9])

    def test_quantile_level_count_mismatch_raises(self):
        """Quantile loss should raise if level count ≠ depth."""
        loss = CostAsymmetricLoss()
        y_true = np.ones((10, 4))
        quantiles = np.ones((10, 4, 3))
        with pytest.raises(ValueError, match="Quantile level count"):
            loss.quantile_loss(y_true, quantiles, [0.1, 0.5, 0.9, 0.95])


# ── Property-Based Tests (Hypothesis) ──────────────────────────────────
# These verify mathematical invariants that MUST hold for any valid inputs.


class TestLossInvariants:
    """Mathematical invariants verified by Hypothesis."""

    @given(
        pair=matched_pair_arrays,
        alpha=valid_alphas,
    )
    @settings(max_examples=200)
    def test_point_loss_nonnegative(self, pair, alpha):
        """Point loss must always be ≥ 0."""
        y_true, y_pred = pair
        loss = CostAsymmetricLoss(alpha=alpha)
        result = loss.point_loss(y_true, y_pred)
        assert result >= 0.0

    @given(
        pair=matched_pair_arrays,
        alpha=valid_alphas,
    )
    @settings(max_examples=200)
    def test_point_loss_zero_when_perfect(self, pair, alpha):
        """Point loss must be zero when forecast is perfect."""
        y_true, _ = pair
        assume(not np.any(np.isnan(y_true)))
        loss = CostAsymmetricLoss(alpha=alpha)
        result = loss.point_loss(y_true, y_true)
        assert abs(result) < 1e-10

    @given(
        pair=matched_pair_arrays,
        alpha=valid_alphas,
    )
    @settings(max_examples=200)
    def test_loss_monotonic_in_alpha(self, pair, alpha):
        """
        For a fixed (y_true, y_pred) pair with under-prediction,
        INCREASING alpha should INCREASE the loss.

        When y < ŷ (over-prediction), increasing α should DECREASE the loss.

        This is the core mathematical property: α controls the penalty
        on under-prediction via the weight α/(1-α) ratio.
        """
        y_true, y_pred = pair
        assume(not np.allclose(y_true, y_pred))

        loss_high = CostAsymmetricLoss(alpha=min(alpha + 0.1, 1.0)).point_loss(y_true, y_pred)
        loss_low = CostAsymmetricLoss(alpha=alpha).point_loss(y_true, y_pred)

        # Determine if the forecast is under-predicting or over-predicting on average
        mean_error = float((y_true - y_pred).mean())

        if mean_error > 0.01:
            # Under-prediction: higher α should increase loss
            assert loss_high >= loss_low - 1e-10, (
                f"Under-prediction (err={mean_error:.4f}): "
                f"loss(α={min(alpha+0.1,1.0):.2f})={loss_high:.6f} "
                f"should be ≥ loss(α={alpha:.2f})={loss_low:.6f}"
            )
        elif mean_error < -0.01:
            # Over-prediction: higher α should decrease loss
            assert loss_low >= loss_high - 1e-10, (
                f"Over-prediction (err={mean_error:.4f}): "
                f"loss(α={alpha:.2f})={loss_low:.6f} "
                f"should be ≥ loss(α={min(alpha+0.1,1.0):.2f})={loss_high:.6f}"
            )

    @given(
        pair=matched_pair_arrays,
    )
    @settings(max_examples=100)
    def test_symmetric_alpha_gives_symmetric_loss(self, pair):
        """
        With α=0.5, over-prediction and under-prediction of equal magnitude
        should have equal loss (symmetry property).
        """
        y_true, _ = pair
        assume(not np.any(np.isnan(y_true)))
        loss = CostAsymmetricLoss(alpha=0.5, per_resource=False)

        delta = np.abs(y_true) * 0.1 + 0.05
        over_pred = y_true + delta
        under_pred = y_true - delta

        loss_over = loss.point_loss(y_true, over_pred)
        loss_under = loss.point_loss(y_true, under_pred)

        assert abs(loss_over - loss_under) < 1e-8, (
            f"Symmetric loss should give equal cost for equal-magnitude "
            f"over and under prediction: {loss_over:.8f} vs {loss_under:.8f}"
        )

    @given(
        y_true=arrays(
            dtype=np.float64,
            shape=_shape_st,
            elements=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        ),
        quantile_levels=quantile_levels_strategy,
        alpha=valid_alphas,
    )
    @settings(max_examples=100)
    def test_quantile_loss_nonnegative(self, y_true, quantile_levels, alpha):
        """Quantile loss must always be ≥ 0."""
        loss = CostAsymmetricLoss(alpha=alpha)

        H, D = y_true.shape
        Q = len(quantile_levels)
        quantiles_arr = np.random.default_rng(42).uniform(0, 1, (H, D, Q))

        result = loss.quantile_loss(y_true, quantiles_arr, quantile_levels)
        assert result >= 0.0

    @given(
        y_true=arrays(
            dtype=np.float64,
            shape=_shape_st,
            elements=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        ),
        quantile_levels=quantile_levels_strategy,
    )
    @settings(max_examples=100)
    def test_perfect_quantiles_give_zero_loss(self, y_true, quantile_levels):
        """
        When quantile forecasts exactly match actuals, loss should be near zero.

        This happens when the forecast distribution is a point mass at the
        actual value — all quantiles equal the actual.
        """
        loss = CostAsymmetricLoss(alpha=0.5, per_resource=False)

        H, D = y_true.shape
        Q = len(quantile_levels)
        # All quantiles = actual → perfect forecast
        perfect_quantiles = np.repeat(y_true[:, :, np.newaxis], Q, axis=2)

        result = loss.quantile_loss(y_true, perfect_quantiles, quantile_levels)
        assert result < 1e-8

    @given(alpha=valid_alphas)
    @settings(max_examples=100)
    def test_premium_loss_higher_for_underprediction(self, alpha):
        """
        Premium SLA (α ≈ 0.90) should penalize under-prediction more
        than standard (α = 0.75) and basic (α = 0.65).
        """
        # Skip cases where the ordering is ambiguous
        assume(alpha > 0.5)

        y_true = np.array([[1.0, 1.0]])
        y_pred = np.array([[0.5, 0.5]])  # 50% under-prediction

        loss_premium = CostAsymmetricLoss(alpha=0.90).point_loss(y_true, y_pred)
        loss_standard = CostAsymmetricLoss(alpha=0.75).point_loss(y_true, y_pred)
        loss_basic = CostAsymmetricLoss(alpha=0.65).point_loss(y_true, y_pred)

        assert loss_premium >= loss_standard, (
            f"Premium ({loss_premium}) should penalize under-prediction ≥ "
            f"Standard ({loss_standard})"
        )
        assert loss_standard >= loss_basic, (
            f"Standard ({loss_standard}) should penalize under-prediction ≥ "
            f"Basic ({loss_basic})"
        )

    @given(
        pair=matched_pair_arrays,
        alpha=valid_alphas,
    )
    @settings(max_examples=100)
    def test_per_resource_mode_higher_or_equal(self, pair, alpha):
        """
        Per-resource loss should be D × the non-per-resource loss
        (since it sums over D dimensions and then averages).
        """
        y_true, y_pred = pair
        loss_flat = CostAsymmetricLoss(alpha=alpha, per_resource=False)
        loss_per = CostAsymmetricLoss(alpha=alpha, per_resource=True)

        result_flat = loss_flat.point_loss(y_true, y_pred)
        result_per = loss_per.point_loss(y_true, y_pred)

        # Per-resource sums means → D × flat (since flat averages all)
        # For 2-D arrays, per_resource sums the per-dimension means,
        # while flat averages everything.
        D = y_true.shape[1]
        expected = result_flat * D
        assert abs(result_per - expected) < 1e-10


# ── Integration: Loss + ForecastConfig ─────────────────────────────────


class TestMakeLossForSLA:
    def test_premium(self):
        loss = make_loss_for_sla("premium")
        assert loss.alpha == 0.90

    def test_standard(self):
        loss = make_loss_for_sla(SLATier.STANDARD)
        assert loss.alpha == 0.75

    def test_basic(self):
        loss = make_loss_for_sla("basic")
        assert loss.alpha == 0.65

    def test_invalid_sla(self):
        with pytest.raises(ValueError):
            make_loss_for_sla("enterprise")
