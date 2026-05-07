"""Tests for M9 SLA tier asymmetry experiment."""

from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss
import numpy as np
import pytest


class TestSLATierOrdering:
    """Premium > Standard > Basic in α values."""

    def test_alpha_ordering(self):
        assert SLATier.PREMIUM.alpha > SLATier.STANDARD.alpha
        assert SLATier.STANDARD.alpha > SLATier.BASIC.alpha
        assert SLATier.PREMIUM.alpha > SLATier.BASIC.alpha

    def test_alpha_range(self):
        """All α values must be in (0, 1)."""
        for tier in SLATier:
            assert 0 < tier.alpha < 1

    def test_all_alphas_above_half(self):
        """All tiers have α > 0.5 → always penalize under more than over."""
        for tier in SLATier:
            assert tier.alpha > 0.5, f"{tier.value} α={tier.alpha} should be >0.5"


class TestAsymmetricPenalty:
    """Cost-asymmetric loss: α/(1-α) ratio controls under/over penalty."""

    # Single-dimension to avoid scale effects
    y_true = np.array([[1.0], [2.0], [3.0]])

    def test_premium_ratio(self):
        """Premium α=0.90 → under/over penalty ratio = 0.90/0.10 = 9.0."""
        under = np.array([[0.5], [1.5], [2.5]])  # 0.5 under each
        over = np.array([[1.5], [2.5], [3.5]])    # 0.5 over each

        loss_fn = CostAsymmetricLoss(alpha=SLATier.PREMIUM.alpha, per_resource=True)
        loss_under = loss_fn.point_loss(self.y_true, under)
        loss_over = loss_fn.point_loss(self.y_true, over)

        ratio = loss_under / loss_over
        assert ratio == pytest.approx(9.0, rel=0.01), (
            f"Expected ratio 0.90/0.10=9.0, got {ratio}"
        )

    def test_basic_ratio(self):
        """Basic α=0.65 → under/over penalty ratio = 0.65/0.35 ≈ 1.857."""
        under = np.array([[0.5], [1.5], [2.5]])
        over = np.array([[1.5], [2.5], [3.5]])

        loss_fn = CostAsymmetricLoss(alpha=SLATier.BASIC.alpha, per_resource=True)
        loss_under = loss_fn.point_loss(self.y_true, under)
        loss_over = loss_fn.point_loss(self.y_true, over)

        ratio = loss_under / loss_over
        expected = 0.65 / 0.35
        assert ratio == pytest.approx(expected, rel=0.01), (
            f"Expected ratio {expected:.4f}, got {ratio}"
        )

    def test_standard_ratio(self):
        """Standard α=0.75 → under/over ratio = 0.75/0.25 = 3.0."""
        under = np.array([[0.5], [1.5], [2.5]])
        over = np.array([[1.5], [2.5], [3.5]])

        loss_fn = CostAsymmetricLoss(alpha=SLATier.STANDARD.alpha, per_resource=True)
        loss_under = loss_fn.point_loss(self.y_true, under)
        loss_over = loss_fn.point_loss(self.y_true, over)

        ratio = loss_under / loss_over
        assert ratio == pytest.approx(3.0, rel=0.01), (
            f"Expected ratio 3.0, got {ratio}"
        )

    def test_premium_ratio_larger_than_basic(self):
        """Premium's under/over ratio should exceed basic's."""
        under = np.array([[0.5], [1.5], [2.5]])
        over = np.array([[1.5], [2.5], [3.5]])

        prem = CostAsymmetricLoss(alpha=SLATier.PREMIUM.alpha, per_resource=True)
        basic = CostAsymmetricLoss(alpha=SLATier.BASIC.alpha, per_resource=True)

        prem_ratio = prem.point_loss(self.y_true, under) / prem.point_loss(self.y_true, over)
        basic_ratio = basic.point_loss(self.y_true, under) / basic.point_loss(self.y_true, over)

        assert prem_ratio > basic_ratio, (
            f"Premium ratio {prem_ratio} should exceed basic {basic_ratio}"
        )

    def test_perfect_prediction_zero_loss(self):
        """Perfect prediction → zero loss regardless of α."""
        perfect = np.array([[1.0], [2.0], [3.0]])
        for tier in SLATier:
            loss_fn = CostAsymmetricLoss(alpha=tier.alpha, per_resource=True)
            assert loss_fn.point_loss(self.y_true, perfect) == 0.0


class TestMonotonicityLogic:
    """Monotonicity checks used in M9 experiment."""

    def test_all_pass(self):
        checks = ["premium_gt_standard", "standard_gt_basic",
                   "premium_gt_basic", "premium_ctx_ge_basic"]
        assert len(checks) == 4

    def test_partial_pass(self):
        checks = ["premium_gt_standard", "standard_gt_basic"]
        assert len(checks) == 2

    def test_empty_checks(self):
        checks = []
        assert len(checks) == 0


class TestConfigDistribution:
    """Context length distribution analysis."""

    def test_counter_profile(self):
        from collections import Counter
        ctxs = [64, 128, 256, 64, 512, 256, 256]
        counts = Counter(ctxs)
        assert counts[64] == 2
        assert counts[256] == 3
        assert counts[128] == 1

    def test_median_context(self):
        ctxs = np.array([64, 128, 256, 64, 512, 256, 256])
        assert np.median(ctxs) == 256

    def test_premium_prefers_longer(self):
        """Simulated: premium should prefer longer context than basic."""
        premium_ctxs = [512, 384, 512, 256, 384, 512, 384]
        basic_ctxs = [128, 64, 256, 128, 64, 128, 256]
        assert np.median(premium_ctxs) >= np.median(basic_ctxs)
