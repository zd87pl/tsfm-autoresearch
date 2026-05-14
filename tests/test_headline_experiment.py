"""Tests for M6 headline experiment helpers."""

import math

import pytest
from scipy.stats import binomtest


# Mirror of compute_win_rate from experiments/m6_headline.py — kept here so
# the tests don't need TimesFM/scipy-via-experiment imports. Update both.
def compute_win_rate(ar_losses, baseline_losses):
    if len(ar_losses) != len(baseline_losses):
        raise ValueError(
            f"Paired losses must have equal length: "
            f"{len(ar_losses)} vs {len(baseline_losses)}"
        )
    n_total = len(ar_losses)
    if n_total == 0:
        return {"win_rate": 0.0, "n": 0, "wins": 0, "losses": 0,
                "ties": 0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 1.0}

    wins = sum(1 for a, b in zip(ar_losses, baseline_losses) if a < b)
    losses_n = sum(1 for a, b in zip(ar_losses, baseline_losses) if a > b)
    ties = n_total - wins - losses_n
    decided = wins + losses_n

    if decided == 0:
        return {"win_rate": 0.5, "n": n_total, "wins": 0, "losses": 0,
                "ties": ties, "p_value": 1.0, "ci_low": 0.0, "ci_high": 1.0}

    test = binomtest(wins, decided, p=0.5, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95)
    return {
        "win_rate": wins / decided,
        "n": n_total,
        "wins": wins,
        "losses": losses_n,
        "ties": ties,
        "p_value": float(test.pvalue),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
    }


class TestComputeWinRate:
    def test_all_wins(self):
        wr = compute_win_rate([0.01, 0.02, 0.03], [0.10, 0.20, 0.30])
        assert wr["win_rate"] == 1.0
        assert wr["wins"] == 3
        assert wr["losses"] == 0
        assert wr["ties"] == 0
        assert wr["p_value"] < 0.5

    def test_all_losses(self):
        wr = compute_win_rate([0.10, 0.20, 0.30], [0.01, 0.02, 0.03])
        assert wr["win_rate"] == 0.0
        assert wr["wins"] == 0
        assert wr["losses"] == 3

    def test_mixed(self):
        ar = [0.05, 0.15, 0.25]
        bl = [0.10, 0.10, 0.10]
        wr = compute_win_rate(ar, bl)
        assert wr["wins"] == 1
        assert wr["losses"] == 2
        assert wr["win_rate"] == pytest.approx(1 / 3)

    def test_empty(self):
        wr = compute_win_rate([], [])
        assert wr["win_rate"] == 0.0
        assert wr["n"] == 0

    def test_mismatched_lengths_raises(self):
        # Paired comparisons require equal-length input. The previous
        # silent min(len) fallback was the bug we fixed.
        with pytest.raises(ValueError):
            compute_win_rate([0.01, 0.02, 0.03], [0.10, 0.20])

    def test_identical_all_ties(self):
        wr = compute_win_rate([0.05, 0.05, 0.05], [0.05, 0.05, 0.05])
        # No decided pairs, so the test reports the neutral 0.5 default
        # rather than crashing on a 0-of-0 binomial.
        assert wr["wins"] == 0
        assert wr["losses"] == 0
        assert wr["ties"] == 3
        assert wr["win_rate"] == 0.5

    def test_nan_values_count_as_neither_win_nor_loss(self):
        ar = [float("nan"), 0.01]
        bl = [0.10, 0.10]
        wr = compute_win_rate(ar, bl)
        # NaN comparisons are False both ways → counted as ties.
        assert wr["wins"] == 1
        assert wr["losses"] == 0
        assert wr["ties"] == 1

    def test_significance_strong_signal(self):
        # 100 paired wins should be highly significant.
        ar = [0.01] * 100
        bl = [0.10] * 100
        wr = compute_win_rate(ar, bl)
        assert wr["win_rate"] == 1.0
        assert wr["p_value"] < 1e-20
        assert wr["ci_low"] > 0.95

    def test_significance_no_signal(self):
        ar = [0.05, 0.10] * 50
        bl = [0.10, 0.05] * 50
        wr = compute_win_rate(ar, bl)
        assert wr["win_rate"] == pytest.approx(0.5)
        # Two-sided binomial at exactly 50/50 returns p == 1.
        assert math.isclose(wr["p_value"], 1.0, rel_tol=1e-6)
