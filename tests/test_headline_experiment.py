"""Tests for M6 headline experiment helpers."""

import numpy as np
import pytest

# Functions from the experiment module (importable without TimesFM)
# compute_win_rate is defined here for testability without TimesFM import
def compute_win_rate(
    ar_losses: list[float],
    baseline_losses: list[float],
) -> float:
    """Imported from experiments/m6_headline.py for testing."""
    n = min(len(ar_losses), len(baseline_losses))
    if n == 0:
        return 0.0
    wins = sum(1 for a, b in zip(ar_losses[:n], baseline_losses[:n]) if a < b)
    return wins / n


class TestComputeWinRate:
    def test_all_wins(self):
        ar = [0.01, 0.02, 0.03]
        bl = [0.10, 0.20, 0.30]
        assert compute_win_rate(ar, bl) == 1.0

    def test_all_losses(self):
        ar = [0.10, 0.20, 0.30]
        bl = [0.01, 0.02, 0.03]
        assert compute_win_rate(ar, bl) == 0.0

    def test_mixed(self):
        ar = [0.05, 0.15, 0.25]
        bl = [0.10, 0.10, 0.10]
        # ar[0] < bl[0] → win; ar[1] > bl[1] → loss; ar[2] > bl[2] → loss
        assert compute_win_rate(ar, bl) == 1 / 3

    def test_empty(self):
        assert compute_win_rate([], []) == 0.0
        assert compute_win_rate([0.1], []) == 0.0
        assert compute_win_rate([], [0.1]) == 0.0

    def test_mismatched_lengths(self):
        ar = [0.01, 0.02, 0.03]
        bl = [0.10, 0.20]
        # Uses min(len) = 2 → ar[0] < bl[0], ar[1] < bl[1] → 2/2 = 1.0
        assert compute_win_rate(ar, bl) == 1.0

    def test_identical(self):
        ar = [0.05, 0.05, 0.05]
        bl = [0.05, 0.05, 0.05]
        # Strict less-than — no wins
        assert compute_win_rate(ar, bl) == 0.0

    def test_nan_values(self):
        """NaN should not win since NaN < X is False for any X."""
        ar = [float("nan"), 0.01]
        bl = [0.10, 0.10]
        # NaN < 0.10 → False, 0.01 < 0.10 → True → 1/2
        assert compute_win_rate(ar, bl) == 0.5
