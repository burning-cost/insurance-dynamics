"""
Tests for RetrospectiveBreakFinder and _pelt module.
"""

import numpy as np
import pytest

from insurance_dynamics.changepoint import RetrospectiveBreakFinder
from insurance_dynamics.changepoint._pelt import find_breaks_pelt
from insurance_dynamics.changepoint.result import BreakResult, BreakInterval


class TestFindBreaksPelt:
    def test_basic_break_detection(self):
        """Simple step function should be detected."""
        rng = np.random.default_rng(42)
        signal = np.concatenate([
            rng.normal(0, 1, 50),
            rng.normal(5, 1, 50),
        ])
        result = find_breaks_pelt(signal, model="l2", penalty=3.0, n_bootstraps=50)
        assert result.n_breaks >= 1
        # Break should be near index 50
        assert any(40 <= b <= 60 for b in result.breaks)

    def test_no_break_constant_signal(self):
        """Constant signal should produce no breaks."""
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.5, 60)
        # High penalty to suppress false positives
        result = find_breaks_pelt(signal, model="l2", penalty=20.0, n_bootstraps=50)
        assert result.n_breaks == 0

    def test_returns_break_result(self):
        signal = np.random.normal(0, 1, 50)
        result = find_breaks_pelt(signal, n_bootstraps=50)
        assert isinstance(result, BreakResult)

    def test_break_cis_have_correct_count(self):
        rng = np.random.default_rng(1)
        signal = np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40)])
        result = find_breaks_pelt(signal, penalty=3.0, n_bootstraps=100)
        assert len(result.break_cis) == len(result.breaks)

    def test_ci_contains_true_break(self):
        """95% CI should contain the true break location."""
        rng = np.random.default_rng(5)
        true_break = 50
        signal = np.concatenate([
            rng.normal(0, 1, true_break),
            rng.normal(3, 1, 50),
        ])
        result = find_breaks_pelt(signal, penalty=3.0, n_bootstraps=200)
        if result.n_breaks > 0:
            ci = result.break_cis[0]
            assert ci.lower <= true_break <= ci.upper, (
                f"CI [{ci.lower}, {ci.upper}] does not contain true break {true_break}"
            )

    def test_ci_lower_leq_upper(self):
        rng = np.random.default_rng(2)
        signal = np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 40)])
        result = find_breaks_pelt(signal, penalty=3.0, n_bootstraps=100)
        for ci in result.break_cis:
            assert ci.lower <= ci.break_index <= ci.upper

    def test_bic_penalty(self):
        """BIC penalty string should work."""
        signal = np.concatenate([
            np.random.normal(0, 1, 40),
            np.random.normal(5, 1, 40),
        ])
        result = find_breaks_pelt(signal, penalty="bic", n_bootstraps=50)
        assert isinstance(result, BreakResult)
        # BIC penalty = log(80) ≈ 4.38
        assert np.isclose(result.penalty, np.log(80), atol=0.01)

    def test_too_short_signal(self):
        """Very short signal should return empty BreakResult without error."""
        signal = np.array([1.0, 2.0, 3.0])
        result = find_breaks_pelt(signal, n_bootstraps=10)
        assert result.n_breaks == 0

    def test_multiple_breaks(self):
        """Three-segment signal with low penalty should find 2 breaks."""
        rng = np.random.default_rng(10)
        signal = np.concatenate([
            rng.normal(0, 0.5, 30),
            rng.normal(5, 0.5, 30),
            rng.normal(2, 0.5, 30),
        ])
        result = find_breaks_pelt(signal, penalty=1.5, n_bootstraps=50)
        assert result.n_breaks >= 2

    def test_model_field_preserved(self):
        signal = np.random.normal(0, 1, 40)
        result = find_breaks_pelt(signal, model="rbf", penalty=5.0, n_bootstraps=20)
        assert result.model == "rbf"

    def test_n_bootstraps_field(self):
        signal = np.random.normal(0, 1, 40)
        result = find_breaks_pelt(signal, n_bootstraps=150, penalty=5.0)
        assert result.n_bootstraps == 150


class TestRetrospectiveBreakFinder:
    def test_basic_usage(self):
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
        finder = RetrospectiveBreakFinder(model="l2", penalty=3.0, n_bootstraps=50)
        result = finder.fit(signal)
        assert isinstance(result, BreakResult)
        assert result.n_breaks >= 1

    def test_with_periods(self):
        rng = np.random.default_rng(1)
        signal = np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)])
        periods = [f"2020-{i:02d}" for i in range(1, 61)]
        finder = RetrospectiveBreakFinder(n_bootstraps=50, penalty=2.0)
        result = finder.fit(signal, periods=periods)
        if result.n_breaks > 0:
            assert result.break_cis[0].period_label is not None
            assert isinstance(result.break_cis[0].period_label, str)

    def test_no_break_high_penalty(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, 60)
        finder = RetrospectiveBreakFinder(penalty=100.0, n_bootstraps=20)
        result = finder.fit(signal)
        assert result.n_breaks == 0

    def test_penalty_preserved(self):
        signal = np.random.normal(0, 1, 50)
        finder = RetrospectiveBreakFinder(penalty=7.5, n_bootstraps=20)
        result = finder.fit(signal)
        assert np.isclose(result.penalty, 7.5)

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(99)
        signal = np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)])
        finder1 = RetrospectiveBreakFinder(seed=42, n_bootstraps=100, penalty=3.0)
        finder2 = RetrospectiveBreakFinder(seed=42, n_bootstraps=100, penalty=3.0)
        r1 = finder1.fit(signal)
        r2 = finder2.fit(signal)
        if r1.n_breaks > 0 and r2.n_breaks > 0:
            assert r1.break_cis[0].lower == r2.break_cis[0].lower
            assert r1.break_cis[0].upper == r2.break_cis[0].upper
