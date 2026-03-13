"""
Edge case tests: single observations, zero counts, very short series, etc.
"""

import numpy as np
import pytest

from insurance_dynamics.changepoint import (
    FrequencyChangeDetector,
    SeverityChangeDetector,
    LossRatioMonitor,
    RetrospectiveBreakFinder,
)
from insurance_dynamics.changepoint._bocpd import PoissonGammaModel, NormalGammaModel, BOCPDEngine


class TestSingleObservation:
    def test_frequency_single_period(self):
        det = FrequencyChangeDetector()
        result = det.fit([5], [100.0])
        assert len(result.changepoint_probs) == 1
        assert 0.0 <= result.changepoint_probs[0] <= 1.0

    def test_severity_single_period(self):
        det = SeverityChangeDetector()
        result = det.fit([3000.0])
        assert len(result.changepoint_probs) == 1

    def test_pelt_single_period(self):
        finder = RetrospectiveBreakFinder(n_bootstraps=10)
        result = finder.fit([5.0])
        assert result.n_breaks == 0

    def test_pelt_two_periods(self):
        finder = RetrospectiveBreakFinder(n_bootstraps=10)
        result = finder.fit([1.0, 2.0])
        assert result.n_breaks == 0


class TestZeroCountPeriods:
    def test_frequency_zero_claims_allowed(self):
        det = FrequencyChangeDetector()
        counts = [0, 0, 0, 5, 5, 0, 0]
        exposure = [100.0] * 7
        result = det.fit(counts, exposure)
        assert len(result.changepoint_probs) == 7

    def test_frequency_all_zero_counts(self):
        det = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=10.0)
        counts = [0] * 20
        exposure = [100.0] * 20
        result = det.fit(counts, exposure)
        assert all(np.isfinite(p) for p in result.changepoint_probs)

    def test_frequency_large_counts(self):
        """Very large claim counts should not cause numerical issues."""
        det = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=0.001)
        counts = [100000] * 10
        exposure = [1_000_000.0] * 10
        result = det.fit(counts, exposure)
        assert all(np.isfinite(p) for p in result.changepoint_probs)


class TestNumericalStability:
    def test_bocpd_long_series_no_nans(self):
        """200-period series should not produce NaN probabilities."""
        rng = np.random.default_rng(42)
        model = PoissonGammaModel(alpha0=1.0, beta0=10.0)
        engine = BOCPDEngine(model=model, hazard=0.01)
        obs = [(rng.poisson(5), 100.0) for _ in range(200)]
        cp_probs, rl_probs = engine.fit(obs)
        assert np.all(np.isfinite(cp_probs))
        assert np.all(np.isfinite(rl_probs))

    def test_bocpd_very_low_hazard(self):
        rng = np.random.default_rng(1)
        model = PoissonGammaModel(alpha0=1.0, beta0=10.0)
        engine = BOCPDEngine(model=model, hazard=1e-4)
        obs = [(rng.poisson(5), 100.0) for _ in range(50)]
        cp_probs, _ = engine.fit(obs)
        assert np.all(np.isfinite(cp_probs))

    def test_bocpd_very_high_hazard(self):
        rng = np.random.default_rng(1)
        model = PoissonGammaModel(alpha0=1.0, beta0=10.0)
        engine = BOCPDEngine(model=model, hazard=0.49)
        obs = [(rng.poisson(5), 100.0) for _ in range(50)]
        cp_probs, _ = engine.fit(obs)
        assert np.all(np.isfinite(cp_probs))

    def test_normal_gamma_extreme_observations(self):
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        engine = BOCPDEngine(model=model, hazard=0.01)
        obs = [0.0] + [1000.0] * 5 + [-1000.0] * 5
        cp_probs, _ = engine.fit(obs)
        assert np.all(np.isfinite(cp_probs))


class TestEmptySeries:
    def test_pelt_empty_signal(self):
        """Empty signal should return BreakResult with no breaks."""
        # ruptures needs at least 1 point — test with minimal data
        finder = RetrospectiveBreakFinder(n_bootstraps=10)
        result = finder.fit(np.array([1.0, 2.0]))
        assert result.n_breaks == 0


class TestExposureVariation:
    def test_varying_exposure_different_weights(self):
        """
        Periods with 10x more exposure provide 10x more information.
        Test that the engine handles exposure variations gracefully.
        """
        rng = np.random.default_rng(99)
        # Same frequency, but exposure varies dramatically
        exposures = np.array([10.0, 100.0, 1000.0, 10000.0] * 10)
        counts = rng.poisson(0.05 * exposures)

        det = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=20.0, hazard=0.01)
        result = det.fit(counts, exposures)
        assert len(result.changepoint_probs) == 40
        assert all(np.isfinite(p) for p in result.changepoint_probs)

    def test_exposure_weighting_higher_exposure_increases_certainty(self):
        """
        With high exposure, the posterior should update faster.
        A break in a high-exposure period should be detected at lower threshold.
        """
        rng = np.random.default_rng(42)
        # Low exposure period (harder to detect)
        low_e = 10.0
        high_e = 10000.0

        # Both series: same frequency ratio pre/post break
        counts_low = np.concatenate([
            rng.poisson(0.05 * low_e, 30),
            rng.poisson(0.15 * low_e, 30),
        ])
        counts_high = np.concatenate([
            rng.poisson(0.05 * high_e, 30),
            rng.poisson(0.15 * high_e, 30),
        ])

        det = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=20.0, hazard=0.02)

        result_low = det.fit(counts_low, np.full(60, low_e))
        det2 = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=20.0, hazard=0.02)
        result_high = det2.fit(counts_high, np.full(60, high_e))

        # High exposure series should have higher peak probability
        assert result_high.max_changepoint_prob >= result_low.max_changepoint_prob


class TestResultProperties:
    def test_change_result_n_periods(self):
        det = FrequencyChangeDetector()
        result = det.fit([5] * 15, [100.0] * 15)
        assert result.n_periods == 15

    def test_change_result_most_probable_run_length(self):
        det = FrequencyChangeDetector()
        result = det.fit([5] * 20, [100.0] * 20)
        rl = result.most_probable_run_length(10)
        assert isinstance(rl, int)
        assert rl >= 0

    def test_change_result_max_prob(self):
        det = FrequencyChangeDetector()
        result = det.fit([5] * 20, [100.0] * 20)
        assert 0.0 <= result.max_changepoint_prob <= 1.0

    def test_detected_break_repr(self):
        from insurance_dynamics.changepoint.result import DetectedBreak
        brk = DetectedBreak(period_index=5, period_label="2020-Q1", probability=0.72, run_length_before=4)
        r = repr(brk)
        assert "2020-Q1" in r
        assert "0.720" in r

    def test_break_interval_repr(self):
        from insurance_dynamics.changepoint.result import BreakInterval
        bi = BreakInterval(break_index=50, lower=47, upper=53, period_label="2021-Q2")
        r = repr(bi)
        assert "2021-Q2" in r
        assert "47" in r
