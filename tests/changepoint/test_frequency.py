"""
Tests for FrequencyChangeDetector.
"""

import numpy as np
import pytest
from datetime import date

from insurance_dynamics.changepoint import FrequencyChangeDetector
from insurance_dynamics.changepoint.result import ChangeResult, DetectedBreak


class TestFrequencyChangeDetectorBasic:
    def _make_detector(self, **kwargs):
        defaults = dict(prior_alpha=2.0, prior_beta=20.0, hazard=0.02, threshold=0.3)
        defaults.update(kwargs)
        return FrequencyChangeDetector(**defaults)

    def test_fit_returns_change_result(self):
        det = self._make_detector()
        counts = [5, 6, 4, 5, 7] * 10
        exposure = [100.0] * 50
        result = det.fit(counts, exposure)
        assert isinstance(result, ChangeResult)

    def test_fit_result_lengths(self):
        det = self._make_detector()
        T = 40
        counts = np.random.poisson(5, T)
        exposure = np.full(T, 100.0)
        result = det.fit(counts, exposure)
        assert len(result.periods) == T
        assert len(result.changepoint_probs) == T
        assert result.run_length_probs.shape[0] == T

    def test_fit_probs_in_range(self):
        det = self._make_detector()
        counts = np.random.poisson(5, 50)
        exposure = np.full(50, 100.0)
        result = det.fit(counts, exposure)
        assert np.all(result.changepoint_probs >= 0)
        assert np.all(result.changepoint_probs <= 1)

    def test_fit_with_period_labels(self):
        det = self._make_detector()
        periods = [f"2020-Q{i%4+1}" for i in range(20)]
        counts = np.random.poisson(5, 20)
        exposure = np.full(20, 100.0)
        result = det.fit(counts, exposure, periods=periods)
        assert result.periods == periods

    def test_mismatched_lengths_raises(self):
        det = self._make_detector()
        with pytest.raises(ValueError, match="same length"):
            det.fit([1, 2, 3], [100.0, 100.0])

    def test_zero_exposure_raises(self):
        det = self._make_detector()
        with pytest.raises(ValueError, match="positive"):
            det.fit([5, 0], [100.0, 0.0])

    def test_detector_type_is_frequency(self):
        det = self._make_detector()
        result = det.fit([5] * 20, [100.0] * 20)
        assert result.detector_type == "frequency"

    def test_no_breaks_constant_series(self):
        """Flat Poisson series should not trigger breaks at threshold=0.5."""
        rng = np.random.default_rng(0)
        det = FrequencyChangeDetector(
            prior_alpha=2.0, prior_beta=20.0, hazard=0.01, threshold=0.5
        )
        counts = rng.poisson(5, 80)
        exposure = np.full(80, 100.0)
        result = det.fit(counts, exposure)
        assert result.n_breaks == 0

    def test_detects_frequency_break(self):
        """Two-regime series: low then high frequency."""
        rng = np.random.default_rng(7)
        exposure = 1000.0
        counts = np.concatenate([
            rng.poisson(0.03 * exposure, 60),
            rng.poisson(0.10 * exposure, 60),
        ])
        exposures = np.full(120, exposure)

        det = FrequencyChangeDetector(
            prior_alpha=1.0, prior_beta=30.0, hazard=0.02, threshold=0.25
        )
        result = det.fit(counts, exposures)
        assert result.n_breaks >= 1

        # Break should be in period range [50, 70]
        break_indices = [b.period_index for b in result.detected_breaks]
        assert any(50 <= idx <= 70 for idx in break_indices), (
            f"Break indices: {break_indices}"
        )

    def test_exposure_weighting_changes_result(self):
        """
        Same claim counts but different exposures should give different
        changepoint probabilities.
        """
        rng = np.random.default_rng(42)
        counts = rng.poisson(5, 40)

        det1 = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=10.0, hazard=0.02)
        det2 = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=10.0, hazard=0.02)

        # Uniform vs varying exposure
        exposure_uniform = np.full(40, 100.0)
        exposure_varying = np.concatenate([np.full(20, 50.0), np.full(20, 200.0)])

        result1 = det1.fit(counts, exposure_uniform)
        result2 = det2.fit(counts, exposure_varying)

        # Results should differ
        assert not np.allclose(result1.changepoint_probs, result2.changepoint_probs)

    def test_update_online(self):
        det = self._make_detector()
        # Fit initial data
        counts = np.random.poisson(5, 20)
        exposure = np.full(20, 100.0)
        det.fit(counts, exposure)
        # Update with new period
        prob = det.update(n=5, exposure=100.0)
        assert 0.0 <= prob <= 1.0

    def test_update_zero_exposure_raises(self):
        det = self._make_detector()
        det.fit([5] * 5, [100.0] * 5)
        with pytest.raises(ValueError, match="positive"):
            det.update(n=5, exposure=0.0)

    def test_posterior_lambda(self):
        det = self._make_detector()
        det.fit([5] * 20, [100.0] * 20)
        alpha, beta = det.posterior_lambda(rl_idx=0)
        assert alpha > 0
        assert beta > 0

    def test_meta_contains_priors(self):
        det = self._make_detector(prior_alpha=3.0, prior_beta=15.0)
        result = det.fit([5] * 10, [100.0] * 10)
        assert result.meta["prior_alpha"] == 3.0
        assert result.meta["prior_beta"] == 15.0


class TestFrequencyWithUKPriors:
    def test_uk_events_enabled(self):
        det = FrequencyChangeDetector(
            hazard=0.01,
            threshold=0.3,
            uk_events=True,
            event_lines=["motor"],
            event_components=["frequency"],
        )
        periods = [date(2020, m, 1) for m in range(1, 13)]
        counts = np.random.poisson(5, 12)
        exposure = np.full(12, 100.0)
        result = det.fit(counts, exposure, periods=periods)
        # Should run without error
        assert isinstance(result, ChangeResult)

    def test_covid_period_has_higher_hazard(self):
        """Periods around March 2020 should have elevated hazard multiplier."""
        from insurance_dynamics.changepoint.priors import UKEventPrior
        prior = UKEventPrior(lines=["motor"], components=["frequency"])
        periods = [
            date(2020, 1, 1),
            date(2020, 3, 23),  # COVID lockdown
            date(2020, 6, 1),
        ]
        hazards = prior.hazard_series(periods, base_hazard=0.01)
        # COVID period should have highest hazard
        assert hazards[1] > hazards[0]

    def test_uk_events_result_has_meta_flag(self):
        det = FrequencyChangeDetector(uk_events=True, event_lines=["motor"])
        periods = [date(2021, m, 1) for m in range(1, 13)]
        counts = np.random.poisson(5, 12)
        exposure = np.full(12, 100.0)
        result = det.fit(counts, exposure, periods=periods)
        assert result.meta["uk_events"] is True
