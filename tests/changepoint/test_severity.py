"""
Tests for SeverityChangeDetector.
"""

import numpy as np
import pytest
from datetime import date

from insurance_dynamics.changepoint import SeverityChangeDetector
from insurance_dynamics.changepoint.result import ChangeResult


class TestSeverityChangeDetectorBasic:
    def _make_detector(self, **kwargs):
        defaults = dict(
            prior_mu=np.log(3000),
            prior_kappa=1.0,
            prior_alpha=2.0,
            prior_beta=1.0,
            hazard=0.02,
            threshold=0.3,
        )
        defaults.update(kwargs)
        return SeverityChangeDetector(**defaults)

    def test_fit_returns_change_result(self):
        det = self._make_detector()
        sevs = np.random.lognormal(mean=8.0, sigma=0.3, size=30)
        result = det.fit(sevs)
        assert isinstance(result, ChangeResult)

    def test_fit_lengths(self):
        det = self._make_detector()
        T = 40
        sevs = np.random.lognormal(8.0, 0.3, T)
        result = det.fit(sevs)
        assert len(result.changepoint_probs) == T
        assert len(result.periods) == T

    def test_probs_in_range(self):
        det = self._make_detector()
        sevs = np.random.lognormal(8.0, 0.3, 50)
        result = det.fit(sevs)
        assert np.all(result.changepoint_probs >= 0)
        assert np.all(result.changepoint_probs <= 1)

    def test_log_transform_applied(self):
        """Log transform should make positive-only series work."""
        det = self._make_detector(log_transform=True)
        # Positive severities
        sevs = np.array([2000.0, 2500.0, 3000.0, 2800.0] * 10)
        result = det.fit(sevs)
        assert isinstance(result, ChangeResult)

    def test_negative_value_raises_when_log_transform(self):
        det = self._make_detector(log_transform=True)
        sevs = np.array([1000.0, -100.0, 2000.0])
        with pytest.raises(ValueError, match="positive"):
            det.fit(sevs)

    def test_no_log_transform(self):
        det = self._make_detector(log_transform=False)
        # Already log-transformed values
        sevs = np.random.normal(8.0, 0.3, 30)
        result = det.fit(sevs)
        assert isinstance(result, ChangeResult)

    def test_detector_type_is_severity(self):
        det = self._make_detector()
        result = det.fit(np.random.lognormal(8.0, 0.3, 20))
        assert result.detector_type == "severity"

    def test_detects_severity_break(self):
        """Step change in mean severity should be detected."""
        rng = np.random.default_rng(99)
        # Pre-break: mean £2000
        pre = rng.lognormal(np.log(2000), 0.2, 50)
        # Post-break: mean £4000 (e.g. Ogden rate change)
        post = rng.lognormal(np.log(4000), 0.2, 50)
        sevs = np.concatenate([pre, post])

        # The break is 2000->4000 (2x severity, 0.7 log units).
        # Max cp_prob is typically ~0.15-0.20 due to the diffuse prior spreading
        # probability over both regimes. Use threshold=0.12 to reliably detect.
        det = SeverityChangeDetector(
            prior_mu=np.log(2000),
            prior_kappa=1.0,
            prior_alpha=2.0,
            prior_beta=0.5,
            hazard=0.02,
            threshold=0.12,
        )
        result = det.fit(sevs)
        assert result.n_breaks >= 1

        break_indices = [b.period_index for b in result.detected_breaks]
        assert any(40 <= idx <= 65 for idx in break_indices), (
            f"Break indices: {break_indices}"
        )

    def test_no_break_stable_series(self):
        """Stable series should not produce breaks at high threshold."""
        rng = np.random.default_rng(0)
        sevs = rng.lognormal(8.0, 0.15, 80)
        det = SeverityChangeDetector(
            prior_mu=8.0, prior_kappa=1.0, prior_alpha=2.0, prior_beta=0.5,
            hazard=0.01, threshold=0.5
        )
        result = det.fit(sevs)
        assert result.n_breaks == 0

    def test_update_online(self):
        det = self._make_detector()
        sevs = np.random.lognormal(8.0, 0.3, 20)
        det.fit(sevs)
        prob = det.update(mean_severity=3000.0)
        assert 0.0 <= prob <= 1.0

    def test_update_negative_raises(self):
        det = self._make_detector(log_transform=True)
        det.fit(np.random.lognormal(8.0, 0.3, 5))
        with pytest.raises(ValueError, match="positive"):
            det.update(mean_severity=-100.0)

    def test_fit_with_claim_counts(self):
        """Passing claim_counts should not raise; counts used for metadata."""
        det = self._make_detector()
        sevs = np.random.lognormal(8.0, 0.3, 20)
        counts = np.random.poisson(10, 20)
        result = det.fit(sevs, claim_counts=counts)
        assert isinstance(result, ChangeResult)

    def test_claim_count_length_mismatch_raises(self):
        det = self._make_detector()
        sevs = np.random.lognormal(8.0, 0.3, 20)
        with pytest.raises(ValueError, match="same length"):
            det.fit(sevs, claim_counts=[10] * 10)

    def test_with_periods(self):
        det = self._make_detector()
        periods = [date(2020, m, 1) for m in range(1, 13)]
        sevs = np.random.lognormal(8.0, 0.3, 12)
        result = det.fit(sevs, periods=periods)
        assert result.periods == periods

    def test_meta_contains_params(self):
        det = self._make_detector(prior_mu=8.5)
        result = det.fit(np.random.lognormal(8.0, 0.3, 10))
        assert result.meta["prior_mu"] == 8.5
        assert result.meta["log_transform"] is True
