"""
Tests for LossRatioMonitor.
"""

import numpy as np
import pytest

from insurance_dynamics.changepoint import LossRatioMonitor
from insurance_dynamics.changepoint.result import MonitorResult


class TestLossRatioMonitor:
    def _make_stable_data(self, T=60, seed=0):
        rng = np.random.default_rng(seed)
        exposure = 1000.0
        counts = rng.poisson(0.05 * exposure, T)
        exposures = np.full(T, exposure)
        premiums = np.full(T, 500000.0)
        # Mean severity £3000
        sevs = rng.lognormal(np.log(3000), 0.3, T)
        lr = (counts * sevs) / premiums
        return counts, exposures, premiums, sevs, lr

    def test_returns_monitor_result(self):
        counts, exposures, premiums, sevs, lr = self._make_stable_data()
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert isinstance(result, MonitorResult)

    def test_monitor_with_freq_and_sev(self):
        counts, exposures, premiums, sevs, lr = self._make_stable_data()
        monitor = LossRatioMonitor()
        result = monitor.monitor(
            claim_counts=counts,
            exposures=exposures,
            mean_severities=sevs,
        )
        assert result.frequency_result is not None
        assert result.severity_result is not None

    def test_monitor_from_loss_ratio(self):
        counts, exposures, premiums, sevs, lr = self._make_stable_data()
        monitor = LossRatioMonitor()
        result = monitor.monitor(
            loss_ratios=lr,
            premiums=premiums,
            claim_counts=counts,
            exposures=exposures,
        )
        assert isinstance(result, MonitorResult)

    def test_recommendation_monitor_stable(self):
        counts, exposures, premiums, sevs, lr = self._make_stable_data()
        monitor = LossRatioMonitor(threshold=0.5)
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert result.recommendation == "monitor"

    def test_recommendation_retrain_after_break(self):
        """Introduce an obvious frequency break."""
        rng = np.random.default_rng(7)
        exposure = 1000.0
        counts = np.concatenate([
            rng.poisson(0.03 * exposure, 60),
            rng.poisson(0.12 * exposure, 60),
        ])
        exposures = np.full(120, exposure)
        monitor = LossRatioMonitor(
            freq_prior_alpha=1.0, freq_prior_beta=30.0,
            hazard=0.02, threshold=0.25
        )
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert result.recommendation == "retrain"

    def test_combined_probs_shape(self):
        counts, exposures, premiums, sevs, lr = self._make_stable_data(T=50)
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert len(result.combined_probs) > 0

    def test_combined_probs_elementwise_max(self):
        """combined_probs should be >= each individual component."""
        counts, exposures, premiums, sevs, lr = self._make_stable_data(T=40)
        monitor = LossRatioMonitor()
        result = monitor.monitor(
            claim_counts=counts, exposures=exposures, mean_severities=sevs
        )
        T = len(result.combined_probs)
        freq_probs = result.frequency_result.changepoint_probs[:T]
        sev_probs = result.severity_result.changepoint_probs[:T]

        assert np.all(result.combined_probs >= freq_probs - 1e-9)
        assert np.all(result.combined_probs >= sev_probs - 1e-9)

    def test_no_data_raises(self):
        monitor = LossRatioMonitor()
        with pytest.raises(ValueError, match="No data"):
            monitor.monitor()

    def test_detected_breaks_deduplicated(self):
        """If both components detect a break at the same period, it should appear once."""
        rng = np.random.default_rng(7)
        exposure = 1000.0
        counts = np.concatenate([
            rng.poisson(0.03 * exposure, 60),
            rng.poisson(0.12 * exposure, 60),
        ])
        exposures = np.full(120, exposure)
        # Same break in severity too
        sevs = np.concatenate([
            rng.lognormal(np.log(2000), 0.2, 60),
            rng.lognormal(np.log(4000), 0.2, 60),
        ])
        monitor = LossRatioMonitor(
            freq_prior_alpha=1.0, freq_prior_beta=30.0,
            sev_prior_mu=np.log(2000), sev_prior_kappa=1.0,
            sev_prior_alpha=2.0, sev_prior_beta=0.5,
            hazard=0.02, threshold=0.25
        )
        result = monitor.monitor(
            claim_counts=counts, exposures=exposures, mean_severities=sevs
        )
        # No duplicate period indices
        indices = [b.period_index for b in result.detected_breaks]
        assert len(indices) == len(set(indices))

    def test_n_breaks_property(self):
        counts, exposures, *_ = self._make_stable_data()
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert result.n_breaks == len(result.detected_breaks)

    def test_with_uk_events(self):
        from datetime import date
        T = 24
        rng = np.random.default_rng(0)
        counts = rng.poisson(5, T)
        exposures = np.full(T, 100.0)
        periods = [date(2020, m % 12 + 1, 1) for m in range(T)]
        monitor = LossRatioMonitor(lines=["motor"], uk_events=True)
        result = monitor.monitor(
            claim_counts=counts, exposures=exposures, periods=periods
        )
        assert isinstance(result, MonitorResult)
