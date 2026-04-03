"""
Extended tests for changepoint module: result dataclasses, report, plot,
loss_ratio monitor edge cases, pelt edge cases, and priors edge cases.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from insurance_dynamics.changepoint import (
    FrequencyChangeDetector,
    LossRatioMonitor,
    SeverityChangeDetector,
    RetrospectiveBreakFinder,
    ConsumerDutyReport,
    UKEventPrior,
)
from insurance_dynamics.changepoint.result import (
    BreakInterval,
    BreakResult,
    ChangeResult,
    DetectedBreak,
    MonitorResult,
)
from insurance_dynamics.changepoint._pelt import (
    _bic_penalty,
    _block_bootstrap,
    _run_pelt,
)
from insurance_dynamics.changepoint.report import _prob_class
from insurance_dynamics.changepoint.priors import UKEvent, UK_EVENTS


# ---------------------------------------------------------------------------
# Result dataclass edge cases
# ---------------------------------------------------------------------------

class TestDetectedBreakRepr:
    def test_repr_with_none_period_label(self):
        brk = DetectedBreak(
            period_index=10,
            period_label=None,
            probability=0.45,
            run_length_before=9,
        )
        r = repr(brk)
        assert "10" in r
        assert "0.450" in r

    def test_repr_with_date_label(self):
        brk = DetectedBreak(
            period_index=5,
            period_label=date(2020, 3, 23),
            probability=0.88,
            run_length_before=4,
        )
        r = repr(brk)
        assert "2020" in r
        assert "0.880" in r

    def test_probability_attribute(self):
        brk = DetectedBreak(
            period_index=3, period_label="Q1", probability=0.72, run_length_before=2
        )
        assert brk.probability == pytest.approx(0.72)


class TestBreakIntervalRepr:
    def test_repr_with_no_period_label(self):
        bi = BreakInterval(break_index=42, lower=39, upper=45)
        r = repr(bi)
        assert "42" in r
        assert "39" in r
        assert "45" in r

    def test_repr_with_period_label(self):
        bi = BreakInterval(break_index=30, lower=28, upper=32, period_label="2021-06")
        r = repr(bi)
        assert "2021-06" in r

    def test_fields_accessible(self):
        bi = BreakInterval(break_index=10, lower=8, upper=12, period_label="test")
        assert bi.break_index == 10
        assert bi.lower == 8
        assert bi.upper == 12
        assert bi.period_label == "test"


class TestChangeResultProperties:
    def _make_result(self, T=20, n_breaks=0):
        rng = np.random.default_rng(0)
        det = FrequencyChangeDetector(threshold=0.5)
        counts = rng.poisson(5, T)
        exposure = np.full(T, 100.0)
        return det.fit(counts, exposure)

    def test_n_periods(self):
        r = self._make_result(T=15)
        assert r.n_periods == 15

    def test_n_breaks_zero(self):
        r = self._make_result()
        assert r.n_breaks == len(r.detected_breaks)

    def test_max_changepoint_prob_empty(self):
        """Edge: ChangeResult with empty changepoint_probs."""
        r = ChangeResult(
            periods=[],
            changepoint_probs=np.array([]),
            run_length_probs=np.empty((0, 0)),
            detected_breaks=[],
            detector_type="frequency",
            hazard_used=0.01,
        )
        assert r.max_changepoint_prob == 0.0

    def test_most_probable_run_length_at_each_step(self):
        r = self._make_result(T=10)
        for t in range(10):
            rl = r.most_probable_run_length(t)
            assert isinstance(rl, int)
            assert rl >= 0

    def test_detector_type_preserved(self):
        det = SeverityChangeDetector()
        sevs = np.random.lognormal(8.0, 0.3, 10)
        r = det.fit(sevs)
        assert r.detector_type == "severity"


class TestBreakResultProperties:
    def test_n_breaks_zero(self):
        r = BreakResult(breaks=[], break_cis=[], n_bootstraps=50, penalty=5.0, model="l2")
        assert r.n_breaks == 0

    def test_n_breaks_positive(self):
        ci = BreakInterval(break_index=30, lower=28, upper=32)
        r = BreakResult(breaks=[30], break_cis=[ci], n_bootstraps=100, penalty=3.0, model="l2")
        assert r.n_breaks == 1

    def test_periods_field(self):
        r = BreakResult(
            breaks=[],
            break_cis=[],
            n_bootstraps=0,
            penalty=0.0,
            model="l2",
            periods=["2020-Q1", "2020-Q2"],
        )
        assert r.periods == ["2020-Q1", "2020-Q2"]


class TestMonitorResultProperties:
    def _make_monitor_result(self):
        rng = np.random.default_rng(7)
        exposure = 1000.0
        counts = rng.poisson(0.05 * exposure, 40)
        exposures = np.full(40, exposure)
        monitor = LossRatioMonitor(threshold=0.5)
        return monitor.monitor(claim_counts=counts, exposures=exposures)

    def test_n_breaks(self):
        r = self._make_monitor_result()
        assert r.n_breaks == len(r.detected_breaks)

    def test_recommendation_is_valid(self):
        r = self._make_monitor_result()
        assert r.recommendation in ("retrain", "monitor")

    def test_combined_probs_nonnegative(self):
        r = self._make_monitor_result()
        assert np.all(r.combined_probs >= 0)

    def test_combined_probs_le_one(self):
        r = self._make_monitor_result()
        assert np.all(r.combined_probs <= 1.0)


# ---------------------------------------------------------------------------
# _bic_penalty
# ---------------------------------------------------------------------------

class TestBicPenalty:
    def test_returns_log_n(self):
        n = 100
        pen = _bic_penalty(n, n_breaks=0)
        assert pen == pytest.approx(np.log(n))

    def test_monotone_in_n(self):
        pens = [_bic_penalty(n, 0) for n in [10, 50, 100, 500]]
        for i in range(len(pens) - 1):
            assert pens[i] < pens[i + 1]

    def test_n_equals_1(self):
        pen = _bic_penalty(1, 0)
        assert pen == pytest.approx(0.0)  # log(1) = 0


# ---------------------------------------------------------------------------
# _block_bootstrap
# ---------------------------------------------------------------------------

class TestBlockBootstrap:
    def test_output_length_matches_input(self):
        rng = np.random.default_rng(42)
        signal = np.arange(50, dtype=float)
        resampled = _block_bootstrap(signal, block_size=5, rng=rng)
        assert len(resampled) == 50

    def test_values_from_original(self):
        rng = np.random.default_rng(1)
        signal = np.arange(30, dtype=float)
        resampled = _block_bootstrap(signal, block_size=5, rng=rng)
        # All values should be integers 0–29 (elements of signal)
        for v in resampled:
            assert v in signal

    def test_block_size_1(self):
        rng = np.random.default_rng(99)
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        resampled = _block_bootstrap(signal, block_size=1, rng=rng)
        assert len(resampled) == 5

    def test_block_size_equals_length(self):
        rng = np.random.default_rng(0)
        signal = np.array([10.0, 20.0, 30.0])
        resampled = _block_bootstrap(signal, block_size=3, rng=rng)
        assert len(resampled) == 3

    def test_2d_signal(self):
        rng = np.random.default_rng(42)
        signal = np.random.normal(0, 1, (30, 2))
        resampled = _block_bootstrap(signal, block_size=5, rng=rng)
        assert resampled.shape[0] == 30
        assert resampled.shape[1] == 2


# ---------------------------------------------------------------------------
# UKEventPrior edge cases
# ---------------------------------------------------------------------------

class TestUKEventPriorEdgeCases:
    def test_empty_events_returns_base_hazard(self):
        prior = UKEventPrior(events=[])
        periods = [date(2020, m, 1) for m in range(1, 13)]
        hazards = prior.hazard_series(periods, base_hazard=0.01)
        assert np.allclose(hazards, 0.01)

    def test_empty_events_multiplier_is_one(self):
        prior = UKEventPrior(events=[])
        mult = prior.hazard_multiplier_for_date(date(2020, 3, 23))
        assert mult == 1.0

    def test_period_to_date_fn_exception_falls_back(self):
        """If period_to_date_fn raises, the period gets base hazard."""
        prior = UKEventPrior()
        def bad_fn(p):
            raise ValueError("bad period")
        hazards = prior.hazard_series(["bad_period"], base_hazard=0.02, period_to_date_fn=bad_fn)
        assert hazards[0] == pytest.approx(0.02)

    def test_window_days_boundary(self):
        """Event exactly at window boundary should match."""
        event = UKEvent(
            name="Test",
            event_date=date(2020, 6, 1),
            affected_lines=["motor"],
            affected_component="frequency",
            hazard_multiplier=10.0,
            confidence="hard",
            window_days=30,
        )
        prior = UKEventPrior(events=[event])
        # Exactly at window boundary: 30 days later
        d_at_boundary = date(2020, 7, 1)
        mult = prior.hazard_multiplier_for_date(d_at_boundary)
        assert mult == 10.0  # |30| <= 30

        # Just past boundary: 31 days later
        d_past = date(2020, 7, 2)
        mult_past = prior.hazard_multiplier_for_date(d_past)
        assert mult_past == 1.0  # |31| > 30

    def test_liability_line_filter(self):
        prior = UKEventPrior(lines=["liability"])
        # All events should affect liability
        for ev in prior.events:
            lines_lower = [l.lower() for l in ev.affected_lines]
            assert "liability" in lines_lower or "all" in lines_lower

    def test_all_components_when_no_filter(self):
        prior = UKEventPrior()
        components = {ev.affected_component for ev in prior.events}
        assert len(components) > 1  # multiple component types

    def test_summary_completeness(self):
        prior = UKEventPrior()
        summary = prior.summary()
        required_keys = {"name", "date", "lines", "component", "multiplier", "confidence", "notes"}
        for ev_dict in summary:
            for key in required_keys:
                assert key in ev_dict, f"Missing key '{key}' in event summary"

    def test_max_hazard_cap_exact(self):
        """base_hazard * multiplier > max_hazard should be capped exactly."""
        event = UKEvent(
            name="Huge",
            event_date=date(2020, 6, 1),
            affected_lines=["motor"],
            affected_component="frequency",
            hazard_multiplier=100.0,
            confidence="hard",
            window_days=30,
        )
        prior = UKEventPrior(events=[event])
        hazards = prior.hazard_series([date(2020, 6, 1)], base_hazard=0.01, max_hazard=0.25)
        assert hazards[0] == pytest.approx(0.25)

    def test_multiple_events_same_date_takes_max(self):
        events = [
            UKEvent("A", date(2020, 6, 1), ["motor"], "frequency", 10.0, "hard", 30),
            UKEvent("B", date(2020, 6, 1), ["motor"], "frequency", 30.0, "hard", 30),
            UKEvent("C", date(2020, 6, 1), ["motor"], "frequency", 20.0, "hard", 30),
        ]
        prior = UKEventPrior(events=events)
        mult = prior.hazard_multiplier_for_date(date(2020, 6, 1))
        assert mult == 30.0

    def test_hazard_series_with_date_objects(self):
        prior = UKEventPrior(lines=["motor"])
        periods = [date(2017, 3, 20)]  # Ogden -0.75% date
        hazards = prior.hazard_series(periods, base_hazard=0.01)
        # Should be elevated (Ogden affects motor, severity — not frequency)
        # But UKEventPrior without component filter will include severity events too
        assert hazards[0] >= 0.01


# ---------------------------------------------------------------------------
# _run_pelt edge cases
# ---------------------------------------------------------------------------

class TestRunPelt:
    def test_no_break_constant_series(self):
        """High penalty on constant series → no breaks."""
        signal = np.ones(50)
        breaks = _run_pelt(signal, model="l2", penalty=100.0)
        assert breaks == []

    def test_break_detected_step_function(self):
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 0.3, 30), rng.normal(5, 0.3, 30)])
        breaks = _run_pelt(signal, model="l2", penalty=2.0)
        assert len(breaks) >= 1
        assert any(25 <= b <= 35 for b in breaks)

    def test_2d_signal_input(self):
        """2-D signal should be accepted by _run_pelt."""
        rng = np.random.default_rng(1)
        signal = rng.normal(0, 1, (60, 2))
        # May or may not detect breaks — just should not crash
        breaks = _run_pelt(signal, model="l2", penalty=5.0)
        assert isinstance(breaks, list)


# ---------------------------------------------------------------------------
# RetrospectiveBreakFinder edge cases
# ---------------------------------------------------------------------------

class TestRetrospectiveBreakFinderExtended:
    def test_short_signal_below_threshold(self):
        """Signal with T < 4 returns empty BreakResult."""
        finder = RetrospectiveBreakFinder(n_bootstraps=10)
        result = finder.fit([1.0, 2.0, 3.0])
        assert result.n_breaks == 0

    def test_periods_list_too_short_for_break_index(self):
        """Break index beyond periods list should produce None period_label."""
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)])
        # Provide too-short periods list
        periods = ["period_1", "period_2"]
        finder = RetrospectiveBreakFinder(n_bootstraps=50, penalty=2.0)
        result = finder.fit(signal, periods=periods)
        # Should not raise
        assert isinstance(result, BreakResult)

    def test_no_breaks_penalty_infinity(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(0, 1, 40)
        finder = RetrospectiveBreakFinder(penalty=1e6, n_bootstraps=20)
        result = finder.fit(signal)
        assert result.n_breaks == 0

    def test_model_normal(self):
        rng = np.random.default_rng(5)
        signal = np.concatenate([rng.normal(0, 0.5, 30), rng.normal(3, 2.0, 30)])
        finder = RetrospectiveBreakFinder(model="normal", penalty=3.0, n_bootstraps=50)
        result = finder.fit(signal)
        assert isinstance(result, BreakResult)

    def test_seed_none_produces_result(self):
        signal = np.concatenate([np.random.normal(0, 1, 30), np.random.normal(5, 1, 30)])
        finder = RetrospectiveBreakFinder(seed=None, n_bootstraps=30, penalty=2.0)
        result = finder.fit(signal)
        assert isinstance(result, BreakResult)

    def test_confidence_90_percent(self):
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40)])
        finder = RetrospectiveBreakFinder(confidence=0.90, n_bootstraps=100, penalty=2.0)
        result = finder.fit(signal)
        # CIs should be narrower at 90% than 95% — just check it runs
        assert isinstance(result, BreakResult)

    def test_block_size_override(self):
        signal = np.concatenate([np.random.normal(0, 1, 40), np.random.normal(5, 1, 40)])
        finder = RetrospectiveBreakFinder(block_size=10, n_bootstraps=30, penalty=2.0)
        result = finder.fit(signal)
        assert isinstance(result, BreakResult)


# ---------------------------------------------------------------------------
# LossRatioMonitor edge cases
# ---------------------------------------------------------------------------

class TestLossRatioMonitorExtended:
    def _make_data(self, T=40, seed=0):
        rng = np.random.default_rng(seed)
        exposure = 1000.0
        counts = rng.poisson(0.05 * exposure, T)
        exposures = np.full(T, exposure)
        premiums = np.full(T, 500_000.0)
        sevs = rng.lognormal(np.log(3000), 0.3, T)
        lr = (counts * sevs) / premiums
        return counts, exposures, premiums, sevs, lr

    def test_severity_only_path(self):
        """Providing only mean_severities runs severity arm only."""
        _, _, _, sevs, _ = self._make_data()
        monitor = LossRatioMonitor(threshold=0.5)
        result = monitor.monitor(mean_severities=sevs)
        assert result.frequency_result is None
        assert result.severity_result is not None

    def test_frequency_only_path(self):
        """Providing only counts+exposures runs frequency arm only."""
        counts, exposures, _, _, _ = self._make_data()
        monitor = LossRatioMonitor(threshold=0.5)
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert result.frequency_result is not None
        assert result.severity_result is None

    def test_loss_ratio_with_some_zero_claims(self):
        """Periods with zero claims produce nan severity → filtered out gracefully."""
        rng = np.random.default_rng(42)
        T = 30
        counts = rng.poisson(3, T)
        counts[5] = 0
        counts[15] = 0
        exposures = np.full(T, 500.0)
        premiums = np.full(T, 250_000.0)
        sevs_all = rng.lognormal(np.log(3000), 0.3, T)
        lr = np.where(counts > 0, (counts * sevs_all) / premiums, 0.0)
        monitor = LossRatioMonitor()
        result = monitor.monitor(
            loss_ratios=lr,
            premiums=premiums,
            claim_counts=counts,
            exposures=exposures,
        )
        assert isinstance(result, __import__('insurance_dynamics').changepoint.result.MonitorResult)

    def test_all_zero_counts_no_severity(self):
        """All zero claims → nan severity everywhere → no severity result."""
        T = 20
        counts = np.zeros(T)
        exposures = np.full(T, 500.0)
        premiums = np.full(T, 250_000.0)
        lr = np.zeros(T)
        monitor = LossRatioMonitor()
        # Should not raise even though all counts are zero
        # Frequency arm should still run
        result = monitor.monitor(
            loss_ratios=lr,
            premiums=premiums,
            claim_counts=counts,
            exposures=exposures,
        )
        assert result.frequency_result is not None

    def test_retrain_threshold_different_from_threshold(self):
        """retrain_threshold can be set independently of threshold."""
        counts, exposures, _, _, _ = self._make_data()
        monitor = LossRatioMonitor(threshold=0.1, retrain_threshold=0.8)
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        # Both are valid settings — just check it runs
        assert result.recommendation in ("retrain", "monitor")

    def test_combined_probs_length_matches_freq_when_only_freq(self):
        counts, exposures, _, _, _ = self._make_data(T=30)
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert len(result.combined_probs) == 30

    def test_detected_breaks_sorted_by_period_index(self):
        rng = np.random.default_rng(7)
        exposure = 1000.0
        counts = np.concatenate([rng.poisson(0.03 * exposure, 40), rng.poisson(0.12 * exposure, 40)])
        exposures = np.full(80, exposure)
        sevs = np.concatenate([rng.lognormal(np.log(2000), 0.2, 40), rng.lognormal(np.log(4000), 0.2, 40)])
        monitor = LossRatioMonitor(
            freq_prior_alpha=1.0, freq_prior_beta=30.0,
            sev_prior_mu=np.log(2000),
            hazard=0.02, threshold=0.25
        )
        result = monitor.monitor(claim_counts=counts, exposures=exposures, mean_severities=sevs)
        indices = [b.period_index for b in result.detected_breaks]
        assert indices == sorted(indices)

    def test_meta_contains_n_breaks(self):
        counts, exposures, _, _, _ = self._make_data()
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        assert "n_freq_breaks" in result.meta
        assert "n_sev_breaks" in result.meta


# ---------------------------------------------------------------------------
# ConsumerDutyReport extended
# ---------------------------------------------------------------------------

class TestConsumerDutyReportExtended:
    def _make_freq_result(self, T=40):
        rng = np.random.default_rng(42)
        exposure = 1000.0
        counts = np.concatenate([
            rng.poisson(0.03 * exposure, T // 2),
            rng.poisson(0.12 * exposure, T // 2),
        ])
        exposures = np.full(T, exposure)
        det = FrequencyChangeDetector(
            prior_alpha=1.0, prior_beta=30.0, hazard=0.02, threshold=0.25
        )
        return det.fit(counts, exposures, periods=[f"2020-{i:02d}" for i in range(T)])

    def test_reviewed_by_in_dict(self):
        result = self._make_freq_result()
        report = ConsumerDutyReport(result, reviewed_by="A. Actuary")
        # reviewed_by is on the object; not directly in to_dict output
        assert report.reviewed_by == "A. Actuary"

    def test_draft_false(self):
        result = self._make_freq_result()
        report = ConsumerDutyReport(result, draft=False)
        html = report.to_html()
        assert "DRAFT" not in html or "Approved" in html

    def test_draft_true_watermark(self):
        result = self._make_freq_result()
        report = ConsumerDutyReport(result, draft=True)
        html = report.to_html()
        assert "DRAFT" in html

    def test_reviewed_by_in_html(self):
        result = self._make_freq_result()
        report = ConsumerDutyReport(result, reviewed_by="Head of Pricing", draft=True)
        html = report.to_html()
        assert "Head of Pricing" in html

    def test_probability_class_high(self):
        """Probability above threshold → high-prob class."""
        cls = _prob_class(0.8, 0.3)
        assert cls == "high-prob"

    def test_probability_class_medium(self):
        """Probability between 0.5x and 1x threshold → medium-prob."""
        cls = _prob_class(0.2, 0.3)  # 0.2 >= 0.15 (=0.5*0.3) and 0.2 < 0.3
        assert cls == "medium-prob"

    def test_probability_class_low(self):
        """Probability below 0.5x threshold → low-prob."""
        cls = _prob_class(0.05, 0.3)  # 0.05 < 0.15
        assert cls == "low-prob"

    def test_match_uk_event_no_events(self):
        result = self._make_freq_result()
        report = ConsumerDutyReport(result, uk_events=[])
        # _match_uk_event with no events returns "—"
        matched = report._match_uk_event("2020-01")
        assert matched == "—"

    def test_match_uk_event_year_match(self):
        result = self._make_freq_result()
        events = [{"name": "COVID", "date": "2020-03-23", "lines": ["motor"],
                   "component": "frequency", "multiplier": 50, "confidence": "hard"}]
        report = ConsumerDutyReport(result, uk_events=events)
        matched = report._match_uk_event("2020-01")
        assert matched == "COVID"

    def test_match_uk_event_no_match(self):
        result = self._make_freq_result()
        events = [{"name": "Ogden 2017", "date": "2017-03-20", "lines": ["motor"],
                   "component": "severity", "multiplier": 40, "confidence": "hard"}]
        report = ConsumerDutyReport(result, uk_events=events)
        matched = report._match_uk_event("2023-06")
        assert matched == "—"

    def test_to_dict_recommendation_from_monitor(self):
        rng = np.random.default_rng(7)
        exposure = 1000.0
        counts = rng.poisson(0.05 * exposure, 40)
        exposures = np.full(40, exposure)
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        report = ConsumerDutyReport(result)
        d = report.to_dict()
        assert d["recommendation"] in ("retrain", "monitor")

    def test_to_dict_severity_result(self):
        """Report from SeverityChangeDetector result."""
        det = SeverityChangeDetector(threshold=0.5)
        sevs = np.random.lognormal(8.0, 0.3, 20)
        result = det.fit(sevs, periods=[f"p{i}" for i in range(20)])
        report = ConsumerDutyReport(result, product="Motor BI")
        d = report.to_dict()
        assert d["product"] == "Motor BI"
        assert isinstance(d, dict)

    def test_html_severity_result(self):
        """to_html() with severity ChangeResult should not crash."""
        det = SeverityChangeDetector(threshold=0.5)
        sevs = np.random.lognormal(8.0, 0.3, 20)
        result = det.fit(sevs, periods=[f"p{i}" for i in range(20)])
        report = ConsumerDutyReport(result)
        html = report.to_html()
        assert "severity" in html.lower()

    def test_threshold_inferred_from_change_result_meta(self):
        """threshold should be inferred from result.meta when not supplied."""
        det = FrequencyChangeDetector(threshold=0.45)
        result = det.fit([5] * 10, [100.0] * 10)
        report = ConsumerDutyReport(result)
        assert report.threshold == pytest.approx(0.45)

    def test_threshold_explicit_overrides_meta(self):
        det = FrequencyChangeDetector(threshold=0.45)
        result = det.fit([5] * 10, [100.0] * 10)
        report = ConsumerDutyReport(result, threshold=0.22)
        assert report.threshold == pytest.approx(0.22)

    def test_to_html_writes_file(self, tmp_path):
        result = self._make_freq_result()
        report = ConsumerDutyReport(result)
        out = tmp_path / "test_report.html"
        report.to_html(path=str(out))
        assert out.exists()
        assert out.stat().st_size > 100

    def test_html_has_period_range(self):
        result = self._make_freq_result(T=20)
        report = ConsumerDutyReport(result)
        html = report.to_html()
        # Should contain the first and last period
        assert "2020-00" in html
        assert "2020-19" in html


# ---------------------------------------------------------------------------
# Plot module (smoke tests — just ensure no crash)
# ---------------------------------------------------------------------------

class TestPlotModule:
    """Smoke tests for plotting functions. Only run if matplotlib available."""

    def _make_freq_result(self, T=30):
        rng = np.random.default_rng(0)
        det = FrequencyChangeDetector(threshold=0.3, hazard=0.02)
        counts = np.concatenate([
            rng.poisson(0.03 * 500, T // 2),
            rng.poisson(0.12 * 500, T // 2),
        ])
        exposures = np.full(T, 500.0)
        return det.fit(counts, exposures, periods=list(range(T)))

    def test_plot_regime_probs_no_crash(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_regime_probs
        except ImportError:
            pytest.skip("matplotlib not available")
        result = self._make_freq_result()
        fig = plot_regime_probs(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_regime_probs_with_ax(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from insurance_dynamics.changepoint.plot import plot_regime_probs
        except ImportError:
            pytest.skip("matplotlib not available")
        result = self._make_freq_result()
        fig, ax = plt.subplots()
        returned = plot_regime_probs(result, ax=ax)
        assert returned is fig
        plt.close("all")

    def test_plot_regime_probs_custom_threshold(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_regime_probs
        except ImportError:
            pytest.skip("matplotlib not available")
        result = self._make_freq_result()
        fig = plot_regime_probs(result, threshold=0.5, title="Custom title")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_run_length_heatmap_no_crash(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_run_length_heatmap
        except ImportError:
            pytest.skip("matplotlib not available")
        result = self._make_freq_result()
        fig = plot_run_length_heatmap(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_run_length_heatmap_with_max_run_length(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_run_length_heatmap
        except ImportError:
            pytest.skip("matplotlib not available")
        result = self._make_freq_result()
        fig = plot_run_length_heatmap(result, max_run_length=5)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_monitor_no_crash(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_monitor
        except ImportError:
            pytest.skip("matplotlib not available")
        rng = np.random.default_rng(0)
        counts = rng.poisson(5, 30)
        exposures = np.full(30, 100.0)
        sevs = rng.lognormal(8.0, 0.3, 30)
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures, mean_severities=sevs)
        fig = plot_monitor(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_monitor_frequency_only(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_monitor
        except ImportError:
            pytest.skip("matplotlib not available")
        rng = np.random.default_rng(0)
        counts = rng.poisson(5, 30)
        exposures = np.full(30, 100.0)
        monitor = LossRatioMonitor()
        result = monitor.monitor(claim_counts=counts, exposures=exposures)
        fig = plot_monitor(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_retrospective_breaks_no_crash(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_retrospective_breaks
        except ImportError:
            pytest.skip("matplotlib not available")
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)])
        finder = RetrospectiveBreakFinder(penalty=2.0, n_bootstraps=30)
        result = finder.fit(signal)
        fig = plot_retrospective_breaks(signal, result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_monitor_no_results_raises(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_dynamics.changepoint.plot import plot_monitor
        except ImportError:
            pytest.skip("matplotlib not available")
        # Manually construct a MonitorResult with no freq or sev result
        empty_result = MonitorResult(
            frequency_result=None,
            severity_result=None,
            combined_probs=np.array([]),
            detected_breaks=[],
            recommendation="monitor",
        )
        with pytest.raises(ValueError, match="no frequency or severity"):
            plot_monitor(empty_result)
