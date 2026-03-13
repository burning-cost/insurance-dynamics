"""
Tests for ConsumerDutyReport.
"""

import json
import pytest
import numpy as np

from insurance_dynamics.changepoint import (
    FrequencyChangeDetector,
    LossRatioMonitor,
    ConsumerDutyReport,
    UKEventPrior,
)
from insurance_dynamics.changepoint.result import ChangeResult, MonitorResult


def _make_freq_result(n_breaks=True) -> ChangeResult:
    rng = np.random.default_rng(42)
    if n_breaks:
        exposure = 1000.0
        counts = np.concatenate([
            rng.poisson(0.03 * exposure, 40),
            rng.poisson(0.12 * exposure, 40),
        ])
    else:
        counts = rng.poisson(0.05 * 1000, 40)
    exposures = np.full(len(counts), 1000.0)

    det = FrequencyChangeDetector(
        prior_alpha=1.0, prior_beta=30.0, hazard=0.02, threshold=0.25
    )
    return det.fit(counts, exposures, periods=[f"2020-{i:02d}" for i in range(len(counts))])


def _make_monitor_result() -> MonitorResult:
    rng = np.random.default_rng(7)
    exposure = 1000.0
    counts = np.concatenate([
        rng.poisson(0.03 * exposure, 40),
        rng.poisson(0.12 * exposure, 40),
    ])
    exposures = np.full(80, exposure)
    monitor = LossRatioMonitor(
        freq_prior_alpha=1.0, freq_prior_beta=30.0, hazard=0.02, threshold=0.25
    )
    return monitor.monitor(claim_counts=counts, exposures=exposures)


class TestConsumerDutyReportDict:
    def test_to_dict_returns_dict(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result, product="Motor")
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        d = report.to_dict()
        required = [
            "product", "segment", "generated_at", "recommendation",
            "detected_breaks", "uk_events", "threshold", "monitoring_frequency",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_json_serialisable(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        d = report.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_recommendation_retrain_when_breaks(self):
        result = _make_freq_result(n_breaks=True)
        report = ConsumerDutyReport(result)
        d = report.to_dict()
        if result.n_breaks > 0:
            assert d["recommendation"] == "retrain"

    def test_recommendation_monitor_when_no_breaks(self):
        result = _make_freq_result(n_breaks=False)
        # Force no breaks by using high threshold
        result.detected_breaks = []
        report = ConsumerDutyReport(result)
        d = report.to_dict()
        assert d["recommendation"] == "monitor"

    def test_product_name_in_dict(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result, product="Home Insurance")
        d = report.to_dict()
        assert d["product"] == "Home Insurance"

    def test_segment_in_dict(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result, segment="Urban, 25-34")
        d = report.to_dict()
        assert d["segment"] == "Urban, 25-34"

    def test_from_monitor_result(self):
        result = _make_monitor_result()
        report = ConsumerDutyReport(result, product="Motor Private")
        d = report.to_dict()
        assert isinstance(d, dict)


class TestConsumerDutyReportHTML:
    def test_to_html_returns_string(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        html = report.to_html()
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_contains_product(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result, product="Motor Private")
        html = report.to_html()
        assert "Motor Private" in html

    def test_html_is_valid_html(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<table" in html

    def test_html_contains_fca_statement(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        html = report.to_html()
        assert "PRIN 2A.9" in html or "Consumer Duty" in html

    def test_html_contains_recommendation(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        html = report.to_html()
        assert "Recommendation" in html or "recommendation" in html.lower()

    def test_to_html_writes_file(self, tmp_path):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        output_path = tmp_path / "report.html"
        report.to_html(path=output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_with_uk_events(self):
        prior = UKEventPrior(lines=["motor"])
        uk_events = prior.summary()
        result = _make_freq_result()
        report = ConsumerDutyReport(result, uk_events=uk_events)
        html = report.to_html()
        assert "Ogden" in html or "GIPP" in html

    def test_from_monitor_result_html(self):
        result = _make_monitor_result()
        report = ConsumerDutyReport(result, product="Motor")
        html = report.to_html()
        assert "Motor" in html

    def test_html_contains_probability_table(self):
        result = _make_freq_result()
        report = ConsumerDutyReport(result)
        html = report.to_html()
        # The prob series table should be present
        assert "P(changepoint)" in html or "changepoint" in html.lower()
