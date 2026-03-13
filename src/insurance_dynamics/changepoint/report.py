"""
Consumer Duty evidence reporting (FCA PRIN 2A.9).

Generates an HTML evidence pack showing the pricing model monitoring output
alongside the UK event calendar, with a compliance statement.

The report is deliberately low-tech: plain HTML table with inline CSS,
no JavaScript, no external dependencies. It needs to survive being emailed
as an attachment, opened in SharePoint preview, and printed to PDF.

Usage
-----
>>> from insurance_changepoint import ConsumerDutyReport
>>> report = ConsumerDutyReport(result, product="Motor Private", segment="All")
>>> report.to_html("monitoring_q4_2024.html")
>>> data = report.to_dict()
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .result import ChangeResult, MonitorResult

try:
    from jinja2 import Environment, BaseLoader
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Consumer Duty Pricing Monitoring Evidence — {{ product }}</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 960px; margin: 2em auto;
         color: #222; font-size: 14px; }
  h1 { color: #1a3a6b; }
  h2 { color: #1a3a6b; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; }
  th { background: #1a3a6b; color: white; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f8f8f8; }
  .break-row { background: #fff3cd !important; font-weight: bold; }
  .high-prob { color: #c0392b; font-weight: bold; }
  .medium-prob { color: #e67e22; }
  .low-prob { color: #27ae60; }
  .compliance { background: #eaf4ea; border: 1px solid #27ae60;
                padding: 1em; margin: 1em 0; border-radius: 4px; }
  .warning { background: #fff3cd; border: 1px solid #e67e22;
             padding: 1em; margin: 1em 0; border-radius: 4px; }
  .metadata { background: #f0f4ff; padding: 0.75em 1em;
              border-left: 4px solid #1a3a6b; margin: 1em 0; }
  .footer { color: #777; font-size: 12px; margin-top: 2em;
            border-top: 1px solid #ccc; padding-top: 0.5em; }
</style>
</head>
<body>

<h1>Consumer Duty Pricing Monitoring Evidence</h1>

<div class="metadata">
  <strong>Product:</strong> {{ product }}<br>
  <strong>Segment:</strong> {{ segment }}<br>
  <strong>Report generated:</strong> {{ generated_at }}<br>
  <strong>Monitoring period:</strong> {{ period_from }} to {{ period_to }}<br>
  <strong>Periods analysed:</strong> {{ n_periods }}<br>
  <strong>Algorithm:</strong> Bayesian Online Changepoint Detection (BOCPD), Poisson-Gamma conjugate
</div>

<h2>Recommendation</h2>
{% if recommendation == "retrain" %}
<div class="warning">
  <strong>Action required: Retrain pricing model.</strong>
  One or more statistically significant regime breaks have been detected in the
  monitoring period. The current pricing model may no longer reflect the
  underlying risk. Review breaks below and assess whether model recalibration
  is required under PRIN 2A.9 fair value obligations.
</div>
{% else %}
<div class="compliance">
  <strong>No action required: Continue monitoring.</strong>
  No statistically significant regime breaks detected in the current monitoring
  period. Pricing model remains appropriate for the observed experience.
  Continue quarterly monitoring.
</div>
{% endif %}

<h2>Changepoint Detection Summary</h2>
<table>
  <tr>
    <th>Component</th>
    <th>Breaks detected</th>
    <th>Max P(changepoint)</th>
    <th>Threshold</th>
  </tr>
  {% for row in summary_rows %}
  <tr>
    <td>{{ row.component }}</td>
    <td>{{ row.n_breaks }}</td>
    <td class="{{ row.prob_class }}">{{ "%.3f"|format(row.max_prob) }}</td>
    <td>{{ "%.2f"|format(row.threshold) }}</td>
  </tr>
  {% endfor %}
</table>

{% if detected_breaks %}
<h2>Detected Regime Breaks</h2>
<table>
  <tr>
    <th>Period</th>
    <th>Component</th>
    <th>P(changepoint)</th>
    <th>Matched UK event</th>
    <th>Notes</th>
  </tr>
  {% for brk in detected_breaks %}
  <tr class="break-row">
    <td>{{ brk.period }}</td>
    <td>{{ brk.component }}</td>
    <td class="{{ brk.prob_class }}">{{ "%.3f"|format(brk.probability) }}</td>
    <td>{{ brk.matched_event }}</td>
    <td>{{ brk.notes }}</td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p><em>No regime breaks detected above the threshold in this period.</em></p>
{% endif %}

<h2>UK Regulatory Event Calendar</h2>
<p>Events from the UK insurance regulatory calendar are shown below. Where a
monitoring period overlaps with a known event, the detection hazard was
increased proportionally (informative prior, not a hard constraint).</p>
<table>
  <tr>
    <th>Event</th>
    <th>Date</th>
    <th>Lines</th>
    <th>Component</th>
    <th>Hazard multiplier</th>
    <th>Confidence</th>
  </tr>
  {% for ev in uk_events %}
  <tr>
    <td>{{ ev.name }}</td>
    <td>{{ ev.date }}</td>
    <td>{{ ev.lines | join(', ') }}</td>
    <td>{{ ev.component }}</td>
    <td>{{ ev.multiplier }}×</td>
    <td>{{ ev.confidence }}</td>
  </tr>
  {% endfor %}
</table>

<h2>Changepoint Probability Series</h2>
<table>
  <tr>
    <th>Period</th>
    <th>P(changepoint) — frequency</th>
    <th>P(changepoint) — severity</th>
    <th>Combined</th>
  </tr>
  {% for row in prob_rows %}
  <tr {% if row.is_break %}class="break-row"{% endif %}>
    <td>{{ row.period }}</td>
    <td class="{{ row.freq_class }}">{{ "%.4f"|format(row.freq_prob) }}</td>
    <td class="{{ row.sev_class }}">{{ "%.4f"|format(row.sev_prob) }}</td>
    <td class="{{ row.combined_class }}">{{ "%.4f"|format(row.combined_prob) }}</td>
  </tr>
  {% endfor %}
</table>

<h2>FCA PRIN 2A.9 Compliance Statement</h2>
<div class="compliance">
  <p>This evidence pack has been produced to support the firm's obligations under
  FCA Principle 12 (Consumer Duty) and PRIN 2A.9 (Fair Value). The pricing
  model monitoring process uses Bayesian online changepoint detection to
  identify structural breaks in claim frequency and severity that would
  indicate the current pricing model no longer reflects underlying risk.</p>

  <p>The monitoring algorithm is applied to all monitored product lines on a
  {{ monitoring_frequency }} basis. Breaks are assessed against the UK
  regulatory event calendar and reviewed by the pricing team. Where a break
  is detected, the pricing model is reviewed and recalibrated as required
  to ensure premiums remain fair value for customers.</p>

  <p><strong>Monitoring frequency:</strong> {{ monitoring_frequency }}<br>
  <strong>Threshold for action:</strong> P(changepoint) ≥ {{ threshold }}<br>
  <strong>Algorithm:</strong> BOCPD (Adams &amp; MacKay 2007), exposure-weighted Poisson-Gamma conjugate<br>
  <strong>Event calendar:</strong> UK regulatory events 2017–2025, sourced from FCA publications and ABI data</p>
</div>

<div class="footer">
  Generated by insurance-changepoint v{{ version }} on {{ generated_at }}.
  This report is produced for internal compliance purposes under FCA Consumer Duty (PRIN 2A.9).
</div>

</body>
</html>
"""


def _prob_class(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "high-prob"
    elif prob >= threshold * 0.5:
        return "medium-prob"
    return "low-prob"


class ConsumerDutyReport:
    """
    FCA PRIN 2A.9 evidence pack for pricing model monitoring.

    Parameters
    ----------
    result :
        Output from LossRatioMonitor.monitor() or a single ChangeResult.
    product :
        Product name for the report header.
    segment :
        Segment description (e.g. "Age 25-34, urban").
    monitoring_frequency :
        How often monitoring is run (e.g. "quarterly").
    threshold :
        Detection threshold used. If not provided, inferred from result.
    uk_events :
        Summary from UKEventPrior.summary(). If not provided, uses empty list.
    version :
        Package version string for the footer.
    """

    def __init__(
        self,
        result: MonitorResult | ChangeResult,
        product: str = "Unspecified",
        segment: str = "All",
        monitoring_frequency: str = "quarterly",
        threshold: float | None = None,
        uk_events: list[dict] | None = None,
        version: str = "0.1.0",
    ) -> None:
        self.result = result
        self.product = product
        self.segment = segment
        self.monitoring_frequency = monitoring_frequency
        self.version = version
        self.uk_events = uk_events or []

        # Infer threshold
        if threshold is not None:
            self.threshold = threshold
        elif isinstance(result, MonitorResult):
            meta = result.meta or {}
            self.threshold = meta.get("threshold", 0.3)
        elif isinstance(result, ChangeResult):
            self.threshold = result.meta.get("threshold", 0.3)
        else:
            self.threshold = 0.3

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable representation of the report."""
        return {
            "product": self.product,
            "segment": self.segment,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recommendation": self._recommendation(),
            "detected_breaks": self._detected_breaks_dicts(),
            "uk_events": self.uk_events,
            "threshold": self.threshold,
            "monitoring_frequency": self.monitoring_frequency,
            "version": self.version,
        }

    def to_html(self, path: str | Path | None = None) -> str:
        """
        Generate the HTML evidence pack.

        Parameters
        ----------
        path :
            If provided, write the HTML to this file.

        Returns
        -------
        str
            HTML string.
        """
        if not _JINJA2_AVAILABLE:
            raise ImportError(
                "jinja2 is required for ConsumerDutyReport.to_html(). "
                "Install it with: pip install jinja2"
            )

        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(_REPORT_TEMPLATE)

        context = self._build_context()
        html = tmpl.render(**context)

        if path is not None:
            Path(path).write_text(html, encoding="utf-8")

        return html

    def _recommendation(self) -> str:
        if isinstance(self.result, MonitorResult):
            return self.result.recommendation
        elif isinstance(self.result, ChangeResult):
            return "retrain" if self.result.n_breaks > 0 else "monitor"
        return "monitor"

    def _detected_breaks_dicts(self) -> list[dict]:
        breaks = []
        if isinstance(self.result, MonitorResult):
            for brk in self.result.detected_breaks:
                breaks.append({
                    "period": str(brk.period_label),
                    "probability": brk.probability,
                    "component": "combined",
                })
        elif isinstance(self.result, ChangeResult):
            for brk in self.result.detected_breaks:
                breaks.append({
                    "period": str(brk.period_label),
                    "probability": brk.probability,
                    "component": self.result.detector_type,
                })
        return breaks

    def _build_context(self) -> dict[str, Any]:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Periods
        if isinstance(self.result, MonitorResult):
            freq_r = self.result.frequency_result
            sev_r = self.result.severity_result
            if freq_r is not None:
                periods = freq_r.periods
            elif sev_r is not None:
                periods = sev_r.periods
            else:
                periods = []
            combined = self.result.combined_probs
            freq_probs = freq_r.changepoint_probs if freq_r is not None else [0.0] * len(periods)
            sev_probs = sev_r.changepoint_probs if sev_r is not None else [0.0] * len(periods)
            recommendation = self.result.recommendation
            all_breaks = self.result.detected_breaks
        else:
            periods = self.result.periods
            combined = self.result.changepoint_probs
            freq_probs = (
                self.result.changepoint_probs
                if self.result.detector_type == "frequency"
                else [0.0] * len(periods)
            )
            sev_probs = (
                self.result.changepoint_probs
                if self.result.detector_type == "severity"
                else [0.0] * len(periods)
            )
            recommendation = "retrain" if self.result.n_breaks > 0 else "monitor"
            all_breaks = self.result.detected_breaks

        period_from = str(periods[0]) if periods else "N/A"
        period_to = str(periods[-1]) if periods else "N/A"
        n_periods = len(periods)

        # Detected breaks rows
        break_indices = {b.period_index for b in all_breaks}
        detected_break_rows = []
        for brk in all_breaks:
            # Try to match a UK event
            matched = self._match_uk_event(str(brk.period_label))
            comp = "combined"
            if isinstance(self.result, ChangeResult):
                comp = self.result.detector_type
            detected_break_rows.append({
                "period": str(brk.period_label),
                "component": comp,
                "probability": brk.probability,
                "prob_class": _prob_class(brk.probability, self.threshold),
                "matched_event": matched,
                "notes": "",
            })

        # Summary rows
        threshold = self.threshold
        if isinstance(self.result, MonitorResult):
            freq_r = self.result.frequency_result
            sev_r = self.result.severity_result
            summary_rows = [
                {
                    "component": "Frequency",
                    "n_breaks": freq_r.n_breaks if freq_r else 0,
                    "max_prob": float(freq_r.max_changepoint_prob) if freq_r else 0.0,
                    "threshold": threshold,
                    "prob_class": _prob_class(
                        float(freq_r.max_changepoint_prob) if freq_r else 0.0,
                        threshold,
                    ),
                },
                {
                    "component": "Severity",
                    "n_breaks": sev_r.n_breaks if sev_r else 0,
                    "max_prob": float(sev_r.max_changepoint_prob) if sev_r else 0.0,
                    "threshold": threshold,
                    "prob_class": _prob_class(
                        float(sev_r.max_changepoint_prob) if sev_r else 0.0,
                        threshold,
                    ),
                },
                {
                    "component": "Combined",
                    "n_breaks": len(all_breaks),
                    "max_prob": float(combined.max()) if len(combined) > 0 else 0.0,
                    "threshold": threshold,
                    "prob_class": _prob_class(
                        float(combined.max()) if len(combined) > 0 else 0.0,
                        threshold,
                    ),
                },
            ]
        else:
            summary_rows = [
                {
                    "component": self.result.detector_type.capitalize(),
                    "n_breaks": self.result.n_breaks,
                    "max_prob": self.result.max_changepoint_prob,
                    "threshold": threshold,
                    "prob_class": _prob_class(
                        self.result.max_changepoint_prob, threshold
                    ),
                }
            ]

        # Probability series rows
        n = min(len(periods), len(combined))
        prob_rows = []
        for i in range(n):
            fp = float(freq_probs[i]) if i < len(freq_probs) else 0.0
            sp = float(sev_probs[i]) if i < len(sev_probs) else 0.0
            cp = float(combined[i])
            prob_rows.append({
                "period": str(periods[i]),
                "freq_prob": fp,
                "sev_prob": sp,
                "combined_prob": cp,
                "is_break": i in break_indices,
                "freq_class": _prob_class(fp, threshold),
                "sev_class": _prob_class(sp, threshold),
                "combined_class": _prob_class(cp, threshold),
            })

        # Format UK events for template
        uk_event_rows = [
            {
                "name": ev.get("name", ""),
                "date": ev.get("date", ""),
                "lines": ev.get("lines", []),
                "component": ev.get("component", ""),
                "multiplier": ev.get("multiplier", 1),
                "confidence": ev.get("confidence", ""),
            }
            for ev in self.uk_events
        ]

        return {
            "product": self.product,
            "segment": self.segment,
            "generated_at": generated_at,
            "period_from": period_from,
            "period_to": period_to,
            "n_periods": n_periods,
            "recommendation": recommendation,
            "summary_rows": summary_rows,
            "detected_breaks": detected_break_rows,
            "uk_events": uk_event_rows,
            "prob_rows": prob_rows,
            "threshold": threshold,
            "monitoring_frequency": self.monitoring_frequency,
            "version": self.version,
        }

    def _match_uk_event(self, period_label: str) -> str:
        """Try to find a UK event near the detected break period."""
        if not self.uk_events:
            return "—"
        # Simple string matching on the year
        for ev in self.uk_events:
            ev_date = ev.get("date", "")
            if ev_date[:4] in period_label or period_label[:4] in ev_date:
                return ev.get("name", "—")
        return "—"
