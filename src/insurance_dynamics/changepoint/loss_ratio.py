"""
Loss ratio monitor: joint frequency and severity changepoint detection.

This combines the frequency and severity detectors into a single workflow
for the most common use case: monitoring loss ratios at a product or segment
level.

The combined signal is the element-wise maximum of the frequency and severity
changepoint probabilities. A break is flagged if either component signals a
regime change above the threshold.

Usage
-----
>>> from insurance_changepoint import LossRatioMonitor
>>> monitor = LossRatioMonitor(lines=['motor'])
>>> result = monitor.monitor(
...     loss_ratios=lr_series,
...     premiums=premium_series,
...     claim_counts=count_series,
...     exposures=exposure_series,
...     periods=period_labels,
... )
>>> print(result.recommendation)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .frequency import FrequencyChangeDetector
from .severity import SeverityChangeDetector
from .result import ChangeResult, DetectedBreak, MonitorResult


class LossRatioMonitor:
    """
    Joint loss ratio monitor combining frequency and severity BOCPD.

    Parameters
    ----------
    freq_prior_alpha, freq_prior_beta :
        Poisson-Gamma priors for frequency.
    sev_prior_mu, sev_prior_kappa, sev_prior_alpha, sev_prior_beta :
        NormalGamma priors for log-severity.
    hazard :
        Base changepoint hazard (shared across both components).
    threshold :
        P(changepoint) threshold for flagging breaks.
    lines :
        UK event calendar lines filter.
    uk_events :
        Whether to apply UK event calendar priors to both detectors.
    retrain_threshold :
        Combined probability above which to recommend 'retrain'.
        Defaults to same as threshold.
    """

    def __init__(
        self,
        freq_prior_alpha: float = 1.0,
        freq_prior_beta: float = 10.0,
        sev_prior_mu: float = 0.0,
        sev_prior_kappa: float = 1.0,
        sev_prior_alpha: float = 2.0,
        sev_prior_beta: float = 1.0,
        hazard: float = 1.0 / 100,
        threshold: float = 0.3,
        lines: list[str] | None = None,
        uk_events: bool = False,
        retrain_threshold: float | None = None,
    ) -> None:
        self.hazard = hazard
        self.threshold = threshold
        self.retrain_threshold = retrain_threshold or threshold
        self.uk_events = uk_events
        self.lines = lines

        self._freq_detector = FrequencyChangeDetector(
            prior_alpha=freq_prior_alpha,
            prior_beta=freq_prior_beta,
            hazard=hazard,
            threshold=threshold,
            uk_events=uk_events,
            event_lines=lines,
            event_components=["frequency"],
        )

        self._sev_detector = SeverityChangeDetector(
            prior_mu=sev_prior_mu,
            prior_kappa=sev_prior_kappa,
            prior_alpha=sev_prior_alpha,
            prior_beta=sev_prior_beta,
            hazard=hazard,
            threshold=threshold,
            uk_events=uk_events,
            event_lines=lines,
            event_components=["severity"],
        )

    def monitor(
        self,
        loss_ratios: list[float] | np.ndarray | None = None,
        premiums: list[float] | np.ndarray | None = None,
        claim_counts: list[float] | np.ndarray | None = None,
        exposures: list[float] | np.ndarray | None = None,
        mean_severities: list[float] | np.ndarray | None = None,
        periods: list[Any] | None = None,
        period_to_date_fn: Any = None,
    ) -> MonitorResult:
        """
        Run joint monitoring.

        You can provide data in two ways:
        1. ``claim_counts`` + ``exposures`` for frequency; ``mean_severities``
           for severity.
        2. ``loss_ratios`` + ``premiums`` + ``claim_counts`` — the monitor will
           derive mean severity as (loss_ratio * premium) / claim_count.

        At least one of (claim_counts + exposures) or (loss_ratios + premiums)
        must be provided. Similarly for severity.

        Parameters
        ----------
        loss_ratios :
            Incurred loss ratio per period (claims / premiums).
        premiums :
            Earned premium per period.
        claim_counts :
            Number of claims per period.
        exposures :
            Earned exposure per period.
        mean_severities :
            Mean claim cost per period. If not provided, derived from
            loss_ratios * premiums / claim_counts.
        periods :
            Period labels.
        period_to_date_fn :
            Callable for event calendar date lookup.

        Returns
        -------
        MonitorResult
        """
        freq_result: ChangeResult | None = None
        sev_result: ChangeResult | None = None

        # Frequency component
        if claim_counts is not None and exposures is not None:
            freq_result = self._freq_detector.fit(
                claim_counts=claim_counts,
                earned_exposure=exposures,
                periods=periods,
                period_to_date_fn=period_to_date_fn,
            )

        # Severity component
        sevs = None
        if mean_severities is not None:
            sevs = np.asarray(mean_severities, dtype=float)
        elif (
            loss_ratios is not None
            and premiums is not None
            and claim_counts is not None
        ):
            lr = np.asarray(loss_ratios, dtype=float)
            prem = np.asarray(premiums, dtype=float)
            cnt = np.asarray(claim_counts, dtype=float)
            # Derive mean severity: avoid division by zero
            with np.errstate(invalid="ignore", divide="ignore"):
                sevs = np.where(cnt > 0, (lr * prem) / cnt, np.nan)
            # Drop NaN periods for severity detector
            valid = ~np.isnan(sevs)
            if valid.any():
                sevs_valid = sevs[valid]
                periods_valid = (
                    [p for p, v in zip(periods, valid) if v]
                    if periods is not None
                    else None
                )
                sev_result = self._sev_detector.fit(
                    mean_severities=sevs_valid,
                    periods=periods_valid,
                    period_to_date_fn=period_to_date_fn,
                )

        if sevs is not None and mean_severities is not None:
            sev_result = self._sev_detector.fit(
                mean_severities=sevs,
                periods=periods,
                period_to_date_fn=period_to_date_fn,
            )

        # Combine signals
        T_freq = len(freq_result.changepoint_probs) if freq_result is not None else 0
        T_sev = len(sev_result.changepoint_probs) if sev_result is not None else 0
        T = max(T_freq, T_sev)

        if T == 0:
            raise ValueError(
                "No data provided. Supply claim_counts+exposures or "
                "loss_ratios+premiums+claim_counts."
            )

        freq_probs = (
            freq_result.changepoint_probs
            if freq_result is not None
            else np.zeros(T)
        )
        sev_probs = (
            sev_result.changepoint_probs
            if sev_result is not None
            else np.zeros(T)
        )

        # Align lengths (may differ if severity had NaN periods)
        if len(freq_probs) != len(sev_probs):
            min_t = min(len(freq_probs), len(sev_probs))
            freq_probs = freq_probs[:min_t]
            sev_probs = sev_probs[:min_t]
            T = min_t

        combined_probs = np.maximum(freq_probs, sev_probs)

        # Deduplicate and merge breaks
        all_breaks: dict[int, DetectedBreak] = {}
        for result in [freq_result, sev_result]:
            if result is None:
                continue
            for brk in result.detected_breaks:
                if brk.period_index not in all_breaks or (
                    brk.probability > all_breaks[brk.period_index].probability
                ):
                    all_breaks[brk.period_index] = brk

        detected = sorted(all_breaks.values(), key=lambda b: b.period_index)

        # Recommendation
        max_combined = float(np.max(combined_probs)) if len(combined_probs) > 0 else 0.0
        recommendation = (
            "retrain" if max_combined >= self.retrain_threshold else "monitor"
        )

        return MonitorResult(
            frequency_result=freq_result,
            severity_result=sev_result,
            combined_probs=combined_probs,
            detected_breaks=detected,
            recommendation=recommendation,
            meta={
                "max_combined_prob": max_combined,
                "n_freq_breaks": freq_result.n_breaks if freq_result else 0,
                "n_sev_breaks": sev_result.n_breaks if sev_result else 0,
                "uk_events": self.uk_events,
            },
        )
