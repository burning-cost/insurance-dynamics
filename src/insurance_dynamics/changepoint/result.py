"""
Result dataclasses for insurance-changepoint.

These are plain data containers — no logic here beyond property helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectedBreak:
    """A single detected regime break from online BOCPD."""

    period_index: int
    """Zero-based index of the period where the break was detected."""

    period_label: Any
    """Label for the period (e.g. date string or integer year-quarter)."""

    probability: float
    """Posterior probability of a changepoint at this period."""

    run_length_before: int
    """Most likely run length of the preceding regime."""

    def __repr__(self) -> str:
        label = self.period_label if self.period_label is not None else self.period_index
        return f"DetectedBreak(period={label}, prob={self.probability:.3f})"


@dataclass
class ChangeResult:
    """
    Output from FrequencyChangeDetector or SeverityChangeDetector.

    Attributes
    ----------
    periods :
        Period labels passed to fit().
    changepoint_probs :
        Array of P(changepoint at t) for each period.
    run_length_probs :
        2-D array of shape (T, T) — posterior distribution over run lengths
        at each time step. Entry [t, ℓ] = P(run_length = ℓ at time t).
    detected_breaks :
        Breaks where probability exceeds the detection threshold.
    detector_type :
        ``'frequency'`` or ``'severity'``.
    hazard_used :
        Effective base hazard used (may vary by period if UK event priors applied).
    meta :
        Arbitrary key-value metadata.
    """

    periods: list[Any]
    changepoint_probs: np.ndarray
    run_length_probs: np.ndarray
    detected_breaks: list[DetectedBreak]
    detector_type: str
    hazard_used: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def n_breaks(self) -> int:
        return len(self.detected_breaks)

    @property
    def max_changepoint_prob(self) -> float:
        if len(self.changepoint_probs) == 0:
            return 0.0
        return float(np.max(self.changepoint_probs))

    def most_probable_run_length(self, t: int) -> int:
        """Return the most probable run length at period t."""
        return int(np.argmax(self.run_length_probs[t]))


@dataclass
class BreakInterval:
    """95% confidence interval on a retrospective break location."""

    break_index: int
    """Point estimate break location (from PELT)."""

    lower: int
    """Lower bound of 95% bootstrap CI."""

    upper: int
    """Upper bound of 95% bootstrap CI."""

    period_label: Any = None
    """Label of the break period if periods were supplied."""

    def __repr__(self) -> str:
        label = self.period_label if self.period_label is not None else self.break_index
        return f"BreakInterval(break={label}, CI=[{self.lower}, {self.upper}])"


@dataclass
class BreakResult:
    """
    Output from RetrospectiveBreakFinder.

    Attributes
    ----------
    breaks :
        List of estimated break indices (PELT point estimates).
    break_cis :
        Bootstrap 95% CI for each break location.
    n_bootstraps :
        Number of bootstrap resamples used.
    penalty :
        Penalty value used.
    model :
        Cost model used ('l2', 'normal', 'rbf', etc.).
    """

    breaks: list[int]
    break_cis: list[BreakInterval]
    n_bootstraps: int
    penalty: float
    model: str
    periods: list[Any] = field(default_factory=list)

    @property
    def n_breaks(self) -> int:
        return len(self.breaks)


@dataclass
class MonitorResult:
    """
    Output from LossRatioMonitor combining frequency and severity signals.

    Attributes
    ----------
    frequency_result :
        ChangeResult from the frequency component.
    severity_result :
        ChangeResult from the severity component.
    combined_probs :
        Element-wise max of frequency and severity changepoint probabilities.
    detected_breaks :
        Union of breaks from both components, deduplicated by period.
    recommendation :
        ``'retrain'`` or ``'monitor'``.
    """

    frequency_result: ChangeResult | None
    severity_result: ChangeResult | None
    combined_probs: np.ndarray
    detected_breaks: list[DetectedBreak]
    recommendation: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_breaks(self) -> int:
        return len(self.detected_breaks)
