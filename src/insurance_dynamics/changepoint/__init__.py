"""insurance_dynamics.changepoint: Bayesian change-point detection for UK insurance pricing.

Key classes
-----------
FrequencyChangeDetector
    Online BOCPD for claim frequency with exposure weighting.
    Handles varying exposure per period via Poisson-Gamma conjugate.

SeverityChangeDetector
    Online BOCPD for claim severity using Normal-Gamma on log-severity.

LossRatioMonitor
    Joint frequency + severity monitoring returning a 'retrain/monitor' recommendation.

RetrospectiveBreakFinder
    Offline PELT with bootstrap confidence intervals on break locations.

UKEventPrior
    UK insurance regulatory event calendar for informative hazard priors.

ConsumerDutyReport
    FCA PRIN 2A.9 evidence pack as HTML and JSON.

Quick start
-----------
>>> from insurance_dynamics.changepoint import FrequencyChangeDetector
>>> detector = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=12.0, hazard=0.01)
>>> result = detector.fit(claim_counts, earned_exposure, periods)
>>> print(result.n_breaks, "breaks detected")

>>> from insurance_dynamics.changepoint import LossRatioMonitor
>>> monitor = LossRatioMonitor(lines=['motor'], uk_events=True)
>>> result = monitor.monitor(claim_counts=counts, exposures=exp, periods=periods)
"""

from .frequency import FrequencyChangeDetector
from .severity import SeverityChangeDetector
from .loss_ratio import LossRatioMonitor
from .retrospective import RetrospectiveBreakFinder
from .priors import UKEventPrior, UKEvent, UK_EVENTS
from .report import ConsumerDutyReport
from .result import ChangeResult, BreakResult, MonitorResult, DetectedBreak, BreakInterval

__version__ = "0.1.0"
__all__ = [
    "FrequencyChangeDetector",
    "SeverityChangeDetector",
    "LossRatioMonitor",
    "RetrospectiveBreakFinder",
    "UKEventPrior",
    "UKEvent",
    "UK_EVENTS",
    "ConsumerDutyReport",
    "ChangeResult",
    "BreakResult",
    "MonitorResult",
    "DetectedBreak",
    "BreakInterval",
    "__version__",
]
