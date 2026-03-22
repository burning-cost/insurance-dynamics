"""insurance-dynamics: Dynamic insurance pricing tools.

Two subpackages:

insurance_dynamics.gas
    GAS (Generalised Autoregressive Score) models — observation-driven
    time-varying parameter models for claim frequency, severity, and loss ratios.

insurance_dynamics.changepoint
    Bayesian change-point detection (BOCPD + PELT) — detect structural breaks
    in insurance time series with UK regulatory event priors and FCA reporting.

Quick start
-----------
>>> from insurance_dynamics.gas import GASModel
>>> from insurance_dynamics.changepoint import FrequencyChangeDetector, LossRatioMonitor
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-dynamics")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

from .gas import (
    GASModel,
    GASResult,
    GASPanel,
    GASPanelResult,
    GASFilter,
    FilterResult,
    gas_forecast,
    ForecastResult,
    bootstrap_ci,
    BootstrapCI,
    compute_diagnostics,
    dawid_sebastiani_score,
    GASDistribution,
    PoissonGAS,
    GammaGAS,
    NegBinGAS,
    LogNormalGAS,
    BetaGAS,
    ZIPGAS,
    DISTRIBUTION_MAP,
    load_motor_frequency,
    load_severity_trend,
    load_loss_ratio,
)

from .changepoint import (
    FrequencyChangeDetector,
    SeverityChangeDetector,
    LossRatioMonitor,
    RetrospectiveBreakFinder,
    UKEventPrior,
    UKEvent,
    UK_EVENTS,
    ConsumerDutyReport,
    ChangeResult,
    BreakResult,
    MonitorResult,
    DetectedBreak,
    BreakInterval,
)

__all__ = [
    # GAS
    "GASModel",
    "GASResult",
    "GASPanel",
    "GASPanelResult",
    "GASFilter",
    "FilterResult",
    "gas_forecast",
    "ForecastResult",
    "bootstrap_ci",
    "BootstrapCI",
    "compute_diagnostics",
    "dawid_sebastiani_score",
    "GASDistribution",
    "PoissonGAS",
    "GammaGAS",
    "NegBinGAS",
    "LogNormalGAS",
    "BetaGAS",
    "ZIPGAS",
    "DISTRIBUTION_MAP",
    "load_motor_frequency",
    "load_severity_trend",
    "load_loss_ratio",
    # Changepoint
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
]
