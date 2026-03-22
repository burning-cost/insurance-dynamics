"""insurance_dynamics.gas: GAS (Generalised Autoregressive Score) models for dynamic insurance pricing.

GAS models track time-varying parameters using the score of the conditional
log-likelihood as a forcing variable. They are observation-driven, producing
a closed-form likelihood and fast MLE via L-BFGS-B.

Key classes
-----------
GASModel
    Main user-facing class. Fits a GAS model to a univariate time series.
GASPanel
    Fits the same model independently to multiple rating cells.

Key functions
-------------
gas_forecast
    h-step-ahead mean-path and simulation forecasts.
bootstrap_ci
    Parametric bootstrap confidence intervals for the filter path.

Distributions supported
-----------------------
poisson     PoissonGAS      Claim frequency
gamma       GammaGAS        Claim severity
negbin      NegBinGAS       Overdispersed frequency
lognormal   LogNormalGAS    Severity (log-normal assumption)
beta        BetaGAS         Loss ratios on (0,1)
zip         ZIPGAS          Zero-inflated frequency

Quick start
-----------
>>> from insurance_dynamics.gas import GASModel
>>> from insurance_dynamics.gas.datasets import load_motor_frequency
>>> data = load_motor_frequency()
>>> model = GASModel('poisson')
>>> result = model.fit(data.y, exposure=data.exposure)
>>> print(result.summary())
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-dynamics")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .model import GASModel, GASResult
from .panel import GASPanel, GASPanelResult
from .filter import GASFilter, FilterResult
from .forecast import gas_forecast, ForecastResult
from .bootstrap import bootstrap_ci, BootstrapCI
from .diagnostics import compute_diagnostics, dawid_sebastiani_score
from .distributions import (
    GASDistribution,
    PoissonGAS,
    GammaGAS,
    NegBinGAS,
    LogNormalGAS,
    BetaGAS,
    ZIPGAS,
    DISTRIBUTION_MAP,
)
from .datasets import load_motor_frequency, load_severity_trend, load_loss_ratio

__all__ = [
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
]
