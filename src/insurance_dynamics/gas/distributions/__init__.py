"""GAS distribution implementations."""

from .base import GASDistribution
from .poisson import PoissonGAS
from .gamma import GammaGAS
from .negative_binomial import NegBinGAS
from .lognormal import LogNormalGAS
from .beta import BetaGAS
from .zip import ZIPGAS

DISTRIBUTION_MAP: dict[str, type[GASDistribution]] = {
    "poisson": PoissonGAS,
    "gamma": GammaGAS,
    "negbin": NegBinGAS,
    "lognormal": LogNormalGAS,
    "beta": BetaGAS,
    "zip": ZIPGAS,
}

__all__ = [
    "GASDistribution",
    "PoissonGAS",
    "GammaGAS",
    "NegBinGAS",
    "LogNormalGAS",
    "BetaGAS",
    "ZIPGAS",
    "DISTRIBUTION_MAP",
]
