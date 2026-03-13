"""Gamma GAS distribution for claim severity modelling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, digamma

from .base import GASDistribution


class GammaGAS(GASDistribution):
    """Gamma distribution with log-linked time-varying mean.

    The Gamma is parameterised as Gamma(shape=a, rate=a/mu) so that the
    mean is mu and the coefficient of variation is 1/sqrt(a). The shape
    parameter ``a`` is treated as static (estimated via MLE jointly with
    the GAS parameters).

    This is the standard choice for aggregate claim severity in UK
    motor and property lines.

    Parameters
    ----------
    shape:
        Fixed shape parameter a > 0. If None, it is estimated from data.
    """

    param_names = ["mean", "shape"]
    default_time_varying = ["mean"]

    def __init__(self, shape: float | None = None) -> None:
        self.shape = shape  # set externally by GASModel after fitting if None

    def _get_shape(self, params: dict) -> float:
        return float(params.get("shape", self.shape if self.shape is not None else 1.0))

    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Score of log Gamma(y | mu, a) w.r.t. f = log(mu).

        d/d(log mu) log p(y | mu, a) = a*(y/mu - 1)
        """
        mu = params["mean"]
        a = self._get_shape(params)
        y_arr = np.asarray(y, dtype=float)
        return {"mean": a * (y_arr / mu - 1.0)}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Fisher information w.r.t. f = log(mu) is the shape parameter a."""
        a = self._get_shape(params)
        return {"mean": float(a)}

    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log Gamma(y | mu, a) using shape-rate parametrisation."""
        mu = params["mean"]
        a = self._get_shape(params)
        y_arr = np.asarray(y, dtype=float)
        # rate = a / mu, so log p = a*log(a/mu) + (a-1)*log(y) - a*y/mu - log Gamma(a)
        return (
            a * np.log(a / mu)
            + (a - 1.0) * np.log(y_arr)
            - a * y_arr / mu
            - gammaln(a)
        )

    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Log link for mean; log link for shape."""
        return np.log(value)

    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Exp for both mean and shape."""
        return np.exp(value)

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Method-of-moments starting values."""
        mu = float(np.mean(y))
        var = float(np.var(y))
        a = mu**2 / var if var > 0 else 1.0
        return {"mean": mu, "shape": a}
