"""Log-Normal GAS distribution for claim severity modelling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import GASDistribution


class LogNormalGAS(GASDistribution):
    """Log-Normal distribution with time-varying log-mean.

    y_t ~ LogNormal(mu_t, sigma) where mu_t is the mean of log(y_t) and
    sigma is the static standard deviation on the log scale.

    The GAS recursion updates mu_t directly (identity link), making it
    equivalent to adaptive exponential smoothing on log-transformed severities.
    This is an alternative to GammaGAS when the severity distribution is
    closer to log-normal in the tail.
    """

    param_names = ["logmean", "logsigma"]
    default_time_varying = ["logmean"]

    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Score w.r.t. mu (identity link on log-mean).

        d/dmu log p(y | mu, sigma) = (log(y) - mu) / sigma^2
        """
        mu = params["logmean"]
        sigma = np.exp(params["logsigma"])
        y_arr = np.asarray(y, dtype=float)
        return {"logmean": (np.log(y_arr) - mu) / sigma**2}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Fisher information w.r.t. mu is 1/sigma^2."""
        sigma = np.exp(params["logsigma"])
        return {"logmean": 1.0 / sigma**2}

    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log LogNormal(y | mu, sigma)."""
        mu = params["logmean"]
        sigma = np.exp(params["logsigma"])
        y_arr = np.asarray(y, dtype=float)
        return (
            -np.log(y_arr)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((np.log(y_arr) - mu) / sigma) ** 2
        )

    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Identity link for log-mean; log link for log-sigma."""
        if param_name == "logmean":
            return value
        return np.log(value)  # logsigma stored as log(sigma)

    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Identity for log-mean; exp for log-sigma."""
        if param_name == "logmean":
            return value
        return np.exp(value)

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Moment estimates on the log scale."""
        log_y = np.log(np.asarray(y, dtype=float))
        return {"logmean": float(np.mean(log_y)), "logsigma": float(np.std(log_y))}
