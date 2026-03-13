"""Poisson GAS distribution for claim frequency modelling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from .base import GASDistribution


class PoissonGAS(GASDistribution):
    """Poisson distribution with log-linked time-varying mean.

    The time-varying parameter f_t = log(mu_t) so that mu_t is always
    positive. The GAS recursion adapts the rate up when observed counts
    exceed the fitted rate and down when they fall short.

    With exposure E_t, the conditional mean is mu_t * E_t and the score
    becomes (y_t / (mu_t * E_t)) - 1, scaled by the Fisher information
    1 / mu_t (for the log-linked parameter).

    This is the primary distribution for monthly or quarterly claim
    frequency monitoring in UK personal lines.
    """

    param_names = ["mean"]
    default_time_varying = ["mean"]

    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Score of log P(y | mu) w.r.t. f = log(mu).

        For the log link: d/df log P(y | exp(f)) = y - mu * E
        where E is the exposure. Fisher info w.r.t. f is mu * E.
        The unit-scaled score is therefore (y / (mu * E)) - 1.
        """
        mu = params["mean"]
        e = exposure if exposure is not None else np.ones_like(np.atleast_1d(y), dtype=float)
        rate = mu * e
        return {"mean": np.asarray(y, dtype=float) / rate - 1.0}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Fisher information of log P(y | mu) w.r.t. f = log(mu).

        For a Poisson with log link, I(f) = mu * E.
        """
        mu = params["mean"]
        e = exposure if exposure is not None else 1.0
        return {"mean": mu * e}

    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log P(y | mu, E) = y*log(mu*E) - mu*E - log(y!)"""
        mu = params["mean"]
        e = exposure if exposure is not None else np.ones_like(np.atleast_1d(y), dtype=float)
        rate = mu * e
        y_arr = np.asarray(y, dtype=float)
        return y_arr * np.log(rate) - rate - gammaln(y_arr + 1.0)

    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Log link: f = log(mu)."""
        return np.log(value)

    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Inverse log link: mu = exp(f)."""
        return np.exp(value)

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Start from the sample mean."""
        return {"mean": float(np.mean(y))}
