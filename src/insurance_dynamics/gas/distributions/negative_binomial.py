"""Negative Binomial GAS distribution for overdispersed count data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, digamma

from .base import GASDistribution


class NegBinGAS(GASDistribution):
    """NB2 (negative binomial) distribution with log-linked time-varying mean.

    Uses the NB2 parametrisation where Var(y) = mu + mu^2 / r, with r being
    the static dispersion parameter (larger r → closer to Poisson). The mean
    mu_t is time-varying via the GAS recursion; r is estimated by MLE.

    Suited to claim frequency data with overdispersion relative to Poisson —
    common when the portfolio contains heterogeneous risk segments.
    """

    param_names = ["mean", "dispersion"]
    default_time_varying = ["mean"]

    def _get_r(self, params: dict) -> float:
        return float(params.get("dispersion", 1.0))

    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Score w.r.t. f = log(mu).

        d/d(log mu) log NB(y | mu, r) = y - (y + r)*mu / (mu + r)
        """
        mu = params["mean"]
        r = self._get_r(params)
        e = exposure if exposure is not None else np.ones_like(np.atleast_1d(y), dtype=float)
        mu_e = mu * e
        y_arr = np.asarray(y, dtype=float)
        return {"mean": y_arr - (y_arr + r) * mu_e / (mu_e + r)}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Fisher information w.r.t. f = log(mu).

        I(f) = mu^2 * r / (mu + r) * E for the log-linked mean.
        """
        mu = params["mean"]
        r = self._get_r(params)
        e = exposure if exposure is not None else 1.0
        mu_e = mu * e
        return {"mean": mu_e**2 * r / (mu_e + r) / mu_e}  # = mu_e * r / (mu_e + r)

    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log NB2(y | mu, r) with exposure offset."""
        mu = params["mean"]
        r = self._get_r(params)
        e = exposure if exposure is not None else np.ones_like(np.atleast_1d(y), dtype=float)
        mu_e = mu * e
        y_arr = np.asarray(y, dtype=float)
        return (
            gammaln(y_arr + r)
            - gammaln(r)
            - gammaln(y_arr + 1.0)
            + r * np.log(r / (r + mu_e))
            + y_arr * np.log(mu_e / (r + mu_e))
        )

    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Log link for both mean and dispersion."""
        return np.log(value)

    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Exp for both mean and dispersion."""
        return np.exp(value)

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Method-of-moments starting values."""
        mu = float(np.mean(y))
        var = float(np.var(y))
        r = mu**2 / max(var - mu, 1e-6)
        return {"mean": mu, "dispersion": max(r, 0.1)}
