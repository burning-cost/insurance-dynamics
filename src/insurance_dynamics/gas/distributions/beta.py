"""Beta GAS distribution for loss ratio modelling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, digamma, polygamma

from .base import GASDistribution


class BetaGAS(GASDistribution):
    """Beta distribution with logit-linked time-varying mean for loss ratios.

    y_t ~ Beta(mu_t * phi, (1 - mu_t) * phi) where mu_t in (0,1) is the
    time-varying mean and phi > 0 is the static precision parameter.

    Loss ratios (claims / premium) live on (0,1) and their evolution over
    time can be tracked with this distribution. The logit link ensures the
    filtered mean stays in (0,1) throughout.

    Score w.r.t. logit(mu) follows Ferrari & Cribari-Neto (2004):
        raw = phi * mu * (1-mu) * [logit(y) + psi((1-mu)*phi) - psi(mu*phi)]
    Fisher information w.r.t. logit(mu):
        I = (phi * mu * (1-mu))^2 * [polygamma(1, mu*phi) + polygamma(1, (1-mu)*phi)]
    where psi is the digamma function and polygamma(1, .) is the trigamma function.
    """

    param_names = ["mean", "precision"]
    default_time_varying = ["mean"]

    def _get_phi(self, params: dict) -> float:
        return float(params.get("precision", 10.0))

    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Raw score w.r.t. logit(mu) (logit link).

        Ferrari & Cribari-Neto (2004) eq. for d/d(logit mu) log Beta(y | mu, phi):
            score = phi * mu * (1-mu) * (log(y/(1-y)) + digamma((1-mu)*phi) - digamma(mu*phi))

        Returns the *raw* (unscaled) score.
        """
        mu = params["mean"]
        phi = self._get_phi(params)
        y_arr = np.asarray(y, dtype=float)
        raw = phi * mu * (1.0 - mu) * (
            np.log(y_arr / (1.0 - y_arr))
            + digamma((1.0 - mu) * phi)
            - digamma(mu * phi)
        )
        return {"mean": raw}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Fisher information w.r.t. logit(mu).

        I(logit(mu)) = (phi * mu * (1-mu))^2 * (trigamma(mu*phi) + trigamma((1-mu)*phi))
        where trigamma = polygamma(1, .).
        """
        mu = params["mean"]
        phi = self._get_phi(params)
        fi = (phi * mu * (1.0 - mu)) ** 2 * (
            polygamma(1, mu * phi) + polygamma(1, (1.0 - mu) * phi)
        )
        return {"mean": fi}

    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log Beta(y | mu, phi) in mean-precision parametrisation."""
        mu = params["mean"]
        phi = self._get_phi(params)
        a = mu * phi
        b = (1.0 - mu) * phi
        y_arr = np.asarray(y, dtype=float)
        result = (
            gammaln(phi)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y_arr)
            + (b - 1.0) * np.log(1.0 - y_arr)
        )
        return np.squeeze(result)

    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Logit link for mean; log link for precision."""
        if param_name == "mean":
            return np.log(value / (1.0 - value))
        return np.log(value)

    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Sigmoid for mean; exp for precision."""
        if param_name == "mean":
            return 1.0 / (1.0 + np.exp(-value))
        return np.exp(value)

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Method-of-moments estimates."""
        y_arr = np.asarray(y, dtype=float)
        mu = float(np.mean(y_arr))
        var = float(np.var(y_arr))
        phi = mu * (1.0 - mu) / max(var, 1e-8) - 1.0
        return {"mean": mu, "precision": max(phi, 1.0)}
