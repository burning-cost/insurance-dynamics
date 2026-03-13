"""Beta GAS distribution for loss ratio modelling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, digamma

from .base import GASDistribution


class BetaGAS(GASDistribution):
    """Beta distribution with logit-linked time-varying mean for loss ratios.

    y_t ~ Beta(mu_t * phi, (1 - mu_t) * phi) where mu_t in (0,1) is the
    time-varying mean and phi > 0 is the static precision parameter.

    Loss ratios (claims / premium) live on (0,1) and their evolution over
    time can be tracked with this distribution. The logit link ensures the
    filtered mean stays in (0,1) throughout.
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
        """Score w.r.t. logit(mu) (logit link).

        For Beta with mean parametrisation, the score w.r.t. logit(mu) is:
        phi * (y - mu) where phi is the precision.
        """
        mu = params["mean"]
        phi = self._get_phi(params)
        y_arr = np.asarray(y, dtype=float)
        return {"mean": phi * (y_arr - mu)}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Fisher information w.r.t. logit(mu).

        I(logit(mu)) = phi * mu * (1 - mu).
        """
        mu = params["mean"]
        phi = self._get_phi(params)
        return {"mean": phi * mu * (1.0 - mu)}

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
        return (
            gammaln(phi)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y_arr)
            + (b - 1.0) * np.log(1.0 - y_arr)
        )

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
