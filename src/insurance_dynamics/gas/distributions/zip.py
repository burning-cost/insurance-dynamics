"""Zero-Inflated Poisson GAS distribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from .base import GASDistribution


class ZIPGAS(GASDistribution):
    """Zero-Inflated Poisson with time-varying mean and zero-inflation probability.

    y_t ~ ZIP(mu_t, pi_t) where:
      P(y=0) = pi_t + (1 - pi_t) * exp(-mu_t)
      P(y=k) = (1 - pi_t) * Poisson(k | mu_t)  for k > 0

    Both mu_t (log-linked) and pi_t (logit-linked) can be made time-varying.
    The zero-inflation probability pi_t tracks structural changes in nil-claim
    rates — useful when excess/deductible changes occur mid-series.

    By default only the mean is time-varying; set time_varying=['mean', 'zeroprob']
    in GASModel to enable both.
    """

    param_names = ["mean", "zeroprob"]
    default_time_varying = ["mean"]

    def _safe_params(self, params: dict) -> tuple:
        mu = params["mean"]
        pi = float(params.get("zeroprob", 0.1))
        pi = np.clip(pi, 1e-8, 1.0 - 1e-8)
        return mu, pi

    def _zero_probs(self, mu: float | NDArray, pi: float) -> NDArray:
        """P(y=0 | mu, pi)."""
        return pi + (1.0 - pi) * np.exp(-mu)

    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Score of log p(y | mu, pi) w.r.t. log(mu) and logit(pi).

        The score is derived component-wise.
        """
        mu, pi = self._safe_params(params)
        e = exposure if exposure is not None else np.ones_like(np.atleast_1d(y), dtype=float)
        mu_e = mu * e
        y_arr = np.asarray(y, dtype=float)
        p0 = self._zero_probs(mu_e, pi)

        is_zero = y_arr == 0

        # Score w.r.t. log(mu)
        # For y=0: d/d(log mu) log p0 = -(1-pi)*mu_e*exp(-mu_e) / p0
        # For y>0: d/d(log mu) log Poisson(y | mu_e) = y - mu_e
        score_mu = np.where(
            is_zero,
            -(1.0 - pi) * mu_e * np.exp(-mu_e) / p0,
            y_arr - mu_e,
        )

        # Score w.r.t. logit(pi) = pi*(1-pi) * d/dpi log p
        # For y=0: d/dpi log p0 = (1 - exp(-mu_e)) / p0
        # For y>0: d/dpi log p(y>0) = -1/(1-pi)
        score_pi_raw = np.where(
            is_zero,
            (1.0 - np.exp(-mu_e)) / p0,
            -1.0 / (1.0 - pi),
        )
        score_pi = pi * (1.0 - pi) * score_pi_raw

        return {"mean": score_mu, "zeroprob": score_pi}

    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Approximate Fisher information via expected squared score.

        Computed analytically for the mean; numerically for pi.
        """
        mu, pi = self._safe_params(params)
        e = exposure if exposure is not None else 1.0
        mu_e = mu * e
        p0 = self._zero_probs(mu_e, pi)

        # Fisher for log(mu): E[(score_mu)^2]
        # E[score_mu^2] = (1-p0) * Var(y>0) term + p0 term (approximate)
        fi_mu = (1.0 - pi) * mu_e  # approximation ignoring zero-inflation correction
        # Correction for zero-inflation
        fi_mu_corr = ((1.0 - pi) * mu_e * np.exp(-mu_e)) ** 2 / p0
        fi_mu = max(fi_mu - fi_mu_corr, 1e-6)

        # Fisher for logit(pi): pi^2*(1-pi)^2 * E[(d log p / dpi)^2]
        fi_pi = (pi * (1.0 - pi)) ** 2 * (
            p0 * ((1.0 - np.exp(-mu_e)) / p0) ** 2
            + (1.0 - p0) * (1.0 / (1.0 - pi)) ** 2
        )

        return {"mean": float(fi_mu), "zeroprob": float(fi_pi)}

    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log ZIP(y | mu, pi) with exposure offset."""
        mu, pi = self._safe_params(params)
        e = exposure if exposure is not None else np.ones_like(np.atleast_1d(y), dtype=float)
        mu_e = mu * e
        y_arr = np.asarray(y, dtype=float)
        p0 = self._zero_probs(mu_e, pi)

        is_zero = y_arr == 0
        ll_zero = np.log(np.maximum(p0, 1e-300))
        ll_pos = np.log(1.0 - pi) + y_arr * np.log(mu_e) - mu_e - gammaln(y_arr + 1.0)

        return np.where(is_zero, ll_zero, ll_pos)

    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Log link for mean; logit link for zero probability."""
        if param_name == "mean":
            return np.log(value)
        return np.log(value / (1.0 - value))

    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Exp for mean; sigmoid for zero probability."""
        if param_name == "mean":
            return np.exp(value)
        return 1.0 / (1.0 + np.exp(-value))

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Estimate from proportion of zeros and sample mean."""
        y_arr = np.asarray(y, dtype=float)
        prop_zero = float(np.mean(y_arr == 0))
        mu = float(np.mean(y_arr[y_arr > 0])) if np.any(y_arr > 0) else 1.0
        # Approximate MOM: prop_zero ≈ pi + (1-pi)*exp(-mu)
        pi = max((prop_zero - np.exp(-mu)) / (1.0 - np.exp(-mu)), 0.01)
        return {"mean": mu, "zeroprob": min(pi, 0.9)}
