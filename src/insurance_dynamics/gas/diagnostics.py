"""Model diagnostics for GAS models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


@dataclass
class DiagnosticsResult:
    """Diagnostic statistics for a fitted GAS model.

    Attributes
    ----------
    pit_values:
        Probability integral transform values u_t = F(y_t | f_t).
        Should be approximately Uniform(0,1) for a well-specified model.
        For discrete distributions, randomised PIT is used.
    ks_statistic:
        Kolmogorov-Smirnov statistic for uniformity of PIT values.
    ks_pvalue:
        p-value for the KS uniformity test.
    ds_score:
        Dawid-Sebastiani proper scoring rule (lower is better).
    acf_values:
        Autocorrelation of score residuals at lags 0-20.
    ljung_box_pvalue:
        p-value for the Ljung-Box test on score residuals.
    """

    pit_values: NDArray[np.float64]
    ks_statistic: float
    ks_pvalue: float
    ds_score: float
    acf_values: NDArray[np.float64]
    ljung_box_pvalue: float

    def summary(self) -> str:
        """Print a short diagnostic summary."""
        lines = [
            "GAS Model Diagnostics",
            "-" * 40,
            f"PIT uniformity (KS): stat={self.ks_statistic:.4f}, p={self.ks_pvalue:.4f}",
            f"Dawid-Sebastiani score: {self.ds_score:.4f}",
            f"Score residual Ljung-Box p: {self.ljung_box_pvalue:.4f}",
        ]
        if self.ks_pvalue > 0.05:
            lines.append("PIT test: PASS (cannot reject uniformity)")
        else:
            lines.append("PIT test: FAIL (distribution misspecified?)")
        if self.ljung_box_pvalue > 0.05:
            lines.append("Score ACF test: PASS (no remaining autocorrelation)")
        else:
            lines.append("Score ACF test: FAIL (residual dynamics remain)")
        return "\n".join(lines)

    def pit_histogram(self, ax=None):
        """Plot PIT histogram with uniform reference line."""
        from .plotting import plot_pit_histogram
        return plot_pit_histogram(self.pit_values, ax=ax)

    def plot_acf(self, ax=None):
        """Plot ACF of score residuals."""
        from .plotting import plot_acf
        return plot_acf(self.acf_values, ax=ax)


def compute_diagnostics(result) -> DiagnosticsResult:
    """Compute diagnostic statistics from a fitted GASResult.

    Parameters
    ----------
    result:
        A fitted GASResult.

    Returns
    -------
    DiagnosticsResult
    """
    from .distributions import PoissonGAS, NegBinGAS, ZIPGAS

    dist = result.distribution
    model = result.model
    filter_path = result.filter_path
    time_varying = model.time_varying

    # P1-2 fix: _y is now stored on GASResult during fit().
    y = getattr(result, "_y", None)

    if y is not None:
        T = len(y)
        pit_vals = np.zeros(T)
        rng = np.random.default_rng(42)

        for t in range(T):
            params_t = {name: float(filter_path[name].iloc[t]) for name in time_varying}
            for sname in model._build_static_param_names():
                params_t[sname] = result.params[sname]

            if isinstance(dist, (PoissonGAS, NegBinGAS, ZIPGAS)):
                pit_vals[t] = _randomised_pit_discrete(dist, float(y[t]), params_t, rng)
            else:
                pit_vals[t] = _pit_continuous(dist, float(y[t]), params_t)
    else:
        sr = result.score_residuals.iloc[:, 0].values
        pit_vals = stats.norm.cdf(sr)

    # KS test for uniformity
    ks_stat, ks_p = stats.kstest(pit_vals, "uniform")

    # P1-3 fix: Dawid-Sebastiani proper scoring rule.
    # Use (y-mu)^2/sigma^2 + 2*log(sigma), not a function of score residuals.
    if y is not None:
        T = len(y)
        mu_arr = np.zeros(T)
        sigma_arr = np.zeros(T)
        for t in range(T):
            params_t = {name: float(filter_path[name].iloc[t]) for name in time_varying}
            for sname in model._build_static_param_names():
                params_t[sname] = result.params[sname]
            mu_arr[t], sigma_arr[t] = _predictive_mean_sigma(dist, params_t)
        sigma_arr = np.where(sigma_arr > 0, sigma_arr, 1e-8)
        ds_mean = dawid_sebastiani_score(
            np.asarray(y, dtype=float), mu_arr, sigma_arr
        )
    else:
        # Fallback when _y is unavailable: use score residuals with unit sigma.
        # This is approximate but avoids crashing.
        sr = result.score_residuals.iloc[:, 0].values
        ds_mean = float(np.mean(sr**2))

    # ACF of score residuals
    sr = result.score_residuals.iloc[:, 0].values
    acf_vals = _compute_acf(sr, nlags=20)

    # Ljung-Box test
    T_sr = len(sr)
    lags = np.arange(1, min(21, T_sr // 4 + 1))
    lb_stat = float(T_sr * (T_sr + 2) * np.sum(
        (acf_vals[1:len(lags) + 1] ** 2) / (T_sr - lags)
    ))
    lb_p = float(1.0 - stats.chi2.cdf(lb_stat, df=len(lags)))

    return DiagnosticsResult(
        pit_values=pit_vals,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        ds_score=ds_mean,
        acf_values=acf_vals,
        ljung_box_pvalue=lb_p,
    )


def _predictive_mean_sigma(dist, params: dict) -> tuple[float, float]:
    """Return (predictive mean, predictive std dev) for a distribution at params.

    Used to compute the Dawid-Sebastiani score without requiring the raw
    score residuals.
    """
    from .distributions import (
        PoissonGAS, NegBinGAS, GammaGAS, LogNormalGAS, BetaGAS, ZIPGAS
    )

    if isinstance(dist, PoissonGAS):
        mu = float(params["mean"])
        return mu, float(np.sqrt(mu))

    elif isinstance(dist, NegBinGAS):
        mu = float(params["mean"])
        r = float(params.get("dispersion", 1.0))
        var = mu + mu**2 / r
        return mu, float(np.sqrt(var))

    elif isinstance(dist, GammaGAS):
        mu = float(params["mean"])
        a = float(params.get("shape", 1.0))
        # Var(Gamma) = mean^2 / shape
        return mu, float(mu / np.sqrt(a))

    elif isinstance(dist, LogNormalGAS):
        mu_log = float(params["logmean"])
        # logsigma is stored as log(sigma_log), so sigma_log = exp(logsigma)
        sigma_log = float(np.exp(params.get("logsigma", np.log(0.5))))
        # Predictive mean on original scale
        mu_orig = float(np.exp(mu_log + 0.5 * sigma_log**2))
        # Predictive std on original scale
        sigma_orig = float(
            np.sqrt((np.exp(sigma_log**2) - 1.0) * np.exp(2.0 * mu_log + sigma_log**2))
        )
        return mu_orig, sigma_orig

    elif isinstance(dist, BetaGAS):
        mu = float(params["mean"])
        phi = float(params.get("precision", 10.0))
        var = mu * (1.0 - mu) / (phi + 1.0)
        return mu, float(np.sqrt(var))

    elif isinstance(dist, ZIPGAS):
        mu = float(params["mean"])
        pi = float(params.get("zeroprob", 0.0))
        mean_y = (1.0 - pi) * mu
        var_y = (1.0 - pi) * (mu + mu**2) - mean_y**2
        return mean_y, float(np.sqrt(max(var_y, 1e-8)))

    else:
        # Generic fallback: mean from first time-varying param, unit sigma
        first_val = next(iter(params.values()), 1.0)
        return float(first_val), 1.0


def _randomised_pit_discrete(
    dist,
    y: float,
    params: dict,
    rng: np.random.Generator,
) -> float:
    """Randomised PIT for discrete distributions."""
    from .distributions import PoissonGAS, NegBinGAS, ZIPGAS
    from scipy.stats import poisson, nbinom

    y_int = int(y)

    if isinstance(dist, PoissonGAS):
        mu = params["mean"]
        f_upper = float(poisson.cdf(y_int, mu))
        f_lower = float(poisson.cdf(y_int - 1, mu)) if y_int > 0 else 0.0
    elif isinstance(dist, NegBinGAS):
        mu = params["mean"]
        r = params.get("dispersion", 1.0)
        p = r / (r + mu)
        f_upper = float(nbinom.cdf(y_int, r, p))
        f_lower = float(nbinom.cdf(y_int - 1, r, p)) if y_int > 0 else 0.0
    else:
        # ZIP: approximate with Poisson
        mu = params["mean"]
        pi = params.get("zeroprob", 0.0)
        f_upper = float(pi + (1 - pi) * poisson.cdf(y_int, mu))
        f_lower = float(pi + (1 - pi) * poisson.cdf(y_int - 1, mu)) if y_int > 0 else float(pi)

    u = rng.uniform()
    return float(f_lower + u * (f_upper - f_lower))


def _pit_continuous(dist, y: float, params: dict) -> float:
    """PIT for continuous distributions."""
    from scipy.stats import gamma, norm, beta as beta_dist
    from .distributions import GammaGAS, LogNormalGAS, BetaGAS

    if isinstance(dist, GammaGAS):
        mu = params["mean"]
        a = params.get("shape", 1.0)
        return float(gamma.cdf(y, a=a, scale=mu / a))
    elif isinstance(dist, LogNormalGAS):
        mu_log = params["logmean"]
        # P0-3 fix: logsigma is log(sigma), so exponentiate to get sigma.
        sigma = float(np.exp(params.get("logsigma", np.log(0.5))))
        return float(norm.cdf(np.log(y), loc=mu_log, scale=sigma))
    elif isinstance(dist, BetaGAS):
        mu = params["mean"]
        phi = params.get("precision", 10.0)
        return float(beta_dist.cdf(y, mu * phi, (1 - mu) * phi))
    else:
        return 0.5  # fallback


def _compute_acf(x: NDArray[np.float64], nlags: int = 20) -> NDArray[np.float64]:
    """Compute sample autocorrelation at lags 0 through nlags."""
    n = len(x)
    x_centred = x - np.mean(x)
    var = float(np.var(x_centred))
    if var == 0:
        return np.zeros(nlags + 1)
    acf = np.zeros(nlags + 1)
    acf[0] = 1.0
    for k in range(1, nlags + 1):
        acf[k] = float(np.dot(x_centred[:-k], x_centred[k:]) / (var * (n - k)))
    return acf


def pit_residuals(
    y: NDArray[np.float64],
    filter_path: pd.DataFrame,
    distribution,
    params: dict[str, float],
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Compute PIT residuals for a given series and filter path."""
    if rng is None:
        rng = np.random.default_rng(42)
    from .distributions import PoissonGAS, NegBinGAS, ZIPGAS

    T = len(y)
    pit_vals = np.zeros(T)
    time_varying = list(filter_path.columns)

    for t in range(T):
        params_t = {name: float(filter_path[name].iloc[t]) for name in time_varying}
        params_t.update({k: v for k, v in params.items() if k not in time_varying})

        if isinstance(distribution, (PoissonGAS, NegBinGAS, ZIPGAS)):
            pit_vals[t] = _randomised_pit_discrete(distribution, float(y[t]), params_t, rng)
        else:
            pit_vals[t] = _pit_continuous(distribution, float(y[t]), params_t)

    return pit_vals


def dawid_sebastiani_score(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> float:
    """Dawid-Sebastiani proper scoring rule.

    DS(F, y) = (y - mu)^2 / sigma^2 + 2 * log(sigma)

    Lower is better. Proper scoring rule for distributional forecasts.
    """
    return float(np.mean(((y - mu) / sigma) ** 2 + 2.0 * np.log(sigma)))
