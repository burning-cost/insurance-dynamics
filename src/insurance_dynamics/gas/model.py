"""GASModel: main user-facing class for fitting GAS models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize

from .distributions.base import GASDistribution
from .distributions import DISTRIBUTION_MAP
from .filter import GASFilter, FilterResult


@dataclass
class GASResult:
    """Result from fitting a GASModel.

    Attributes
    ----------
    filter_path:
        pd.DataFrame with one column per time-varying parameter, showing
        the filtered values at each observation.
    log_likelihood:
        Total log-likelihood at the MLE.
    aic:
        Akaike information criterion.
    bic:
        Bayesian information criterion.
    params:
        Fitted parameter dict (all GAS + static parameters).
    std_errors:
        Standard errors from the numerical Hessian.
    score_residuals:
        Standardised score residuals — should be approximately iid(0,1)
        for a well-specified model.
    n_obs:
        Number of observations.
    """

    filter_path: pd.DataFrame
    log_likelihood: float
    aic: float
    bic: float
    params: dict[str, float]
    std_errors: dict[str, float]
    score_residuals: pd.DataFrame
    n_obs: int
    distribution: GASDistribution
    model: "GASModel"
    _raw_result: optimize.OptimizeResult | None = field(default=None, repr=False)

    @property
    def trend_index(self) -> pd.DataFrame:
        """Time-varying parameter paths relative to the first observation (=100).

        This is the format actuaries already understand from chain-ladder
        development factor analysis.
        """
        base = self.filter_path.iloc[0]
        return (self.filter_path / base) * 100.0

    def relativities(self, base: str = "mean") -> pd.DataFrame:
        """Time-varying relativities relative to the mean of the filter path.

        Parameters
        ----------
        base:
            ``'mean'`` divides by the time-average; ``'first'`` divides by
            the first observation (same as trend_index / 100).

        Returns
        -------
        pd.DataFrame with the same shape as filter_path.
        """
        if base == "mean":
            divisor = self.filter_path.mean()
        elif base == "first":
            divisor = self.filter_path.iloc[0]
        else:
            raise ValueError(f"base must be 'mean' or 'first', got '{base}'")
        return self.filter_path / divisor

    def summary(self) -> str:
        """Formatted coefficient table."""
        lines = [
            f"GAS Model ({self.model.distribution_name})",
            f"Observations: {self.n_obs}",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.aic:.4f}  BIC: {self.bic:.4f}",
            "",
            f"{'Parameter':<25} {'Estimate':>12} {'Std Error':>12} {'z-value':>10}",
            "-" * 62,
        ]
        for name, est in self.params.items():
            se = self.std_errors.get(name, float("nan"))
            z = est / se if se and se > 0 else float("nan")
            lines.append(f"  {name:<23} {est:>12.6f} {se:>12.6f} {z:>10.3f}")
        return "\n".join(lines)

    def forecast(
        self,
        h: int = 6,
        method: str = "mean_path",
        quantiles: list[float] | None = None,
        n_sim: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> "ForecastResult":
        """Produce h-step-ahead forecasts from the fitted model.

        Parameters
        ----------
        h:
            Number of periods to forecast.
        method:
            ``'mean_path'`` propagates the mean; ``'simulate'`` draws
            sample paths.
        quantiles:
            Quantile levels for prediction intervals (simulate only).
        n_sim:
            Number of simulation paths.
        rng:
            Random number generator for reproducibility.

        Returns
        -------
        ForecastResult
        """
        from .forecast import gas_forecast

        return gas_forecast(
            result=self,
            h=h,
            method=method,
            quantiles=quantiles or [0.1, 0.5, 0.9],
            n_sim=n_sim,
            rng=rng,
        )

    def diagnostics(self) -> "DiagnosticsResult":
        """Compute model diagnostics.

        Returns
        -------
        DiagnosticsResult with PIT residuals, coverage test and ACF.
        """
        from .diagnostics import compute_diagnostics

        return compute_diagnostics(self)

    def bootstrap_ci(
        self,
        method: str = "parametric",
        n_boot: int = 500,
        confidence: float = 0.95,
        rng: np.random.Generator | None = None,
    ) -> "BootstrapCI":
        """Parametric bootstrap confidence intervals for the filter path.

        Parameters
        ----------
        method:
            ``'parametric'`` samples from the fitted distribution.
        n_boot:
            Number of bootstrap replications.
        confidence:
            Nominal coverage of the interval.
        rng:
            Random number generator.

        Returns
        -------
        BootstrapCI
        """
        from .bootstrap import bootstrap_ci

        return bootstrap_ci(
            result=self,
            method=method,
            n_boot=n_boot,
            confidence=confidence,
            rng=rng,
        )


class GASModel:
    """GAS (Generalised Autoregressive Score) model for time-varying parameters.

    Fits a score-driven model to a univariate time series via maximum
    likelihood. The GAS recursion is:

        f_{t+1} = omega + alpha * S(f_t) * nabla(y_t, f_t) + phi * f_t

    where nabla is the score of the log-likelihood w.r.t. f, and S is a
    scaling matrix derived from the Fisher information.

    Parameters
    ----------
    distribution:
        Distribution name (``'poisson'``, ``'gamma'``, ``'negbin'``,
        ``'lognormal'``, ``'beta'``, ``'zip'``) or a GASDistribution instance.
    p:
        Number of score lags. Default 1.
    q:
        Number of AR lags (persistence). Default 1.
    scaling:
        Score scaling method. One of:
        ``'unit'`` — no scaling;
        ``'fisher_inv'`` — multiply by inverse Fisher information (default);
        ``'fisher_inv_sqrt'`` — robust alternative.
    time_varying:
        Which distribution parameters to make time-varying. If None,
        uses the distribution's default.

    Examples
    --------
    >>> from insurance_gas import GASModel
    >>> model = GASModel('poisson')
    >>> result = model.fit(y, exposure=exposure)
    >>> print(result.summary())
    """

    def __init__(
        self,
        distribution: str | GASDistribution = "poisson",
        p: int = 1,
        q: int = 1,
        scaling: str = "fisher_inv",
        time_varying: list[str] | None = None,
    ) -> None:
        if isinstance(distribution, str):
            dist_name = distribution.lower()
            if dist_name not in DISTRIBUTION_MAP:
                raise ValueError(
                    f"Unknown distribution '{distribution}'. "
                    f"Choose from: {list(DISTRIBUTION_MAP.keys())}"
                )
            self.distribution_name = dist_name
            self.distribution = DISTRIBUTION_MAP[dist_name]()
        else:
            self.distribution = distribution
            self.distribution_name = type(distribution).__name__

        self.p = p
        self.q = q
        self.scaling = scaling
        self.time_varying = time_varying or self.distribution.default_time_varying

        self._filter = GASFilter(
            distribution=self.distribution,
            time_varying=self.time_varying,
            scaling=scaling,
            p=p,
            q=q,
        )

        self._fitted: GASResult | None = None

    def _build_param_names(self) -> list[str]:
        """Return ordered list of all optimisation parameter names."""
        names = []
        for tv_name in self.time_varying:
            names.append(f"omega_{tv_name}")
            for i in range(self.p):
                names.append(f"alpha_{tv_name}_{i+1}")
            for j in range(self.q):
                names.append(f"phi_{tv_name}_{j+1}")
        return names

    def _build_static_param_names(self) -> list[str]:
        """Return names of static (non-time-varying) distribution parameters."""
        return [
            p for p in self.distribution.param_names if p not in self.time_varying
        ]

    def _unpack_x(
        self,
        x: NDArray[np.float64],
        param_names: list[str],
        static_names: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Split optimisation vector into GAS params and static params."""
        n_gas = len(param_names)
        gas_x = x[:n_gas]
        static_x = x[n_gas:]

        gas_params = dict(zip(param_names, gas_x.tolist()))
        static_params = {}
        for i, name in enumerate(static_names):
            static_params[name] = float(
                self.distribution.unlink(name, static_x[i])
            )

        return gas_params, static_params

    def _neg_log_likelihood(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None,
        param_names: list[str],
        static_names: list[str],
    ) -> float:
        """Negative log-likelihood for scipy.optimize.minimize."""
        try:
            gas_params, static_params = self._unpack_x(x, param_names, static_names)

            # Stationarity constraint: phi sum < 1 in magnitude
            for tv_name in self.time_varying:
                phi_vals = [gas_params[f"phi_{tv_name}_{j+1}"] for j in range(self.q)]
                if abs(sum(phi_vals)) >= 1.0:
                    return 1e10

            result = self._filter.run(y, gas_params, static_params, exposure=exposure)
            nll = -np.sum(result.log_likelihoods)
            return float(nll) if np.isfinite(nll) else 1e10
        except Exception:
            return 1e10

    def fit(
        self,
        y: Sequence[float] | NDArray[np.float64],
        exposure: Sequence[float] | NDArray[np.float64] | None = None,
        max_iter: int = 1000,
    ) -> GASResult:
        """Fit the GAS model via maximum likelihood (L-BFGS-B).

        Parameters
        ----------
        y:
            Observed time series of length T.
        exposure:
            Optional per-observation exposure values (for count distributions).
        max_iter:
            Maximum iterations for the optimiser.

        Returns
        -------
        GASResult with filter path, parameter estimates, and diagnostics.
        """
        y = np.asarray(y, dtype=float)
        T = len(y)
        if T < 4:
            raise ValueError(f"Need at least 4 observations, got {T}.")

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            if len(exposure) != T:
                raise ValueError(f"exposure length {len(exposure)} != y length {T}.")

        # Initialise distribution from data
        init_dist = self.distribution.initial_params(y)

        param_names = self._build_param_names()
        static_names = self._build_static_param_names()

        # Build initial x0
        x0_gas = []
        for tv_name in self.time_varying:
            init_f = self.distribution.link(tv_name, init_dist.get(tv_name, 1.0))
            x0_gas.append(float(init_f) * (1.0 - 0.9))  # omega ≈ intercept
            for i in range(self.p):
                x0_gas.append(0.1)  # alpha
            for j in range(self.q):
                x0_gas.append(0.9 if j == 0 else 0.0)  # phi

        x0_static = []
        for name in static_names:
            val = init_dist.get(name, 1.0)
            x0_static.append(float(self.distribution.link(name, val)))

        x0 = np.array(x0_gas + x0_static)

        # Bounds: phi in (-1, 1) via tanh transformation applied after
        # For now we enforce phi sum < 1 inside the objective.
        result = optimize.minimize(
            self._neg_log_likelihood,
            x0,
            args=(y, exposure, param_names, static_names),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6},
        )

        if not result.success:
            warnings.warn(
                f"Optimisation did not converge: {result.message}. "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        gas_params, static_params = self._unpack_x(
            result.x, param_names, static_names
        )

        # Run filter at MLE
        filter_result = self._filter.run(y, gas_params, static_params, exposure=exposure)

        # Hessian-based standard errors
        std_errors = self._compute_std_errors(
            result.x, y, exposure, param_names, static_names
        )

        # All fitted parameters (GAS + static on natural scale)
        all_params: dict[str, float] = dict(gas_params)
        all_params.update(static_params)

        # Standard errors for static params — transform back
        all_se: dict[str, float] = {}
        for name in param_names:
            all_se[name] = std_errors.get(name, float("nan"))
        for i, name in enumerate(static_names):
            all_se[name] = std_errors.get(f"static_{i}", float("nan"))

        # Build filter_path DataFrame
        filter_df = pd.DataFrame(filter_result.filter_paths)

        score_df = pd.DataFrame(filter_result.score_residuals)

        n_params = len(x0)
        ll = float(np.sum(filter_result.log_likelihoods))
        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + n_params * np.log(T)

        self._fitted = GASResult(
            filter_path=filter_df,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            params=all_params,
            std_errors=all_se,
            score_residuals=score_df,
            n_obs=T,
            distribution=self.distribution,
            model=self,
            _raw_result=result,
        )
        return self._fitted

    def _compute_std_errors(
        self,
        x_opt: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None,
        param_names: list[str],
        static_names: list[str],
    ) -> dict[str, float]:
        """Numerical Hessian-based standard errors."""
        try:
            eps = 1e-5
            n = len(x_opt)
            H = np.zeros((n, n))
            f0 = self._neg_log_likelihood(
                x_opt, y, exposure, param_names, static_names
            )
            for i in range(n):
                for j in range(i, n):
                    ei = np.zeros(n)
                    ej = np.zeros(n)
                    ei[i] = eps
                    ej[j] = eps
                    if i == j:
                        fpp = self._neg_log_likelihood(x_opt + ei + ej, y, exposure, param_names, static_names)
                        fpm = self._neg_log_likelihood(x_opt + ei - ej, y, exposure, param_names, static_names)
                        H[i, j] = (fpp - 2 * f0 + fpm) / eps**2
                    else:
                        fpp = self._neg_log_likelihood(x_opt + ei + ej, y, exposure, param_names, static_names)
                        fpm = self._neg_log_likelihood(x_opt + ei - ej, y, exposure, param_names, static_names)
                        fmp = self._neg_log_likelihood(x_opt - ei + ej, y, exposure, param_names, static_names)
                        fmm = self._neg_log_likelihood(x_opt - ei - ej, y, exposure, param_names, static_names)
                        H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)

            # Covariance = inv(H)
            try:
                cov = np.linalg.inv(H)
                variances = np.diag(cov)
                variances = np.where(variances > 0, variances, np.nan)
                ses = np.sqrt(variances)
            except np.linalg.LinAlgError:
                ses = np.full(n, float("nan"))

            result = {}
            for i, name in enumerate(param_names):
                result[name] = float(ses[i]) if i < len(ses) else float("nan")
            for j, name in enumerate(static_names):
                idx = len(param_names) + j
                result[f"static_{j}"] = float(ses[idx]) if idx < len(ses) else float("nan")
            return result

        except Exception:
            all_names = param_names + [f"static_{j}" for j in range(len(static_names))]
            return {name: float("nan") for name in all_names}
