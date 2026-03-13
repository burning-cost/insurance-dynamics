"""Parametric bootstrap confidence intervals for GAS filter paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BootstrapCI:
    """Bootstrap confidence intervals for the filter path.

    Attributes
    ----------
    filter_lower:
        Lower confidence band (DataFrame, same shape as filter_path).
    filter_upper:
        Upper confidence band.
    filter_median:
        Bootstrap median (may differ slightly from MLE estimate).
    confidence:
        Nominal coverage level.
    n_boot:
        Number of bootstrap replications used.
    """

    filter_lower: pd.DataFrame
    filter_upper: pd.DataFrame
    filter_median: pd.DataFrame
    confidence: float
    n_boot: int

    def plot(self, param: str | None = None, ax=None):
        """Plot the bootstrap confidence band."""
        import matplotlib.pyplot as plt
        from .plotting import plot_filter

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        # Overlay CI on top of median
        if param is None:
            param = self.filter_median.columns[0]
        ax.fill_between(
            range(len(self.filter_lower)),
            self.filter_lower[param].values,
            self.filter_upper[param].values,
            alpha=0.3,
            label=f"{int(self.confidence * 100)}% bootstrap CI",
        )
        ax.plot(self.filter_median[param].values, label="Bootstrap median")
        ax.legend()
        return ax


def bootstrap_ci(
    result,
    method: str = "parametric",
    n_boot: int = 500,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Parametric bootstrap confidence intervals for the GAS filter path.

    Generates synthetic datasets from the fitted model, refits the model to
    each, and collects the distribution of filter paths. This captures both
    parameter uncertainty and filtering uncertainty.

    Parameters
    ----------
    result:
        Fitted GASResult.
    method:
        Only ``'parametric'`` is currently supported.
    n_boot:
        Number of bootstrap replications.
    confidence:
        Nominal coverage (e.g. 0.95 for 95% CI).
    rng:
        Random number generator for reproducibility.

    Returns
    -------
    BootstrapCI
    """
    from .forecast import _draw_sample

    if rng is None:
        rng = np.random.default_rng(42)

    model = result.model
    dist = result.distribution
    T = result.n_obs
    time_varying = model.time_varying

    # Collect bootstrap filter paths
    boot_paths: dict[str, list[NDArray]] = {name: [] for name in time_varying}

    for b in range(n_boot):
        # Generate synthetic series from fitted filter path
        y_boot = np.zeros(T)
        for t in range(T):
            params_t = {name: float(result.filter_path[name].iloc[t]) for name in time_varying}
            # Add static params
            for sname in model._build_static_param_names():
                params_t[sname] = result.params[sname]
            y_boot[t] = _draw_sample(dist, params_t, rng)

        # Refit model
        try:
            boot_model = type(model)(
                distribution=model.distribution_name if hasattr(model, "distribution_name") else "poisson",
                p=model.p,
                q=model.q,
                scaling=model.scaling,
                time_varying=model.time_varying,
            )
            boot_result = boot_model.fit(y_boot, max_iter=300)
            for name in time_varying:
                boot_paths[name].append(boot_result.filter_path[name].values)
        except Exception:
            continue  # skip failed fits

    if not any(boot_paths[name] for name in time_varying):
        raise RuntimeError("All bootstrap replications failed.")

    alpha = 1.0 - confidence
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    lower_data: dict[str, NDArray] = {}
    upper_data: dict[str, NDArray] = {}
    median_data: dict[str, NDArray] = {}

    for name in time_varying:
        paths_arr = np.array(boot_paths[name])  # (n_boot, T)
        lower_data[name] = np.quantile(paths_arr, lower_q, axis=0)
        upper_data[name] = np.quantile(paths_arr, upper_q, axis=0)
        median_data[name] = np.quantile(paths_arr, 0.5, axis=0)

    return BootstrapCI(
        filter_lower=pd.DataFrame(lower_data),
        filter_upper=pd.DataFrame(upper_data),
        filter_median=pd.DataFrame(median_data),
        confidence=confidence,
        n_boot=n_boot,
    )
