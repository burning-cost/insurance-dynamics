"""Plotting functions for GAS model output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_filter(
    filter_path: pd.DataFrame,
    param: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ax=None,
    **kwargs,
):
    """Plot the time-varying filter path.

    Parameters
    ----------
    filter_path:
        DataFrame with one column per time-varying parameter.
    param:
        Which parameter to plot. Uses the first column if None.
    title:
        Plot title.
    ylabel:
        Y-axis label.
    ax:
        Matplotlib axes object. Creates a new figure if None.

    Returns
    -------
    Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if param is None:
        param = filter_path.columns[0]

    ax.plot(filter_path[param].values, **kwargs)
    ax.set_title(title or f"GAS Filter Path: {param}")
    ax.set_xlabel("Period")
    ax.set_ylabel(ylabel or param)
    return ax


def plot_pit_histogram(
    pit_values: NDArray[np.float64],
    bins: int = 20,
    ax=None,
) -> object:
    """Histogram of PIT values with uniform reference line.

    A well-specified model produces a near-uniform PIT distribution.
    Any U-shape or systematic bias indicates distributional misspecification.

    Parameters
    ----------
    pit_values:
        Array of PIT values in [0, 1].
    bins:
        Number of histogram bins.
    ax:
        Matplotlib axes.

    Returns
    -------
    Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.hist(pit_values, bins=bins, density=True, alpha=0.7, label="PIT values")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Uniform(0,1)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram (should be uniform for well-specified model)")
    ax.legend()
    return ax


def plot_forecast_fan(
    forecast,
    param: str | None = None,
    history: NDArray[np.float64] | None = None,
    ax=None,
) -> object:
    """Fan chart of a GAS model forecast.

    Parameters
    ----------
    forecast:
        ForecastResult object.
    param:
        Which parameter to plot.
    history:
        Optional historical values to prepend to the chart.
    ax:
        Matplotlib axes.

    Returns
    -------
    Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if param is None:
        param = next(iter(forecast.mean_path))

    h = forecast.h
    x_fc = np.arange(h)

    if history is not None:
        x_hist = np.arange(-len(history), 0)
        ax.plot(x_hist, history, color="black", linewidth=1.5, label="History")

    ax.plot(x_fc, forecast.mean_path[param], color="steelblue", linewidth=2, label="Forecast mean")

    # Shade quantile bands in pairs
    quantiles = sorted(forecast.quantiles.keys())
    mid = len(quantiles) // 2
    colours = ["#aec6e8", "#5e9fcb", "#2171b5"]

    for i, q_low in enumerate(quantiles[:mid]):
        q_high = quantiles[-(i + 1)]
        ax.fill_between(
            x_fc,
            forecast.quantiles[q_low][param],
            forecast.quantiles[q_high][param],
            alpha=0.3,
            color=colours[i % len(colours)],
            label=f"{int(q_low * 100)}-{int(q_high * 100)}% CI",
        )

    ax.axvline(-0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(param)
    ax.set_title(f"GAS Forecast: {param}")
    ax.legend()
    return ax


def plot_acf(
    acf_values: NDArray[np.float64],
    title: str = "Score Residuals ACF",
    ax=None,
) -> object:
    """Bar plot of autocorrelation function values.

    Parameters
    ----------
    acf_values:
        ACF at lags 0, 1, ..., nlags.
    title:
        Plot title.
    ax:
        Matplotlib axes.

    Returns
    -------
    Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    lags = np.arange(len(acf_values))
    n = len(acf_values)
    conf_band = 1.96 / np.sqrt(n)

    ax.bar(lags[1:], acf_values[1:], color="steelblue", alpha=0.8)
    ax.axhline(conf_band, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf_band, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    return ax
