"""Synthetic datasets for testing and demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataset:
    """A synthetic dataset with known ground-truth parameters.

    Attributes
    ----------
    y:
        Observed values.
    exposure:
        Exposure values (where applicable).
    filter_truth:
        True time-varying parameter values (for evaluation).
    description:
        Human-readable description of the data-generating process.
    params:
        True GAS parameters used to generate the data.
    """

    y: np.ndarray
    exposure: np.ndarray | None
    filter_truth: dict[str, np.ndarray]
    description: str
    params: dict[str, float]


def load_motor_frequency(
    T: int = 60,
    seed: int = 42,
    trend_break: bool = True,
) -> SyntheticDataset:
    """Monthly motor claim frequency with seasonal pattern and a trend break.

    The data-generating process is a Poisson GAS(1,1) with:
    - Seasonal component (winter claims 30% higher than summer)
    - Linear upward trend for first 36 months
    - A step change in frequency at month 37 (e.g., model change or
      portfolio shift)

    Parameters
    ----------
    T:
        Number of periods. Default 60 (5 years monthly).
    seed:
        Random seed.
    trend_break:
        If True, includes a step-change at period T//2.

    Returns
    -------
    SyntheticDataset with Poisson-distributed monthly claim counts.
    """
    rng = np.random.default_rng(seed)

    # True GAS parameters
    omega = 0.02
    alpha = 0.15
    phi = 0.85

    # Seasonal pattern (winter effect)
    months = np.arange(T) % 12
    seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * months / 12)

    # Base exposure (earned car years)
    exposure = rng.uniform(800, 1200, T)

    # True time-varying log-rate
    f_true = np.zeros(T)
    f_true[0] = omega / (1.0 - phi)  # unconditional mean
    f = f_true[0]

    y = np.zeros(T, dtype=int)
    mu_true = np.zeros(T)

    for t in range(T):
        mu = np.exp(f) * seasonal[t]
        if trend_break and t == T // 2:
            mu *= 1.4  # 40% step increase
        rate = mu * exposure[t]
        y[t] = rng.poisson(rate)
        mu_true[t] = mu

        # GAS update
        score = y[t] / rate - 1.0
        f_next = omega + alpha * score + phi * f
        f_true[t] = f
        f = f_next

    return SyntheticDataset(
        y=y.astype(float),
        exposure=exposure,
        filter_truth={"mean": mu_true},
        description=(
            f"Synthetic motor frequency: T={T} monthly periods, "
            f"Poisson GAS(1,1), seasonal pattern (winter peak), "
            f"{'step change at period ' + str(T//2) if trend_break else 'no trend break'}."
        ),
        params={"omega_mean": omega, "alpha_mean_1": alpha, "phi_mean_1": phi},
    )


def load_severity_trend(
    T: int = 40,
    seed: int = 42,
    inflation_rate: float = 0.05,
) -> SyntheticDataset:
    """Quarterly severity with persistent inflation trend.

    The data-generating process is a Gamma GAS(1,1) with log-linked mean.
    The mean severity rises by approximately ``inflation_rate`` per period
    (compound), interrupted by moderate noise.

    Parameters
    ----------
    T:
        Number of periods. Default 40 (10 years quarterly).
    seed:
        Random seed.
    inflation_rate:
        Quarterly inflation rate as a fraction (0.05 = 5% per quarter).

    Returns
    -------
    SyntheticDataset with Gamma-distributed severity observations.
    """
    rng = np.random.default_rng(seed)

    # True GAS parameters
    omega = np.log(1.0 + inflation_rate) * (1.0 - 0.88)
    alpha = 0.08
    phi = 0.88
    shape = 3.0  # Gamma shape

    f = omega / (1.0 - phi)
    f_true = np.zeros(T)
    y = np.zeros(T)
    mu_true = np.zeros(T)

    for t in range(T):
        mu = np.exp(f)
        y[t] = rng.gamma(shape=shape, scale=mu / shape)
        mu_true[t] = mu

        score = shape * (y[t] / mu - 1.0)
        f_next = omega + alpha * (score / shape) + phi * f  # fisher_inv scaling
        f_true[t] = f
        f = f_next

    return SyntheticDataset(
        y=y,
        exposure=None,
        filter_truth={"mean": mu_true},
        description=(
            f"Synthetic severity trend: T={T} quarterly periods, "
            f"Gamma GAS(1,1), quarterly inflation of {inflation_rate*100:.1f}%, "
            f"shape={shape}."
        ),
        params={
            "omega_mean": float(omega),
            "alpha_mean_1": alpha,
            "phi_mean_1": phi,
            "shape": shape,
        },
    )


def load_loss_ratio(
    T: int = 48,
    seed: int = 42,
) -> SyntheticDataset:
    """Monthly loss ratio on [0,1] with time-varying mean.

    Simulated from a Beta GAS(1,1) with a gradual deterioration followed
    by stabilisation. Useful for testing BetaGAS.

    Parameters
    ----------
    T:
        Number of periods.
    seed:
        Random seed.

    Returns
    -------
    SyntheticDataset with Beta-distributed loss ratios.
    """
    rng = np.random.default_rng(seed)

    omega = 0.01
    alpha = 0.1
    phi = 0.88
    precision = 15.0

    # Start at loss ratio 0.65
    f0 = np.log(0.65 / 0.35)
    f = f0
    f_true = np.zeros(T)
    y = np.zeros(T)
    mu_true = np.zeros(T)

    for t in range(T):
        mu = 1.0 / (1.0 + np.exp(-f))
        # Clamp
        mu = np.clip(mu, 1e-4, 1.0 - 1e-4)
        y[t] = rng.beta(mu * precision, (1.0 - mu) * precision)
        mu_true[t] = mu

        score = precision * (y[t] - mu)
        fi = precision * mu * (1.0 - mu)
        f_next = omega + alpha * (score / fi) + phi * f
        f_true[t] = f
        f = f_next

    return SyntheticDataset(
        y=np.clip(y, 1e-6, 1.0 - 1e-6),
        exposure=None,
        filter_truth={"mean": mu_true},
        description=(
            f"Synthetic loss ratio: T={T} monthly periods, Beta GAS(1,1), "
            f"logit-linked mean, precision={precision}."
        ),
        params={
            "omega_mean": omega,
            "alpha_mean_1": alpha,
            "phi_mean_1": phi,
            "precision": precision,
        },
    )
