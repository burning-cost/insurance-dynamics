"""
PELT wrapper with bootstrap confidence intervals on break locations.

Wraps the ruptures library for retrospective changepoint detection,
adding two things:
1. Bootstrap CI on break locations (which ruptures does not provide)
2. A consistent interface returning BreakResult with BreakInterval objects

Why not use ruptures directly? Ruptures finds break locations but gives no
uncertainty. For Consumer Duty evidence, knowing that a break is "around
period 47 ± 3" is much more defensible than asserting exactly period 47.

The bootstrap approach: resample the series 1000x with replacement (block
bootstrap preserving local autocorrelation), refit PELT on each, and collect
the distribution of detected break locations. The 2.5th and 97.5th percentiles
of this distribution form the 95% CI.

Block bootstrap block size defaults to max(5, T//20) to respect short-run
autocorrelation typical in insurance loss ratios.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import ruptures as rpt
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False

from .result import BreakResult, BreakInterval


def _check_ruptures() -> None:
    if not _RUPTURES_AVAILABLE:
        raise ImportError(
            "ruptures is required for RetrospectiveBreakFinder. "
            "Install it with: pip install ruptures"
        )


def _bic_penalty(n: int, n_breaks: int, n_params_per_segment: int = 1) -> float:
    """
    BIC penalty for ruptures.

    ruptures expects penalty = (sigma^2 or 1) * log(n) * n_params.
    For l2 cost, AIC penalty = 2, BIC penalty = log(n).
    We use log(n) * n_bkps_attempted is not right; ruptures adds
    pen * n_bkps to the cost, so pen = log(n) is the standard BIC.
    """
    return float(np.log(n))


def _run_pelt(
    signal: np.ndarray,
    model: str,
    penalty: float,
) -> list[int]:
    """
    Run PELT and return break indices (excluding the last point T).
    """
    algo = rpt.Pelt(model=model, min_size=2, jump=1)
    # ruptures expects shape (T,) or (T, d)
    if signal.ndim == 1:
        signal_2d = signal.reshape(-1, 1)
    else:
        signal_2d = signal
    algo.fit(signal_2d)
    result = algo.predict(pen=penalty)
    # ruptures returns [..., T] — drop the last element
    return result[:-1]


def _block_bootstrap(
    signal: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Block bootstrap resample of a 1-D or 2-D signal.
    """
    T = len(signal)
    n_blocks = int(np.ceil(T / block_size))
    # Sample block starting positions with replacement
    starts = rng.integers(0, max(1, T - block_size + 1), size=n_blocks)
    blocks = [signal[s : s + block_size] for s in starts]
    resampled = np.concatenate(blocks)[:T]
    return resampled


def find_breaks_pelt(
    signal: np.ndarray,
    model: str = "l2",
    penalty: float | str = "bic",
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    block_size: int | None = None,
    seed: int | None = 42,
) -> BreakResult:
    """
    Retrospective break detection with bootstrap confidence intervals.

    Parameters
    ----------
    signal :
        1-D array of observations (loss ratios, log-severities, etc.).
    model :
        ruptures cost model. 'l2' for Gaussian mean changes, 'rbf' for
        distribution changes, 'normal' for Gaussian mean+variance changes.
    penalty :
        Penalty value for PELT. 'bic' (default) uses log(T). Can be a
        float for manual tuning.
    n_bootstraps :
        Number of bootstrap resamples for CI estimation. 200 is adequate
        for exploration, 1000 for reporting.
    confidence :
        CI level (default 0.95 for 95% CI).
    block_size :
        Bootstrap block size. Defaults to max(5, T//20).
    seed :
        Random seed for reproducibility.

    Returns
    -------
    BreakResult
    """
    _check_ruptures()

    signal = np.asarray(signal, dtype=float)
    T = len(signal)

    if T < 4:
        return BreakResult(
            breaks=[],
            break_cis=[],
            n_bootstraps=0,
            penalty=0.0,
            model=model,
        )

    # Resolve penalty
    if penalty == "bic":
        pen_value = _bic_penalty(T, n_breaks=0)
    else:
        pen_value = float(penalty)

    # Point estimates from PELT
    point_breaks = _run_pelt(signal, model, pen_value)

    if not point_breaks:
        return BreakResult(
            breaks=[],
            break_cis=[],
            n_bootstraps=n_bootstraps,
            penalty=pen_value,
            model=model,
        )

    # Bootstrap CI
    bs = int(block_size) if block_size is not None else max(5, T // 20)
    rng = np.random.default_rng(seed)

    # For each point break, collect bootstrap break positions closest to it
    # We use a matching strategy: for each bootstrap replicate, find the
    # break closest to each original break estimate.
    n_orig = len(point_breaks)
    bootstrap_positions: list[list[int]] = [[] for _ in range(n_orig)]

    for _ in range(n_bootstraps):
        resampled = _block_bootstrap(signal, bs, rng)
        bs_breaks = _run_pelt(resampled, model, pen_value)
        if not bs_breaks:
            continue
        # Match each original break to the closest bootstrap break
        for j, orig_break in enumerate(point_breaks):
            dists = [abs(b - orig_break) for b in bs_breaks]
            closest = bs_breaks[np.argmin(dists)]
            bootstrap_positions[j].append(closest)

    # Compute CIs
    alpha = 1.0 - confidence
    break_cis = []
    for j, orig_break in enumerate(point_breaks):
        positions = bootstrap_positions[j]
        if len(positions) < 10:
            # Too few bootstrap detections — wide CI
            lower = max(0, orig_break - bs)
            upper = min(T - 1, orig_break + bs)
        else:
            arr = np.array(positions)
            lower = int(np.percentile(arr, 100 * alpha / 2))
            upper = int(np.percentile(arr, 100 * (1 - alpha / 2)))
        break_cis.append(
            BreakInterval(
                break_index=orig_break,
                lower=lower,
                upper=upper,
            )
        )

    return BreakResult(
        breaks=point_breaks,
        break_cis=break_cis,
        n_bootstraps=n_bootstraps,
        penalty=pen_value,
        model=model,
    )
