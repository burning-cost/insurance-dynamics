"""
Retrospective break finder: PELT with bootstrap confidence intervals.

A thin wrapper around _pelt.find_breaks_pelt() that presents the same
class-based interface as the online detectors.

Usage
-----
>>> from insurance_changepoint import RetrospectiveBreakFinder
>>> finder = RetrospectiveBreakFinder(model='l2', penalty='bic')
>>> breaks = finder.fit(loss_ratio_series)
>>> print(breaks.breaks)        # [24, 67, 103]
>>> print(breaks.break_cis)     # [BreakInterval(...), ...]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._pelt import find_breaks_pelt
from .result import BreakResult, BreakInterval


class RetrospectiveBreakFinder:
    """
    Retrospective changepoint detection using PELT with bootstrap CIs.

    Uses ruptures as the PELT backend. Adds bootstrap confidence intervals
    on break locations by resampling the series and refitting PELT.

    Parameters
    ----------
    model :
        ruptures cost model: 'l2' (Gaussian mean), 'rbf' (kernel-based),
        'normal' (Gaussian mean+variance), 'ar' (autoregressive).
        'l2' is usually the right choice for smoothed loss ratios.
    penalty :
        Penalty for PELT. 'bic' (default) uses log(T). Higher penalty =
        fewer breaks. Can be a float for manual tuning.
    n_bootstraps :
        Number of bootstrap resamples for CI estimation. 1000 is the
        minimum for reporting; use 200 for quick exploration.
    confidence :
        CI coverage level. Default 0.95.
    block_size :
        Block bootstrap block size. Defaults to max(5, T//20).
    seed :
        Random seed.
    """

    def __init__(
        self,
        model: str = "l2",
        penalty: float | str = "bic",
        n_bootstraps: int = 1000,
        confidence: float = 0.95,
        block_size: int | None = None,
        seed: int | None = 42,
    ) -> None:
        self.model = model
        self.penalty = penalty
        self.n_bootstraps = n_bootstraps
        self.confidence = confidence
        self.block_size = block_size
        self.seed = seed

    def fit(
        self,
        series: list[float] | np.ndarray,
        periods: list[Any] | None = None,
    ) -> BreakResult:
        """
        Find retrospective breaks in a series.

        Parameters
        ----------
        series :
            1-D array of observations.
        periods :
            Optional period labels. If provided, attached to BreakInterval
            objects as period_label.

        Returns
        -------
        BreakResult
        """
        arr = np.asarray(series, dtype=float)
        result = find_breaks_pelt(
            signal=arr,
            model=self.model,
            penalty=self.penalty,
            n_bootstraps=self.n_bootstraps,
            confidence=self.confidence,
            block_size=self.block_size,
            seed=self.seed,
        )

        # Attach period labels if provided
        if periods is not None and result.break_cis:
            updated_cis = []
            for ci in result.break_cis:
                label = periods[ci.break_index] if ci.break_index < len(periods) else None
                updated_cis.append(
                    BreakInterval(
                        break_index=ci.break_index,
                        lower=ci.lower,
                        upper=ci.upper,
                        period_label=label,
                    )
                )
            result = BreakResult(
                breaks=result.breaks,
                break_cis=updated_cis,
                n_bootstraps=result.n_bootstraps,
                penalty=result.penalty,
                model=result.model,
                periods=list(periods),
            )

        return result
