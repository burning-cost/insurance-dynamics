"""Panel GAS: fit the same model to multiple rating cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import GASModel, GASResult


@dataclass
class GASPanelResult:
    """Results from a panel GAS fit.

    Attributes
    ----------
    filter_paths:
        Dict mapping cell_id -> filter_path DataFrame.
    trend_indices:
        Dict mapping cell_id -> trend_index DataFrame.
    results:
        Dict mapping cell_id -> GASResult.
    failed_cells:
        List of cell IDs where fitting failed.
    """

    filter_paths: dict[str | int, pd.DataFrame]
    trend_indices: dict[str | int, pd.DataFrame]
    results: dict[str | int, GASResult]
    failed_cells: list[str | int]

    def summary_frame(self, param: str | None = None) -> pd.DataFrame:
        """Return a wide DataFrame of filter paths: rows=period, cols=cell_id.

        Parameters
        ----------
        param:
            Which time-varying parameter to extract. Uses the first if None.
        """
        frames = {}
        for cell_id, fp in self.filter_paths.items():
            if param is None:
                param = fp.columns[0]
            frames[cell_id] = fp[param].values

        lengths = {k: len(v) for k, v in frames.items()}
        max_len = max(lengths.values())
        for k in frames:
            v = frames[k]
            if len(v) < max_len:
                frames[k] = np.concatenate([v, np.full(max_len - len(v), np.nan)])

        return pd.DataFrame(frames)

    def trend_summary(self, param: str | None = None) -> pd.DataFrame:
        """Wide DataFrame of trend indices per cell."""
        frames = {}
        for cell_id, ti in self.trend_indices.items():
            if param is None:
                param = ti.columns[0]
            frames[cell_id] = ti[param].values

        max_len = max(len(v) for v in frames.values())
        for k in frames:
            v = frames[k]
            if len(v) < max_len:
                frames[k] = np.concatenate([v, np.full(max_len - len(v), np.nan)])

        return pd.DataFrame(frames)


class GASPanel:
    """Fit the same GAS model independently to multiple time series.

    Typical use: one series per vehicle class, territory, or rating cell.
    Fits the model separately to each cell and returns aligned filter paths
    for comparison.

    Parameters
    ----------
    distribution:
        Distribution name or GASDistribution instance.
    p, q:
        GAS filter orders.
    scaling:
        Score scaling method.
    time_varying:
        Time-varying parameter names.

    Examples
    --------
    >>> panel = GASPanel('poisson')
    >>> result = panel.fit(
    ...     data,
    ...     period_col='period',
    ...     cell_col='vehicle_class',
    ...     y_col='claims',
    ...     exposure_col='exposure',
    ... )
    >>> result.trend_summary()
    """

    def __init__(
        self,
        distribution: str = "poisson",
        p: int = 1,
        q: int = 1,
        scaling: str = "fisher_inv",
        time_varying: list[str] | None = None,
    ) -> None:
        self.distribution = distribution
        self.p = p
        self.q = q
        self.scaling = scaling
        self.time_varying = time_varying

    def fit(
        self,
        data: pd.DataFrame,
        y_col: str = "claims",
        period_col: str = "period",
        cell_col: str = "cell_id",
        exposure_col: str | None = None,
        max_iter: int = 500,
        verbose: bool = False,
    ) -> GASPanelResult:
        """Fit GAS model to each cell in the panel data.

        Parameters
        ----------
        data:
            DataFrame with at least columns for period, cell ID, and
            the response variable.
        y_col:
            Column name for the response (claims, severity, loss ratio).
        period_col:
            Column name for the time period.
        cell_col:
            Column name for the rating cell identifier.
        exposure_col:
            Optional column name for exposure values.
        max_iter:
            Maximum optimiser iterations per cell.
        verbose:
            Print progress.

        Returns
        -------
        GASPanelResult
        """
        cells = data[cell_col].unique()
        filter_paths: dict = {}
        trend_indices: dict = {}
        results: dict = {}
        failed: list = []

        for cell_id in cells:
            cell_data = data[data[cell_col] == cell_id].sort_values(period_col)
            y = cell_data[y_col].values.astype(float)
            exposure = None
            if exposure_col is not None:
                exposure = cell_data[exposure_col].values.astype(float)

            if len(y) < 4:
                if verbose:
                    print(f"Cell {cell_id}: too few observations ({len(y)}), skipping.")
                failed.append(cell_id)
                continue

            model = GASModel(
                distribution=self.distribution,
                p=self.p,
                q=self.q,
                scaling=self.scaling,
                time_varying=self.time_varying,
            )

            try:
                res = model.fit(y, exposure=exposure, max_iter=max_iter)
                filter_paths[cell_id] = res.filter_path
                trend_indices[cell_id] = res.trend_index
                results[cell_id] = res
                if verbose:
                    print(f"Cell {cell_id}: LL={res.log_likelihood:.2f}")
            except Exception as e:
                if verbose:
                    print(f"Cell {cell_id}: fit failed — {e}")
                failed.append(cell_id)

        return GASPanelResult(
            filter_paths=filter_paths,
            trend_indices=trend_indices,
            results=results,
            failed_cells=failed,
        )
