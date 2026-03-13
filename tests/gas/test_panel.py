"""Tests for GASPanel."""

import numpy as np
import pandas as pd
import pytest

from insurance_dynamics.gas import GASPanel
from insurance_dynamics.gas.panel import GASPanelResult
from insurance_dynamics.gas.datasets import load_motor_frequency


def _make_panel_data(n_cells: int = 4, T: int = 36, seed: int = 1) -> pd.DataFrame:
    """Synthetic panel with multiple rating cells."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cells):
        base_rate = rng.uniform(1.0, 5.0)
        exposure = rng.uniform(500, 1500, T)
        claims = rng.poisson(base_rate * exposure / 1000.0).astype(float)
        for t in range(T):
            rows.append({
                "period": t,
                "cell_id": f"cell_{i}",
                "claims": claims[t],
                "exposure": exposure[t] / 1000.0,
            })
    return pd.DataFrame(rows)


class TestGASPanelConstruction:
    def test_default_construction(self):
        p = GASPanel("poisson")
        assert p.distribution == "poisson"

    def test_custom_parameters(self):
        p = GASPanel("gamma", p=1, q=1, scaling="unit")
        assert p.scaling == "unit"


class TestGASPanelFit:
    def setup_method(self):
        self.data = _make_panel_data(n_cells=3, T=36)

    def test_fit_returns_result(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        assert isinstance(r, GASPanelResult)

    def test_all_cells_fitted(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        n_cells = self.data["cell_id"].nunique()
        assert len(r.filter_paths) + len(r.failed_cells) == n_cells

    def test_filter_paths_positive(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        for cell_id, fp in r.filter_paths.items():
            assert np.all(fp["mean"].values > 0), f"Negative filter for {cell_id}"

    def test_trend_indices_start_at_100(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        for cell_id, ti in r.trend_indices.items():
            assert ti["mean"].iloc[0] == pytest.approx(100.0), f"Cell {cell_id}"

    def test_summary_frame(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        sf = r.summary_frame()
        assert isinstance(sf, pd.DataFrame)
        assert len(sf) == 36

    def test_trend_summary(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        ts = r.trend_summary()
        assert isinstance(ts, pd.DataFrame)

    def test_short_cell_skipped(self):
        """Cells with fewer than 4 observations should be in failed_cells."""
        short_row = pd.DataFrame({
            "period": [0, 1, 2],
            "cell_id": ["short_cell", "short_cell", "short_cell"],
            "claims": [1.0, 2.0, 1.0],
            "exposure": [1.0, 1.0, 1.0],
        })
        data = pd.concat([self.data, short_row], ignore_index=True)
        panel = GASPanel("poisson")
        r = panel.fit(
            data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col="exposure",
        )
        assert "short_cell" in r.failed_cells

    def test_no_exposure_column(self):
        panel = GASPanel("poisson")
        r = panel.fit(
            self.data,
            y_col="claims",
            period_col="period",
            cell_col="cell_id",
            exposure_col=None,
        )
        assert len(r.filter_paths) > 0
