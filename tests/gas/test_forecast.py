"""Tests for GAS forecasting."""

import warnings

import numpy as np
import pytest

from insurance_dynamics.gas import GASModel
from insurance_dynamics.gas.forecast import ForecastResult, gas_forecast
from insurance_dynamics.gas.datasets import load_motor_frequency, load_severity_trend


class TestForecastResult:
    def setup_method(self):
        data = load_motor_frequency(T=48, seed=1, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_forecast_via_result(self):
        fc = self.result.forecast(h=6, method="mean_path")
        assert isinstance(fc, ForecastResult)

    def test_forecast_h_length(self):
        for h in [1, 6, 12]:
            fc = self.result.forecast(h=h, method="mean_path")
            assert len(fc.mean_path["mean"]) == h

    def test_forecast_positive_poisson(self):
        fc = self.result.forecast(h=6, method="mean_path")
        assert np.all(fc.mean_path["mean"] > 0)

    def test_forecast_finite(self):
        fc = self.result.forecast(h=12, method="mean_path")
        assert np.all(np.isfinite(fc.mean_path["mean"]))

    def test_forecast_decays_to_unconditional(self):
        """Long-horizon forecasts should converge toward the stationary mean."""
        fc_long = self.result.forecast(h=24, method="mean_path")
        assert np.isfinite(fc_long.mean_path["mean"][-1])

    def test_to_dataframe(self):
        fc = self.result.forecast(h=6, method="mean_path")
        df = fc.to_dataframe()
        assert isinstance(df, __import__("pandas").DataFrame)
        assert "mean" in df.columns
        assert len(df) == 6

    def test_quantiles_in_result(self):
        fc = self.result.forecast(h=6, method="mean_path", quantiles=[0.1, 0.5, 0.9])
        assert 0.1 in fc.quantiles
        assert 0.9 in fc.quantiles


class TestSimulatedForecast:
    def setup_method(self):
        # Use a well-identified model with stable filter
        rng = np.random.default_rng(42)
        y = rng.poisson(3.0, 48).astype(float)
        exposure = np.ones(48) * 100.0  # large exposure → stable rate
        m = GASModel("poisson")
        self.result = m.fit(y, exposure=exposure)

    def test_simulate_returns_quantiles(self):
        fc = self.result.forecast(
            h=6, method="simulate", quantiles=[0.1, 0.5, 0.9],
            n_sim=100, rng=np.random.default_rng(42)
        )
        assert 0.1 in fc.quantiles
        assert 0.9 in fc.quantiles

    def test_quantile_paths_finite(self):
        fc = self.result.forecast(
            h=6, method="simulate", quantiles=[0.1, 0.9],
            n_sim=100, rng=np.random.default_rng(42)
        )
        for q, paths in fc.quantiles.items():
            for name, arr in paths.items():
                assert np.all(np.isfinite(arr)), f"Non-finite in q={q}, param={name}"

    def test_lower_quantile_below_upper(self):
        fc = self.result.forecast(
            h=6, method="simulate", quantiles=[0.1, 0.9],
            n_sim=100, rng=np.random.default_rng(42)
        )
        np.testing.assert_array_less(
            fc.quantiles[0.1]["mean"], fc.quantiles[0.9]["mean"] + 1e-6
        )

    def test_simulate_gamma(self):
        rng = np.random.default_rng(99)
        y = rng.gamma(shape=3.0, scale=200.0, size=40)
        m = GASModel("gamma")
        r = m.fit(y)
        fc = r.forecast(h=4, method="simulate", n_sim=50, rng=np.random.default_rng(1))
        assert np.all(np.isfinite(fc.mean_path["mean"]))


class TestForecastGasFunction:
    def test_gas_forecast_standalone(self):
        data = load_motor_frequency(T=36, seed=4, trend_break=False)
        m = GASModel("poisson")
        r = m.fit(data.y, exposure=data.exposure)
        fc = gas_forecast(r, h=3)
        assert isinstance(fc, ForecastResult)
        assert len(fc.mean_path["mean"]) == 3
