"""Tests for GAS diagnostics."""

import numpy as np
import pytest

from insurance_dynamics.gas import GASModel
from insurance_dynamics.gas.diagnostics import (
    DiagnosticsResult,
    compute_diagnostics,
    dawid_sebastiani_score,
    pit_residuals,
    _compute_acf,
)
from insurance_dynamics.gas.datasets import load_motor_frequency, load_severity_trend


class TestDiagnosticsBasic:
    def setup_method(self):
        data = load_motor_frequency(T=48, seed=1, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_diagnostics_via_result(self):
        diag = self.result.diagnostics()
        assert isinstance(diag, DiagnosticsResult)

    def test_pit_values_length(self):
        diag = self.result.diagnostics()
        assert len(diag.pit_values) == self.result.n_obs

    def test_pit_values_in_unit_interval(self):
        diag = self.result.diagnostics()
        assert np.all(diag.pit_values >= 0.0)
        assert np.all(diag.pit_values <= 1.0)

    def test_ks_statistic_in_range(self):
        diag = self.result.diagnostics()
        assert 0.0 <= diag.ks_statistic <= 1.0

    def test_ks_pvalue_in_range(self):
        diag = self.result.diagnostics()
        assert 0.0 <= diag.ks_pvalue <= 1.0

    def test_ds_score_finite(self):
        diag = self.result.diagnostics()
        assert np.isfinite(diag.ds_score)

    def test_ljung_box_pvalue_in_range(self):
        diag = self.result.diagnostics()
        assert 0.0 <= diag.ljung_box_pvalue <= 1.0

    def test_summary_string(self):
        diag = self.result.diagnostics()
        s = diag.summary()
        assert "PIT" in s
        assert "Dawid-Sebastiani" in s


class TestACF:
    def test_acf_lag0_is_one(self):
        x = np.random.default_rng(1).standard_normal(100)
        acf = _compute_acf(x, nlags=10)
        assert acf[0] == pytest.approx(1.0)

    def test_acf_length(self):
        x = np.ones(50)
        acf = _compute_acf(x, nlags=15)
        assert len(acf) == 16

    def test_acf_iid_near_zero(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(500)
        acf = _compute_acf(x, nlags=20)
        # 95% of lags beyond 0 should be inside ±0.1 for iid noise
        assert np.mean(np.abs(acf[1:]) < 0.15) > 0.75

    def test_acf_ar1_has_decay(self):
        """AR(1) process should have geometrically decaying ACF."""
        rng = np.random.default_rng(7)
        n = 500
        phi = 0.8
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        acf = _compute_acf(x, nlags=5)
        assert acf[1] > 0.5  # should be near phi


class TestDawidSebastiani:
    def test_perfect_forecast(self):
        """When mu = y and sigma is calibrated, DS should be small."""
        n = 100
        y = np.ones(n) * 3.0
        mu = np.ones(n) * 3.0
        sigma = np.ones(n) * 1.0
        ds = dawid_sebastiani_score(y, mu, sigma)
        assert np.isfinite(ds)
        assert ds < 10.0  # roughly log(1) + 0 = 0 → ~0.0

    def test_bad_forecast_worse(self):
        """Biased forecast should have worse (higher) DS than unbiased."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(200) + 5.0
        mu_good = np.full(200, 5.0)
        mu_bad = np.full(200, 1.0)
        sigma = np.ones(200)
        ds_good = dawid_sebastiani_score(y, mu_good, sigma)
        ds_bad = dawid_sebastiani_score(y, mu_bad, sigma)
        assert ds_good < ds_bad

    def test_returns_float(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        result = dawid_sebastiani_score(y, mu, sigma)
        assert isinstance(result, float)


class TestPITResiduals:
    def test_pit_standalone_poisson(self):
        rng = np.random.default_rng(42)
        y = rng.poisson(3.0, 30).astype(float)
        from insurance_dynamics.gas.distributions import PoissonGAS
        dist = PoissonGAS()
        import pandas as pd
        fp = pd.DataFrame({"mean": np.full(30, 3.0)})
        pits = pit_residuals(y, fp, dist, {}, rng=rng)
        assert len(pits) == 30
        assert np.all(pits >= 0.0)
        assert np.all(pits <= 1.0)

    def test_pit_standalone_gamma(self):
        rng = np.random.default_rng(99)
        y = rng.gamma(shape=3.0, scale=200.0, size=30)
        from insurance_dynamics.gas.distributions import GammaGAS
        dist = GammaGAS()
        import pandas as pd
        fp = pd.DataFrame({"mean": np.full(30, 600.0)})
        pits = pit_residuals(y, fp, dist, {"shape": 3.0}, rng=rng)
        assert len(pits) == 30
        assert np.all(pits >= 0.0)
        assert np.all(pits <= 1.0)
