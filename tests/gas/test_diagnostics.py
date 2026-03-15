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


# ---------------------------------------------------------------------------
# P1-2 regression: _y stored on GASResult
# ---------------------------------------------------------------------------

class TestP12YStoredOnResult:
    """P1-2: GASResult._y must be set by fit() so PIT branches execute correctly."""

    def test_p1_2_y_attribute_set_after_fit(self):
        data = load_motor_frequency(T=24, seed=5, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        assert result._y is not None
        assert len(result._y) == result.n_obs

    def test_p1_2_y_values_match_training_data(self):
        data = load_motor_frequency(T=24, seed=6, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        np.testing.assert_array_equal(result._y, np.asarray(data.y, dtype=float))

    def test_p1_2_pit_uses_correct_branch_for_poisson(self):
        """When _y is set, diagnostics use randomised PIT, not Gaussian CDF.

        The Gaussian CDF fallback (used before P1-2 fix) maps score residuals
        directly through norm.cdf.  The correct Poisson PIT applies the
        randomised-PIT procedure via the Poisson CDF — the outputs differ.
        We verify that the PIT values are NOT identical to norm.cdf(score_residuals),
        confirming the correct branch executes.
        """
        from scipy import stats as scipy_stats
        data = load_motor_frequency(T=48, seed=7, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        diag = result.diagnostics()
        assert np.all(diag.pit_values >= 0.0)
        assert np.all(diag.pit_values <= 1.0)
        # The Gaussian CDF fallback would give norm.cdf(score_residuals).
        # The correct Poisson PIT gives different values — the arrays must differ.
        gaussian_fallback = scipy_stats.norm.cdf(result.score_residuals.iloc[:, 0].values)
        assert not np.allclose(diag.pit_values, gaussian_fallback, atol=1e-6), (
            "PIT values match Gaussian fallback — _y branch not executing (P1-2 unfixed)"
        )

    def test_p1_2_exposure_stored_on_result(self):
        data = load_motor_frequency(T=24, seed=8, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        assert result._exposure is not None
        assert len(result._exposure) == result.n_obs


# ---------------------------------------------------------------------------
# P0-3 regression: LogNormal PIT uses exp(logsigma)
# ---------------------------------------------------------------------------

class TestP03LognormalPIT:
    """P0-3: _pit_continuous must exponentiate logsigma before passing to norm.cdf."""

    def test_p0_3_lognormal_pit_values_in_unit_interval(self):
        """Before fix, norm.cdf received negative sigma for logsigma < 0, returning NaN."""
        from insurance_dynamics.gas.diagnostics import _pit_continuous
        from insurance_dynamics.gas.distributions import LogNormalGAS

        dist = LogNormalGAS()
        # logsigma = log(0.5) ≈ -0.693 — this was the failure case
        params = {"logmean": 6.0, "logsigma": np.log(0.5)}
        y = float(np.exp(6.0))
        pit = _pit_continuous(dist, y, params)
        assert np.isfinite(pit), "PIT returned NaN — logsigma not exponentiated (P0-3 unfixed)"
        assert 0.0 <= pit <= 1.0

    def test_p0_3_lognormal_pit_at_median_is_near_half(self):
        """y = exp(logmean) is the log-normal median; PIT at median ≈ 0.5."""
        from insurance_dynamics.gas.diagnostics import _pit_continuous
        from insurance_dynamics.gas.distributions import LogNormalGAS

        dist = LogNormalGAS()
        logmean = 6.0
        logsigma = np.log(0.4)  # sigma_log = 0.4
        params = {"logmean": logmean, "logsigma": logsigma}
        y = float(np.exp(logmean))  # this is the median, not the mean
        pit = _pit_continuous(dist, y, params)
        assert pit == pytest.approx(0.5, abs=0.01)

    def test_p0_3_lognormal_diagnostics_no_nan(self):
        """Full diagnostics pipeline must not produce NaN ds_score for lognormal."""
        data = load_severity_trend(T=48, seed=2)
        m = GASModel("lognormal")
        result = m.fit(data.y)
        diag = result.diagnostics()
        assert np.isfinite(diag.ds_score)
        assert not np.any(np.isnan(diag.pit_values))


# ---------------------------------------------------------------------------
# P1-3 regression: DS score uses correct Dawid-Sebastiani formula
# ---------------------------------------------------------------------------

class TestP13DSScore:
    """P1-3: ds_score must use (y-mu)^2/sigma^2 + 2*log(sigma), not sr^2 + 2*log|sr|."""

    def test_p1_3_ds_score_equals_dawid_sebastiani_function(self):
        """The ds_score attribute must be numerically consistent with the
        standalone dawid_sebastiani_score() function evaluated on the same data."""
        data = load_motor_frequency(T=48, seed=10, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        diag = result.diagnostics()

        # Recompute DS manually using the stored predictive quantities
        from insurance_dynamics.gas.diagnostics import _predictive_mean_sigma
        T = result.n_obs
        y_arr = result._y
        mu_arr = np.zeros(T)
        sigma_arr = np.zeros(T)
        for t in range(T):
            params_t = {name: float(result.filter_path[name].iloc[t])
                        for name in result.model.time_varying}
            for sname in result.model._build_static_param_names():
                params_t[sname] = result.params[sname]
            mu_arr[t], sigma_arr[t] = _predictive_mean_sigma(result.distribution, params_t)
        sigma_arr = np.where(sigma_arr > 0, sigma_arr, 1e-8)
        expected_ds = dawid_sebastiani_score(y_arr.astype(float), mu_arr, sigma_arr)
        assert diag.ds_score == pytest.approx(expected_ds, rel=1e-6)

    def test_p1_3_good_model_ds_better_than_bad(self):
        """A well-fitted model should have a lower (better) DS score than a constant
        model with the wrong mean."""
        rng = np.random.default_rng(42)
        y = rng.poisson(3.0, 60).astype(float)
        mu_good = np.full(60, 3.0)
        mu_bad = np.full(60, 10.0)
        sigma = np.sqrt(mu_good)
        ds_good = dawid_sebastiani_score(y, mu_good, sigma)
        ds_bad = dawid_sebastiani_score(y, mu_bad, np.sqrt(mu_bad))
        assert ds_good < ds_bad


# ---------------------------------------------------------------------------
# ACF tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dawid-Sebastiani standalone function
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PIT residuals standalone
# ---------------------------------------------------------------------------

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
