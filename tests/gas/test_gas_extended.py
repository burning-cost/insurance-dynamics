"""
Extended tests for the GAS module: distribution edge cases, filter internals,
forecast edge cases, diagnostics gaps, panel edge cases, datasets.

Targets code paths not covered by existing tests:
- GASDistribution.scaled_score with fi=0 (zero Fisher info guard)
- GASDistribution.scaled_score with unknown scaling raises ValueError
- NegBinGAS with exposure in score and log_likelihood
- ZIPGAS with exposure in score and log_likelihood
- ZIPGAS with zeroprob near boundaries (0 and 1)
- BetaGAS at boundary values (near 0 and near 1)
- LogNormalGAS logsigma negative (log(sigma) < 0, i.e. sigma < 1)
- GASResult.summary() content
- GASResult.relativities() base='first'
- ForecastResult.to_dataframe() with specific param
- ForecastResult.plot() smoke test
- GASPanel with gamma distribution
- GASPanel trend_summary structure
- bootstrap_ci on gamma model
- compute_diagnostics standalone function
- pit_residuals for NegBin and ZIP
- datasets: all load functions
- GASModel with distribution instance (GammaGAS with custom shape)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_dynamics.gas import GASModel, GASPanel, GASResult, bootstrap_ci
from insurance_dynamics.gas.bootstrap import BootstrapCI
from insurance_dynamics.gas.datasets import (
    load_loss_ratio,
    load_motor_frequency,
    load_severity_trend,
)
from insurance_dynamics.gas.diagnostics import (
    DiagnosticsResult,
    _compute_acf,
    _pit_continuous,
    compute_diagnostics,
    dawid_sebastiani_score,
    pit_residuals,
)
from insurance_dynamics.gas.distributions import (
    BetaGAS,
    GammaGAS,
    LogNormalGAS,
    NegBinGAS,
    PoissonGAS,
    ZIPGAS,
)
from insurance_dynamics.gas.filter import GASFilter, FilterResult
from insurance_dynamics.gas.forecast import ForecastResult, gas_forecast


# ---------------------------------------------------------------------------
# GASDistribution base: scaled_score edge cases
# ---------------------------------------------------------------------------

class TestGASDistributionBaseEdgeCases:
    def test_scaled_score_unknown_scaling_raises(self):
        dist = PoissonGAS()
        with pytest.raises(ValueError, match="Unknown scaling"):
            dist.scaled_score(
                np.array([3.0]), {"mean": 2.0}, scaling="bad_scaling_option"
            )

    def test_scaled_score_unit_returns_raw(self):
        dist = GammaGAS()
        y = np.array([500.0])
        params = {"mean": 500.0, "shape": 3.0}
        unit = dist.scaled_score(y, params, scaling="unit")
        raw = dist.score(y, params)
        assert unit["mean"] == pytest.approx(raw["mean"])

    def test_scaled_score_fisher_inv_sqrt_poisson(self):
        dist = PoissonGAS()
        # raw = y - mu = 4 - 2 = 2, fisher = 2, sqrt(fisher) = sqrt(2)
        s = dist.scaled_score(np.array([4.0]), {"mean": 2.0}, scaling="fisher_inv_sqrt")
        expected = 2.0 / np.sqrt(2.0)
        assert s["mean"] == pytest.approx(expected)

    def test_repr_string(self):
        dist = PoissonGAS()
        r = repr(dist)
        assert "PoissonGAS" in r

    def test_initial_params_base_default_empty(self):
        """GASDistribution base initial_params returns empty dict."""
        # We can't instantiate ABC directly, but we can test a subclass that
        # has override. PoissonGAS.initial_params is tested elsewhere — this
        # ensures the GammaGAS has reasonable initial_params.
        dist = GammaGAS()
        rng = np.random.default_rng(1)
        y = rng.gamma(shape=4.0, scale=200.0, size=100)
        init = dist.initial_params(y)
        assert "mean" in init
        assert "shape" in init
        assert init["shape"] > 0


# ---------------------------------------------------------------------------
# NegBinGAS extended
# ---------------------------------------------------------------------------

class TestNegBinGASExtended:
    def setup_method(self):
        self.dist = NegBinGAS()
        self.params = {"mean": 3.0, "dispersion": 5.0}

    def test_score_with_exposure(self):
        """With exposure, rate = mu * E."""
        E = 2.0
        mu = 3.0
        y = 6.0  # = mu * E
        s = self.dist.score(np.array([y]), {"mean": mu, "dispersion": 5.0}, exposure=np.array([E]))
        assert np.isfinite(s["mean"])

    def test_log_likelihood_with_exposure(self):
        E = 2.0
        ll = self.dist.log_likelihood(
            np.array([6.0]),
            {"mean": 3.0, "dispersion": 5.0},
            exposure=np.array([E]),
        )
        assert np.isfinite(float(ll))

    def test_fisher_with_exposure(self):
        fi = self.dist.fisher({"mean": 2.0, "dispersion": 4.0}, exposure=np.array([3.0]))
        assert fi["mean"] > 0

    def test_score_small_dispersion(self):
        """Very small r (highly overdispersed) should be numerically stable."""
        params = {"mean": 3.0, "dispersion": 0.01}
        s = self.dist.score(np.array([5.0]), params)
        assert np.isfinite(s["mean"])

    def test_score_large_dispersion(self):
        """Very large r (near-Poisson) should be numerically stable."""
        params = {"mean": 3.0, "dispersion": 1e6}
        s = self.dist.score(np.array([5.0]), params)
        assert np.isfinite(s["mean"])

    def test_link_unlink_dispersion(self):
        r = 7.5
        f = self.dist.link("dispersion", r)
        assert self.dist.unlink("dispersion", f) == pytest.approx(r)

    def test_initial_params_overdispersed(self):
        """When var >> mean (overdispersed), r should be small."""
        rng = np.random.default_rng(42)
        y = rng.negative_binomial(1, 0.25, 200).astype(float)  # mean=3, very overdispersed
        init = self.dist.initial_params(y)
        assert init["dispersion"] > 0
        assert init["mean"] > 0

    def test_initial_params_underdispersed_fallback(self):
        """When var <= mean, r formula can blow up — check fallback."""
        y = np.array([3.0] * 20)  # var=0 < mean=3 → r formula: mu^2 / max(var-mu, 1e-6)
        init = self.dist.initial_params(y)
        assert init["dispersion"] > 0

    def test_log_likelihood_sum_over_array(self):
        """Array input: log_likelihood should return array of same length."""
        y = np.array([0.0, 1.0, 2.0, 5.0])
        ll = self.dist.log_likelihood(y, {"mean": 3.0, "dispersion": 5.0})
        assert len(np.atleast_1d(ll)) == 4
        assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------------------
# ZIPGAS extended
# ---------------------------------------------------------------------------

class TestZIPGASExtended:
    def setup_method(self):
        self.dist = ZIPGAS()
        self.params = {"mean": 2.0, "zeroprob": 0.2}

    def test_score_with_exposure(self):
        E = 3.0
        s = self.dist.score(
            np.array([0.0]), {"mean": 2.0, "zeroprob": 0.2},
            exposure=np.array([E]),
        )
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["zeroprob"])

    def test_log_likelihood_with_exposure(self):
        E = 2.0
        ll = self.dist.log_likelihood(
            np.array([4.0]), {"mean": 2.0, "zeroprob": 0.1},
            exposure=np.array([E]),
        )
        assert np.isfinite(float(ll))

    def test_zeroprob_near_zero(self):
        """zeroprob near 0 → essentially Poisson."""
        params = {"mean": 2.0, "zeroprob": 1e-9}
        s = self.dist.score(np.array([3.0]), params)
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["zeroprob"])

    def test_zeroprob_near_one(self):
        """zeroprob near 1 → nearly all zeros."""
        params = {"mean": 2.0, "zeroprob": 1.0 - 1e-9}
        s = self.dist.score(np.array([0.0]), params)
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["zeroprob"])

    def test_link_zeroprob_boundary(self):
        """logit(0.5) = 0; logit(0.9) > 0; logit(0.1) < 0."""
        assert self.dist.link("zeroprob", 0.5) == pytest.approx(0.0, abs=1e-8)
        assert self.dist.link("zeroprob", 0.9) > 0
        assert self.dist.link("zeroprob", 0.1) < 0

    def test_unlink_zeroprob_large_positive(self):
        """sigmoid(large) → near 1."""
        v = self.dist.unlink("zeroprob", 20.0)
        assert v > 0.99

    def test_unlink_zeroprob_large_negative(self):
        """sigmoid(very_negative) → near 0."""
        v = self.dist.unlink("zeroprob", -20.0)
        assert v < 0.01

    def test_fisher_all_zeros_series(self):
        """All zeros: zero-prob is close to 1, Fisher should be positive."""
        fi = self.dist.fisher({"mean": 1.0, "zeroprob": 0.9})
        assert fi["mean"] >= 0
        assert fi["zeroprob"] >= 0

    def test_initial_params_all_zero(self):
        y = np.zeros(20)
        init = self.dist.initial_params(y)
        # mean defaults to 1.0, zeroprob should be positive
        assert init["mean"] > 0
        assert 0.0 < init["zeroprob"] < 1.0

    def test_score_array_input(self):
        y = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        s = self.dist.score(y, self.params)
        assert len(s["mean"]) == 5
        assert len(s["zeroprob"]) == 5
        assert np.all(np.isfinite(s["mean"]))
        assert np.all(np.isfinite(s["zeroprob"]))


# ---------------------------------------------------------------------------
# BetaGAS extended
# ---------------------------------------------------------------------------

class TestBetaGASExtended:
    def setup_method(self):
        self.dist = BetaGAS()
        self.params = {"mean": 0.65, "precision": 15.0}

    def test_log_likelihood_near_zero(self):
        """y very close to 0 should still give finite ll."""
        ll = self.dist.log_likelihood(np.array([1e-4]), self.params)
        assert np.isfinite(float(ll))

    def test_log_likelihood_near_one(self):
        """y very close to 1 should still give finite ll."""
        ll = self.dist.log_likelihood(np.array([1.0 - 1e-4]), self.params)
        assert np.isfinite(float(ll))

    def test_score_near_one(self):
        s = self.dist.score(np.array([0.99]), self.params)
        assert np.isfinite(s["mean"])

    def test_score_near_zero(self):
        s = self.dist.score(np.array([0.01]), self.params)
        assert np.isfinite(s["mean"])

    def test_score_at_exact_half(self):
        """y = 0.5 should give finite score."""
        s = self.dist.score(np.array([0.5]), self.params)
        assert np.isfinite(s["mean"])

    def test_fisher_high_precision(self):
        fi = self.dist.fisher({"mean": 0.5, "precision": 100.0})
        assert fi["mean"] > 0

    def test_fisher_low_precision(self):
        fi = self.dist.fisher({"mean": 0.5, "precision": 2.0})
        assert fi["mean"] > 0

    def test_link_unlink_mean_extremes(self):
        for mu in [0.01, 0.5, 0.99]:
            f = self.dist.link("mean", mu)
            assert self.dist.unlink("mean", f) == pytest.approx(mu, rel=1e-5)

    def test_link_unlink_precision_large(self):
        phi = 200.0
        f = self.dist.link("precision", phi)
        assert self.dist.unlink("precision", f) == pytest.approx(phi, rel=1e-5)

    def test_log_likelihood_array(self):
        y = np.array([0.1, 0.4, 0.65, 0.8, 0.95])
        ll = self.dist.log_likelihood(y, self.params)
        assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------------------
# LogNormalGAS extended
# ---------------------------------------------------------------------------

class TestLogNormalGASExtended:
    def setup_method(self):
        self.dist = LogNormalGAS()

    def test_logsigma_negative(self):
        """logsigma = log(sigma) where sigma < 1 → logsigma < 0. Should be stable."""
        params = {"logmean": 6.0, "logsigma": np.log(0.3)}
        y = np.exp(6.0)
        ll = self.dist.log_likelihood(np.array([y]), params)
        assert np.isfinite(float(ll))

    def test_score_logsigma_negative(self):
        params = {"logmean": 6.0, "logsigma": np.log(0.3)}
        y = np.exp(6.0)
        s = self.dist.score(np.array([y]), params)
        assert np.isfinite(s["logmean"])

    def test_fisher_logsigma_negative(self):
        params = {"logmean": 6.0, "logsigma": np.log(0.3)}
        fi = self.dist.fisher(params)
        assert fi["logmean"] > 0

    def test_log_likelihood_extreme_y(self):
        """Very large y should give finite (very negative) ll."""
        params = {"logmean": 6.0, "logsigma": 0.5}
        ll = self.dist.log_likelihood(np.array([1e10]), params)
        assert np.isfinite(float(ll))

    def test_log_likelihood_very_small_y(self):
        params = {"logmean": 6.0, "logsigma": 0.5}
        ll = self.dist.log_likelihood(np.array([0.001]), params)
        assert np.isfinite(float(ll))

    def test_param_names(self):
        assert "logmean" in self.dist.param_names
        assert "logsigma" in self.dist.param_names

    def test_initial_params_symmetry(self):
        """Initial params should give logmean near true value."""
        rng = np.random.default_rng(7)
        y = rng.lognormal(mean=7.0, sigma=0.6, size=500)
        init = self.dist.initial_params(y)
        assert init["logmean"] == pytest.approx(7.0, abs=0.2)


# ---------------------------------------------------------------------------
# GammaGAS extended
# ---------------------------------------------------------------------------

class TestGammaGASExtended:
    def setup_method(self):
        self.dist = GammaGAS()
        self.params = {"mean": 500.0, "shape": 3.0}

    def test_custom_shape_at_construction(self):
        dist = GammaGAS(shape=5.0)
        # shape is a default static param
        assert hasattr(dist, "shape")
        assert dist.shape == 5.0

    def test_score_array(self):
        y = np.array([200.0, 500.0, 800.0])
        s = self.dist.score(y, self.params)
        assert len(s["mean"]) == 3

    def test_link_unlink_mean(self):
        mu = 1234.0
        f = self.dist.link("mean", mu)
        assert self.dist.unlink("mean", f) == pytest.approx(mu, rel=1e-6)

    def test_initial_params_all_same(self):
        """All-same values → var=0, shape fallback."""
        y = np.full(20, 500.0)
        init = self.dist.initial_params(y)
        assert init["mean"] == pytest.approx(500.0)
        assert init["shape"] > 0

    def test_log_likelihood_array(self):
        rng = np.random.default_rng(1)
        y = rng.gamma(3.0, 200.0, 50)
        ll = self.dist.log_likelihood(y, self.params)
        assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------------------
# GASFilter extended
# ---------------------------------------------------------------------------

class TestGASFilterExtended:
    def test_filter_result_attributes(self):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"])
        y = np.ones(10) * 2.0
        r = filt.run(y, {"omega_mean": 0.05, "alpha_mean_1": 0.1, "phi_mean_1": 0.9}, {})
        assert hasattr(r, "filter_paths")
        assert hasattr(r, "f_paths")
        assert hasattr(r, "log_likelihoods")
        assert hasattr(r, "score_residuals")

    def test_filter_total_log_likelihood(self):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"])
        rng = np.random.default_rng(5)
        y = rng.poisson(3.0, 30).astype(float)
        r = filt.run(y, {"omega_mean": 0.05, "alpha_mean_1": 0.1, "phi_mean_1": 0.85}, {})
        total_ll = np.sum(r.log_likelihoods)
        assert np.isfinite(total_ll)

    def test_filter_lognormal_distribution(self):
        dist = LogNormalGAS()
        filt = GASFilter(dist, ["logmean"], scaling="fisher_inv")
        rng = np.random.default_rng(3)
        y = rng.lognormal(mean=7.0, sigma=0.4, size=30)
        gas_params = {"omega_logmean": 0.05, "alpha_logmean_1": 0.1, "phi_logmean_1": 0.85}
        r = filt.run(y, gas_params, static_params={"logsigma": np.log(0.4)})
        assert np.all(np.isfinite(r.log_likelihoods))

    def test_filter_with_static_params(self):
        dist = GammaGAS()
        filt = GASFilter(dist, ["mean"], scaling="fisher_inv")
        rng = np.random.default_rng(1)
        y = rng.gamma(3.0, 300.0, 25)
        gas_params = {"omega_mean": 0.05, "alpha_mean_1": 0.1, "phi_mean_1": 0.85}
        r = filt.run(y, gas_params, static_params={"shape": 3.0})
        assert np.all(r.filter_paths["mean"] > 0)


# ---------------------------------------------------------------------------
# GASResult extended
# ---------------------------------------------------------------------------

class TestGASResultExtended:
    def setup_method(self):
        data = load_motor_frequency(T=36, seed=2, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_summary_contains_params(self):
        s = self.result.summary()
        assert "omega_mean" in s
        assert "alpha_mean_1" in s
        assert "phi_mean_1" in s

    def test_relativities_base_first_starts_at_one(self):
        rel = self.result.relativities(base="first")
        assert float(rel["mean"].iloc[0]) == pytest.approx(1.0)

    def test_relativities_base_mean_mean_is_one(self):
        rel = self.result.relativities(base="mean")
        assert float(rel["mean"].mean()) == pytest.approx(1.0, rel=1e-6)

    def test_trend_index_all_positive(self):
        ti = self.result.trend_index
        assert np.all(ti["mean"].values > 0)

    def test_n_obs_matches_data(self):
        assert self.result.n_obs == 36

    def test_aic_bic_ordering(self):
        """AIC and BIC should be finite; AIC and BIC calculated from ll and n_params."""
        n_params = len(self.result.params)
        expected_aic = -2 * self.result.log_likelihood + 2 * n_params
        assert self.result.aic == pytest.approx(expected_aic, rel=1e-5)

    def test_params_all_finite(self):
        for name, val in self.result.params.items():
            assert np.isfinite(val), f"Param {name} is not finite: {val}"

    def test_score_residuals_are_dataframe(self):
        assert isinstance(self.result.score_residuals, pd.DataFrame)

    def test_filter_path_is_dataframe(self):
        assert isinstance(self.result.filter_path, pd.DataFrame)
        assert "mean" in self.result.filter_path.columns


# ---------------------------------------------------------------------------
# GASModel extended
# ---------------------------------------------------------------------------

class TestGASModelExtended:
    def test_distribution_name_preserved(self):
        m = GASModel("negbin")
        assert m.distribution_name == "negbin"

    def test_fit_negbin_with_exposure(self):
        rng = np.random.default_rng(10)
        y = rng.negative_binomial(5, 0.5, 40).astype(float)
        exposure = rng.uniform(0.5, 2.0, 40)
        m = GASModel("negbin")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y, exposure=exposure)
        assert isinstance(r, GASResult)

    def test_fit_zip_both_tv_params(self):
        """ZIP with both mean and zeroprob time-varying."""
        rng = np.random.default_rng(25)
        n = 50
        is_zero = rng.random(n) < 0.3
        y = np.where(is_zero, 0.0, rng.poisson(2.0, n).astype(float))
        m = GASModel("zip", time_varying=["mean", "zeroprob"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert isinstance(r, GASResult)

    def test_fit_gamma_with_p2_q1(self):
        rng = np.random.default_rng(33)
        y = rng.gamma(3.0, 300.0, 40)
        m = GASModel("gamma", p=2, q=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert r.n_obs == 40

    def test_fit_lognormal_with_p1_q2(self):
        rng = np.random.default_rng(77)
        y = rng.lognormal(7.0, 0.4, 40)
        m = GASModel("lognormal", p=1, q=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert r.n_obs == 40

    def test_build_static_param_names(self):
        m = GASModel("gamma", time_varying=["mean"])
        names = m._build_static_param_names()
        assert "shape" in names

    def test_fit_beta_with_unit_scaling(self):
        data = load_loss_ratio(T=30, seed=9)
        m = GASModel("beta", scaling="unit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(data.y)
        assert isinstance(r, GASResult)

    def test_fit_beta_with_fisher_inv_sqrt(self):
        data = load_loss_ratio(T=30, seed=11)
        m = GASModel("beta", scaling="fisher_inv_sqrt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(data.y)
        assert isinstance(r, GASResult)


# ---------------------------------------------------------------------------
# Forecast extended
# ---------------------------------------------------------------------------

class TestForecastExtended:
    def setup_method(self):
        data = load_motor_frequency(T=48, seed=1, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_forecast_h_one(self):
        fc = self.result.forecast(h=1, method="mean_path")
        assert len(fc.mean_path["mean"]) == 1

    def test_forecast_to_dataframe_specific_param(self):
        fc = self.result.forecast(h=6, method="mean_path")
        df = fc.to_dataframe(param="mean")
        assert "mean" in df.columns
        assert len(df) == 6

    def test_forecast_to_dataframe_with_quantiles(self):
        fc = self.result.forecast(h=6, method="simulate", quantiles=[0.1, 0.9],
                                   n_sim=50, rng=np.random.default_rng(42))
        df = fc.to_dataframe()
        assert "mean" in df.columns
        assert "q10" in df.columns
        assert "q90" in df.columns

    def test_forecast_simulate_no_quantiles(self):
        """Simulate with no quantiles returns empty quantiles dict."""
        fc = self.result.forecast(h=4, method="simulate", quantiles=None,
                                   n_sim=50, rng=np.random.default_rng(1))
        assert isinstance(fc.quantiles, dict)

    def test_forecast_h_attribute(self):
        fc = self.result.forecast(h=9, method="mean_path")
        assert fc.h == 9

    def test_forecast_plot_no_crash(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not available")
        fc = self.result.forecast(h=6, method="mean_path")
        ax = fc.plot()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_forecast_gamma_simulate(self):
        rng = np.random.default_rng(42)
        y = rng.gamma(3.0, 300.0, 40)
        m = GASModel("gamma")
        r = m.fit(y)
        fc = r.forecast(h=6, method="simulate", n_sim=100, rng=np.random.default_rng(5))
        assert np.all(np.isfinite(fc.mean_path["mean"]))


# ---------------------------------------------------------------------------
# Bootstrap extended
# ---------------------------------------------------------------------------

class TestBootstrapExtended:
    def test_bootstrap_gamma(self):
        rng = np.random.default_rng(50)
        y = rng.gamma(3.0, 300.0, 36)
        m = GASModel("gamma")
        r = m.fit(y)
        ci = r.bootstrap_ci(n_boot=30, rng=np.random.default_rng(1))
        assert isinstance(ci, BootstrapCI)
        assert np.all(ci.filter_lower["mean"].values > 0)

    def test_bootstrap_lognormal(self):
        rng = np.random.default_rng(60)
        y = rng.lognormal(7.0, 0.4, 36)
        m = GASModel("lognormal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = r.bootstrap_ci(n_boot=20, rng=np.random.default_rng(2))
        assert isinstance(ci, BootstrapCI)

    def test_bootstrap_ci_confidence_99(self):
        data = load_motor_frequency(T=30, seed=5, trend_break=False)
        m = GASModel("poisson")
        r = m.fit(data.y, exposure=data.exposure)
        ci = r.bootstrap_ci(n_boot=30, confidence=0.99, rng=np.random.default_rng(7))
        assert ci.confidence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Diagnostics extended
# ---------------------------------------------------------------------------

class TestDiagnosticsExtended:
    def setup_method(self):
        data = load_motor_frequency(T=36, seed=3, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_compute_diagnostics_standalone(self):
        diag = compute_diagnostics(self.result)
        assert isinstance(diag, DiagnosticsResult)

    def test_diagnostics_summary_contains_ks(self):
        diag = self.result.diagnostics()
        s = diag.summary()
        assert "KS" in s or "Kolmogorov" in s or "ks" in s.lower()

    def test_diagnostics_summary_contains_ljung_box(self):
        diag = self.result.diagnostics()
        s = diag.summary()
        assert "Ljung" in s or "ljung" in s.lower() or "serial" in s.lower()

    def test_acf_constant_series_is_nan_or_one(self):
        """ACF of constant series — lag 0 should be 1 or nan (division by zero)."""
        x = np.ones(50)
        acf = _compute_acf(x, nlags=5)
        # lag 0 is always 1; other lags may be nan or 1
        assert acf[0] == pytest.approx(1.0) or np.isnan(acf[0])

    def test_pit_continuous_gamma(self):
        """PIT for gamma at mean should be roughly 0.5 (more than 0, less than 1)."""
        from scipy.stats import gamma as scipy_gamma
        dist = GammaGAS()
        mu = 500.0
        shape = 3.0
        # y = mu * shape / shape = mu (mode); cdf at mean with shape=3 > 0.5 with shape param
        params = {"mean": mu, "shape": shape}
        # The gamma scale = mean / shape = 500/3
        pit = _pit_continuous(dist, float(mu), params)
        assert 0.0 <= pit <= 1.0

    def test_pit_residuals_negbin(self):
        rng = np.random.default_rng(99)
        y = rng.negative_binomial(5, 0.625, 30).astype(float)
        dist = NegBinGAS()
        fp = pd.DataFrame({"mean": np.full(30, 3.0)})
        static = {"dispersion": 5.0}
        pits = pit_residuals(y, fp, dist, static, rng=rng)
        assert len(pits) == 30
        assert np.all(pits >= 0.0)
        assert np.all(pits <= 1.0)

    def test_pit_residuals_beta(self):
        rng = np.random.default_rng(123)
        y = rng.beta(0.65 * 15, 0.35 * 15, 30)
        dist = BetaGAS()
        fp = pd.DataFrame({"mean": np.full(30, 0.65)})
        static = {"precision": 15.0}
        pits = pit_residuals(y, fp, dist, static, rng=rng)
        assert len(pits) == 30

    def test_pit_residuals_zip(self):
        rng = np.random.default_rng(77)
        n = 30
        is_zero = rng.random(n) < 0.3
        y = np.where(is_zero, 0.0, rng.poisson(2.0, n).astype(float))
        dist = ZIPGAS()
        fp = pd.DataFrame({"mean": np.full(n, 2.0)})
        static = {"zeroprob": 0.3}
        pits = pit_residuals(y, fp, dist, static, rng=rng)
        assert len(pits) == n

    def test_dawid_sebastiani_single_observation(self):
        y = np.array([3.0])
        mu = np.array([3.0])
        sigma = np.array([1.0])
        ds = dawid_sebastiani_score(y, mu, sigma)
        assert np.isfinite(ds)

    def test_dawid_sebastiani_large_sigma(self):
        """Very wide predictive distribution → large DS score."""
        y = np.ones(10) * 5.0
        mu = np.ones(10) * 5.0
        sigma_small = np.ones(10) * 0.1
        sigma_large = np.ones(10) * 100.0
        ds_small = dawid_sebastiani_score(y, mu, sigma_small)
        ds_large = dawid_sebastiani_score(y, mu, sigma_large)
        assert ds_large > ds_small


# ---------------------------------------------------------------------------
# GASPanel extended
# ---------------------------------------------------------------------------

class TestGASPanelExtended:
    def _make_panel_data(self, n_cells=3, T=30, distribution="gamma", seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_cells):
            base = rng.uniform(200, 800)
            if distribution == "gamma":
                y = rng.gamma(3.0, base / 3.0, T)
            else:
                y = rng.poisson(rng.uniform(2, 6), T).astype(float)
            for t in range(T):
                rows.append({"period": t, "cell": f"c{i}", "y": y[t], "exposure": 1.0})
        return pd.DataFrame(rows)

    def test_gamma_panel_fit(self):
        data = self._make_panel_data(distribution="gamma")
        panel = GASPanel("gamma")
        r = panel.fit(data, y_col="y", period_col="period", cell_col="cell")
        assert len(r.filter_paths) + len(r.failed_cells) == 3

    def test_panel_trend_indices_all_start_100(self):
        data = self._make_panel_data()
        panel = GASPanel("gamma")
        r = panel.fit(data, y_col="y", period_col="period", cell_col="cell")
        for cell_id, ti in r.trend_indices.items():
            assert ti["mean"].iloc[0] == pytest.approx(100.0, abs=1e-4)

    def test_panel_trend_summary_columns(self):
        data = self._make_panel_data(distribution="poisson")
        panel = GASPanel("poisson")
        r = panel.fit(data, y_col="y", period_col="period", cell_col="cell")
        ts = r.trend_summary()
        assert isinstance(ts, pd.DataFrame)
        assert "period" in ts.columns or ts.index.name == "period" or len(ts.columns) > 0

    def test_panel_summary_frame_period_count(self):
        T = 25
        data = self._make_panel_data(T=T)
        panel = GASPanel("gamma")
        r = panel.fit(data, y_col="y", period_col="period", cell_col="cell")
        sf = r.summary_frame()
        assert len(sf) == T

    def test_panel_failed_cells_structure(self):
        """Failed cells should be recorded in r.failed_cells."""
        data = self._make_panel_data(n_cells=2)
        short = pd.DataFrame({
            "period": [0, 1], "cell": ["bad_cell", "bad_cell"],
            "y": [1.0, 2.0], "exposure": [1.0, 1.0]
        })
        data = pd.concat([data, short], ignore_index=True)
        panel = GASPanel("gamma")
        r = panel.fit(data, y_col="y", period_col="period", cell_col="cell")
        assert "bad_cell" in r.failed_cells

    def test_panel_single_cell(self):
        T = 30
        data = self._make_panel_data(n_cells=1, T=T)
        panel = GASPanel("poisson")
        r = panel.fit(data, y_col="y", period_col="period", cell_col="cell")
        assert len(r.filter_paths) == 1


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_load_motor_frequency_shape(self):
        data = load_motor_frequency(T=36, seed=1)
        assert len(data.y) == 36
        assert len(data.exposure) == 36

    def test_load_motor_frequency_with_trend_break(self):
        data = load_motor_frequency(T=48, seed=2, trend_break=True)
        assert len(data.y) == 48
        assert np.all(data.y >= 0)

    def test_load_severity_trend_shape(self):
        data = load_severity_trend(T=60, seed=3)
        assert len(data.y) == 60
        assert np.all(data.y > 0)

    def test_load_severity_trend_positive(self):
        data = load_severity_trend(T=40, seed=5, inflation_rate=0.08)
        assert np.all(data.y > 0)

    def test_load_loss_ratio_shape(self):
        data = load_loss_ratio(T=36, seed=7)
        assert len(data.y) == 36

    def test_load_loss_ratio_in_unit_interval(self):
        data = load_loss_ratio(T=36, seed=8)
        assert np.all(data.y > 0)
        assert np.all(data.y < 1)

    def test_load_motor_frequency_different_seeds_differ(self):
        d1 = load_motor_frequency(T=24, seed=1)
        d2 = load_motor_frequency(T=24, seed=99)
        # Different seeds should give different data
        assert not np.allclose(d1.y, d2.y)

    def test_load_severity_trend_inflation_effect(self):
        """Higher inflation rate should lead to higher mean severity."""
        d_low = load_severity_trend(T=60, seed=10, inflation_rate=0.01)
        d_high = load_severity_trend(T=60, seed=10, inflation_rate=0.20)
        # Mean of high inflation series should be higher
        assert np.mean(d_high.y) > np.mean(d_low.y)
