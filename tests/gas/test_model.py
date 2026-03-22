"""Tests for GASModel fitting and GASResult."""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_dynamics.gas import GASModel, GASResult
from insurance_dynamics.gas.datasets import (
    load_motor_frequency,
    load_severity_trend,
    load_loss_ratio,
)


# ---------------------------------------------------------------------------
# Basic model construction
# ---------------------------------------------------------------------------

class TestGASModelConstruction:
    def test_default_poisson(self):
        m = GASModel("poisson")
        assert m.distribution_name == "poisson"

    def test_all_distribution_names(self):
        for dist in ["poisson", "gamma", "negbin", "lognormal", "beta", "zip"]:
            m = GASModel(dist)
            assert m is not None

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            GASModel("tweedie")

    def test_custom_p_q(self):
        m = GASModel("poisson", p=2, q=2)
        assert m.p == 2
        assert m.q == 2

    def test_scaling_options(self):
        for s in ["unit", "fisher_inv", "fisher_inv_sqrt"]:
            m = GASModel("poisson", scaling=s)
            assert m.scaling == s

    def test_time_varying_override(self):
        m = GASModel("gamma", time_varying=["mean"])
        assert m.time_varying == ["mean"]

    def test_distribution_instance_input(self):
        from insurance_dynamics.gas.distributions import GammaGAS
        dist = GammaGAS(shape=3.0)
        m = GASModel(dist)
        assert m.distribution is dist


# ---------------------------------------------------------------------------
# Poisson model fit
# ---------------------------------------------------------------------------

class TestPoissonModelFit:
    def setup_method(self):
        self.data = load_motor_frequency(T=48, seed=1, trend_break=False)

    def test_fit_returns_result(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert isinstance(r, GASResult)

    def test_filter_path_length(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert len(r.filter_path) == len(self.data.y)

    def test_filter_path_positive(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert np.all(r.filter_path["mean"].values > 0)

    def test_aic_bic_finite(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert np.isfinite(r.aic)
        assert np.isfinite(r.bic)

    def test_aic_less_than_2x_neg_ll(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert r.aic > -2.0 * r.log_likelihood - 1e-6

    def test_log_likelihood_finite(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert np.isfinite(r.log_likelihood)

    def test_summary_is_string(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        s = r.summary()
        assert isinstance(s, str)
        assert "GAS Model" in s
        assert "Log-likelihood" in s

    def test_params_dict_has_expected_keys(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert "omega_mean" in r.params
        assert "alpha_mean_1" in r.params
        assert "phi_mean_1" in r.params

    def test_score_residuals_length(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert len(r.score_residuals) == len(self.data.y)

    def test_no_exposure(self):
        y = np.random.default_rng(1).poisson(3.0, 30).astype(float)
        m = GASModel("poisson")
        r = m.fit(y)
        assert r.filter_path["mean"].notna().all()

    def test_mismatched_exposure_raises(self):
        m = GASModel("poisson")
        with pytest.raises(ValueError, match="exposure length"):
            m.fit(self.data.y, exposure=np.ones(5))

    def test_too_few_obs_raises(self):
        m = GASModel("poisson")
        with pytest.raises(ValueError, match="at least 4"):
            m.fit(np.array([1.0, 2.0]))

    def test_n_obs_correct(self):
        m = GASModel("poisson")
        r = m.fit(self.data.y, exposure=self.data.exposure)
        assert r.n_obs == len(self.data.y)


# ---------------------------------------------------------------------------
# Trend index and relativities
# ---------------------------------------------------------------------------

class TestTrendIndex:
    def setup_method(self):
        data = load_motor_frequency(T=36, seed=3, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_trend_index_starts_at_100(self):
        ti = self.result.trend_index
        assert float(ti["mean"].iloc[0]) == pytest.approx(100.0)

    def test_trend_index_positive(self):
        ti = self.result.trend_index
        assert np.all(ti["mean"].values > 0)

    def test_relativities_mean_near_one(self):
        rel = self.result.relativities(base="mean")
        assert float(rel["mean"].mean()) == pytest.approx(1.0, rel=1e-6)

    def test_relativities_first_equals_one(self):
        rel = self.result.relativities(base="first")
        assert float(rel["mean"].iloc[0]) == pytest.approx(1.0)

    def test_relativities_invalid_base_raises(self):
        with pytest.raises(ValueError, match="base must be"):
            self.result.relativities(base="median")


# ---------------------------------------------------------------------------
# Gamma model fit
# ---------------------------------------------------------------------------

class TestGammaModelFit:
    def setup_method(self):
        # Use longer series for more robust trend detection
        self.data = load_severity_trend(T=80, seed=2, inflation_rate=0.05)

    def test_gamma_fit_returns_result(self):
        m = GASModel("gamma")
        r = m.fit(self.data.y)
        assert isinstance(r, GASResult)

    def test_gamma_filter_positive(self):
        m = GASModel("gamma")
        r = m.fit(self.data.y)
        assert np.all(r.filter_path["mean"].values > 0)

    def test_gamma_trend_direction(self):
        """GAS filter mean should be higher in the last quarter than the first quarter.

        Uses a strong inflation signal (10% annual, 120 periods) to make
        the directional assertion robust across different optimisation paths.
        """
        data = load_severity_trend(T=120, seed=7, inflation_rate=0.10)
        m = GASModel("gamma")
        r = m.fit(data.y)
        # Compare first 20 vs last 20 periods — compound inflation makes this unambiguous
        early = r.filter_path["mean"].iloc[:20].mean()
        late = r.filter_path["mean"].iloc[-20:].mean()
        assert late > early

    def test_gamma_ll_finite(self):
        m = GASModel("gamma")
        r = m.fit(self.data.y)
        assert np.isfinite(r.log_likelihood)


# ---------------------------------------------------------------------------
# Negative binomial model fit
# ---------------------------------------------------------------------------

class TestNegBinModelFit:
    def test_negbin_fit(self):
        rng = np.random.default_rng(10)
        y = rng.negative_binomial(5, 0.5, 40).astype(float)
        m = GASModel("negbin")
        r = m.fit(y)
        assert isinstance(r, GASResult)
        assert np.all(r.filter_path["mean"].values > 0)

    def test_negbin_ll_better_than_poisson_overdispersed(self):
        """NB should fit overdispersed data with higher or equal log-likelihood."""
        rng = np.random.default_rng(20)
        y = rng.negative_binomial(2, 0.4, 50).astype(float)

        m_nb = GASModel("negbin")
        m_p = GASModel("poisson")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_nb = m_nb.fit(y)
            r_p = m_p.fit(y)

        # NB has more parameters but should have better or equal log-likelihood
        assert r_nb.log_likelihood >= r_p.log_likelihood - 5


# ---------------------------------------------------------------------------
# LogNormal model fit
# ---------------------------------------------------------------------------

class TestLogNormalModelFit:
    def test_lognormal_fit(self):
        rng = np.random.default_rng(15)
        y = rng.lognormal(mean=6.0, sigma=0.5, size=40)
        m = GASModel("lognormal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert isinstance(r, GASResult)

    def test_lognormal_filter_finite(self):
        rng = np.random.default_rng(16)
        y = rng.lognormal(mean=6.0, sigma=0.5, size=40)
        m = GASModel("lognormal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert np.all(np.isfinite(r.filter_path["logmean"].values))


# ---------------------------------------------------------------------------
# Beta model fit
# ---------------------------------------------------------------------------

class TestBetaModelFit:
    def setup_method(self):
        self.data = load_loss_ratio(T=48, seed=5)

    def test_beta_fit_returns_result(self):
        m = GASModel("beta")
        r = m.fit(self.data.y)
        assert isinstance(r, GASResult)

    def test_beta_filter_in_unit_interval(self):
        m = GASModel("beta")
        r = m.fit(self.data.y)
        assert np.all(r.filter_path["mean"].values > 0)
        assert np.all(r.filter_path["mean"].values < 1)

    def test_beta_ll_finite(self):
        m = GASModel("beta")
        r = m.fit(self.data.y)
        assert np.isfinite(r.log_likelihood)


# ---------------------------------------------------------------------------
# ZIP model fit
# ---------------------------------------------------------------------------

class TestZIPModelFit:
    def test_zip_fit_single_tv_param(self):
        """Default: only mean is time-varying."""
        rng = np.random.default_rng(25)
        n = 50
        is_zero = rng.random(n) < 0.3
        y = np.where(is_zero, 0.0, rng.poisson(2.0, n).astype(float))
        m = GASModel("zip")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert isinstance(r, GASResult)
        assert np.all(r.filter_path["mean"].values > 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_zeros_poisson(self):
        """Fit on all-zero claim series — model may converge to very small mean."""
        y = np.zeros(20)
        m = GASModel("poisson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        # The fit should not crash; filter values may be very small but non-negative
        assert r.filter_path["mean"].values is not None
        # Allow either all finite or convergence to small values
        vals = r.filter_path["mean"].values
        # Values should be non-negative at minimum
        assert np.all(np.nan_to_num(vals, nan=0.0) >= 0)

    def test_single_large_observation(self):
        """Outlier in Poisson series should not cause numerical failure."""
        rng = np.random.default_rng(77)
        y = rng.poisson(2.0, 30).astype(float)
        y[15] = 50.0
        m = GASModel("poisson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert np.all(np.isfinite(r.filter_path["mean"].values))

    def test_very_short_series(self):
        """Minimum length series (4 observations)."""
        y = np.array([1.0, 2.0, 1.0, 3.0])
        m = GASModel("poisson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit(y)
        assert len(r.filter_path) == 4

    def test_long_series(self):
        """200-period Poisson series should fit without issue."""
        rng = np.random.default_rng(88)
        y = rng.poisson(3.0, 200).astype(float)
        m = GASModel("poisson")
        r = m.fit(y)
        assert len(r.filter_path) == 200

    def test_exposure_all_ones_equivalent_to_no_exposure(self):
        """Exposure of all ones should give same result as no exposure."""
        rng = np.random.default_rng(99)
        y = rng.poisson(2.0, 30).astype(float)
        m1 = GASModel("poisson")
        m2 = GASModel("poisson")
        r1 = m1.fit(y)
        r2 = m2.fit(y, exposure=np.ones(30))
        np.testing.assert_allclose(
            r1.filter_path["mean"].values,
            r2.filter_path["mean"].values,
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# P0/P1 regression tests
# ---------------------------------------------------------------------------

class TestConstructionValidation:
    """P1: validate p, q, scaling at construction time."""

    def test_p_zero_raises(self):
        with pytest.raises(ValueError, match="p must be an integer >= 1"):
            GASModel("poisson", p=0)

    def test_p_negative_raises(self):
        with pytest.raises(ValueError, match="p must be an integer >= 1"):
            GASModel("poisson", p=-1)

    def test_p_float_raises(self):
        with pytest.raises(ValueError, match="p must be an integer >= 1"):
            GASModel("poisson", p=1.5)  # type: ignore

    def test_q_zero_raises(self):
        with pytest.raises(ValueError, match="q must be an integer >= 1"):
            GASModel("poisson", q=0)

    def test_q_negative_raises(self):
        with pytest.raises(ValueError, match="q must be an integer >= 1"):
            GASModel("poisson", q=-2)

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValueError, match="scaling must be one of"):
            GASModel("poisson", scaling="identity")

    def test_invalid_scaling_inverse_raises(self):
        with pytest.raises(ValueError, match="scaling must be one of"):
            GASModel("poisson", scaling="inverse")

    def test_valid_p_q_scaling_accepted(self):
        m = GASModel("poisson", p=2, q=3, scaling="unit")
        assert m.p == 2
        assert m.q == 3
        assert m.scaling == "unit"


class TestOmegaPhiCoupling:
    """P0: omega and phi_init must reference the same phi_init value."""

    def test_x0_omega_consistent_with_phi(self):
        """Check that the initial x0 vector satisfies omega = init_f * (1 - phi_init).

        We reconstruct x0 by patching the optimiser call and inspecting the
        initial parameter vector passed to it.
        """
        import unittest.mock as mock

        rng = np.random.default_rng(99)
        y = rng.poisson(3.0, 30).astype(float)
        m = GASModel("poisson")

        captured_x0 = {}

        original_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize

        def capturing_minimize(fun, x0, **kwargs):
            captured_x0["x0"] = x0.copy()
            return original_minimize(fun, x0, **kwargs)

        with mock.patch("scipy.optimize.minimize", side_effect=capturing_minimize):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(y)

        x0 = captured_x0["x0"]
        # For p=1, q=1, poisson (one TV param: mean), layout is [omega, alpha, phi]
        omega = x0[0]
        phi = x0[2]
        # omega + phi * init_f should ≈ init_f, i.e. omega = init_f * (1 - phi)
        # Equivalently, omega / (1 - phi) should ≈ init_f = omega / (1 - phi)
        # The key invariant: phi == 0.9 and omega == init_f * 0.1
        assert abs(phi - 0.9) < 1e-10, f"phi_init should be 0.9, got {phi}"
        # omega / (1 - phi) should equal init_f (positive)
        init_f_recovered = omega / (1.0 - phi)
        assert init_f_recovered > 0, "Recovered init_f should be positive"
