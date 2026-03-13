"""Tests for individual GAS distributions."""

import numpy as np
import pytest

from insurance_dynamics.gas.distributions import (
    PoissonGAS,
    GammaGAS,
    NegBinGAS,
    LogNormalGAS,
    BetaGAS,
    ZIPGAS,
    DISTRIBUTION_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_params(dist_cls, **kwargs):
    """Build a minimal params dict for a distribution."""
    if dist_cls is PoissonGAS:
        return {"mean": kwargs.get("mean", 2.0)}
    elif dist_cls is GammaGAS:
        return {"mean": kwargs.get("mean", 500.0), "shape": kwargs.get("shape", 3.0)}
    elif dist_cls is NegBinGAS:
        return {"mean": kwargs.get("mean", 2.0), "dispersion": kwargs.get("dispersion", 5.0)}
    elif dist_cls is LogNormalGAS:
        return {"logmean": kwargs.get("logmean", 6.0), "logsigma": kwargs.get("logsigma", 0.5)}
    elif dist_cls is BetaGAS:
        return {"mean": kwargs.get("mean", 0.65), "precision": kwargs.get("precision", 15.0)}
    elif dist_cls is ZIPGAS:
        return {"mean": kwargs.get("mean", 2.0), "zeroprob": kwargs.get("zeroprob", 0.2)}


# ---------------------------------------------------------------------------
# PoissonGAS
# ---------------------------------------------------------------------------

class TestPoissonGAS:
    def setup_method(self):
        self.dist = PoissonGAS()
        self.params = {"mean": 2.0}

    def test_score_zero_count(self):
        """y=0: score should be 0/mu - 1 = -1."""
        s = self.dist.score(np.array([0.0]), self.params)
        assert s["mean"] == pytest.approx(-1.0)

    def test_score_equals_mu_count(self):
        """y=mu: score should be 0."""
        s = self.dist.score(np.array([2.0]), {"mean": 2.0})
        assert s["mean"] == pytest.approx(0.0)

    def test_score_large_count(self):
        """y=6, mu=2: score = 6/2 - 1 = 2."""
        s = self.dist.score(np.array([6.0]), {"mean": 2.0})
        assert s["mean"] == pytest.approx(2.0)

    def test_fisher_positive(self):
        fi = self.dist.fisher(self.params)
        assert fi["mean"] > 0

    def test_fisher_equals_mu(self):
        fi = self.dist.fisher({"mean": 3.0})
        assert fi["mean"] == pytest.approx(3.0)

    def test_fisher_with_exposure(self):
        fi = self.dist.fisher({"mean": 2.0}, exposure=np.array([5.0]))
        assert fi["mean"] == pytest.approx(10.0)

    def test_log_likelihood_positive_count(self):
        ll = self.dist.log_likelihood(np.array([2.0]), {"mean": 2.0})
        # log(2^2 * exp(-2) / 2!) = 2*log(2) - 2 - log(2)
        expected = 2.0 * np.log(2.0) - 2.0 - np.log(2.0)
        assert float(ll) == pytest.approx(expected, rel=1e-6)

    def test_log_likelihood_zero_count(self):
        ll = self.dist.log_likelihood(np.array([0.0]), {"mean": 2.0})
        assert float(ll) == pytest.approx(-2.0, rel=1e-6)

    def test_log_likelihood_with_exposure(self):
        """rate = mu * E = 2 * 3 = 6; y=6"""
        ll = self.dist.log_likelihood(
            np.array([6.0]), {"mean": 2.0}, exposure=np.array([3.0])
        )
        # log P(6 | 6) = 6*log(6) - 6 - log(6!)
        from scipy.special import gammaln
        expected = 6.0 * np.log(6.0) - 6.0 - gammaln(7.0)
        assert float(ll) == pytest.approx(expected, rel=1e-6)

    def test_link_unlink_roundtrip(self):
        mu = 3.5
        f = self.dist.link("mean", mu)
        assert self.dist.unlink("mean", f) == pytest.approx(mu)

    def test_link_is_log(self):
        assert self.dist.link("mean", np.e) == pytest.approx(1.0)

    def test_initial_params_returns_mean(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        init = self.dist.initial_params(y)
        assert init["mean"] == pytest.approx(2.5)

    def test_scaled_score_unit(self):
        s = self.dist.scaled_score(np.array([4.0]), {"mean": 2.0}, scaling="unit")
        assert s["mean"] == pytest.approx(1.0)  # 4/2 - 1 = 1

    def test_scaled_score_fisher_inv(self):
        # fisher_inv: scale by 1/mu = 0.5
        s = self.dist.scaled_score(np.array([4.0]), {"mean": 2.0}, scaling="fisher_inv")
        assert s["mean"] == pytest.approx(1.0 / 2.0)

    def test_scaled_score_fisher_inv_sqrt(self):
        s = self.dist.scaled_score(np.array([4.0]), {"mean": 2.0}, scaling="fisher_inv_sqrt")
        expected = 1.0 / np.sqrt(2.0)
        assert s["mean"] == pytest.approx(expected)

    def test_score_array_input(self):
        y = np.array([1.0, 2.0, 3.0])
        s = self.dist.score(y, {"mean": 2.0})
        expected = y / 2.0 - 1.0
        np.testing.assert_allclose(s["mean"], expected)

    def test_exposure_weighted_score(self):
        """With exposure E=2, effective rate = mu*E = 4. y=4: score = 4/4 - 1 = 0."""
        s = self.dist.score(np.array([4.0]), {"mean": 2.0}, exposure=np.array([2.0]))
        assert s["mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GammaGAS
# ---------------------------------------------------------------------------

class TestGammaGAS:
    def setup_method(self):
        self.dist = GammaGAS()
        self.params = {"mean": 500.0, "shape": 3.0}

    def test_score_at_mean(self):
        """y = mu: score should be zero."""
        s = self.dist.score(np.array([500.0]), self.params)
        assert s["mean"] == pytest.approx(0.0, abs=1e-10)

    def test_score_above_mean(self):
        s = self.dist.score(np.array([600.0]), self.params)
        assert s["mean"] > 0

    def test_score_below_mean(self):
        s = self.dist.score(np.array([400.0]), self.params)
        assert s["mean"] < 0

    def test_score_magnitude(self):
        """a*(y/mu - 1) with a=3, y=600, mu=500 → 3*(1.2-1) = 0.6"""
        s = self.dist.score(np.array([600.0]), self.params)
        assert s["mean"] == pytest.approx(3.0 * (600.0 / 500.0 - 1.0))

    def test_fisher_equals_shape(self):
        fi = self.dist.fisher(self.params)
        assert fi["mean"] == pytest.approx(3.0)

    def test_log_likelihood_positive(self):
        ll = self.dist.log_likelihood(np.array([500.0]), self.params)
        assert np.isfinite(float(ll))

    def test_log_likelihood_decreases_away_from_mean(self):
        ll_at_mean = float(self.dist.log_likelihood(np.array([500.0]), self.params))
        ll_far = float(self.dist.log_likelihood(np.array([2000.0]), self.params))
        assert ll_at_mean > ll_far

    def test_link_unlink(self):
        mu = 350.0
        f = self.dist.link("mean", mu)
        assert self.dist.unlink("mean", f) == pytest.approx(mu)

    def test_initial_params_reasonable(self):
        rng = np.random.default_rng(1)
        y = rng.gamma(shape=3.0, scale=200.0, size=200)
        init = self.dist.initial_params(y)
        assert init["mean"] > 0
        assert init["shape"] > 0

    def test_fisher_inv_scaling(self):
        """With fisher_inv, scaled score = raw_score / fisher = shape * (y/mu-1) / shape."""
        s_unit = self.dist.scaled_score(np.array([600.0]), self.params, scaling="unit")
        s_fi = self.dist.scaled_score(np.array([600.0]), self.params, scaling="fisher_inv")
        a = 3.0
        assert s_fi["mean"] == pytest.approx(s_unit["mean"] / a)


# ---------------------------------------------------------------------------
# NegBinGAS
# ---------------------------------------------------------------------------

class TestNegBinGAS:
    def setup_method(self):
        self.dist = NegBinGAS()
        self.params = {"mean": 3.0, "dispersion": 5.0}

    def test_score_at_mean(self):
        """E[y - (y+r)*mu/(mu+r)] at y=mu should give 0 in expectation."""
        # Score at the expected value
        mu, r = 3.0, 5.0
        expected = mu - (mu + r) * mu / (mu + r)
        s = self.dist.score(np.array([mu]), self.params)
        assert float(s["mean"]) == pytest.approx(expected, rel=1e-6)

    def test_log_likelihood_finite(self):
        ll = self.dist.log_likelihood(np.array([3.0]), self.params)
        assert np.isfinite(float(ll))

    def test_log_likelihood_zero(self):
        ll = self.dist.log_likelihood(np.array([0.0]), self.params)
        assert np.isfinite(float(ll))

    def test_fisher_positive(self):
        fi = self.dist.fisher(self.params)
        assert fi["mean"] > 0

    def test_approaches_poisson_large_r(self):
        """Large r: NB approaches Poisson. Check log-likelihood."""
        from insurance_dynamics.gas.distributions import PoissonGAS
        nb_params = {"mean": 2.0, "dispersion": 1e6}
        ll_nb = float(self.dist.log_likelihood(np.array([3.0]), nb_params))

        p_dist = PoissonGAS()
        ll_p = float(p_dist.log_likelihood(np.array([3.0]), {"mean": 2.0}))
        assert ll_nb == pytest.approx(ll_p, abs=0.1)

    def test_link_unlink(self):
        val = 4.5
        assert self.dist.unlink("mean", self.dist.link("mean", val)) == pytest.approx(val)

    def test_initial_params(self):
        rng = np.random.default_rng(42)
        y = rng.negative_binomial(5, 0.625, 200).astype(float)  # mean=3, dispersion~5
        init = self.dist.initial_params(y)
        assert init["mean"] > 0
        assert init["dispersion"] > 0


# ---------------------------------------------------------------------------
# LogNormalGAS
# ---------------------------------------------------------------------------

class TestLogNormalGAS:
    def setup_method(self):
        self.dist = LogNormalGAS()
        self.params = {"logmean": 6.0, "logsigma": 0.5}

    def test_score_at_mean(self):
        """y = exp(mu): log(y) - mu = 0 → score = 0."""
        y = np.exp(6.0)
        s = self.dist.score(np.array([y]), self.params)
        assert s["logmean"] == pytest.approx(0.0, abs=1e-10)

    def test_score_above_mean(self):
        y = np.exp(6.5)  # log(y) > 6
        s = self.dist.score(np.array([y]), self.params)
        assert s["logmean"] > 0

    def test_fisher_positive(self):
        fi = self.dist.fisher(self.params)
        assert fi["logmean"] > 0

    def test_fisher_equals_inv_sigma_sq(self):
        sigma = np.exp(0.5)
        fi = self.dist.fisher(self.params)
        assert fi["logmean"] == pytest.approx(1.0 / sigma**2)

    def test_log_likelihood_finite(self):
        y = np.exp(6.0)
        ll = self.dist.log_likelihood(np.array([y]), self.params)
        assert np.isfinite(float(ll))

    def test_link_unlink_logmean(self):
        val = 5.5
        f = self.dist.link("logmean", val)
        assert self.dist.unlink("logmean", f) == pytest.approx(val)

    def test_link_unlink_logsigma(self):
        val = 0.7
        # logsigma is stored as log(sigma) — link is log(sigma), unlink is exp
        # But initial_params returns sigma directly...
        # Actually: link("logsigma", sigma) = log(sigma), unlink returns exp(f)
        sigma = 0.7
        f = self.dist.link("logsigma", sigma)
        assert self.dist.unlink("logsigma", f) == pytest.approx(sigma)

    def test_initial_params_reasonable(self):
        rng = np.random.default_rng(5)
        y = rng.lognormal(mean=6.0, sigma=0.5, size=200)
        init = self.dist.initial_params(y)
        assert init["logmean"] == pytest.approx(6.0, abs=0.3)
        assert init["logsigma"] == pytest.approx(0.5, abs=0.1)

    def test_scaled_score_fisher_inv(self):
        """fisher_inv scaling: (log(y) - mu) / sigma^2 * sigma^2 = log(y) - mu."""
        y = np.exp(6.5)
        s_unit = self.dist.scaled_score(np.array([y]), self.params, scaling="unit")
        s_fi = self.dist.scaled_score(np.array([y]), self.params, scaling="fisher_inv")
        sigma = np.exp(0.5)
        assert s_fi["logmean"] == pytest.approx(s_unit["logmean"] * sigma**2)


# ---------------------------------------------------------------------------
# BetaGAS
# ---------------------------------------------------------------------------

class TestBetaGAS:
    def setup_method(self):
        self.dist = BetaGAS()
        self.params = {"mean": 0.65, "precision": 15.0}

    def test_score_at_mean(self):
        s = self.dist.score(np.array([0.65]), self.params)
        assert s["mean"] == pytest.approx(0.0, abs=1e-10)

    def test_score_above_mean(self):
        s = self.dist.score(np.array([0.8]), self.params)
        assert s["mean"] > 0

    def test_score_below_mean(self):
        s = self.dist.score(np.array([0.5]), self.params)
        assert s["mean"] < 0

    def test_score_magnitude(self):
        """phi * (y - mu) with phi=15, y=0.8, mu=0.65."""
        s = self.dist.score(np.array([0.8]), self.params)
        assert s["mean"] == pytest.approx(15.0 * (0.8 - 0.65))

    def test_fisher_positive(self):
        fi = self.dist.fisher(self.params)
        assert fi["mean"] > 0

    def test_fisher_formula(self):
        mu, phi = 0.65, 15.0
        fi = self.dist.fisher(self.params)
        expected = phi * mu * (1.0 - mu)
        assert fi["mean"] == pytest.approx(expected)

    def test_log_likelihood_finite(self):
        ll = self.dist.log_likelihood(np.array([0.65]), self.params)
        assert np.isfinite(float(ll))

    def test_log_likelihood_boundary_near(self):
        """Near-zero and near-one values should still return finite ll."""
        ll = self.dist.log_likelihood(np.array([0.001]), self.params)
        assert np.isfinite(float(ll))

    def test_link_unlink_mean(self):
        mu = 0.72
        f = self.dist.link("mean", mu)
        assert self.dist.unlink("mean", f) == pytest.approx(mu, rel=1e-6)

    def test_link_unlink_precision(self):
        phi = 20.0
        f = self.dist.link("precision", phi)
        assert self.dist.unlink("precision", f) == pytest.approx(phi)

    def test_initial_params(self):
        rng = np.random.default_rng(7)
        y = rng.beta(0.65 * 15, 0.35 * 15, 200)
        init = self.dist.initial_params(y)
        assert 0.0 < init["mean"] < 1.0
        assert init["precision"] >= 1.0


# ---------------------------------------------------------------------------
# ZIPGAS
# ---------------------------------------------------------------------------

class TestZIPGAS:
    def setup_method(self):
        self.dist = ZIPGAS()
        self.params = {"mean": 2.0, "zeroprob": 0.2}

    def test_score_zero_obs(self):
        """Score for y=0 should involve both components."""
        s = self.dist.score(np.array([0.0]), self.params)
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["zeroprob"])

    def test_score_positive_obs(self):
        s = self.dist.score(np.array([3.0]), self.params)
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["zeroprob"])

    def test_log_likelihood_zero(self):
        ll = self.dist.log_likelihood(np.array([0.0]), self.params)
        # P(0) = 0.2 + 0.8 * exp(-2)
        expected = np.log(0.2 + 0.8 * np.exp(-2.0))
        assert float(ll) == pytest.approx(expected, rel=1e-6)

    def test_log_likelihood_positive(self):
        ll = self.dist.log_likelihood(np.array([2.0]), self.params)
        from scipy.special import gammaln
        expected = np.log(0.8) + 2.0 * np.log(2.0) - 2.0 - gammaln(3.0)
        assert float(ll) == pytest.approx(expected, rel=1e-6)

    def test_log_likelihood_finite(self):
        for y in [0.0, 1.0, 3.0, 10.0]:
            ll = self.dist.log_likelihood(np.array([y]), self.params)
            assert np.isfinite(float(ll))

    def test_fisher_positive(self):
        fi = self.dist.fisher(self.params)
        assert fi["mean"] > 0
        assert fi["zeroprob"] > 0

    def test_link_unlink_mean(self):
        mu = 3.0
        f = self.dist.link("mean", mu)
        assert self.dist.unlink("mean", f) == pytest.approx(mu)

    def test_link_unlink_zeroprob(self):
        pi = 0.3
        f = self.dist.link("zeroprob", pi)
        assert self.dist.unlink("zeroprob", f) == pytest.approx(pi, rel=1e-6)

    def test_initial_params(self):
        rng = np.random.default_rng(12)
        y = rng.choice([0.0, 1.0, 2.0, 3.0], 100, p=[0.4, 0.3, 0.2, 0.1])
        init = self.dist.initial_params(y)
        assert init["mean"] > 0
        assert 0.0 < init["zeroprob"] < 1.0


# ---------------------------------------------------------------------------
# Distribution map
# ---------------------------------------------------------------------------

class TestDistributionMap:
    def test_all_keys_present(self):
        for key in ["poisson", "gamma", "negbin", "lognormal", "beta", "zip"]:
            assert key in DISTRIBUTION_MAP

    def test_all_instantiable(self):
        for dist_cls in DISTRIBUTION_MAP.values():
            d = dist_cls()
            assert hasattr(d, "score")
            assert hasattr(d, "fisher")
            assert hasattr(d, "log_likelihood")
            assert hasattr(d, "link")
            assert hasattr(d, "unlink")
