"""
Tests for the core BOCPD engine and conjugate models.
"""

import numpy as np
import pytest
from scipy.special import gammaln

from insurance_dynamics.changepoint._bocpd import (
    PoissonGammaModel,
    NormalGammaModel,
    BOCPDEngine,
    _logsumexp,
)


# ─── _logsumexp ──────────────────────────────────────────────────────────────

class TestLogSumExp:
    def test_single_value(self):
        assert np.isclose(_logsumexp(np.array([0.0])), 0.0)

    def test_two_values(self):
        # log(exp(0) + exp(0)) = log(2)
        result = _logsumexp(np.array([0.0, 0.0]))
        assert np.isclose(result, np.log(2))

    def test_large_values_stable(self):
        # Should not overflow
        arr = np.array([1000.0, 1001.0, 999.0])
        result = _logsumexp(arr)
        assert np.isfinite(result)
        # Manual: log(exp(1000)+exp(1001)+exp(999))
        # = 1001 + log(exp(-1)+1+exp(-2)) ≈ 1001.46
        assert 1001.0 < result < 1002.0

    def test_empty_returns_neginf(self):
        result = _logsumexp(np.array([]))
        assert result == -np.inf

    def test_all_neginf(self):
        result = _logsumexp(np.array([-np.inf, -np.inf]))
        assert result == -np.inf


# ─── PoissonGammaModel ────────────────────────────────────────────────────────

class TestPoissonGammaModel:
    def test_log_pred_prior(self):
        model = PoissonGammaModel(alpha0=2.0, beta0=10.0)
        # At rl_idx=0 (prior), should give finite log-prob
        lp = model.log_pred((3, 100.0), rl_idx=0)
        assert np.isfinite(lp)

    def test_log_pred_zero_claims(self):
        model = PoissonGammaModel(alpha0=1.0, beta0=10.0)
        lp = model.log_pred((0, 100.0), rl_idx=0)
        assert np.isfinite(lp)
        # P(0 claims) should be > 0
        assert lp > -np.inf

    def test_log_pred_increases_with_higher_exposure(self):
        """More exposure should make zero claims LESS likely."""
        model = PoissonGammaModel(alpha0=2.0, beta0=5.0)
        lp_small = model.log_pred((0, 10.0), rl_idx=0)
        lp_large = model.log_pred((0, 1000.0), rl_idx=0)
        assert lp_small > lp_large

    def test_update_adds_entries(self):
        model = PoissonGammaModel(alpha0=1.0, beta0=5.0)
        assert model.current_run_length_count == 1
        model.update((2, 50.0))
        assert model.current_run_length_count == 2
        model.update((1, 50.0))
        assert model.current_run_length_count == 3

    def test_update_sufficient_stats(self):
        """After one update with (n=5, e=100), alpha at rl=1 should be alpha0+5."""
        model = PoissonGammaModel(alpha0=2.0, beta0=10.0)
        model.update((5, 100.0))
        # rl_idx=0 is the new prior (after changepoint), rl_idx=1 has the data
        a1, b1 = model.get_posterior_params(1)
        assert np.isclose(a1, 2.0 + 5.0)
        assert np.isclose(b1, 10.0 + 100.0)

    def test_update_prior_entry_fresh(self):
        """After update, rl_idx=0 should still be the prior."""
        model = PoissonGammaModel(alpha0=2.0, beta0=10.0)
        model.update((5, 100.0))
        a0, b0 = model.get_posterior_params(0)
        assert np.isclose(a0, 2.0)
        assert np.isclose(b0, 10.0)

    def test_trim(self):
        model = PoissonGammaModel(alpha0=1.0, beta0=5.0)
        for i in range(20):
            model.update((1, 10.0))
        assert model.current_run_length_count == 21
        model.trim(10)
        assert model.current_run_length_count == 11

    def test_reset_to_prior(self):
        model = PoissonGammaModel(alpha0=1.0, beta0=5.0)
        for i in range(5):
            model.update((1, 10.0))
        model.reset_to_prior()
        assert model.current_run_length_count == 1

    def test_log_pred_valid_negbinomial(self):
        """Check log_pred sums to ≤ 1 (probability) across several n values."""
        model = PoissonGammaModel(alpha0=3.0, beta0=20.0)
        # Approximate: sum P(n=0..100) should be close to 1
        total = sum(np.exp(model.log_pred((n, 50.0), 0)) for n in range(200))
        assert 0.98 < total <= 1.001


# ─── NormalGammaModel ─────────────────────────────────────────────────────────

class TestNormalGammaModel:
    def test_log_pred_prior(self):
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        lp = model.log_pred(0.5, rl_idx=0)
        assert np.isfinite(lp)

    def test_log_pred_finite_for_extreme_values(self):
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        lp_near = model.log_pred(0.1, 0)
        lp_far = model.log_pred(100.0, 0)
        assert np.isfinite(lp_near)
        assert np.isfinite(lp_far)
        assert lp_near > lp_far  # closer to prior mean = higher prob

    def test_update_adds_entries(self):
        model = NormalGammaModel()
        assert len(model._mus) == 1
        model.update(1.5)
        assert len(model._mus) == 2
        model.update(2.0)
        assert len(model._mus) == 3

    def test_posterior_mean_tracks_data(self):
        """After several identical observations, posterior mean should shift toward them."""
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        for _ in range(10):
            model.update(5.0)
        # At the longest run length, mu should be close to 5.0
        last_idx = len(model._mus) - 1
        mu, _, _, _ = model.get_posterior_params(last_idx)
        assert mu > 3.0  # should be pulled toward 5.0

    def test_reset_to_prior(self):
        model = NormalGammaModel(mu0=1.0)
        for _ in range(5):
            model.update(3.0)
        model.reset_to_prior()
        assert len(model._mus) == 1
        assert np.isclose(model._mus[0], 1.0)

    def test_trim(self):
        model = NormalGammaModel()
        for _ in range(15):
            model.update(1.0)
        model.trim(5)
        assert len(model._mus) == 6


# ─── BOCPDEngine ──────────────────────────────────────────────────────────────

class TestBOCPDEngine:
    def _make_freq_engine(self, hazard=0.01):
        model = PoissonGammaModel(alpha0=2.0, beta0=20.0)
        return BOCPDEngine(model=model, hazard=hazard)

    def test_hazard_validation(self):
        model = PoissonGammaModel()
        with pytest.raises(ValueError):
            BOCPDEngine(model=model, hazard=0.0)
        with pytest.raises(ValueError):
            BOCPDEngine(model=model, hazard=1.1)

    def test_single_step_returns_probability(self):
        engine = self._make_freq_engine()
        prob = engine.step((5, 100.0))
        assert 0.0 <= prob <= 1.0

    def test_fit_returns_correct_shape(self):
        engine = self._make_freq_engine()
        obs = [(np.random.poisson(5), 100.0) for _ in range(30)]
        cp_probs, rl_probs = engine.fit(obs)
        assert cp_probs.shape == (30,)
        assert rl_probs.shape[0] == 30

    def test_changepoint_probs_in_range(self):
        engine = self._make_freq_engine()
        obs = [(np.random.poisson(5), 100.0) for _ in range(50)]
        cp_probs, _ = engine.fit(obs)
        assert np.all(cp_probs >= 0.0)
        assert np.all(cp_probs <= 1.0)

    def test_detects_obvious_break(self):
        """
        Simulate 50 periods at λ=0.05 then 50 at λ=0.15.
        The engine should show elevated P(changepoint) around period 50.
        """
        rng = np.random.default_rng(42)
        exposure = 1000.0
        counts_pre = rng.poisson(0.05 * exposure, size=50)
        counts_post = rng.poisson(0.15 * exposure, size=50)
        counts = np.concatenate([counts_pre, counts_post])

        obs = [(c, exposure) for c in counts]

        engine = self._make_freq_engine(hazard=0.02)
        cp_probs, _ = engine.fit(obs)

        # Peak should be around index 50 (±10)
        peak_idx = int(np.argmax(cp_probs))
        assert 40 <= peak_idx <= 65, (
            f"Peak at {peak_idx}, expected near 50. "
            f"Max prob: {cp_probs.max():.3f}"
        )
        assert cp_probs[peak_idx] > 0.2, (
            f"Peak probability {cp_probs.max():.3f} too low"
        )

    def test_no_breaks_constant_series(self):
        """Constant Poisson series should produce low changepoint probabilities."""
        rng = np.random.default_rng(123)
        exposure = 500.0
        counts = rng.poisson(0.04 * exposure, size=60)
        obs = [(c, exposure) for c in counts]

        engine = self._make_freq_engine(hazard=0.01)
        cp_probs, _ = engine.fit(obs)

        # Median should be low — no structural break
        assert np.median(cp_probs) < 0.3

    def test_per_step_hazard_override(self):
        """A very high hazard at a specific step should boost P(changepoint)."""
        engine = self._make_freq_engine(hazard=0.01)
        obs = [(5, 100.0)] * 20
        hazards = [0.01] * 10 + [0.49] + [0.01] * 9
        cp_probs, _ = engine.fit(obs, hazards=hazards)
        # Period 10 (index 10) should be highest
        # (though the data is flat, the prior pushes it)
        assert cp_probs[10] > cp_probs[0]

    def test_reset_reinitialises(self):
        engine = self._make_freq_engine()
        engine.fit([(5, 100.0)] * 10)
        engine.reset()
        assert engine._t == 0
        assert len(engine._log_R) == 1

    def test_run_length_probs_normalised(self):
        """Each row of run_length_probs should sum to ≈1."""
        engine = self._make_freq_engine()
        obs = [(5, 100.0)] * 20
        _, rl_probs = engine.fit(obs)
        row_sums = rl_probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_exposure_weighting(self):
        """
        Two periods with same lambda but different exposure should give
        different log-predictive probabilities.
        """
        model = PoissonGammaModel(alpha0=2.0, beta0=10.0)
        # 5 claims / 100 exposure vs 5 claims / 10 exposure — very different rates
        lp_high_exposure = model.log_pred((5, 100.0), 0)
        lp_low_exposure = model.log_pred((5, 10.0), 0)
        assert lp_high_exposure != lp_low_exposure

    def test_max_run_length_bounds_memory(self):
        model = PoissonGammaModel()
        engine = BOCPDEngine(model=model, hazard=0.01, max_run_length=20)
        obs = [(5, 100.0)] * 100
        engine.fit(obs)
        # Internal log_R should not grow beyond max_run_length+1
        assert len(engine._log_R) <= 21
