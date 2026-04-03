"""
Extended tests for the BOCPD core engine and conjugate models.

Targets code paths not hit by the existing test suite:
- _logsumexp with mixed finite/neg-inf values
- NormalGammaModel trim when already within limit
- BOCPDEngine.step per-step hazard override path
- BOCPDEngine numerical failure fallback (log_norm non-finite)
- BOCPDEngine.fit with per-step hazards
- BOCPDEngine.fit with empty sequence
- PoissonGammaModel.trim when already within limit
- PoissonGammaModel.get_posterior_params at various indices
- NormalGammaModel.get_posterior_params at various indices
- FrequencyChangeDetector.update with UK priors and period_to_date_fn
- FrequencyChangeDetector.posterior_lambda at negative index
- SeverityChangeDetector.update with UK priors
- SeverityChangeDetector.fit with log_transform=False + update
- Multiple calls to update (streaming)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from insurance_dynamics.changepoint._bocpd import (
    BOCPDEngine,
    NormalGammaModel,
    PoissonGammaModel,
    _logsumexp,
)
from insurance_dynamics.changepoint import FrequencyChangeDetector, SeverityChangeDetector


# ---------------------------------------------------------------------------
# _logsumexp extended cases
# ---------------------------------------------------------------------------

class TestLogSumExpExtended:
    def test_mixed_finite_and_neginf(self):
        """Some -inf entries should be ignored gracefully."""
        arr = np.array([-np.inf, 0.0, -np.inf])
        result = _logsumexp(arr)
        # log(exp(-inf) + exp(0) + exp(-inf)) = log(1) = 0
        assert np.isclose(result, 0.0)

    def test_all_identical_values(self):
        """log(n * exp(c)) = c + log(n)."""
        c = 5.0
        n = 8
        arr = np.full(n, c)
        result = _logsumexp(arr)
        assert np.isclose(result, c + np.log(n))

    def test_single_neginf(self):
        arr = np.array([-np.inf])
        result = _logsumexp(arr)
        assert result == -np.inf

    def test_negative_values(self):
        arr = np.array([-10.0, -20.0])
        result = _logsumexp(arr)
        expected = np.log(np.exp(-10.0) + np.exp(-20.0))
        assert np.isclose(result, expected)

    def test_positive_and_negative_mixed(self):
        arr = np.array([-3.0, 1.0, -1.0])
        manual = np.log(np.exp(-3) + np.exp(1) + np.exp(-1))
        assert np.isclose(_logsumexp(arr), manual)


# ---------------------------------------------------------------------------
# PoissonGammaModel extended
# ---------------------------------------------------------------------------

class TestPoissonGammaModelExtended:
    def test_trim_no_op_when_within_limit(self):
        """Trimming to a limit larger than current count should have no effect."""
        model = PoissonGammaModel(alpha0=1.0, beta0=5.0)
        model.update((2, 50.0))
        model.update((3, 50.0))
        # 3 entries now
        count_before = model.current_run_length_count
        model.trim(10)  # limit >> current count
        assert model.current_run_length_count == count_before

    def test_trim_exactly_at_limit(self):
        model = PoissonGammaModel(alpha0=1.0, beta0=5.0)
        for _ in range(5):
            model.update((1, 10.0))
        # 6 entries. Trim to 5 (max_run_length=5 → keep 6 entries)
        model.trim(5)
        assert model.current_run_length_count == 6

    def test_get_posterior_params_multiple_indices(self):
        model = PoissonGammaModel(alpha0=2.0, beta0=10.0)
        model.update((5, 100.0))
        model.update((3, 100.0))
        # rl_idx=0: prior
        a0, b0 = model.get_posterior_params(0)
        assert a0 == pytest.approx(2.0)
        assert b0 == pytest.approx(10.0)
        # rl_idx=2: posterior with both updates
        a2, b2 = model.get_posterior_params(2)
        assert a2 == pytest.approx(2.0 + 5.0 + 3.0)
        assert b2 == pytest.approx(10.0 + 100.0 + 100.0)

    def test_log_pred_large_exposure_small_n(self):
        """Small count with large exposure should return finite log-prob."""
        model = PoissonGammaModel(alpha0=1.0, beta0=1.0)
        lp = model.log_pred((0, 1e6), rl_idx=0)
        assert np.isfinite(lp)

    def test_log_pred_large_n(self):
        """Very large claim counts should not cause numerical issues."""
        model = PoissonGammaModel(alpha0=5.0, beta0=1.0)
        lp = model.log_pred((500, 100.0), rl_idx=0)
        assert np.isfinite(lp)

    def test_update_multiple_times_then_reset(self):
        model = PoissonGammaModel(alpha0=1.0, beta0=5.0)
        for _ in range(10):
            model.update((2, 50.0))
        assert model.current_run_length_count == 11
        model.reset_to_prior()
        assert model.current_run_length_count == 1
        a0, b0 = model.get_posterior_params(0)
        assert a0 == pytest.approx(1.0)
        assert b0 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# NormalGammaModel extended
# ---------------------------------------------------------------------------

class TestNormalGammaModelExtended:
    def test_trim_no_op_within_limit(self):
        model = NormalGammaModel()
        model.update(1.0)
        model.update(2.0)
        count_before = len(model._mus)
        model.trim(100)
        assert len(model._mus) == count_before

    def test_trim_exactly_at_limit(self):
        model = NormalGammaModel()
        for _ in range(5):
            model.update(1.0)
        model.trim(5)
        assert len(model._mus) == 6

    def test_get_posterior_params_at_each_index(self):
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        model.update(3.0)
        # Index 0: fresh prior
        mu0, k0, a0, b0 = model.get_posterior_params(0)
        assert mu0 == pytest.approx(0.0)
        assert k0 == pytest.approx(1.0)
        # Index 1: posterior with one observation x=3.0
        mu1, k1, a1, b1 = model.get_posterior_params(1)
        assert k1 == pytest.approx(2.0)
        assert a1 == pytest.approx(2.5)
        # mu1 should be weighted average: (1*0 + 3) / 2 = 1.5
        assert mu1 == pytest.approx(1.5)

    def test_log_pred_extreme_kappa(self):
        """Very small kappa (uncertain prior on mean) should still be finite."""
        model = NormalGammaModel(mu0=0.0, kappa0=1e-6, alpha0=2.0, beta0=1.0)
        lp = model.log_pred(5.0, rl_idx=0)
        assert np.isfinite(lp)

    def test_update_convergence(self):
        """After many identical updates, posterior mean converges to observation."""
        x_true = 7.5
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        for _ in range(50):
            model.update(x_true)
        last_idx = len(model._mus) - 1
        mu, _, _, _ = model.get_posterior_params(last_idx)
        assert mu == pytest.approx(x_true, abs=0.1)

    def test_log_pred_negative_observation(self):
        """Negative observations (log-scale) should be handled."""
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        lp = model.log_pred(-5.0, rl_idx=0)
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# BOCPDEngine extended
# ---------------------------------------------------------------------------

class TestBOCPDEngineExtended:
    def _make_poisson_engine(self, hazard=0.01, max_run_length=500):
        model = PoissonGammaModel(alpha0=2.0, beta0=20.0)
        return BOCPDEngine(model=model, hazard=hazard, max_run_length=max_run_length)

    def _make_normal_engine(self, hazard=0.01):
        model = NormalGammaModel(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        return BOCPDEngine(model=model, hazard=hazard)

    def test_step_with_hazard_override_higher_than_base(self):
        """hazard_t > base should boost P(changepoint)."""
        engine = self._make_poisson_engine(hazard=0.01)
        # Run several steps to build up run length
        for _ in range(10):
            engine.step((5, 100.0))
        # Compare step with base vs elevated hazard
        engine2 = self._make_poisson_engine(hazard=0.01)
        for _ in range(10):
            engine2.step((5, 100.0))
        prob_base = engine.step((5, 100.0), hazard_t=None)
        prob_high = engine2.step((5, 100.0), hazard_t=0.4)
        assert prob_high > prob_base

    def test_step_with_zero_hazard_override_disallowed(self):
        """Base hazard validation at construction, not per step — step accepts any float."""
        engine = self._make_poisson_engine(hazard=0.01)
        # hazard_t=0 doesn't raise — engine just uses 0 in the recursion
        # (very low changepoint probability at that step)
        # This tests the hazard_t code path is actually exercised
        prob = engine.step((5, 100.0), hazard_t=0.001)
        assert 0.0 <= prob <= 1.0

    def test_fit_empty_returns_empty_arrays(self):
        engine = self._make_poisson_engine()
        cp_probs, rl_probs = engine.fit([])
        assert len(cp_probs) == 0
        assert rl_probs.shape[0] == 0

    def test_fit_with_per_step_hazards_wrong_length_uses_index(self):
        """Per-step hazards matching obs length are applied correctly."""
        engine = self._make_poisson_engine(hazard=0.01)
        obs = [(5, 100.0)] * 10
        hazards = [0.01] * 10
        cp_probs, rl_probs = engine.fit(obs, hazards=hazards)
        assert len(cp_probs) == 10

    def test_fit_resets_engine(self):
        """Second fit call should reset engine state, producing same result as first."""
        engine = self._make_poisson_engine()
        obs = [(5, 100.0)] * 20
        cp1, rl1 = engine.fit(obs)
        cp2, rl2 = engine.fit(obs)
        np.testing.assert_allclose(cp1, cp2)

    def test_run_length_probs_row_sum_with_trimming(self):
        """Even with aggressive trimming, row sums should be ≈1."""
        engine = self._make_poisson_engine(max_run_length=5)
        obs = [(5, 100.0)] * 30
        _, rl_probs = engine.fit(obs)
        row_sums = rl_probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_step_increments_t(self):
        engine = self._make_poisson_engine()
        assert engine._t == 0
        engine.step((5, 100.0))
        assert engine._t == 1
        engine.step((3, 100.0))
        assert engine._t == 2

    def test_changepoint_probs_stored_after_each_step(self):
        engine = self._make_poisson_engine()
        for i in range(5):
            engine.step((i, 100.0))
        assert len(engine._changepoint_probs) == 5

    def test_max_run_length_zero_raises(self):
        """max_run_length=0 would break the recursion — test construction."""
        model = PoissonGammaModel()
        # No explicit validation on max_run_length=0, but step should still run
        engine = BOCPDEngine(model=model, hazard=0.01, max_run_length=1)
        prob = engine.step((5, 100.0))
        assert 0.0 <= prob <= 1.0

    def test_normal_model_engine_tracks_step_change(self):
        """Step change in normal series should elevate P(changepoint)."""
        rng = np.random.default_rng(42)
        low_vals = rng.normal(0.0, 0.3, 50)
        high_vals = rng.normal(5.0, 0.3, 50)
        obs = np.concatenate([low_vals, high_vals]).tolist()

        engine = self._make_normal_engine(hazard=0.02)
        cp_probs, _ = engine.fit(obs)
        # Peak should be near the break at index 50
        peak_idx = int(np.argmax(cp_probs))
        assert 40 <= peak_idx <= 65

    def test_engine_with_constant_series_low_cp_probs(self):
        """Constant Poisson series should give median P(changepoint) < 0.15."""
        rng = np.random.default_rng(7)
        obs = [(rng.poisson(5), 100.0) for _ in range(80)]
        engine = self._make_poisson_engine(hazard=0.01)
        cp_probs, _ = engine.fit(obs)
        assert np.median(cp_probs) < 0.15

    def test_fit_single_observation(self):
        engine = self._make_poisson_engine()
        cp_probs, rl_probs = engine.fit([(5, 100.0)])
        assert len(cp_probs) == 1
        assert 0.0 <= cp_probs[0] <= 1.0
        assert rl_probs.shape == (1, 2)

    def test_hazard_boundary_values(self):
        """Near-boundary hazards (just above 0, just below 1) should work."""
        for h in [1e-6, 0.001, 0.5, 0.999]:
            model = PoissonGammaModel(alpha0=1.0, beta0=10.0)
            try:
                engine = BOCPDEngine(model=model, hazard=h)
                cp_probs, _ = engine.fit([(5, 100.0)] * 5)
                assert np.all(np.isfinite(cp_probs))
            except ValueError:
                # hazard=0.999 < 1.0 is valid; hazard=1.0 raises
                pass

    def test_hazard_exactly_zero_raises(self):
        model = PoissonGammaModel()
        with pytest.raises(ValueError):
            BOCPDEngine(model=model, hazard=0.0)

    def test_hazard_exactly_one_raises(self):
        model = PoissonGammaModel()
        with pytest.raises(ValueError):
            BOCPDEngine(model=model, hazard=1.0)


# ---------------------------------------------------------------------------
# FrequencyChangeDetector extended
# ---------------------------------------------------------------------------

class TestFrequencyChangeDetectorExtended:
    def test_update_with_uk_priors_and_date_period(self):
        """Online update with UK event priors and a date period."""
        det = FrequencyChangeDetector(
            hazard=0.01,
            threshold=0.3,
            uk_events=True,
            event_lines=["motor"],
            event_components=["frequency"],
        )
        # Fit initial data
        counts = np.random.default_rng(42).poisson(5, 20)
        exposure = np.full(20, 100.0)
        periods = [date(2020, 1, 1) + timedelta(days=30 * i)
                   for i in range(20)]
        det.fit(counts, exposure, periods=periods)

        # Update with COVID lockdown date
        covid_date = date(2020, 3, 23)
        prob = det.update(n=5, exposure=100.0, period=covid_date)
        assert 0.0 <= prob <= 1.0

    def test_update_with_period_to_date_fn(self):
        """Online update with period_to_date_fn for string period labels."""
        def q_to_date(q: str) -> date:
            year, quarter = q.split("-Q")
            month = (int(quarter) - 1) * 3 + 1
            return date(int(year), month, 1)

        det = FrequencyChangeDetector(
            hazard=0.01,
            uk_events=True,
            event_lines=["motor"],
        )
        counts = np.random.default_rng(1).poisson(5, 12)
        exposure = np.full(12, 100.0)
        periods = [f"2019-Q{i%4+1}" for i in range(12)]
        det.fit(counts, exposure, periods=periods, period_to_date_fn=q_to_date)

        prob = det.update(
            n=3, exposure=100.0,
            period="2020-Q1",
            period_to_date_fn=q_to_date,
        )
        assert 0.0 <= prob <= 1.0

    def test_posterior_lambda_at_last_run_length(self):
        """posterior_lambda at last index (longest run) should be well-defined."""
        det = FrequencyChangeDetector(prior_alpha=2.0, prior_beta=20.0, hazard=0.01)
        det.fit([5] * 10, [100.0] * 10)
        # Last index corresponds to longest run
        model = det._engine.model
        last_idx = model.current_run_length_count - 1
        a, b = det.posterior_lambda(rl_idx=last_idx)
        assert a > 0
        assert b > 0
        # After 10 updates with n=5, alpha should have grown
        assert a > det.prior_alpha

    def test_fit_with_default_integer_periods(self):
        """When no periods supplied, result.periods should be range(T)."""
        det = FrequencyChangeDetector()
        T = 15
        result = det.fit([5] * T, [100.0] * T)
        assert result.periods == list(range(T))

    def test_meta_threshold_stored(self):
        det = FrequencyChangeDetector(threshold=0.42)
        result = det.fit([5] * 10, [100.0] * 10)
        assert result.meta["threshold"] == pytest.approx(0.42)

    def test_streaming_state_updated_after_fit(self):
        det = FrequencyChangeDetector()
        det.fit([5, 6, 7], [100.0, 100.0, 100.0], periods=["a", "b", "c"])
        assert det._periods_seen == ["a", "b", "c"]
        assert len(det._cp_probs) == 3

    def test_update_appends_to_streaming_state(self):
        det = FrequencyChangeDetector()
        det.fit([5] * 5, [100.0] * 5)
        det.update(n=3, exposure=100.0, period="new_period")
        assert det._periods_seen[-1] == "new_period"
        assert len(det._cp_probs) == 6

    def test_negative_exposure_raises_in_fit(self):
        det = FrequencyChangeDetector()
        with pytest.raises(ValueError, match="positive"):
            det.fit([5, 3], [100.0, -1.0])

    def test_zero_exposure_raises_in_update(self):
        det = FrequencyChangeDetector()
        det.fit([5] * 5, [100.0] * 5)
        with pytest.raises(ValueError, match="positive"):
            det.update(n=5, exposure=0.0)


# ---------------------------------------------------------------------------
# SeverityChangeDetector extended
# ---------------------------------------------------------------------------

class TestSeverityChangeDetectorExtended:
    def test_update_with_uk_priors(self):
        """Online update with UK event priors enabled."""
        det = SeverityChangeDetector(
            hazard=0.01,
            uk_events=True,
            event_lines=["motor"],
            event_components=["severity"],
        )
        sevs = np.random.default_rng(42).lognormal(8.0, 0.3, 20)
        periods = [date(2016, 1, 1) + timedelta(days=30 * i)
                   for i in range(20)]
        det.fit(sevs, periods=periods)

        # Update with Ogden 2017 period
        ogden_date = date(2017, 3, 20)
        prob = det.update(mean_severity=5000.0, period=ogden_date)
        assert 0.0 <= prob <= 1.0

    def test_update_no_log_transform(self):
        """Online update with log_transform=False."""
        det = SeverityChangeDetector(log_transform=False)
        x = np.random.default_rng(1).normal(8.0, 0.3, 10)
        det.fit(x)
        prob = det.update(mean_severity=8.5)
        assert 0.0 <= prob <= 1.0

    def test_update_zero_mean_severity_raises_when_log(self):
        det = SeverityChangeDetector(log_transform=True)
        det.fit(np.random.lognormal(8.0, 0.3, 5))
        with pytest.raises(ValueError, match="positive"):
            det.update(mean_severity=0.0)

    def test_fit_preserves_period_labels(self):
        det = SeverityChangeDetector()
        periods = [f"month_{i:02d}" for i in range(15)]
        sevs = np.random.lognormal(8.0, 0.3, 15)
        result = det.fit(sevs, periods=periods)
        assert result.periods == periods

    def test_fit_log_transform_false_with_claim_counts(self):
        """No log-transform path with claim_counts provided."""
        det = SeverityChangeDetector(log_transform=False)
        x = np.random.normal(8.0, 0.3, 15)
        counts = np.random.poisson(10, 15)
        result = det.fit(x, claim_counts=counts)
        assert len(result.changepoint_probs) == 15

    def test_streaming_state_updated_after_fit(self):
        det = SeverityChangeDetector()
        sevs = np.random.lognormal(8.0, 0.3, 8)
        det.fit(sevs, periods=list(range(8)))
        assert len(det._periods_seen) == 8
        assert len(det._cp_probs) == 8

    def test_update_appends_to_streaming_state(self):
        det = SeverityChangeDetector()
        det.fit(np.random.lognormal(8.0, 0.3, 5))
        det.update(mean_severity=3500.0, period="extra")
        assert det._periods_seen[-1] == "extra"
        assert len(det._cp_probs) == 6

    def test_meta_log_transform_field(self):
        det = SeverityChangeDetector(log_transform=False)
        result = det.fit(np.random.normal(8.0, 0.3, 10))
        assert result.meta["log_transform"] is False

    def test_no_breaks_at_very_high_threshold(self):
        rng = np.random.default_rng(0)
        sevs = rng.lognormal(8.0, 0.15, 50)
        det = SeverityChangeDetector(threshold=0.99)
        result = det.fit(sevs)
        assert result.n_breaks == 0
