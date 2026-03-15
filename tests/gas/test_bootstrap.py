"""Tests for parametric bootstrap confidence intervals."""

import numpy as np
import pytest

from insurance_dynamics.gas import GASModel, bootstrap_ci
from insurance_dynamics.gas.bootstrap import BootstrapCI
from insurance_dynamics.gas.datasets import load_motor_frequency


class TestBootstrapCI:
    def setup_method(self):
        data = load_motor_frequency(T=36, seed=1, trend_break=False)
        m = GASModel("poisson")
        self.result = m.fit(data.y, exposure=data.exposure)

    def test_bootstrap_via_result(self):
        ci = self.result.bootstrap_ci(n_boot=50, rng=np.random.default_rng(1))
        assert isinstance(ci, BootstrapCI)

    def test_lower_below_upper(self):
        ci = self.result.bootstrap_ci(n_boot=50, rng=np.random.default_rng(2))
        np.testing.assert_array_less(
            ci.filter_lower["mean"].values,
            ci.filter_upper["mean"].values + 1e-8,
        )

    def test_ci_length(self):
        ci = self.result.bootstrap_ci(n_boot=50, rng=np.random.default_rng(3))
        assert len(ci.filter_lower) == self.result.n_obs
        assert len(ci.filter_upper) == self.result.n_obs
        assert len(ci.filter_median) == self.result.n_obs

    def test_confidence_attribute(self):
        ci = self.result.bootstrap_ci(n_boot=50, confidence=0.90, rng=np.random.default_rng(4))
        assert ci.confidence == pytest.approx(0.90)

    def test_n_boot_attribute(self):
        n = 30
        ci = self.result.bootstrap_ci(n_boot=n, rng=np.random.default_rng(5))
        assert ci.n_boot <= n  # could be fewer if some failed

    def test_lower_positive(self):
        ci = self.result.bootstrap_ci(n_boot=50, rng=np.random.default_rng(6))
        assert np.all(ci.filter_lower["mean"].values > 0)

    def test_median_within_ci(self):
        ci = self.result.bootstrap_ci(n_boot=50, rng=np.random.default_rng(7))
        lower = ci.filter_lower["mean"].values
        upper = ci.filter_upper["mean"].values
        median = ci.filter_median["mean"].values
        assert np.all(lower <= median + 1e-8)
        assert np.all(median <= upper + 1e-8)

    def test_standalone_function(self):
        ci = bootstrap_ci(self.result, n_boot=30, rng=np.random.default_rng(8))
        assert isinstance(ci, BootstrapCI)


# ---------------------------------------------------------------------------
# P1-1 regression: bootstrap passes exposure to boot_model.fit()
# ---------------------------------------------------------------------------

class TestP11BootstrapExposure:
    """P1-1: bootstrap_ci must pass original exposure to each refit, otherwise
    Poisson counts are not on the same scale as the original fit."""

    def test_p1_1_exposure_stored_on_result(self):
        data = load_motor_frequency(T=24, seed=20, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        assert result._exposure is not None
        np.testing.assert_array_almost_equal(
            result._exposure, np.asarray(data.exposure, dtype=float)
        )

    def test_p1_1_bootstrap_with_exposure_produces_finite_ci(self):
        """With exposure properly passed, bootstrap CIs must be finite and positive."""
        data = load_motor_frequency(T=30, seed=21, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        ci = result.bootstrap_ci(n_boot=30, rng=np.random.default_rng(21))
        assert np.all(np.isfinite(ci.filter_lower["mean"].values))
        assert np.all(np.isfinite(ci.filter_upper["mean"].values))
        assert np.all(ci.filter_lower["mean"].values > 0)

    def test_p1_1_bootstrap_no_exposure_still_works(self):
        """Without exposure, bootstrap should still work (no regression)."""
        rng = np.random.default_rng(22)
        y = rng.poisson(3.0, 30).astype(float)
        m = GASModel("poisson")
        result = m.fit(y)
        assert result._exposure is None
        ci = result.bootstrap_ci(n_boot=20, rng=np.random.default_rng(23))
        assert isinstance(ci, BootstrapCI)

    def test_p1_1_bootstrap_ci_width_reasonable_with_exposure(self):
        """CI width should be non-trivially larger than zero for each period."""
        data = load_motor_frequency(T=36, seed=24, trend_break=False)
        m = GASModel("poisson")
        result = m.fit(data.y, exposure=data.exposure)
        ci = result.bootstrap_ci(n_boot=40, rng=np.random.default_rng(24))
        widths = ci.filter_upper["mean"].values - ci.filter_lower["mean"].values
        # All widths must be non-negative; at least some must be > 0
        assert np.all(widths >= -1e-8)
        assert np.mean(widths) > 1e-4
