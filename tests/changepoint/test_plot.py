"""
Tests for plotting functions. Uses matplotlib non-interactive backend.
"""

import numpy as np
import pytest

# Use non-interactive backend before importing matplotlib
import matplotlib
matplotlib.use("Agg")

from insurance_dynamics.changepoint import FrequencyChangeDetector, LossRatioMonitor, RetrospectiveBreakFinder
from insurance_dynamics.changepoint.plot import (
    plot_regime_probs,
    plot_monitor,
    plot_run_length_heatmap,
    plot_retrospective_breaks,
)


def _make_freq_result():
    rng = np.random.default_rng(0)
    det = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=20.0, hazard=0.02, threshold=0.3)
    counts = np.concatenate([rng.poisson(30, 40), rng.poisson(90, 40)])
    exposure = np.full(80, 1000.0)
    return det.fit(counts, exposure, periods=[f"2020-{i:02d}" for i in range(80)])


def _make_monitor_result():
    rng = np.random.default_rng(7)
    exposure = 1000.0
    counts = np.concatenate([rng.poisson(30, 40), rng.poisson(90, 40)])
    exposures = np.full(80, exposure)
    monitor = LossRatioMonitor(
        freq_prior_alpha=1.0, freq_prior_beta=30.0, hazard=0.02, threshold=0.25
    )
    return monitor.monitor(claim_counts=counts, exposures=exposures)


class TestPlotRegimeProbs:
    def test_returns_figure(self):
        result = _make_freq_result()
        fig = plot_regime_probs(result)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_custom_threshold(self):
        result = _make_freq_result()
        fig = plot_regime_probs(result, threshold=0.5)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_custom_title(self):
        result = _make_freq_result()
        fig = plot_regime_probs(result, title="Test Chart")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_existing_ax(self):
        import matplotlib.pyplot as plt
        result = _make_freq_result()
        fig, ax = plt.subplots()
        returned_fig = plot_regime_probs(result, ax=ax)
        assert returned_fig is fig
        plt.close("all")


class TestPlotMonitor:
    def test_returns_figure(self):
        result = _make_monitor_result()
        fig = plot_monitor(result)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_no_results_raises(self):
        from insurance_dynamics.changepoint.result import MonitorResult
        import numpy as np
        result = MonitorResult(
            frequency_result=None,
            severity_result=None,
            combined_probs=np.array([]),
            detected_breaks=[],
            recommendation="monitor",
        )
        with pytest.raises(ValueError, match="no frequency or severity"):
            plot_monitor(result)


class TestPlotRunLengthHeatmap:
    def test_returns_figure(self):
        result = _make_freq_result()
        fig = plot_run_length_heatmap(result)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_max_run_length(self):
        result = _make_freq_result()
        fig = plot_run_length_heatmap(result, max_run_length=20)
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotRetrospectiveBreaks:
    def test_returns_figure(self):
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 40)])
        finder = RetrospectiveBreakFinder(n_bootstraps=50, penalty=3.0)
        result = finder.fit(signal)
        fig = plot_retrospective_breaks(signal, result)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_with_periods(self):
        rng = np.random.default_rng(1)
        signal = np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)])
        periods = list(range(60))
        finder = RetrospectiveBreakFinder(n_bootstraps=50, penalty=2.0)
        result = finder.fit(signal, periods=periods)
        fig = plot_retrospective_breaks(signal, result, periods=periods)
        import matplotlib.pyplot as plt
        plt.close("all")
