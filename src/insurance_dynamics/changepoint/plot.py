"""
Regime probability plots for insurance-changepoint.

All plotting functions return matplotlib Figure objects — no side effects,
no plt.show() calls. The caller decides what to do with the figure.

Usage
-----
>>> from insurance_changepoint import FrequencyChangeDetector
>>> from insurance_changepoint.plot import plot_regime_probs, plot_run_length_heatmap
>>> result = detector.fit(counts, exposures, periods)
>>> fig = plot_regime_probs(result)
>>> fig.savefig("monitoring_q4_2024.png", dpi=150)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from .result import ChangeResult, MonitorResult, BreakResult


def _check_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def plot_regime_probs(
    result: ChangeResult,
    figsize: tuple[float, float] = (12, 4),
    threshold: float | None = None,
    title: str | None = None,
    ax: Any = None,
) -> Any:
    """
    Plot changepoint probability over time with detected breaks marked.

    Parameters
    ----------
    result :
        Output from FrequencyChangeDetector.fit() or SeverityChangeDetector.fit().
    figsize :
        Figure size (width, height) in inches.
    threshold :
        Detection threshold line. If None, inferred from result.meta.
    title :
        Plot title. If None, a default is generated.
    ax :
        Existing matplotlib Axes to plot on. If None, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    thresh = threshold or result.meta.get("threshold", 0.3)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = np.arange(len(result.changepoint_probs))
    labels = result.periods

    ax.fill_between(x, 0, result.changepoint_probs, alpha=0.3, color="#1a3a6b",
                    label="P(changepoint)")
    ax.plot(x, result.changepoint_probs, color="#1a3a6b", linewidth=1.2)

    # Threshold line
    ax.axhline(thresh, color="#e74c3c", linestyle="--", linewidth=0.9,
               label=f"Threshold ({thresh:.2f})")

    # Mark detected breaks
    for brk in result.detected_breaks:
        ax.axvline(brk.period_index, color="#e74c3c", linestyle=":", linewidth=1.0,
                   alpha=0.7)
        ax.annotate(
            str(brk.period_label),
            xy=(brk.period_index, brk.probability),
            xytext=(brk.period_index + 0.5, min(brk.probability + 0.05, 0.95)),
            fontsize=8,
            color="#c0392b",
            ha="left",
        )

    # X-axis labels — thin out if many periods
    step = max(1, len(x) // 12)
    tick_positions = x[::step]
    tick_labels = [str(labels[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(changepoint)", fontsize=10)
    ax.set_xlabel("Period", fontsize=10)

    default_title = (
        f"Changepoint Probability — {result.detector_type.capitalize()}"
    )
    ax.set_title(title or default_title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_monitor(
    result: MonitorResult,
    figsize: tuple[float, float] = (12, 7),
    threshold: float | None = None,
    title: str | None = None,
) -> Any:
    """
    Two-panel plot for LossRatioMonitor output.

    Top panel: frequency changepoint probabilities.
    Bottom panel: severity changepoint probabilities.
    Combined breaks marked on both.

    Parameters
    ----------
    result :
        Output from LossRatioMonitor.monitor().
    figsize :
        Figure size.
    threshold :
        Detection threshold. Inferred from meta if None.
    title :
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    thresh = threshold or 0.3

    n_panels = sum([
        result.frequency_result is not None,
        result.severity_result is not None,
    ])

    if n_panels == 0:
        raise ValueError("MonitorResult has no frequency or severity results.")

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if result.frequency_result is not None:
        plot_regime_probs(
            result.frequency_result,
            threshold=thresh,
            ax=axes[panel_idx],
        )
        panel_idx += 1

    if result.severity_result is not None:
        plot_regime_probs(
            result.severity_result,
            threshold=thresh,
            ax=axes[panel_idx],
        )
        panel_idx += 1

    fig.suptitle(
        title or f"Loss Ratio Monitor — Recommendation: {result.recommendation.upper()}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_run_length_heatmap(
    result: ChangeResult,
    figsize: tuple[float, float] = (12, 6),
    max_run_length: int | None = None,
    title: str | None = None,
) -> Any:
    """
    Heatmap of run-length posterior distribution over time.

    Each column is a time period, each row is a run length. Dark colour = high
    probability. Detected breaks show up as vertical stripes near rl=0.

    Parameters
    ----------
    result :
        ChangeResult with run_length_probs populated.
    figsize :
        Figure size.
    max_run_length :
        Cap on run lengths shown. Defaults to T//2.
    title :
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    rl_probs = result.run_length_probs
    T, max_rl = rl_probs.shape

    cap = max_run_length or min(max_rl, T // 2 + 1)
    rl_probs_clipped = rl_probs[:, :cap]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        rl_probs_clipped.T,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="P(run length)")

    # X-axis
    step = max(1, T // 12)
    tick_positions = list(range(0, T, step))
    tick_labels = [str(result.periods[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Run length (periods)", fontsize=10)
    ax.set_xlabel("Period", fontsize=10)
    ax.set_title(
        title or f"Run-Length Posterior — {result.detector_type.capitalize()}",
        fontsize=11,
    )

    # Mark detected breaks
    for brk in result.detected_breaks:
        ax.axvline(brk.period_index, color="#e74c3c", linestyle="--",
                   linewidth=1.0, alpha=0.8)

    fig.tight_layout()
    return fig


def plot_retrospective_breaks(
    signal: np.ndarray,
    break_result: BreakResult,
    periods: list[Any] | None = None,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
    ylabel: str = "Value",
) -> Any:
    """
    Plot the original series with retrospective PELT break locations and CIs.

    Parameters
    ----------
    signal :
        Original time series.
    break_result :
        Output from RetrospectiveBreakFinder.fit().
    periods :
        Period labels. Defaults to integer index.
    figsize :
        Figure size.
    title :
        Plot title.
    ylabel :
        Y-axis label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    T = len(signal)
    x = np.arange(T)
    labs = [str(periods[i]) if periods else str(i) for i in range(T)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, signal, color="#2c3e50", linewidth=1.2, label="Series")

    for ci in break_result.break_cis:
        # Shaded CI
        ax.axvspan(ci.lower, ci.upper, alpha=0.2, color="#e74c3c")
        # Point estimate
        ax.axvline(ci.break_index, color="#e74c3c", linewidth=1.5,
                   linestyle="--", label="Break (PELT)")

    # Avoid duplicate legend entries
    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    step = max(1, T // 12)
    tick_positions = list(range(0, T, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labs[i] for i in tick_positions], rotation=45,
                       ha="right", fontsize=8)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("Period", fontsize=10)
    ax.set_title(
        title or f"Retrospective Break Detection — {break_result.model} model, "
                 f"{break_result.n_breaks} break(s)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
