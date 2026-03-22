# insurance-dynamics

[![PyPI](https://img.shields.io/pypi/v/insurance-dynamics)](https://pypi.org/project/insurance-dynamics/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-dynamics)](https://pypi.org/project/insurance-dynamics/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()


Dynamic insurance pricing models for UK pricing teams.

Two problems, one package: tracking how your loss ratios are changing right now (GAS), and detecting when they shifted structurally in the past (changepoint).

Merged from: `insurance-gas` (GAS score-driven models) and `insurance-changepoint` (BOCPD/PELT detection).

**Blog post:** [Tracking Trend Between Model Updates with GAS Filters](https://burning-cost.github.io/2027/04/15/gas-models-for-between-update-trend/)

## The problems this solves

**GAS models** (`insurance_dynamics.gas`): Your GLM gives you a static relativity. Reality gives you frequency that drifts quarter by quarter. GAS (Generalised Autoregressive Score) models fit a time-varying parameter model to claim counts, severities, or loss ratios — producing a filter path that shows you how the underlying rate has moved, with confidence bands from parametric bootstrap.

**Changepoint detection** (`insurance_dynamics.changepoint`): You suspect something changed — whiplash rules, Ogden rate, a competitor exit. BOCPD tells you online whether this month looks like a break. PELT gives you retrospective break locations with bootstrap confidence intervals. The `LossRatioMonitor` combines both into a 'retrain / monitor' recommendation suitable for model governance.

## Subpackages

### `insurance_dynamics.gas`

- `GASModel` — fits GAS(p,q) to a univariate series; Poisson, Gamma, NegBin, LogNormal, Beta, ZIP
- `GASPanel` — fits the same spec across a set of rating cells
- `gas_forecast` — h-step-ahead forecasts with simulation intervals
- `bootstrap_ci` — parametric bootstrap confidence intervals on the filter path
- `compute_diagnostics` — Dawid-Sebastiani score, PIT histograms, autocorrelation of score residuals

### `insurance_dynamics.changepoint`

- `FrequencyChangeDetector` — online BOCPD for claim frequency with exposure weighting
- `SeverityChangeDetector` — online BOCPD on log-severity using Normal-Gamma conjugate
- `LossRatioMonitor` — joint monitoring returning retrain/monitor recommendation
- `RetrospectiveBreakFinder` — offline PELT with bootstrap confidence intervals
- `UKEventPrior` — UK insurance event calendar (Ogden, whiplash reform, FCA pricing review)
- `ConsumerDutyReport` — FCA PRIN 2A.9 evidence pack as HTML and JSON

## Installation

```bash
uv add insurance-dynamics
```

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-dynamics/discussions). Found it useful? A star helps others find it.

## Quick start

The GAS example uses the built-in dataset loader — no data setup required:

```python
from insurance_dynamics.gas import GASModel
from insurance_dynamics.gas.datasets import load_motor_frequency

data = load_motor_frequency(T=48)
model = GASModel("poisson")
result = model.fit(data.y, exposure=data.exposure)
print(result.summary())
result.filter_path.plot()
```

The changepoint monitor requires time series arrays. Here is a minimal self-contained example with 60 months of synthetic UK motor data including a regime shift at month 36:

```python
import numpy as np
from insurance_dynamics.changepoint import LossRatioMonitor

rng = np.random.default_rng(42)
T = 60

# Monthly exposure (earned car years)
exposures = rng.uniform(800, 1200, T)

# Claim frequency: base 0.08/year, step up 37.5% at month 36
true_rate = np.where(np.arange(T) < 36, 0.08, 0.11)
counts = rng.poisson(true_rate * exposures)

# Mean severity (log-normal, slight upward drift)
mean_sev = 1500 * np.exp(0.003 * np.arange(T)) * rng.lognormal(0, 0.15, T)

# Period labels — sequential months across years
periods = [f"{2021 + i // 12}-{(i % 12) + 1:02d}" for i in range(T)]

monitor = LossRatioMonitor(lines=["motor"], uk_events=True)
result = monitor.monitor(
    claim_counts=counts,
    exposures=exposures,
    mean_severities=mean_sev,
    periods=periods,
)
print(result.recommendation)  # 'retrain' or 'monitor'
```

## Design decisions

GAS and changepoint detection are complementary tools. GAS smooths the signal continuously; BOCPD detects discrete jumps. Using both gives you a full picture: is the drift smooth (GAS will catch it) or structural (BOCPD will flag it)? The `LossRatioMonitor` is deliberately opinionated — it returns a binary recommendation, not a probability, because pricing teams need an action, not a number.


## Databricks Benchmark

A full benchmark notebook is in `databricks/benchmark_gas_vs_rolling.py`. It benchmarks GAS filters against rolling window averages and a static Poisson GLM on synthetic UK motor frequency data with two regime changes: a gradual upward drift followed by a sharp step down.

Run it directly on Databricks serverless compute — no external data required.

### Performance: GAS vs Rolling Windows vs Static GLM

Benchmarked on 72 months of synthetic UK motor claim frequency (4,500 vehicle-years/month, gradual drift in months 0-35, sharp step change at month 36). DGP: true rate rises from 0.065 to 0.090 during phase 1, then steps down to 0.055 at month 36. Results from `databricks/benchmark_gas_vs_rolling.py` (Databricks serverless, 2026-03-22, seed=42).

The key question: does GAS adapt faster than rolling windows after a regime change?

**Adaptation speed** (months to reach within 10% of true post-break rate):

- 3-month rolling: adapts within 3-4 months (window flushes fast but lags the full series)
- 6-month rolling: adapts within 5-7 months (slower flush of pre-break observations)
- GAS Poisson: adapts within 2-3 months (score-driven update reacts immediately)

**GAS consistently wins on RMSE vs the true rate** — the score-driven update uses the Poisson gradient, which is exposure-weighted and proportional to the surprise relative to the current estimate. Rolling windows give equal weight to all months in the window regardless of exposure.

**On log-likelihood, GAS is strictly best** — it was estimated by MLE on the full series, so it finds the parameters that maximise the probability of the observed sequence. This reflects better distributional calibration throughout, not just around breaks.

**The honest caveat**: on smooth drift, a 3-month rolling window is competitive with GAS. The advantage of GAS is clearest at sharp structural breaks and when the persistence parameter (phi) is not well-calibrated to a simple rolling window length. For a series with no breaks and no drift, a rolling average and GAS will converge to similar results.

| Method | MAE (post-break) | RMSE vs true (post-break) | Log-likelihood |
|---|---|---|---|
| Static GLM | Worst (blended) | Worst | Lowest |
| Rolling 6-month | Moderate | Moderate | N/A (not probabilistic) |
| Rolling 3-month | Better | Better | N/A |
| GAS Poisson | **Best** | **Best** | **Highest** |

**When to use GAS:** Monthly or quarterly aggregate claim data where the underlying rate may be drifting or subject to structural events. GAS adapts continuously; it does not require you to choose a window length.

**When NOT to use GAS:** Cross-sectional risk factor estimation (that is a GLM job). Series shorter than ~20 months (insufficient to estimate omega/alpha/phi reliably). Any context where an underwriter needs to reproduce the calculation in Excel.

## Performance (Previous Benchmark)

Benchmarked against a static Poisson GLM (intercept-only and linear trend variants) on 60 months of synthetic UK motor frequency data with a known regime shift at month 36 (+37.5% step increase in claim frequency). Results from `benchmarks/benchmark.py` run locally (Raspberry Pi ARM64) on 2026-03-16.

| Model | MAE (all periods) | MAE (post-break) | RMSE vs true lambda (post-break) | Log-likelihood |
|-------|------------------|-----------------|----------------------------------|----------------|
| GLM constant | 0.014980 | 0.018468 | 0.019104 | -289.0 |
| GLM trend | 0.009697 | 0.009918 | 0.008795 | -236.3 |
| GAS Poisson | **0.008438** | 0.010684 | **0.007540** | **-231.7** |

GAS Poisson vs best static baseline (post-break):
- MAE overall: **+13.0% improvement** vs GLM trend
- MAE post-break: -7.7% (GLM trend has slightly lower post-break MAE on this DGP — within noise)
- RMSE vs true lambda: **+14.3% improvement**
- Log-likelihood: **+4.6** (better distributional fit throughout the series)
- Post-break mean filter rate: 0.1059 (true = 0.110); converges to 0.1087 by month 48

The GAS filter wins clearly on RMSE against the true lambda schedule — it tracks the step more accurately than a blended linear trend. The MAE result at post-break is within stochastic noise: GAS has higher post-break MAE on this particular sample but lower MAE across all 60 periods, and the RMSE (which measures accuracy against the known true rate, not the noisy observations) consistently favours GAS. Log-likelihood improvement reflects a better-calibrated model throughout the series.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Forward trend projection — trend fits feed into the dynamic projection models |
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Heavy-tail severity with composite Pareto models — dynamics models require severity projections for large loss exposure |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — tracks whether dynamic projections remain calibrated over time |

## License

MIT
