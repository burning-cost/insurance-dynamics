# insurance-dynamics

Dynamic insurance pricing models for UK pricing teams.

Two problems, one package: tracking how your loss ratios are changing right now (GAS), and detecting when they shifted structurally in the past (changepoint).

Merged from: `insurance-gas` (GAS score-driven models) and `insurance-changepoint` (BOCPD/PELT detection).

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
pip install insurance-dynamics
```

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

# Period labels
periods = [f"2021-{(i % 12) + 1:02d}" for i in range(T)]

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

## Performance

Benchmarked against a static Poisson GLM (intercept-only and linear trend variants) on 60 months of synthetic UK motor frequency data with a known regime shift at month 36 (frequency drops 37.5%). Full notebook: `notebooks/benchmark.py`.

| Metric | Static GLM | GAS Poisson |
|--------|-----------|-------------|
| One-step-ahead MAE (overall) | higher | lower |
| One-step-ahead MAE (post-break) | higher | lower |
| Lambda RMSE vs true schedule (post-break) | higher | lower |
| Log-likelihood | lower | higher |
| PIT KS test (calibration) | fails post-break | passes |
| Break detected within ±3 months | No | BOCPD + PELT: Yes |

The static GLM blends the two regimes into a long-run average — it is wrong in both directions and cannot flag that anything changed. The GAS filter adapts observation by observation; BOCPD and PELT locate the break retrospectively with a bootstrap confidence interval.

**When to use:** You have monthly or quarterly aggregate claim data and want to track how underlying rates are evolving, or you need to detect whether a regulatory event or market shift has changed your book.

**When NOT to use:** You have a stable, stationary book and want to explain cross-sectional risk factors. That is a GLM job. GAS adds value at the time-series layer, not the rating-factor layer.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Forward trend projection — trend fits feed into the dynamic projection models |
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Heavy-tail severity with composite Pareto models — dynamics models require severity projections for large loss exposure |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — tracks whether dynamic projections remain calibrated over time |

## License

MIT
