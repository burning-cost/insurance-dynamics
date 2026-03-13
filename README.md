# insurance-dynamics

Dynamic insurance pricing models for UK pricing teams.

Two problems, one package: tracking how your loss ratios are changing right now (GAS), and detecting when they shifted structurally in the past (changepoint).

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

```python
from insurance_dynamics.gas import GASModel
from insurance_dynamics.gas.datasets import load_motor_frequency

data = load_motor_frequency(T=48)
model = GASModel("poisson")
result = model.fit(data.y, exposure=data.exposure)
print(result.summary())
result.filter_path.plot()
```

```python
from insurance_dynamics.changepoint import LossRatioMonitor

monitor = LossRatioMonitor(lines=["motor"], uk_events=True)
result = monitor.monitor(
    claim_counts=counts,
    exposures=exposures,
    periods=periods,
)
print(result.recommendation)  # 'retrain' or 'monitor'
```

## Design decisions

GAS and changepoint detection are complementary tools. GAS smooths the signal continuously; BOCPD detects discrete jumps. Using both gives you a full picture: is the drift smooth (GAS will catch it) or structural (BOCPD will flag it)? The `LossRatioMonitor` is deliberately opinionated — it returns a binary recommendation, not a probability, because pricing teams need an action, not a number.

## License

MIT
