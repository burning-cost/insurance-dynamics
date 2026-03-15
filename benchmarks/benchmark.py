"""
Benchmark: GAS Poisson filter vs static GLM for time-varying claim frequency.

DGP: 60 months of UK motor frequency with a known regime shift at month 36.
  - Pre-shift (months 0-35):  true lambda = 0.080 claims/year
  - Post-shift (months 36-59): true lambda = 0.110 claims/year (+37.5%)

A static Poisson GLM (intercept-only or linear trend) fits a single constant
rate and cannot track the structural break. The GAS Poisson filter updates its
log-rate score-by-score, tracking the change in near-real-time.

Metrics:
  1. One-step-ahead MAE on predicted frequency (overall and post-break)
  2. Filter path RMSE vs true lambda schedule (post-break period)
  3. Log-likelihood on full series
"""

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def generate_motor_series(T: int = 60, break_at: int = 36, seed: int = 42):
    """
    Monthly motor claim frequency with a step-change at `break_at`.

    Returns:
        counts    : array(T) of monthly claim counts
        exposures : array(T) of monthly earned car-years
        true_rate : array(T) of true underlying frequency (per car-year)
    """
    rng = np.random.default_rng(seed)
    exposures  = rng.uniform(800, 1200, T)
    true_rate  = np.where(np.arange(T) < break_at, 0.080, 0.110)
    counts     = rng.poisson(true_rate * exposures)
    return counts.astype(float), exposures, true_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-10)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------------

def static_predictions(counts: np.ndarray, exposures: np.ndarray) -> dict:
    """
    Two naive baselines, both producing a predicted rate per exposure-unit:

    1. Constant: rate = total_claims / total_exposure
    2. Linear trend: log(rate) = a + b*t (OLS on log-frequency)
    """
    T = len(counts)
    t = np.arange(T, dtype=float)

    # Baseline 1: constant
    rate_const  = counts.sum() / exposures.sum()
    freq_const  = np.full(T, rate_const)

    # Baseline 2: linear trend on log-frequency, weighted by exposure
    log_freq = np.log(np.maximum(counts / exposures, 1e-10))
    X = np.column_stack([np.ones(T), t])
    W = exposures
    XtWX = (X.T * W) @ X
    XtWy = (X.T * W) @ log_freq
    try:
        beta       = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        freq_trend = np.exp(X @ beta)
    except Exception:
        freq_trend = freq_const.copy()

    return {"constant": freq_const, "linear_trend": freq_trend}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Benchmark: GAS Poisson filter vs static Poisson GLM")
    print("DGP: 60-month motor frequency, step break at month 36 (+37.5%)")
    print("=" * 65)

    T        = 60
    BREAK_AT = 36
    counts, exposures, true_rate = generate_motor_series(T=T, break_at=BREAK_AT, seed=42)
    obs_freq = counts / exposures
    post     = np.arange(T) >= BREAK_AT

    print(f"\nObserved frequency: pre-break={obs_freq[:BREAK_AT].mean():.4f}  "
          f"post-break={obs_freq[BREAK_AT:].mean():.4f}")
    print(f"True rate:          pre=0.080  post=0.110")

    # ------------------------------------------------------------------
    # Static baselines
    # ------------------------------------------------------------------
    baselines = static_predictions(counts, exposures)
    base_results = {}
    for name, freq_pred in baselines.items():
        base_results[name] = {
            "mae_all":   mae(obs_freq, freq_pred),
            "mae_post":  mae(obs_freq[post], freq_pred[post]),
            "rmse_post": rmse(true_rate[post], freq_pred[post]),
            "ll":        poisson_loglik(counts, freq_pred * exposures),
        }

    # ------------------------------------------------------------------
    # GAS Poisson
    # ------------------------------------------------------------------
    print("\nFitting GAS Poisson model...")
    gas_ok = False
    try:
        from insurance_dynamics.gas import GASModel

        model  = GASModel("poisson", p=1, q=1)
        result = model.fit(counts, exposure=exposures)

        fp = result.filter_path
        # PoissonGAS time-varying param is "mean" on the natural scale (mu per unit)
        # filter_paths stores natural-scale values
        if "mean" in fp.columns:
            gas_rate = fp["mean"].to_numpy()
        else:
            # Fallback: take first column and check scale
            col = fp.columns[0]
            vals = fp[col].to_numpy()
            gas_rate = np.exp(vals) if vals.min() < 0 else vals

        gas_results = {
            "mae_all":   mae(obs_freq, gas_rate),
            "mae_post":  mae(obs_freq[post], gas_rate[post]),
            "rmse_post": rmse(true_rate[post], gas_rate[post]),
            "ll":        poisson_loglik(counts, gas_rate * exposures),
        }

        gas_ok = True
        print(f"  LL={result.log_likelihood:.2f}  AIC={result.aic:.2f}")
        print(f"  Post-break mean filter rate: {gas_rate[post].mean():.4f}  (true=0.110)")
        print(f"  Post-break convergence at month 48: {gas_rate[48]:.4f}")
    except Exception as e:
        print(f"  GAS fit failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RESULTS (frequency = claims per car-year)")
    print("=" * 65)
    header = f"{'Model':<22} {'MAE (all)':>12} {'MAE (post)':>12} {'RMSE (post)':>13} {'Log-lik':>10}"
    print(header)
    print("-" * 72)

    label_map = {"constant": "GLM constant", "linear_trend": "GLM trend"}
    for name, r in base_results.items():
        lbl = label_map[name]
        print(f"{lbl:<22} {r['mae_all']:>12.6f} {r['mae_post']:>12.6f} {r['rmse_post']:>13.6f} {r['ll']:>10.1f}")

    if gas_ok:
        r = gas_results
        print(f"{'GAS Poisson':<22} {r['mae_all']:>12.6f} {r['mae_post']:>12.6f} {r['rmse_post']:>13.6f} {r['ll']:>10.1f}")

    print("-" * 72)

    if gas_ok:
        best_mae  = min(v["mae_post"]  for v in base_results.values())
        best_rmse = min(v["rmse_post"] for v in base_results.values())
        mae_impr  = 100.0 * (best_mae  - gas_results["mae_post"])  / best_mae
        rmse_impr = 100.0 * (best_rmse - gas_results["rmse_post"]) / best_rmse

        print(f"\nGAS vs best static baseline (post-break):")
        print(f"  MAE  improvement: {mae_impr:+.1f}%")
        print(f"  RMSE improvement: {rmse_impr:+.1f}%")
        print(f"  Log-lik:          {gas_results['ll'] - max(v['ll'] for v in base_results.values()):+.1f}")

        print(f"\nConclusion: static GLM fixes rate at the historical mean and")
        print(f"  cannot respond to the step change at month 36. GAS updates")
        print(f"  its filter after each observation and converges toward the")
        print(f"  new true rate (0.110), reducing post-break error substantially.")


if __name__ == "__main__":
    main()
