"""
Core BOCPD recursion engine.

Implements the Adams & MacKay (2007) online Bayesian changepoint detection
algorithm with log-sum-exp numerics throughout to avoid underflow.

The recursion operates over run-length distributions: at each time step t,
we maintain log P(r_t = ℓ | x_{1:t}) for all possible run lengths ℓ ∈ {0, ..., t}.

Correct Adams & MacKay recursion (confirmed against paper, Eq. 1):

  P(r_{t+1} = 0     | x_{1:t+1}) ∝ H * P(x_{t+1} | prior)
  P(r_{t+1} = ℓ+1  | x_{1:t+1}) ∝ (1-H) * P(r_t=ℓ | x_{1:t}) * P(x_{t+1} | rl=ℓ)

The changepoint entry uses ONLY the prior predictive, not a weighted average
over run lengths. This is because a changepoint resets sufficient statistics
to the prior regardless of which run was active. An incorrect formulation
(weighted average) yields P(r_{t+1}=0) = H always (constants cancel in
normalisation), making detection impossible.

Key extension: hazard H(t) can vary per period, which is how we incorporate
the UK event prior calendar.

References
----------
Adams, R.P. & MacKay, D.J.C. (2007). Bayesian online changepoint detection.
arXiv:0710.3742.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.special import gammaln, betaln


def _logsumexp(log_weights: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    if len(log_weights) == 0:
        return -np.inf
    c = np.max(log_weights)
    if not np.isfinite(c):
        return -np.inf
    return float(c + np.log(np.sum(np.exp(log_weights - c))))


class _ConjugateModel(ABC):
    """
    Abstract base for conjugate predictive models used within BOCPD.

    Each model maintains a set of sufficient statistics indexed by run length.
    At run length ℓ, the sufficient stats represent data from the last ℓ periods.

    Methods
    -------
    log_pred(obs, rl_idx) :
        Log predictive probability of obs given sufficient stats at run-length
        index rl_idx.
    update(obs) :
        Add a new observation to all existing run-length sufficient stats,
        and add a fresh prior stats entry for run length 0.
    reset_to_prior() :
        Reset to a single prior sufficient stat (for run length 0 at time 0).
    """

    @abstractmethod
    def log_pred(self, obs: Any, rl_idx: int) -> float:
        """Log predictive P(obs | sufficient stats at run-length index rl_idx)."""
        ...

    @abstractmethod
    def update(self, obs: Any) -> None:
        """Extend sufficient stats with new observation."""
        ...

    @abstractmethod
    def reset_to_prior(self) -> None:
        """Reset to single prior entry (called at t=0)."""
        ...

    @abstractmethod
    def trim(self, max_run_length: int) -> None:
        """Drop entries beyond max_run_length to bound memory."""
        ...


class PoissonGammaModel(_ConjugateModel):
    """
    Poisson-Gamma conjugate for insurance claim frequency.

    Observation model: n_t | λ, e_t ~ Poisson(λ * e_t)
    Prior / running posterior: λ ~ Gamma(α, β)  (rate parameterisation)
    Predictive: n_{t+1} | α, β, e_{t+1} ~ NegBin(α, β/(β+e_{t+1}))

    The key novelty: we track e (exposure) as a sufficient statistic so
    each period can have a different exposure (earned premium, vehicle-years,
    policy-years, etc.).

    Parameters
    ----------
    alpha0 :
        Prior Gamma shape.
    beta0 :
        Prior Gamma rate (i.e. 1/mean when shape=1 means E[λ]=alpha/beta).
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 10.0) -> None:
        self.alpha0 = alpha0
        self.beta0 = beta0
        # _alphas[i] and _betas[i] are the sufficient stats for run-length i
        self._alphas: list[float] = []
        self._betas: list[float] = []
        self.reset_to_prior()

    def reset_to_prior(self) -> None:
        self._alphas = [self.alpha0]
        self._betas = [self.beta0]

    def log_pred(self, obs: tuple[float, float], rl_idx: int) -> float:
        """
        Log P(n | alpha_rl, beta_rl, exposure).

        Parameters
        ----------
        obs :
            (n, exposure) tuple — claims count and earned exposure.
        rl_idx :
            Index into the sufficient-stats arrays.
        """
        n, e = obs
        n = float(n)
        e = float(e)

        a = self._alphas[rl_idx]
        b = self._betas[rl_idx]

        # NegBin log-pmf: log Gamma(a+n) - log Gamma(a) - log n!
        #   + a * log(b/(b+e)) + n * log(e/(b+e))
        log_p = (
            gammaln(a + n)
            - gammaln(a)
            - gammaln(n + 1)
            + a * np.log(b / (b + e))
            + n * np.log(e / (b + e))
        )
        return float(log_p)

    def update(self, obs: tuple[float, float]) -> None:
        """
        Extend all existing sufficient stats with new observation,
        then prepend a fresh prior entry for run length 0.
        """
        n, e = obs
        n = float(n)
        e = float(e)

        # Update all existing run lengths
        new_alphas = [a + n for a in self._alphas]
        new_betas = [b + e for b in self._betas]

        # Prepend prior for new run length 0 (changepoint just happened)
        self._alphas = [self.alpha0] + new_alphas
        self._betas = [self.beta0] + new_betas

    def trim(self, max_run_length: int) -> None:
        if len(self._alphas) > max_run_length + 1:
            self._alphas = self._alphas[: max_run_length + 1]
            self._betas = self._betas[: max_run_length + 1]

    @property
    def current_run_length_count(self) -> int:
        return len(self._alphas)

    def get_posterior_params(self, rl_idx: int) -> tuple[float, float]:
        """Return (alpha, beta) posterior params at run-length index rl_idx."""
        return self._alphas[rl_idx], self._betas[rl_idx]


class NormalGammaModel(_ConjugateModel):
    """
    Normal-Gamma conjugate for log-severity (or any continuous series).

    Observation model: x_t | μ, τ ~ Normal(μ, 1/τ)
    Prior: μ, τ ~ NormalGamma(μ₀, κ₀, α₀, β₀)
    Predictive: Student-t with 2α₀ degrees of freedom.

    We work on raw (possibly log-transformed) values. Caller is responsible
    for taking log(severity) before passing to this model.

    Parameters
    ----------
    mu0, kappa0, alpha0, beta0 :
        NormalGamma prior hyperparameters.
        - mu0: prior mean
        - kappa0: prior pseudo-observations for mean
        - alpha0: Gamma shape for precision
        - beta0: Gamma rate for precision
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self._mus: list[float] = []
        self._kappas: list[float] = []
        self._alphas: list[float] = []
        self._betas: list[float] = []
        self.reset_to_prior()

    def reset_to_prior(self) -> None:
        self._mus = [self.mu0]
        self._kappas = [self.kappa0]
        self._alphas = [self.alpha0]
        self._betas = [self.beta0]

    def log_pred(self, obs: float, rl_idx: int) -> float:
        """
        Log predictive P(x | NormalGamma params at rl_idx).
        Uses Student-t predictive distribution.
        """
        x = float(obs)
        mu = self._mus[rl_idx]
        kappa = self._kappas[rl_idx]
        alpha = self._alphas[rl_idx]
        beta = self._betas[rl_idx]

        # Student-t predictive: t_{2α}(μ, β(κ+1)/(ακ))
        # log pdf of t_ν(x; μ, σ²):
        #   log Gamma((ν+1)/2) - log Gamma(ν/2) - 0.5*log(νπσ²)
        #   - (ν+1)/2 * log(1 + (x-μ)²/(νσ²))
        nu = 2.0 * alpha
        sigma2 = beta * (kappa + 1.0) / (alpha * kappa)

        log_p = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * sigma2)
            - (nu + 1.0) / 2.0 * np.log(1.0 + (x - mu) ** 2 / (nu * sigma2))
        )
        return float(log_p)

    def update(self, obs: float) -> None:
        """
        Extend sufficient stats with new observation using NormalGamma
        conjugate update rule, then prepend fresh prior.
        """
        x = float(obs)

        new_mus = []
        new_kappas = []
        new_alphas = []
        new_betas = []

        for mu, kappa, alpha, beta in zip(
            self._mus, self._kappas, self._alphas, self._betas
        ):
            kappa_new = kappa + 1.0
            mu_new = (kappa * mu + x) / kappa_new
            alpha_new = alpha + 0.5
            beta_new = beta + (kappa * (x - mu) ** 2) / (2.0 * kappa_new)

            new_mus.append(mu_new)
            new_kappas.append(kappa_new)
            new_alphas.append(alpha_new)
            new_betas.append(beta_new)

        self._mus = [self.mu0] + new_mus
        self._kappas = [self.kappa0] + new_kappas
        self._alphas = [self.alpha0] + new_alphas
        self._betas = [self.beta0] + new_betas

    def trim(self, max_run_length: int) -> None:
        if len(self._mus) > max_run_length + 1:
            self._mus = self._mus[: max_run_length + 1]
            self._kappas = self._kappas[: max_run_length + 1]
            self._alphas = self._alphas[: max_run_length + 1]
            self._betas = self._betas[: max_run_length + 1]

    def get_posterior_params(self, rl_idx: int) -> tuple[float, float, float, float]:
        return (
            self._mus[rl_idx],
            self._kappas[rl_idx],
            self._alphas[rl_idx],
            self._betas[rl_idx],
        )


class BOCPDEngine:
    """
    Core BOCPD run-length recursion.

    This class is model-agnostic: it operates on log-likelihood contributions
    provided by a _ConjugateModel. The caller (FrequencyChangeDetector etc.)
    instantiates the correct conjugate model and passes data in the right format.

    Parameters
    ----------
    model :
        Conjugate predictive model implementing _ConjugateModel.
    hazard :
        Base hazard (probability of changepoint per period). Typical values
        for monthly data: 1/36 to 1/120.
    max_run_length :
        Cap on tracked run lengths to bound O(T²) memory. Default 500.
    """

    def __init__(
        self,
        model: _ConjugateModel,
        hazard: float = 1.0 / 100,
        max_run_length: int = 500,
    ) -> None:
        if not 0.0 < hazard < 1.0:
            raise ValueError(f"hazard must be in (0, 1), got {hazard}")
        self.model = model
        self.hazard = hazard
        self.max_run_length = max_run_length

        # log R[ℓ] = log P(r_t = ℓ | x_{1:t})
        # Initialise at t=0: run length 0, probability 1
        self._log_R: np.ndarray = np.array([0.0])  # length 1 at t=0

        self._t: int = 0
        self._changepoint_probs: list[float] = []
        self._run_length_probs_store: list[np.ndarray] = []

    def reset(self) -> None:
        """Reset engine to initial state."""
        self._log_R = np.array([0.0])
        self._t = 0
        self._changepoint_probs = []
        self._run_length_probs_store = []
        self.model.reset_to_prior()

    def step(self, obs: Any, hazard_t: float | None = None) -> float:
        """
        Process one observation and return P(changepoint at this step).

        Parameters
        ----------
        obs :
            Observation in the format expected by self.model.log_pred().
        hazard_t :
            Per-step hazard override. If None, uses self.hazard.

        Returns
        -------
        float
            P(changepoint at this time step).
        """
        H = hazard_t if hazard_t is not None else self.hazard

        # Number of possible run lengths before seeing this observation
        # _log_R has length ℓ_{max} = t+1 at time t
        current_len = len(self._log_R)

        # Compute log predictive for each run length.
        # rl_idx=0 is always the prior (model was just reset / changepoint).
        # rl_idx>0 are posteriors conditioned on data from the current run.
        log_preds = np.array(
            [self.model.log_pred(obs, rl_idx) for rl_idx in range(current_len)]
        )

        # Adams & MacKay (2007) recursion:
        #
        #   P(r_{t+1}=0    | x_{1:t+1}) ∝ H * P(x_{t+1} | prior)
        #   P(r_{t+1}=ℓ+1 | x_{1:t+1}) ∝ (1-H) * P(r_t=ℓ) * P(x_{t+1} | rl=ℓ)
        #
        # The changepoint entry uses ONLY the prior predictive (rl_idx=0),
        # not a weighted sum. This is because a changepoint resets the
        # sufficient statistics to the prior regardless of which run we came from.
        log_prior_pred = log_preds[0]  # log P(x_{t+1} | prior)

        # Changepoint entry: run-length resets to 0
        log_cp = np.log(H) + log_prior_pred

        # Growth entries: run-length increments by 1
        # log P(r_{t+1}=ℓ+1) ∝ log(1-H) + log_R[ℓ] + log_pred(rl=ℓ)
        log_growth = np.log(1.0 - H) + self._log_R + log_preds

        # New log_R: entry 0 = changepoint mass, entries 1..t+1 = growth
        log_R_new = np.empty(current_len + 1)
        log_R_new[0] = log_cp
        log_R_new[1:] = log_growth

        # Normalise
        log_norm = _logsumexp(log_R_new)
        if not np.isfinite(log_norm):
            # Numerical failure — reset to uniform
            log_R_new = -np.log(len(log_R_new)) * np.ones(len(log_R_new))
            log_norm = 0.0
        log_R_new -= log_norm

        # Trim to bound memory
        if len(log_R_new) > self.max_run_length + 1:
            # Keep the most probable entries — just truncate oldest
            log_R_new = log_R_new[: self.max_run_length + 1]
            # Re-normalise after trim
            log_norm = _logsumexp(log_R_new)
            log_R_new -= log_norm

        self._log_R = log_R_new
        self.model.update(obs)
        self.model.trim(self.max_run_length)

        # P(changepoint at t) = probability that run length just reset to 0
        # which equals exp(log_R_new[0])
        prob_cp = float(np.exp(log_R_new[0]))

        self._changepoint_probs.append(prob_cp)
        # Store normalised run-length distribution
        self._run_length_probs_store.append(np.exp(log_R_new))

        self._t += 1
        return prob_cp

    def fit(
        self, observations: list[Any], hazards: list[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a sequence of observations.

        Parameters
        ----------
        observations :
            List of observations in the format expected by model.log_pred().
        hazards :
            Optional per-step hazard values. Must match len(observations).

        Returns
        -------
        changepoint_probs :
            Array of shape (T,) with P(changepoint at t) for each step.
        run_length_probs :
            2-D array of shape (T, T) — padded with zeros. Entry [t, ℓ]
            is P(run_length=ℓ at time t).
        """
        self.reset()
        T = len(observations)

        for i, obs in enumerate(observations):
            h = hazards[i] if hazards is not None else None
            self.step(obs, hazard_t=h)

        changepoint_probs = np.array(self._changepoint_probs)

        # Build run_length_probs matrix — each row padded to T+1
        max_len = max(len(r) for r in self._run_length_probs_store) if T > 0 else 1
        run_length_probs = np.zeros((T, max_len))
        for t, rl_dist in enumerate(self._run_length_probs_store):
            run_length_probs[t, : len(rl_dist)] = rl_dist

        return changepoint_probs, run_length_probs
