"""
Frequency change detector: Poisson-Gamma BOCPD for insurance claim counts.

This is the primary novel contribution of insurance-changepoint. No existing
Python package handles exposure-weighted Poisson BOCPD. The exposure weighting
matters because insurance periods differ in size (earning patterns, mid-term
adjustments, portfolio growth).

Usage
-----
>>> from insurance_changepoint import FrequencyChangeDetector
>>> detector = FrequencyChangeDetector(prior_alpha=1.0, prior_beta=12.0)
>>> result = detector.fit(claim_counts, earned_exposure, periods)
>>> print(result.n_breaks, "regime breaks detected")
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from ._bocpd import BOCPDEngine, PoissonGammaModel
from .priors import UKEventPrior
from .result import ChangeResult, DetectedBreak


class FrequencyChangeDetector:
    """
    Online Bayesian changepoint detection for insurance claim frequency.

    Models claim count n_t with exposure e_t per period as:
        n_t | λ ~ Poisson(λ * e_t)
        λ ~ Gamma(α₀, β₀)

    The Poisson-Gamma conjugate gives a NegativeBinomial predictive,
    making the algorithm exact (no MCMC required) and O(T²) in the
    offline case, O(1) per update in the online case.

    Parameters
    ----------
    prior_alpha :
        Gamma shape hyperparameter. Controls prior mean of λ via E[λ] = α/β.
        For monthly motor frequency ~5%, try alpha=1, beta=20.
    prior_beta :
        Gamma rate hyperparameter.
    hazard :
        Base probability of a changepoint per period. 1/100 means you expect
        a structural break roughly every 100 periods (good for monthly data
        covering 8+ years).
    threshold :
        P(changepoint) threshold above which a break is flagged.
    max_run_length :
        Cap on tracked run lengths. Higher = more memory, lower = small
        approximation error for very long runs. 500 is fine for 20 years monthly.
    uk_events :
        Whether to apply UK event calendar priors. If True, requires periods
        to be date objects or period_to_date_fn to be provided on fit().
    event_lines :
        Lines of business for UK event filtering. e.g. ['motor'].
    event_components :
        Component filter for UK events. e.g. ['frequency'].
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 10.0,
        hazard: float = 1.0 / 100,
        threshold: float = 0.3,
        max_run_length: int = 500,
        uk_events: bool = False,
        event_lines: list[str] | None = None,
        event_components: list[str] | None = None,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.hazard = hazard
        self.threshold = threshold
        self.max_run_length = max_run_length
        self.uk_events = uk_events
        self.event_lines = event_lines or (["motor"] if uk_events else None)
        self.event_components = event_components or (
            ["frequency"] if uk_events else None
        )

        self._model = PoissonGammaModel(
            alpha0=prior_alpha, beta0=prior_beta
        )
        self._engine = BOCPDEngine(
            model=self._model,
            hazard=hazard,
            max_run_length=max_run_length,
        )

        self._uk_prior: UKEventPrior | None = None
        if uk_events:
            self._uk_prior = UKEventPrior(
                lines=self.event_lines, components=self.event_components
            )

        # Streaming state for online use
        self._periods_seen: list[Any] = []
        self._cp_probs: list[float] = []

    def fit(
        self,
        claim_counts: list[float] | np.ndarray,
        earned_exposure: list[float] | np.ndarray,
        periods: list[Any] | None = None,
        period_to_date_fn: Any = None,
    ) -> ChangeResult:
        """
        Fit the model to a full historical series.

        Parameters
        ----------
        claim_counts :
            Number of claims per period.
        earned_exposure :
            Earned exposure per period (vehicle-years, policy-years,
            earned premium, etc.). Must be positive.
        periods :
            Optional period labels (dates, strings, integers). If dates
            and uk_events=True, the UK event calendar is applied.
        period_to_date_fn :
            Callable to convert period labels to dates for event calendar.

        Returns
        -------
        ChangeResult
        """
        counts = np.asarray(claim_counts, dtype=float)
        exposure = np.asarray(earned_exposure, dtype=float)

        if len(counts) != len(exposure):
            raise ValueError(
                f"claim_counts (len={len(counts)}) and earned_exposure "
                f"(len={len(exposure)}) must have the same length."
            )
        if np.any(exposure <= 0):
            raise ValueError("All exposure values must be positive.")

        T = len(counts)
        periods_list = list(periods) if periods is not None else list(range(T))

        # Compute per-period hazards
        hazards: list[float] | None = None
        if self._uk_prior is not None:
            hazards_arr = self._uk_prior.hazard_series(
                periods_list,
                base_hazard=self.hazard,
                period_to_date_fn=period_to_date_fn,
            )
            hazards = hazards_arr.tolist()

        # Run BOCPD
        observations = list(zip(counts.tolist(), exposure.tolist()))
        cp_probs, rl_probs = self._engine.fit(observations, hazards=hazards)

        # Detect breaks above threshold
        detected = []
        for t, prob in enumerate(cp_probs):
            if prob >= self.threshold:
                rl_idx = int(np.argmax(rl_probs[t]))
                detected.append(
                    DetectedBreak(
                        period_index=t,
                        period_label=periods_list[t],
                        probability=float(prob),
                        run_length_before=rl_idx,
                    )
                )

        # Store state for online updates
        self._periods_seen = periods_list[:]
        self._cp_probs = cp_probs.tolist()

        return ChangeResult(
            periods=periods_list,
            changepoint_probs=cp_probs,
            run_length_probs=rl_probs,
            detected_breaks=detected,
            detector_type="frequency",
            hazard_used=self.hazard,
            meta={
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "threshold": self.threshold,
                "uk_events": self.uk_events,
            },
        )

    def update(
        self,
        n: float,
        exposure: float,
        period: Any = None,
        period_to_date_fn: Any = None,
    ) -> float:
        """
        Online update with a single new period.

        Parameters
        ----------
        n :
            Claims count for the new period.
        exposure :
            Earned exposure for the new period.
        period :
            Period label for the new period.
        period_to_date_fn :
            Callable to convert period label to date for event calendar.

        Returns
        -------
        float
            P(changepoint at this period).
        """
        if exposure <= 0:
            raise ValueError("exposure must be positive.")

        hazard_t: float | None = None
        if self._uk_prior is not None and period is not None:
            hazards_arr = self._uk_prior.hazard_series(
                [period],
                base_hazard=self.hazard,
                period_to_date_fn=period_to_date_fn,
            )
            hazard_t = float(hazards_arr[0])

        prob = self._engine.step((float(n), float(exposure)), hazard_t=hazard_t)
        self._cp_probs.append(prob)
        self._periods_seen.append(period)
        return prob

    def posterior_lambda(
        self, rl_idx: int = 0
    ) -> tuple[float, float]:
        """
        Return posterior Gamma(alpha, beta) for λ at a given run-length index.

        Parameters
        ----------
        rl_idx :
            Index into current run-length distribution. 0 = most recent segment
            after a changepoint; -1 = longest run.

        Returns
        -------
        (alpha, beta) posterior parameters.
        """
        model = self._engine.model
        if not isinstance(model, PoissonGammaModel):
            raise TypeError("Unexpected model type")
        return model.get_posterior_params(rl_idx)
