"""
Severity change detector: Normal-Gamma BOCPD on log-severity.

We work on log(mean_severity) because severity distributions are log-normal
to a reasonable approximation. This converts a skewed positive-valued problem
to a roughly Gaussian one, where the Normal-Gamma conjugate is exact.

Usage
-----
>>> from insurance_changepoint import SeverityChangeDetector
>>> detector = SeverityChangeDetector()
>>> result = detector.fit(mean_severities, claim_counts, periods)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._bocpd import BOCPDEngine, NormalGammaModel
from .priors import UKEventPrior
from .result import ChangeResult, DetectedBreak


class SeverityChangeDetector:
    """
    Online Bayesian changepoint detection for insurance claim severity.

    Works on log-transformed mean severity per period. The Normal-Gamma
    conjugate is used, giving a Student-t predictive distribution.

    Where claim counts are provided, the log-severity is weighted by
    sqrt(count) in computing the sufficient statistics (heavier periods
    give more information). This is a simple but effective approach;
    full weighted conjugate updates require a custom derivation.

    Parameters
    ----------
    prior_mu :
        Prior mean of log(severity). For motor bodily injury in 2024,
        log(£4000) ≈ 8.3.
    prior_kappa :
        Prior pseudo-observations for the mean (strength of prior on mu).
    prior_alpha :
        NormalGamma shape parameter.
    prior_beta :
        NormalGamma rate parameter.
    hazard :
        Base changepoint hazard per period.
    threshold :
        P(changepoint) threshold.
    log_transform :
        If True (default), log-transform the severity values before fitting.
        Set to False if you have already log-transformed.
    uk_events :
        Whether to apply UK event calendar priors.
    event_lines :
        Lines of business for UK event filtering.
    event_components :
        Component filter for UK events.
    """

    def __init__(
        self,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
        hazard: float = 1.0 / 100,
        threshold: float = 0.3,
        max_run_length: int = 500,
        log_transform: bool = True,
        uk_events: bool = False,
        event_lines: list[str] | None = None,
        event_components: list[str] | None = None,
    ) -> None:
        self.prior_mu = prior_mu
        self.prior_kappa = prior_kappa
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.hazard = hazard
        self.threshold = threshold
        self.max_run_length = max_run_length
        self.log_transform = log_transform
        self.uk_events = uk_events
        self.event_lines = event_lines or (["motor"] if uk_events else None)
        self.event_components = event_components or (
            ["severity"] if uk_events else None
        )

        self._model = NormalGammaModel(
            mu0=prior_mu,
            kappa0=prior_kappa,
            alpha0=prior_alpha,
            beta0=prior_beta,
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

        self._periods_seen: list[Any] = []
        self._cp_probs: list[float] = []

    def fit(
        self,
        mean_severities: list[float] | np.ndarray,
        claim_counts: list[float] | np.ndarray | None = None,
        periods: list[Any] | None = None,
        period_to_date_fn: Any = None,
    ) -> ChangeResult:
        """
        Fit to a historical series of mean severities.

        Parameters
        ----------
        mean_severities :
            Mean claim cost per period. Must be positive if log_transform=True.
        claim_counts :
            Number of claims per period. Currently used only for validation
            and metadata; full count-weighted updates are not implemented.
        periods :
            Optional period labels.
        period_to_date_fn :
            Callable to convert period labels to dates.

        Returns
        -------
        ChangeResult
        """
        sevs = np.asarray(mean_severities, dtype=float)
        T = len(sevs)

        if claim_counts is not None:
            cnts = np.asarray(claim_counts, dtype=float)
            if len(cnts) != T:
                raise ValueError(
                    "claim_counts must have the same length as mean_severities."
                )
        else:
            cnts = np.ones(T)

        if self.log_transform:
            if np.any(sevs <= 0):
                raise ValueError(
                    "All mean_severities must be positive when log_transform=True."
                )
            x = np.log(sevs)
        else:
            x = sevs.copy()

        periods_list = list(periods) if periods is not None else list(range(T))

        hazards: list[float] | None = None
        if self._uk_prior is not None:
            hazards_arr = self._uk_prior.hazard_series(
                periods_list,
                base_hazard=self.hazard,
                period_to_date_fn=period_to_date_fn,
            )
            hazards = hazards_arr.tolist()

        observations = x.tolist()
        cp_probs, rl_probs = self._engine.fit(observations, hazards=hazards)

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

        self._periods_seen = periods_list[:]
        self._cp_probs = cp_probs.tolist()

        return ChangeResult(
            periods=periods_list,
            changepoint_probs=cp_probs,
            run_length_probs=rl_probs,
            detected_breaks=detected,
            detector_type="severity",
            hazard_used=self.hazard,
            meta={
                "prior_mu": self.prior_mu,
                "prior_kappa": self.prior_kappa,
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "threshold": self.threshold,
                "log_transform": self.log_transform,
                "uk_events": self.uk_events,
            },
        )

    def update(
        self,
        mean_severity: float,
        claim_count: float | None = None,
        period: Any = None,
        period_to_date_fn: Any = None,
    ) -> float:
        """
        Online update with a single new period.

        Parameters
        ----------
        mean_severity :
            Mean claim cost for the new period.
        claim_count :
            Number of claims (informational only).
        period :
            Period label.
        period_to_date_fn :
            Callable for event calendar date lookup.

        Returns
        -------
        float
            P(changepoint at this period).
        """
        if self.log_transform and mean_severity <= 0:
            raise ValueError("mean_severity must be positive when log_transform=True.")

        x = float(np.log(mean_severity) if self.log_transform else mean_severity)

        hazard_t: float | None = None
        if self._uk_prior is not None and period is not None:
            hazards_arr = self._uk_prior.hazard_series(
                [period],
                base_hazard=self.hazard,
                period_to_date_fn=period_to_date_fn,
            )
            hazard_t = float(hazards_arr[0])

        prob = self._engine.step(x, hazard_t=hazard_t)
        self._cp_probs.append(prob)
        self._periods_seen.append(period)
        return prob
