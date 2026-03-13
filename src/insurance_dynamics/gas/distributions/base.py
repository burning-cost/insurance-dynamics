"""Base class for GAS distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class GASDistribution(ABC):
    """Abstract base class for distributions used in GAS models.

    Each distribution provides the ingredients the GAS filter needs:
    the score (gradient of log-likelihood with respect to the time-varying
    parameter), Fisher information for scaling, and the log-likelihood itself.

    Parameters are always held on the *natural* scale internally. The link
    and unlink methods convert to/from the unconstrained scale used during
    optimisation.
    """

    # Names of all parameters (time-varying and static).
    param_names: list[str] = []

    # Subset of param_names that are time-varying by default.
    default_time_varying: list[str] = []

    @abstractmethod
    def score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Gradient of the log-likelihood w.r.t. each time-varying parameter.

        Parameters
        ----------
        y:
            Observations at time t.
        params:
            Dict mapping parameter name -> current value (natural scale).
        exposure:
            Optional exposure offset (used by count distributions).

        Returns
        -------
        Dict mapping parameter name -> score value at time t.
        """

    @abstractmethod
    def fisher(
        self,
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Expected Fisher information for each time-varying parameter.

        Returns
        -------
        Dict mapping parameter name -> Fisher information value.
        """

    @abstractmethod
    def log_likelihood(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        exposure: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Log-likelihood contribution at each observation.

        Returns
        -------
        Array of per-observation log-likelihoods.
        """

    @abstractmethod
    def link(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Map from natural scale to unconstrained (optimisation) scale."""

    @abstractmethod
    def unlink(self, param_name: str, value: float | NDArray) -> float | NDArray:
        """Map from unconstrained scale back to natural scale."""

    def scaled_score(
        self,
        y: NDArray[np.float64],
        params: dict[str, NDArray[np.float64]],
        scaling: str = "fisher_inv",
        exposure: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Score pre-multiplied by the chosen scaling matrix.

        Parameters
        ----------
        scaling:
            ``'unit'`` — no scaling (raw score).
            ``'fisher_inv'`` — multiply by inverse Fisher information.
            ``'fisher_inv_sqrt'`` — multiply by inverse square-root of Fisher.

        Returns
        -------
        Dict mapping parameter name -> scaled score.
        """
        raw = self.score(y, params, exposure=exposure)
        if scaling == "unit":
            return raw

        fish = self.fisher(params, exposure=exposure)
        result: dict[str, NDArray[np.float64]] = {}
        for name, s in raw.items():
            fi = fish[name]
            if scaling == "fisher_inv":
                result[name] = s / fi if fi != 0 else s
            elif scaling == "fisher_inv_sqrt":
                result[name] = s / np.sqrt(fi) if fi != 0 else s
            else:
                raise ValueError(f"Unknown scaling '{scaling}'. Use 'unit', 'fisher_inv', or 'fisher_inv_sqrt'.")
        return result

    def initial_params(self, y: NDArray[np.float64]) -> dict[str, float]:
        """Sensible starting values for static parameters from data."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
