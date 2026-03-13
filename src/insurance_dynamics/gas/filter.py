"""Core GAS filter recursion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .distributions.base import GASDistribution


@dataclass
class FilterResult:
    """Output from the GAS filter forward pass.

    Attributes
    ----------
    filter_paths:
        Dict mapping time-varying parameter name to its filtered values
        (length T, natural scale).
    f_paths:
        Same paths on the unconstrained (link) scale — what the GAS
        recursion actually operates on.
    log_likelihoods:
        Per-observation log-likelihood contributions.
    """

    filter_paths: dict[str, NDArray[np.float64]]
    f_paths: dict[str, NDArray[np.float64]]
    log_likelihoods: NDArray[np.float64]
    score_residuals: dict[str, NDArray[np.float64]] = field(default_factory=dict)


class GASFilter:
    """Forward pass of the GAS(p, q) recursion.

    The recursion for each time-varying parameter f (on the link scale) is:

        f_{t+1} = omega + alpha_1 * S(f_t) * nabla_t + ... (p lags)
                         + phi_1 * f_t + ... (q lags)

    where nabla_t = d/df log p(y_t | f_t) is the score and S(f_t) is the
    scaling matrix (inverse Fisher information for ``fisher_inv``).

    Parameters
    ----------
    distribution:
        A fitted GASDistribution instance.
    time_varying:
        Names of parameters to make time-varying.
    scaling:
        One of ``'unit'``, ``'fisher_inv'``, ``'fisher_inv_sqrt'``.
    p:
        Number of score lags.
    q:
        Number of AR lags.
    """

    def __init__(
        self,
        distribution: GASDistribution,
        time_varying: list[str],
        scaling: str = "fisher_inv",
        p: int = 1,
        q: int = 1,
    ) -> None:
        self.distribution = distribution
        self.time_varying = time_varying
        self.scaling = scaling
        self.p = p
        self.q = q

    def _make_params(
        self,
        f_tv: dict[str, float],
        static_params: dict[str, float],
    ) -> dict[str, NDArray]:
        """Combine time-varying (natural scale) and static parameters."""
        params = dict(static_params)
        for name, fval in f_tv.items():
            params[name] = self.distribution.unlink(name, fval)
        return params

    def run(
        self,
        y: NDArray[np.float64],
        gas_params: dict[str, float],
        static_params: dict[str, float],
        exposure: NDArray[np.float64] | None = None,
        f0: dict[str, float] | None = None,
    ) -> FilterResult:
        """Run the GAS filter forward pass.

        Parameters
        ----------
        y:
            Observations (length T).
        gas_params:
            GAS model parameters: for each time-varying parameter ``name``,
            expects keys ``omega_{name}``, ``alpha_{name}_1`` ..
            ``alpha_{name}_p``, ``phi_{name}_1`` .. ``phi_{name}_q``.
        static_params:
            Static distribution parameters (e.g. shape for Gamma).
        exposure:
            Optional per-observation exposure values (length T).
        f0:
            Initial filter values on the link scale. Defaults to the
            unconditional mean omega / (1 - sum(phi)).

        Returns
        -------
        FilterResult
        """
        T = len(y)
        y = np.asarray(y, dtype=float)
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)

        # Extract GAS parameters per time-varying variable
        omega: dict[str, float] = {}
        alpha: dict[str, list[float]] = {}
        phi: dict[str, list[float]] = {}

        for name in self.time_varying:
            omega[name] = gas_params[f"omega_{name}"]
            alpha[name] = [gas_params[f"alpha_{name}_{i+1}"] for i in range(self.p)]
            phi[name] = [gas_params[f"phi_{name}_{i+1}"] for i in range(self.q)]

        # Initialise filter state
        f_current: dict[str, float] = {}
        for name in self.time_varying:
            if f0 is not None and name in f0:
                f_current[name] = f0[name]
            else:
                phi_sum = sum(phi[name])
                if abs(phi_sum) < 1.0:
                    f_current[name] = omega[name] / (1.0 - phi_sum)
                else:
                    f_current[name] = omega[name]

        # Store histories for AR lags
        f_history: dict[str, list[float]] = {n: [f_current[n]] * self.q for n in self.time_varying}
        score_history: dict[str, list[float]] = {n: [0.0] * self.p for n in self.time_varying}

        # Output arrays
        f_paths: dict[str, list[float]] = {n: [] for n in self.time_varying}
        log_lls: list[float] = []
        scaled_scores: dict[str, list[float]] = {n: [] for n in self.time_varying}

        for t in range(T):
            # Record current state
            for name in self.time_varying:
                f_paths[name].append(f_current[name])

            # Build full parameter dict (time-varying on natural scale + static)
            params = self._make_params(f_current, static_params)

            # Exposure at t
            exp_t = exposure[t] if exposure is not None else None

            # Log-likelihood
            ll = self.distribution.log_likelihood(
                np.array([y[t]]), params, exposure=np.array([exp_t]) if exp_t is not None else None
            )
            log_lls.append(float(np.squeeze(ll)))

            # Scaled score for each time-varying parameter
            ss = self.distribution.scaled_score(
                np.array([y[t]]),
                params,
                scaling=self.scaling,
                exposure=np.array([exp_t]) if exp_t is not None else None,
            )
            for name in self.time_varying:
                score_val = float(np.squeeze(ss[name]))
                scaled_scores[name].append(score_val)
                score_history[name].append(score_val)

            # Update f for next period
            f_next: dict[str, float] = {}
            for name in self.time_varying:
                val = omega[name]
                for i in range(self.p):
                    val += alpha[name][i] * score_history[name][-(i + 1)]
                for j in range(self.q):
                    val += phi[name][j] * f_history[name][-(j + 1)]
                f_next[name] = val
                f_history[name].append(val)

            f_current = f_next

        # Convert to arrays
        filter_paths = {
            name: self.distribution.unlink(name, np.array(f_paths[name]))
            for name in self.time_varying
        }
        f_path_arrays = {name: np.array(f_paths[name]) for name in self.time_varying}

        return FilterResult(
            filter_paths=filter_paths,
            f_paths=f_path_arrays,
            log_likelihoods=np.array(log_lls),
            score_residuals={n: np.array(v) for n, v in scaled_scores.items()},
        )
