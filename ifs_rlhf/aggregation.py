"""
Intuitionistic Fuzzy Weighted Averaging (IFWA) operator.

Paper Definition 3 / Eqs. (3)–(5):

  μ̃ = 1 − ∏_k (1 − μ_k)^{w_k}
  ν̃ = ∏_k ν_k^{w_k}
  π̃ = 1 − μ̃ − ν̃

Paper Proposition 1 (verified by the tests):
  Idempotency, Boundedness, Monotonicity, Symmetry (equal weights).
"""

from __future__ import annotations
import numpy as np
from .ifn import IFN


_EPS = 1e-3   # Laplace smoothing ε (paper §4.4)


def _laplace_smooth(mu: float, nu: float, eps: float = _EPS) -> tuple[float, float]:
    """
    Apply symmetric Laplace smoothing to prevent ν = 0 collapsing ∏ ν_k^{w_k}.

    After adding ε to both, renormalise so that μ + ν ≤ 1 still holds.
    """
    mu_s = mu + eps
    nu_s = nu + eps
    total = mu_s + nu_s
    if total > 1.0:
        mu_s /= total
        nu_s /= total
    return mu_s, nu_s


def ifwa(
    ifns: list[IFN],
    weights: np.ndarray | list[float] | None = None,
    laplace_eps: float = _EPS,
) -> IFN:
    """
    IFWA aggregation of K IFNs with weight vector W.

    Args:
        ifns:        List of K IFN objects (one per annotator).
        weights:     1-D array of non-negative weights summing to 1.
                     Defaults to uniform (1/K each).
        laplace_eps: Smoothing constant ε to prevent ν_k = 0 degeneration.

    Returns:
        Aggregated IFN (μ̃, ν̃, π̃).
    """
    K = len(ifns)
    if K == 0:
        raise ValueError("Need at least one IFN to aggregate.")

    if weights is None:
        weights = np.full(K, 1.0 / K)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (K,):
            raise ValueError(f"Weight vector must have length K={K}.")
        if weights.min() < 0:
            raise ValueError("Weights must be non-negative.")
        weights = weights / weights.sum()   # normalise to be safe

    # Laplace-smooth each IFN before aggregation
    mu_arr = np.empty(K)
    nu_arr = np.empty(K)
    for i, ifn in enumerate(ifns):
        mu_arr[i], nu_arr[i] = _laplace_smooth(ifn.mu, ifn.nu, eps=laplace_eps)

    # μ̃ = 1 − ∏ (1 − μ_k)^{w_k}
    agg_mu = 1.0 - float(np.prod((1.0 - mu_arr) ** weights))
    # ν̃ = ∏ ν_k^{w_k}
    agg_nu = float(np.prod(nu_arr ** weights))

    return IFN(mu=agg_mu, nu=agg_nu)


# ---------------------------------------------------------------------------
# Batch version (for entire datasets)
# ---------------------------------------------------------------------------

def batch_ifwa(
    mu_matrix: np.ndarray,
    nu_matrix: np.ndarray,
    weights: np.ndarray | None = None,
    laplace_eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised IFWA for a full dataset.

    Args:
        mu_matrix: (N, K) array of membership values.
        nu_matrix: (N, K) array of non-membership values.
        weights:   (K,) weight vector; uniform if None.
        laplace_eps: Laplace smoothing constant.

    Returns:
        (mu_tilde, nu_tilde, pi_tilde) each of shape (N,).
    """
    N, K = mu_matrix.shape
    if weights is None:
        weights = np.full(K, 1.0 / K)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    # Laplace smoothing
    mu_s = mu_matrix + laplace_eps
    nu_s = nu_matrix + laplace_eps
    total = mu_s + nu_s
    overflow = total > 1.0
    mu_s[overflow] /= total[overflow]
    nu_s[overflow] /= total[overflow]

    # μ̃ = 1 − ∏_k (1 − μ_k)^{w_k}
    log_one_minus_mu = np.log(np.clip(1.0 - mu_s, 1e-12, None))  # (N, K)
    agg_mu = 1.0 - np.exp(log_one_minus_mu @ weights)             # (N,)

    # ν̃ = ∏_k ν_k^{w_k}
    log_nu = np.log(np.clip(nu_s, 1e-12, None))                  # (N, K)
    agg_nu = np.exp(log_nu @ weights)                              # (N,)

    # Clip numerical noise
    agg_mu = np.clip(agg_mu, 0.0, 1.0)
    agg_nu = np.clip(agg_nu, 0.0, 1.0)
    total_agg = agg_mu + agg_nu
    overflow = total_agg > 1.0
    agg_mu[overflow] /= total_agg[overflow]
    agg_nu[overflow] /= total_agg[overflow]

    agg_pi = np.clip(1.0 - agg_mu - agg_nu, 0.0, 1.0)
    return agg_mu, agg_nu, agg_pi
