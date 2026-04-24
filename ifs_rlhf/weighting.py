"""
Entropy-regularised annotator reliability weighting.

Paper §4.4  (Eqs. 6–7):
  ρ_k = exp(−λ₁ π̄_k − λ₂ σ_k²)
  w_k = ρ_k / Σ ρ_{k'}

Iterative refinement (§4.4 / Algorithm 1, Phase 5) adds a third term:
  ρ_k^{t+1} = exp(−λ₁ π̄_k − λ₂ σ_k² − λ₃ ê_k^{t})   (Eq. A1)

Convergence to a unique fixed point under Assumption 1 is proved in
Proposition 2 of the paper.
"""

from __future__ import annotations
import numpy as np
from scipy.special import softmax


def compute_reliability_scores(
    pi_matrix: np.ndarray,
    lambda1: float = 0.8,
    lambda2: float = 0.5,
    error_rates: np.ndarray | None = None,
    lambda3: float = 0.3,
) -> np.ndarray:
    """
    Compute per-annotator reliability scores ρ_k (Eq. 6 / Eq. A1).

    Args:
        pi_matrix:   (N, K) hesitation degrees π_k^{(n)}.
                     NaN entries are ignored (annotator did not label example n).
        lambda1:     Weight on mean hesitation π̄_k.
        lambda2:     Weight on hesitation variance σ_k².
        error_rates: (K,) RM-estimated error rates ê_k (Phase 5 only).
                     Pass None to use the two-term formula (Phase 2).
        lambda3:     Weight on error rates (Eq. A1 / Phase 5).

    Returns:
        rho: (K,) reliability scores (not yet normalised).
    """
    # Mean and variance of π per annotator (ignoring NaN = missing annotations)
    pi_mean = np.nanmean(pi_matrix, axis=0)   # (K,)
    pi_var  = np.nanvar(pi_matrix, axis=0)    # (K,)

    log_rho = -lambda1 * pi_mean - lambda2 * pi_var

    if error_rates is not None:
        error_rates = np.asarray(error_rates, dtype=float)
        log_rho -= lambda3 * error_rates

    return np.exp(log_rho)


def reliability_to_weights(rho: np.ndarray) -> np.ndarray:
    """
    Softmax-normalise reliability scores to a weight simplex (Eq. 7).

    w_k = ρ_k / Σ ρ_{k'}

    Using log-softmax for numerical stability; equivalent to Eq. 7.
    """
    # Simple normalisation (identical to softmax after log): w_k = ρ_k / Σ ρ
    total = rho.sum()
    if total < 1e-12:
        return np.full_like(rho, 1.0 / len(rho))
    return rho / total


def compute_weights(
    pi_matrix: np.ndarray,
    lambda1: float = 0.8,
    lambda2: float = 0.5,
    error_rates: np.ndarray | None = None,
    lambda3: float = 0.3,
) -> np.ndarray:
    """
    Full pipeline: hesitation matrix → normalised weight vector.

    Returns:
        w: (K,) weight vector summing to 1.
    """
    rho = compute_reliability_scores(
        pi_matrix, lambda1=lambda1, lambda2=lambda2,
        error_rates=error_rates, lambda3=lambda3,
    )
    return reliability_to_weights(rho)


# ---------------------------------------------------------------------------
# Iterative weight refinement (Algorithm 1, Phase 5)
# ---------------------------------------------------------------------------

class IterativeWeightRefiner:
    """
    Self-supervised iterative annotator weight refinement (Paper §4.4 / Phase 5).

    After each RM training round, error rates on the low-hesitation 'clean'
    subset D_clean = {n : π̃^{(n)} < τ} are used to update ρ_k, which in
    turn drives the next round of IFWA aggregation and RM training.

    Convergence is guaranteed (Proposition 2) when λ₃ L_e L_φ < 1.
    In practice convergence is observed within 2–3 rounds (Figure 5).
    """

    def __init__(
        self,
        pi_matrix: np.ndarray,
        lambda1: float = 0.8,
        lambda2: float = 0.5,
        lambda3: float = 0.3,
        tau: float = 0.15,
    ):
        """
        Args:
            pi_matrix: (N, K) hesitation matrix from IFN mapping.
            lambda1:   Mean-hesitation penalty coefficient.
            lambda2:   Hesitation-variance penalty coefficient.
            lambda3:   RM error-rate penalty coefficient.
            tau:       Hesitation threshold for D_clean.
        """
        self.pi_matrix = pi_matrix
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.tau = tau

        # Initial weights (Phase 2: no error rates yet)
        self.weights = compute_weights(pi_matrix, lambda1, lambda2)
        self.history: list[np.ndarray] = [self.weights.copy()]

    def refine(
        self,
        pi_tilde: np.ndarray,
        annotator_labels: np.ndarray,
        rm_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        One refinement step (Phase 5).

        Args:
            pi_tilde:          (N,) aggregated hesitation after current IFWA.
            annotator_labels:  (N, K) hard votes {0, 1, NaN} per annotator.
            rm_predictions:    (N,) binary RM predictions (0 or 1).

        Returns:
            Updated (K,) weight vector.
        """
        clean_mask = pi_tilde < self.tau
        N_clean = clean_mask.sum()

        if N_clean < 10:
            # Not enough clean examples – keep current weights
            return self.weights

        clean_preds = rm_predictions[clean_mask]           # (N_clean,)
        clean_labels = annotator_labels[clean_mask, :]     # (N_clean, K)

        K = clean_labels.shape[1]
        error_rates = np.empty(K)
        for k in range(K):
            valid = ~np.isnan(clean_labels[:, k])
            if valid.sum() < 5:
                error_rates[k] = 0.5
                continue
            error_rates[k] = float(
                (clean_labels[valid, k] != clean_preds[valid]).mean()
            )

        self.weights = compute_weights(
            self.pi_matrix, self.lambda1, self.lambda2,
            error_rates=error_rates, lambda3=self.lambda3,
        )
        self.history.append(self.weights.copy())
        return self.weights

    def weight_history(self) -> np.ndarray:
        """Return stacked weight history of shape (rounds, K)."""
        return np.stack(self.history, axis=0)
