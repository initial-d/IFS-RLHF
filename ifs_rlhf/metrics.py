"""
IFS dataset quality metrics (Paper §3.3).

Three indicators defined over a collection of aggregated IFNs:

  Conf    = 1 − (1/n) Σ π̃_i               annotation confidence
  Clarity = (1/n) Σ |μ̃_i − ν̃_i|          net directional strength
  IFS-IAA = 1 − (2 / K(K−1)) Σ_{i<j} d_IFS(A_i, A_j)

IFS distance (Definition in paper):
  d_IFS(A_i, A_j) = sqrt{ (1/2) [(μ_i−μ_j)² + (ν_i−ν_j)² + (π_i−π_j)²] }
"""

from __future__ import annotations
import numpy as np


def ifs_distance(
    mu_i: float, nu_i: float,
    mu_j: float, nu_j: float,
) -> float:
    """
    Normalised Euclidean distance in IFS space (Definition in §3.3).

    d_IFS(A_i, A_j) = sqrt{ (1/2) [(Δμ)² + (Δν)² + (Δπ)²] }
    """
    pi_i = 1.0 - mu_i - nu_i
    pi_j = 1.0 - mu_j - nu_j
    return float(np.sqrt(0.5 * ((mu_i - mu_j)**2 + (nu_i - nu_j)**2 + (pi_i - pi_j)**2)))


def annotation_confidence(pi_tilde: np.ndarray) -> float:
    """
    Conf = 1 − mean(π̃)  ∈ [0, 1].

    Higher values ↔ annotators were on average decisive.
    """
    return float(1.0 - np.mean(pi_tilde))


def preference_clarity(mu_tilde: np.ndarray, nu_tilde: np.ndarray) -> float:
    """
    Clarity = mean |μ̃ − ν̃|  ∈ [0, 1].

    Measures how unambiguous the net direction of preference is,
    independent of hesitation level.
    """
    return float(np.mean(np.abs(mu_tilde - nu_tilde)))


def ifs_inter_annotator_agreement(
    mu_matrix: np.ndarray,
    nu_matrix: np.ndarray,
) -> float:
    """
    IFS-IAA = 1 − (2 / K(K−1)) Σ_{i<j} d_IFS(A_i, A_j)

    Computed as the mean over all annotator *pairs* per example,
    then averaged across examples.

    Args:
        mu_matrix: (N, K) per-annotator membership values.
        nu_matrix: (N, K) per-annotator non-membership values.

    Returns:
        Scalar IFS-IAA ∈ [0, 1].
    """
    N, K = mu_matrix.shape
    if K < 2:
        raise ValueError("Need at least 2 annotators to compute IFS-IAA.")

    pi_matrix = 1.0 - mu_matrix - nu_matrix

    pair_dists = []
    for i in range(K):
        for j in range(i + 1, K):
            d = np.sqrt(0.5 * (
                (mu_matrix[:, i] - mu_matrix[:, j])**2
                + (nu_matrix[:, i] - nu_matrix[:, j])**2
                + (pi_matrix[:, i] - pi_matrix[:, j])**2
            ))  # (N,)
            pair_dists.append(d)

    mean_pair_dist = float(np.mean(pair_dists))
    return 1.0 - mean_pair_dist


def dataset_quality_report(
    mu_tilde: np.ndarray,
    nu_tilde: np.ndarray,
    pi_tilde: np.ndarray,
    mu_matrix: np.ndarray | None = None,
    nu_matrix: np.ndarray | None = None,
) -> dict:
    """
    Compute all IFS quality metrics for a dataset.

    Args:
        mu_tilde, nu_tilde, pi_tilde: (N,) aggregated IFN components.
        mu_matrix, nu_matrix:          (N, K) per-annotator IFN components
                                       required for IFS-IAA; optional.

    Returns:
        Dictionary with metric names and values.
    """
    report = {
        "Conf":    annotation_confidence(pi_tilde),
        "Clarity": preference_clarity(mu_tilde, nu_tilde),
        "mean_pi": float(pi_tilde.mean()),
        "pct_high_hesitation (>0.4)": float((pi_tilde > 0.4).mean()),
    }
    if mu_matrix is not None and nu_matrix is not None:
        report["IFS-IAA"] = ifs_inter_annotator_agreement(mu_matrix, nu_matrix)
    return report
