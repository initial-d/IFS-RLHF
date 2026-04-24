"""
Credal Reward Sets and reward partial identification.

Paper §3.4 – Definition 4, Proposition 3, Theorem 1.

A Credal Reward Set is the interval of preference probabilities consistent
with the annotation evidence:
    C^{(n)} = [μ̃^{(n)}, 1 − ν̃^{(n)}]  ⊂ [0, 1]

Width = π̃^{(n)}  (aggregated hesitation).

Key results proved in the paper
--------------------------------
Proposition 3 (Identification Consistency):
  1. IFS soft label  s = μ̃/(μ̃+ν̃)  always lies inside C^{(n)}.
  2. Hard label  ℓ ∈ {0,1}  lies *outside* C^{(n)} whenever π̃>0 and ν̃>0.

Theorem 1 (Strict Excess Risk):
  The hard (or ε-smoothed) label incurs strictly higher minimax cross-
  entropy risk over C^{(n)} than the IFS soft label.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CredalRewardSet:
    """
    Credal Reward Set for a single preference example.

    Attributes:
        lower:  μ̃  – lower bound (Belief / support probability).
        upper:  1−ν̃ – upper bound (Plausibility).
        width:  π̃   – hesitation / partial identification interval width.
        soft_label: s = μ̃/(μ̃+ν̃) – IFS-consistent soft label.
    """
    mu_tilde: float
    nu_tilde: float

    @property
    def lower(self) -> float:
        return self.mu_tilde

    @property
    def upper(self) -> float:
        return 1.0 - self.nu_tilde

    @property
    def width(self) -> float:
        """π̃ = upper − lower = 1 − μ̃ − ν̃."""
        return max(0.0, self.upper - self.lower)

    @property
    def soft_label(self) -> float:
        """s = μ̃ / (μ̃ + ν̃); always inside C (Proposition 3, Part 1)."""
        denom = self.mu_tilde + self.nu_tilde
        return self.mu_tilde / denom if denom > 1e-9 else 0.5

    def contains(self, p: float) -> bool:
        """Check whether probability p ∈ C^{(n)}."""
        return self.lower <= p <= self.upper

    # ------------------------------------------------------------------
    # Proposition 3 verification
    # ------------------------------------------------------------------

    def soft_label_is_consistent(self) -> bool:
        """
        Paper Proposition 3, Part 1:
            s^{(n)} ∈ C^{(n)}  always.
        """
        return self.contains(self.soft_label)

    def hard_label_is_inconsistent(self) -> bool | None:
        """
        Paper Proposition 3, Part 2:
            ℓ ∈ {0,1} lies outside C^{(n)} whenever π̃>0 and ν̃>0.

        Returns:
            True  if the hard label is provably outside C.
            False if it happens to fall inside (only when ν̃=0 or π̃=0).
            None  if the condition cannot be determined (degenerate).
        """
        if self.width <= 0:
            return None   # C collapses to a point; hard label equals it
        hard_label = 1.0 if self.mu_tilde > self.nu_tilde else 0.0
        return not self.contains(hard_label)

    def minimax_cross_entropy(self, p_hat: float) -> float:
        """
        Worst-case cross-entropy loss of a predicted label p_hat over C.

        max_{p* ∈ C} L_CE(p_hat; p*) =
            max(L_CE(p_hat; lower), L_CE(p_hat; upper))
        because L_CE is linear in p*.
        """
        def ce(p_hat: float, p_star: float) -> float:
            p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
            return -(p_star * np.log(p_hat) + (1 - p_star) * np.log(1 - p_hat))

        return max(ce(p_hat, self.lower), ce(p_hat, self.upper))


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def build_credal_sets(
    mu_tilde: np.ndarray,
    nu_tilde: np.ndarray,
) -> list[CredalRewardSet]:
    """Build a list of CredalRewardSets from batch arrays."""
    return [
        CredalRewardSet(float(mu), float(nu))
        for mu, nu in zip(mu_tilde, nu_tilde)
    ]


def identification_consistency_stats(
    mu_tilde: np.ndarray,
    nu_tilde: np.ndarray,
) -> dict:
    """
    Compute dataset-level identification-consistency statistics.

    Returns a dict with:
      'soft_pct_consistent':   fraction where s^{(n)} ∈ C (should be 1.0)
      'hard_pct_inconsistent': fraction where ℓ ∉ C (expected > 0 when π̃>0)
      'mean_width':            mean credal interval width = mean π̃
      'pct_partially_identified': fraction with π̃ > 0
    """
    sets = build_credal_sets(mu_tilde, nu_tilde)

    soft_ok = [s.soft_label_is_consistent() for s in sets]
    hard_bad = [s.hard_label_is_inconsistent() for s in sets if s.hard_label_is_inconsistent() is not None]
    widths = np.array([s.width for s in sets])

    return {
        "soft_pct_consistent":    float(np.mean(soft_ok)),
        "hard_pct_inconsistent":  float(np.mean(hard_bad)) if hard_bad else 0.0,
        "mean_width":             float(widths.mean()),
        "pct_partially_identified": float((widths > 1e-6).mean()),
    }
