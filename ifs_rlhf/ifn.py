"""
Intuitionistic Fuzzy Numbers (IFN) and annotation mapping.

Paper §4.3 – IFN Mapping from Raw Annotations (Eq. 1).

An IFN is a triple (μ, ν, π) with:
  μ  ∈ [0, 1]  – membership (support for A)
  ν  ∈ [0, 1]  – non-membership (opposition / support for B)
  π  = 1−μ−ν  – hesitation (epistemic uncertainty)
  constraint: μ + ν ≤ 1  ⟺  π ≥ 0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


Vote = Literal["A", "B", "tie"]


@dataclass
class IFN:
    """A single Intuitionistic Fuzzy Number."""
    mu: float   # membership   (support for A ≻ B)
    nu: float   # non-membership (support for B ≻ A)

    def __post_init__(self) -> None:
        if self.mu < 0 or self.nu < 0:
            raise ValueError("μ and ν must be ≥ 0.")
        if self.mu + self.nu > 1.0 + 1e-9:
            raise ValueError(f"IFN constraint violated: μ+ν = {self.mu+self.nu:.4f} > 1.")
        # clip floating-point noise
        total = self.mu + self.nu
        if total > 1.0:
            self.mu /= total
            self.nu /= total

    @property
    def pi(self) -> float:
        """Hesitation degree π = 1 − μ − ν."""
        return max(0.0, 1.0 - self.mu - self.nu)

    @property
    def soft_label(self) -> float:
        """s = μ / (μ + ν); the IFS-consistent soft preference probability."""
        denom = self.mu + self.nu
        return self.mu / denom if denom > 1e-9 else 0.5

    def __repr__(self) -> str:
        return f"IFN(μ={self.mu:.3f}, ν={self.nu:.3f}, π={self.pi:.3f})"


def map_annotation_to_ifn(
    vote: Vote,
    confidence: float,
    delta: float = 0.05,
    clip_low: float = 0.10,
    clip_high: float | None = None,
) -> IFN:
    """
    Map a single annotator's (vote, confidence) pair to an IFN.

    Paper Eq. (1):
      vote == 'A'   → (c_k, δ)
      vote == 'B'   → (δ, c_k)
      vote == 'tie' → ((1−c_k)/2, (1−c_k)/2)

    Args:
        vote:       Ordinal judgment: 'A', 'B', or 'tie'.
        confidence: Self-reported or proxy confidence ∈ [0, 1].
        delta:      Skepticism floor δ ∈ (0, 0.5]; prevents ν = 0 collapsing
                    the IFWA product.  Default 0.05.
        clip_low:   Minimum confidence after clipping.  Default 0.10.
        clip_high:  Maximum confidence; defaults to 1 − δ.

    Returns:
        IFN with the μ, ν, π components.
    """
    if clip_high is None:
        clip_high = 1.0 - delta
    c = float(np.clip(confidence, clip_low, clip_high))

    if vote == "A":
        mu, nu = c, delta
    elif vote == "B":
        mu, nu = delta, c
    elif vote == "tie":
        half = (1.0 - c) / 2.0
        mu, nu = half, half
    else:
        raise ValueError(f"Unknown vote: {vote!r}. Expected 'A', 'B', or 'tie'.")

    return IFN(mu=mu, nu=nu)


def response_time_to_confidence(
    response_time_s: float,
    baseline_s: float = 30.0,
) -> float:
    """
    Convert response time to a confidence proxy (Drift-Diffusion Model proxy).

    c_k = min(1, t_base / t_k)   (Paper §4.3, Eq. 2)

    Faster responses → higher confidence.
    """
    return float(min(1.0, baseline_s / max(response_time_s, 1e-3)))


def logit_confidence_from_lm(
    prob_A: float,
    prob_B: float,
) -> tuple[float, float]:
    """
    Derive confidence and hesitation from LLM judge token probabilities.

    Used for RLAIF / Chatbot-Arena style elicitation (Paper §4.2, bullet 3):
      c_k  = max(P(A), P(B))
      π_k  = 1 − P(A) − P(B)   (residual mass on neither token)

    Returns:
        (confidence, hesitation)
    """
    hesitation = max(0.0, 1.0 - prob_A - prob_B)
    confidence = max(prob_A, prob_B)
    return confidence, hesitation
