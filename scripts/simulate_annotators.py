"""
Synthetic annotator construction (Paper §5.1).

Five annotator types:
  Expert         – low flip rate (p=0.05), high confidence
  Novice-1       – moderate flip rate (p=0.25), moderate confidence
  Novice-2       – high flip rate (p=0.35), lower confidence
  Length-biased  – always prefers longer response, high uniform confidence
  Hesitant       – correct 70% but always votes 'tie' with low confidence

Confidence generation: c_k ~ clip(N(1−p_flip, σ²), 0.10, 0.95)
This gives Pearson r ≈ 0.65–0.70 between c_k and annotator correctness.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal


AnnotatorType = Literal["expert", "novice1", "novice2", "length_biased", "hesitant"]


@dataclass
class SyntheticAnnotatorSpec:
    name: str
    flip_prob: float          # p_k^flip
    conf_mean: float          # mean of confidence distribution
    conf_std: float           # std of confidence distribution
    always_tie: bool = False  # Hesitant annotator
    length_bias: bool = False # Length-biased annotator


ANNOTATOR_SPECS: dict[AnnotatorType, SyntheticAnnotatorSpec] = {
    "expert":        SyntheticAnnotatorSpec("Expert",        flip_prob=0.05, conf_mean=0.90, conf_std=0.10),
    "novice1":       SyntheticAnnotatorSpec("Novice-1",      flip_prob=0.25, conf_mean=0.72, conf_std=0.12),
    "novice2":       SyntheticAnnotatorSpec("Novice-2",      flip_prob=0.35, conf_mean=0.62, conf_std=0.12),
    "length_biased": SyntheticAnnotatorSpec("Length-biased", flip_prob=0.00, conf_mean=0.85, conf_std=0.10, length_bias=True),
    "hesitant":      SyntheticAnnotatorSpec("Hesitant",      flip_prob=0.30, conf_mean=0.10, conf_std=0.00, always_tie=True),
}


def _sample_confidence(
    spec: SyntheticAnnotatorSpec,
    N: int,
    rng: np.random.Generator,
    calibration: Literal["well", "miscalibrated", "inverse"] = "well",
) -> np.ndarray:
    """
    Sample per-instance confidence under three calibration regimes (§5.1).
    """
    if spec.always_tie:
        return np.full(N, spec.conf_mean)

    if calibration == "well":
        c = rng.normal(spec.conf_mean, spec.conf_std, size=N)
    elif calibration == "miscalibrated":
        c = rng.uniform(0.3, 1.0, size=N)
    elif calibration == "inverse":
        # Noisiest annotators are most confident (adversarial)
        c = np.full(N, spec.flip_prob)
    else:
        raise ValueError(f"Unknown calibration: {calibration!r}")

    return np.clip(c, 0.10, 0.95)


def simulate_annotations(
    ground_truth: np.ndarray,
    annotator_types: list[AnnotatorType] | None = None,
    noise_override: dict[str, float] | None = None,
    calibration: Literal["well", "miscalibrated", "inverse"] = "well",
    response_lengths: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic multi-annotator annotations.

    Args:
        ground_truth:   (N,) binary ground-truth labels (0 or 1).
        annotator_types: List of annotator type keys.  Defaults to all five.
        noise_override: Override flip probabilities, e.g. {'novice1': 0.10}.
        calibration:    Confidence calibration regime.
        response_lengths: (N,) length difference |len_A − len_B|; used by
                         length-biased annotator to determine preference.
                         If None, random length preference is simulated.
        seed:           Random seed.

    Returns:
        votes:      (N, K) array of vote strings ('A', 'B', 'tie') as object.
        confs:      (N, K) float array of confidence scores.
        flip_mask:  (N, K) bool array marking which labels were flipped.
    """
    if annotator_types is None:
        annotator_types = list(ANNOTATOR_SPECS.keys())

    rng = np.random.default_rng(seed)
    N = len(ground_truth)
    K = len(annotator_types)

    votes = np.empty((N, K), dtype=object)
    confs = np.empty((N, K), dtype=float)
    flipped = np.zeros((N, K), dtype=bool)

    for k, atype in enumerate(annotator_types):
        spec = ANNOTATOR_SPECS[atype].copy() if hasattr(ANNOTATOR_SPECS[atype], 'copy') else ANNOTATOR_SPECS[atype]

        # Allow per-run flip probability override
        flip_p = spec.flip_prob
        if noise_override and spec.name in noise_override:
            flip_p = noise_override[spec.name]
        if noise_override and atype in noise_override:
            flip_p = noise_override[atype]

        c = _sample_confidence(spec, N, rng, calibration=calibration)
        confs[:, k] = c

        if spec.always_tie:
            # Hesitant: always outputs tie regardless of ground truth
            votes[:, k] = "tie"
            continue

        if spec.length_bias:
            # Length-biased: prefer longer response
            if response_lengths is not None:
                # Positive length diff → A is longer → vote A
                length_vote = np.where(response_lengths >= 0, "A", "B")
            else:
                length_vote = rng.choice(["A", "B"], size=N)
            votes[:, k] = length_vote
            continue

        # Standard annotator: flip ground-truth with probability p
        flip = rng.random(N) < flip_p
        flipped[:, k] = flip
        labels = np.where(flip, 1 - ground_truth, ground_truth)
        votes[:, k] = np.where(labels == 1, "A", "B")

    return votes, confs, flipped


@dataclass
class SyntheticAnnotatorSpec:
    """Re-declared with __copy__ for safe mutation."""
    name: str
    flip_prob: float
    conf_mean: float
    conf_std: float
    always_tie: bool = False
    length_bias: bool = False

    def copy(self) -> "SyntheticAnnotatorSpec":
        import copy
        return copy.copy(self)


# Re-populate with the corrected class
ANNOTATOR_SPECS: dict[AnnotatorType, SyntheticAnnotatorSpec] = {
    "expert":        SyntheticAnnotatorSpec("Expert",        flip_prob=0.05, conf_mean=0.90, conf_std=0.10),
    "novice1":       SyntheticAnnotatorSpec("Novice-1",      flip_prob=0.25, conf_mean=0.72, conf_std=0.12),
    "novice2":       SyntheticAnnotatorSpec("Novice-2",      flip_prob=0.35, conf_mean=0.62, conf_std=0.12),
    "length_biased": SyntheticAnnotatorSpec("Length-biased", flip_prob=0.00, conf_mean=0.85, conf_std=0.10, length_bias=True),
    "hesitant":      SyntheticAnnotatorSpec("Hesitant",      flip_prob=0.30, conf_mean=0.10, conf_std=0.00, always_tie=True),
}


def majority_vote(votes: np.ndarray) -> np.ndarray:
    """
    Baseline majority-vote aggregation.

    Args:
        votes: (N, K) object array of 'A', 'B', or 'tie'.

    Returns:
        (N,) binary labels {0=B wins, 1=A wins}; ties → 0.5 treated as 0.
    """
    N, K = votes.shape
    results = np.zeros(N, dtype=int)
    for n in range(N):
        count_A = (votes[n] == "A").sum()
        count_B = (votes[n] == "B").sum()
        results[n] = int(count_A > count_B)
    return results
