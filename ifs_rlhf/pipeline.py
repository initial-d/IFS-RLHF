"""
IFS-RLHF pipeline – Algorithm 1 in the paper.

Phases:
  1. IFN Mapping          (§4.3 / Eq. 1)
  2. Annotator Weighting  (§4.4 / Eqs. 6–7)
  3. IFWA Aggregation     (§4.4 / Eqs. 3–5)
  4. RM Training          (§4.5 / IFS-weighted loss)
  5. Iterative Refinement (§4.4 / Phase 5, optional)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .ifn import IFN, Vote, map_annotation_to_ifn
from .aggregation import batch_ifwa
from .weighting import compute_weights, IterativeWeightRefiner
from .credal import identification_consistency_stats
from .metrics import dataset_quality_report


@dataclass
class AnnotationRecord:
    """A single annotator's contribution to one preference example."""
    annotator_id: int
    vote: Vote
    confidence: float     # ∈ [0, 1]


@dataclass
class PreferenceExample:
    """One (x, y_A, y_B) triple with K annotator records."""
    example_id: int
    annotations: list[AnnotationRecord]

    # Filled after IFN mapping
    ifns: list[IFN] = field(default_factory=list)
    # Filled after aggregation
    mu_tilde: float = 0.0
    nu_tilde: float = 0.0
    pi_tilde: float = 0.0

    @property
    def soft_label(self) -> float:
        denom = self.mu_tilde + self.nu_tilde
        return self.mu_tilde / denom if denom > 1e-9 else 0.5

    @property
    def hard_label(self) -> int:
        return int(self.mu_tilde > self.nu_tilde)


@dataclass
class IFSRLHFConfig:
    """Hyperparameters matching paper experiments."""
    delta: float = 0.05       # skepticism floor (IFN mapping)
    lambda1: float = 0.8      # mean-hesitation weight
    lambda2: float = 0.5      # hesitation-variance weight
    lambda3: float = 0.3      # error-rate weight (Phase 5)
    gamma: float = 2.0        # focusing exponent (RM loss)
    beta: float = 0.01        # KL regularisation
    tau: float = 0.15         # D_clean threshold
    laplace_eps: float = 1e-3 # IFWA Laplace smoothing


class IFSRLHFPipeline:
    """
    Pure-NumPy implementation of Phases 1–3 and quality diagnostics.

    RM training (Phase 4) is handled by separate PyTorch training loops
    in train/train_rm.py so that this class stays framework-agnostic.
    """

    def __init__(self, config: IFSRLHFConfig | None = None) -> None:
        self.config = config or IFSRLHFConfig()
        self.weights_: Optional[np.ndarray] = None
        self.refiner_: Optional[IterativeWeightRefiner] = None

        # Filled after fit()
        self.mu_matrix_: Optional[np.ndarray] = None
        self.nu_matrix_: Optional[np.ndarray] = None
        self.pi_matrix_: Optional[np.ndarray] = None
        self.mu_tilde_: Optional[np.ndarray] = None
        self.nu_tilde_: Optional[np.ndarray] = None
        self.pi_tilde_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Phase 1: IFN Mapping
    # ------------------------------------------------------------------

    def _map_ifns(self, examples: list[PreferenceExample]) -> None:
        """Map raw annotations to IFN matrices (N × K)."""
        N = len(examples)
        K = max(len(ex.annotations) for ex in examples)
        cfg = self.config

        mu = np.full((N, K), np.nan)
        nu = np.full((N, K), np.nan)

        for n, ex in enumerate(examples):
            for ann in ex.annotations:
                k = ann.annotator_id
                ifn = map_annotation_to_ifn(
                    ann.vote, ann.confidence, delta=cfg.delta
                )
                ex.ifns.append(ifn)
                mu[n, k] = ifn.mu
                nu[n, k] = ifn.nu

        # Replace NaN (missing annotations) with neutral (0.5, 0.5) → π=0
        # so they don't distort aggregation; treat as abstentions
        mu_filled = np.where(np.isnan(mu), 0.5, mu)
        nu_filled = np.where(np.isnan(nu), 0.5, nu)

        self.mu_matrix_ = mu_filled
        self.nu_matrix_ = nu_filled
        self.pi_matrix_ = np.clip(1.0 - mu_filled - nu_filled, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Phase 2: Annotator Weighting
    # ------------------------------------------------------------------

    def _compute_weights(self) -> None:
        cfg = self.config
        self.weights_ = compute_weights(
            self.pi_matrix_, lambda1=cfg.lambda1, lambda2=cfg.lambda2
        )
        self.refiner_ = IterativeWeightRefiner(
            self.pi_matrix_, lambda1=cfg.lambda1,
            lambda2=cfg.lambda2, lambda3=cfg.lambda3, tau=cfg.tau,
        )

    # ------------------------------------------------------------------
    # Phase 3: IFWA Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self) -> None:
        cfg = self.config
        mu_t, nu_t, pi_t = batch_ifwa(
            self.mu_matrix_, self.nu_matrix_,
            weights=self.weights_, laplace_eps=cfg.laplace_eps,
        )
        self.mu_tilde_ = mu_t
        self.nu_tilde_ = nu_t
        self.pi_tilde_ = pi_t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, examples: list[PreferenceExample]) -> "IFSRLHFPipeline":
        """
        Run Phases 1–3 on the dataset.

        After calling fit(), access aggregated IFNs via:
          self.mu_tilde_, self.nu_tilde_, self.pi_tilde_   (N,) arrays
          self.soft_labels()     → (N,) soft labels s^{(n)}
          self.hard_labels()     → (N,) hard labels ℓ^{(n)} ∈ {0,1}
          self.instance_weights(gamma)  → (N,) loss weights (1−π̃)^γ
        """
        self._map_ifns(examples)
        self._compute_weights()
        self._aggregate()

        # Write aggregated values back to examples (for downstream use)
        for n, ex in enumerate(examples):
            ex.mu_tilde = float(self.mu_tilde_[n])
            ex.nu_tilde = float(self.nu_tilde_[n])
            ex.pi_tilde = float(self.pi_tilde_[n])

        return self

    def refine_weights(
        self,
        annotator_labels: np.ndarray,
        rm_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Phase 5: iterative weight refinement using RM error rates.

        Args:
            annotator_labels: (N, K) binary votes {0, 1, NaN}.
            rm_predictions:   (N,) binary RM predictions.

        Returns:
            Updated (K,) weight vector.
        """
        if self.refiner_ is None:
            raise RuntimeError("Call fit() before refine_weights().")

        new_weights = self.refiner_.refine(
            self.pi_tilde_, annotator_labels, rm_predictions
        )
        self.weights_ = new_weights
        self._aggregate()
        return new_weights

    def soft_labels(self) -> np.ndarray:
        """s^{(n)} = μ̃ / (μ̃ + ν̃)."""
        denom = np.clip(self.mu_tilde_ + self.nu_tilde_, 1e-9, None)
        return self.mu_tilde_ / denom

    def hard_labels(self) -> np.ndarray:
        """ℓ^{(n)} = 1[μ̃ > ν̃]."""
        return (self.mu_tilde_ > self.nu_tilde_).astype(int)

    def instance_weights(self, gamma: float | None = None) -> np.ndarray:
        """(1 − π̃)^γ per-example loss weights."""
        if gamma is None:
            gamma = self.config.gamma
        return np.clip(1.0 - self.pi_tilde_, 0.0, 1.0) ** gamma

    def quality_report(self) -> dict:
        """IFS dataset quality metrics (Paper §3.3)."""
        return dataset_quality_report(
            self.mu_tilde_, self.nu_tilde_, self.pi_tilde_,
            mu_matrix=self.mu_matrix_, nu_matrix=self.nu_matrix_,
        )

    def consistency_report(self) -> dict:
        """Identification-consistency stats (Proposition 3)."""
        return identification_consistency_stats(self.mu_tilde_, self.nu_tilde_)
