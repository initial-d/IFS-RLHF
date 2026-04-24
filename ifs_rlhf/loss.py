"""
IFS-weighted reward model loss and IFS-DPO loss.

Paper §4.5 – IFS-Weighted Ranking Loss:
  L_IFS = −(1/N) Σ_n (1−π̃_n)^γ [μ̃_n log σ(Δr_n) + ν̃_n log(1−σ(Δr_n))]

Key properties:
  • π̃_n → 1  ⟹  example contributes ≈ 0  (uncertain pairs down-weighted)
  • π̃_n = 0 with crisp labels  ⟹  reduces to standard Bradley-Terry L_BT
  • Gradient fixed point: σ(Δr_n) = μ̃_n / (μ̃_n + ν̃_n) = s_n  (Gradient Prop.)

Paper §4.6 – IFS-DPO loss (Eq. 8):
  L_IFS-DPO = −(1/N) Σ_n (1−π̃_n)^γ · [μ̃_n − ν̃_n] ·
                  log σ(β log p_θ(y_A|x)/p_ref(y_A|x) − β log p_θ(y_B|x)/p_ref(y_B|x))
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bradley_terry_loss(
    delta_r: torch.Tensor,
) -> torch.Tensor:
    """
    Standard Bradley-Terry loss (hard-label baseline).
    L_BT = −log σ(Δr)
    """
    return F.binary_cross_entropy_with_logits(delta_r, torch.ones_like(delta_r))


# ---------------------------------------------------------------------------
# IFS-Weighted Ranking Loss
# ---------------------------------------------------------------------------

class IFSRankingLoss(nn.Module):
    """
    IFS-weighted reward-model ranking loss (Paper, main loss equation).

    L_IFS = −(1/N) Σ_n w_n [μ̃_n log σ(Δr_n) + ν̃_n log(1−σ(Δr_n))]
    where w_n = (1 − π̃_n)^γ.

    Args:
        gamma:      Focusing exponent γ ≥ 0.  γ=0 gives unweighted soft-label
                    CE; γ=2 is the paper default for RM training.
        reduction:  'mean' or 'sum'.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        delta_r: torch.Tensor,
        mu_tilde: torch.Tensor,
        nu_tilde: torch.Tensor,
        pi_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            delta_r:   (N,) r_φ(x, y_A) − r_φ(x, y_B) for each example.
            mu_tilde:  (N,) aggregated membership.
            nu_tilde:  (N,) aggregated non-membership.
            pi_tilde:  (N,) aggregated hesitation  (= 1 − μ̃ − ν̃).

        Returns:
            Scalar loss.
        """
        # Hesitation-based instance weight  w_n = (1 − π̃_n)^γ
        instance_weight = (1.0 - pi_tilde).clamp(min=0.0) ** self.gamma  # (N,)

        # log σ(Δr) and log(1 − σ(Δr)) = log σ(−Δr)
        log_prob_A = F.logsigmoid(delta_r)          # log P(A ≻ B)
        log_prob_B = F.logsigmoid(-delta_r)         # log P(B ≻ A)

        # Soft cross-entropy term
        soft_ce = -(mu_tilde * log_prob_A + nu_tilde * log_prob_B)  # (N,)

        # Weighted loss
        loss = instance_weight * soft_ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def gradient_check(
        self,
        delta_r: torch.Tensor,
        mu_tilde: torch.Tensor,
        nu_tilde: torch.Tensor,
        pi_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient of L_IFS w.r.t. Δr_n (Gradient Proposition in paper):

          ∂L/∂Δr_n = −(1−π̃_n)^γ [μ̃_n (1−σ(Δr_n)) − ν̃_n σ(Δr_n)]

        Gradient is zero when σ(Δr_n) = μ̃_n / (μ̃_n + ν̃_n).
        """
        w = (1.0 - pi_tilde).clamp(min=0.0) ** self.gamma
        sigma = torch.sigmoid(delta_r)
        grad = -w * (mu_tilde * (1.0 - sigma) - nu_tilde * sigma)
        return grad


# ---------------------------------------------------------------------------
# IFS-DPO Loss
# ---------------------------------------------------------------------------

class IFSDPOLoss(nn.Module):
    """
    IFS-weighted DPO loss (Paper §4.6, Eq. 8).

    Responses are assumed relabelled so that y_A has higher IFS score
    (μ̃_n > ν̃_n), guaranteeing the margin [μ̃_n − ν̃_n] ≥ 0.

    L_IFS-DPO = −(1/N) Σ_n (1−π̃_n)^γ · (μ̃_n − ν̃_n) ·
                    log σ(β · log_ratio_diff_n)

    where log_ratio_diff_n =
        log p_θ(y_A|x)/p_ref(y_A|x) − log p_θ(y_B|x)/p_ref(y_B|x).

    Note: γ=1 is recommended for IFS-DPO because the margin (μ̃−ν̃) already
    provides difficulty weighting; γ=2 over-suppresses borderline pairs.

    Args:
        beta:   KL regularisation temperature β (paper default 0.1).
        gamma:  Focusing exponent; paper recommends γ=1 for DPO.
    """

    def __init__(self, beta: float = 0.1, gamma: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        log_ratio_A_policy: torch.Tensor,
        log_ratio_B_policy: torch.Tensor,
        mu_tilde: torch.Tensor,
        nu_tilde: torch.Tensor,
        pi_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            log_ratio_A_policy: (N,) log p_θ(y_A|x) − log p_ref(y_A|x).
            log_ratio_B_policy: (N,) log p_θ(y_B|x) − log p_ref(y_B|x).
            mu_tilde:  (N,) aggregated membership  (for the relabelled A side).
            nu_tilde:  (N,) aggregated non-membership.
            pi_tilde:  (N,) aggregated hesitation.

        Returns:
            Scalar loss.
        """
        # Hesitation weight  (1 − π̃_n)^γ
        instance_weight = (1.0 - pi_tilde).clamp(min=0.0) ** self.gamma  # (N,)

        # Label-confidence margin  [μ̃_n − ν̃_n]  ≥ 0 by relabelling convention
        margin = (mu_tilde - nu_tilde).clamp(min=0.0)  # (N,)

        # β · (log-ratio for A − log-ratio for B)
        log_ratio_diff = self.beta * (log_ratio_A_policy - log_ratio_B_policy)  # (N,)

        loss = -instance_weight * margin * F.logsigmoid(log_ratio_diff)

        return loss.mean()


# ---------------------------------------------------------------------------
# Convenience: KL regularisation term (paper §4.5, L_total)
# ---------------------------------------------------------------------------

def kl_regularisation(
    reward: torch.Tensor,
    reward_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Squared-distance KL penalty relative to the SFT reward model.

    L_KL = E[(r_φ − r_{φ_0})²]
    """
    return F.mse_loss(reward, reward_ref.detach())
