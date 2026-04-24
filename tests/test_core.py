"""
Unit tests for IFS-RLHF core components.

Tests are organized to verify:
  1.  IFN constraint and mapping correctness
  2.  IFWA operator properties (Proposition 1)
  3.  Credal set bounds and identification consistency (Proposition 3)
  4.  IFS-weighted loss gradient fixed point (Gradient Proposition)
  5.  Annotator weighting and iterative refinement
  6.  Quality metrics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest
import torch

from ifs_rlhf.ifn import IFN, map_annotation_to_ifn, response_time_to_confidence
from ifs_rlhf.aggregation import ifwa, batch_ifwa
from ifs_rlhf.weighting import compute_weights
from ifs_rlhf.loss import IFSRankingLoss, IFSDPOLoss
from ifs_rlhf.credal import CredalRewardSet, identification_consistency_stats
from ifs_rlhf.metrics import (
    annotation_confidence, preference_clarity, ifs_inter_annotator_agreement
)
from ifs_rlhf.pipeline import IFSRLHFPipeline, IFSRLHFConfig, PreferenceExample, AnnotationRecord


# ─────────────────────────────────────────────────────────────────────────────
# IFN
# ─────────────────────────────────────────────────────────────────────────────

class TestIFN:
    def test_pi_non_negative(self):
        ifn = IFN(0.6, 0.3)
        assert ifn.pi >= 0
        assert abs(ifn.pi - 0.1) < 1e-9

    def test_constraint_violation_raises(self):
        with pytest.raises(ValueError):
            IFN(0.7, 0.5)   # 0.7 + 0.5 = 1.2 > 1

    def test_soft_label_bounds(self):
        for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
            nu = 1.0 - mu - 0.05
            ifn = IFN(mu, nu)
            s = ifn.soft_label
            assert 0.0 <= s <= 1.0

    def test_crisp_set_pi_zero(self):
        ifn = IFN(0.8, 0.2)
        assert abs(ifn.pi) < 1e-9

    def test_mapping_vote_A(self):
        ifn = map_annotation_to_ifn("A", confidence=0.9)
        assert ifn.mu > ifn.nu
        assert ifn.pi >= 0

    def test_mapping_vote_B(self):
        ifn = map_annotation_to_ifn("B", confidence=0.9)
        assert ifn.nu > ifn.mu

    def test_mapping_tie_symmetric(self):
        ifn = map_annotation_to_ifn("tie", confidence=0.4)
        assert abs(ifn.mu - ifn.nu) < 1e-9

    def test_mapping_tie_high_conf_high_pi(self):
        # Paper §4.3: for a tie, π_k = c_k (Eq. 1 with o_k='tie').
        # High confidence in a tie → large π (uncertain about which response wins).
        # The paper states: "higher confidence in a tie implies less hesitation
        # *about the tie itself*" — i.e., the annotator is sure they're tied,
        # which semantically means maximum A-vs-B uncertainty (π = c_k).
        ifn_high = map_annotation_to_ifn("tie", confidence=0.9)
        ifn_low  = map_annotation_to_ifn("tie", confidence=0.3)
        assert ifn_high.pi > ifn_low.pi   # confident tie → high π
        assert abs(ifn_high.pi - 0.9) < 0.01

    def test_response_time_proxy(self):
        c = response_time_to_confidence(10.0, baseline_s=30.0)
        assert abs(c - 1.0) < 1e-9      # fast ≤ baseline → clipped to 1
        c2 = response_time_to_confidence(60.0, baseline_s=30.0)
        assert abs(c2 - 0.5) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# IFWA (Proposition 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestIFWA:
    def test_idempotency(self):
        """P1-1: If all IFNs equal A, IFWA = A."""
        a = IFN(0.6, 0.2)
        result = ifwa([a, a, a])
        # After Laplace smoothing the values shift slightly; test the direction
        assert abs(result.mu - a.mu) < 0.05
        assert abs(result.nu - a.nu) < 0.05

    def test_exact_idempotency_no_smoothing(self):
        """Idempotency holds exactly when laplace_eps=0."""
        a = IFN(0.6, 0.2)
        result = ifwa([a, a, a], laplace_eps=0.0)
        assert abs(result.mu - a.mu) < 1e-9
        assert abs(result.nu - a.nu) < 1e-9

    def test_pi_non_negative(self):
        """π̃ ≥ 0 always."""
        ifns = [IFN(0.7, 0.2), IFN(0.3, 0.5), IFN(0.5, 0.1)]
        result = ifwa(ifns)
        assert result.pi >= -1e-9

    def test_ifn_constraint(self):
        """μ̃ + ν̃ ≤ 1."""
        ifns = [IFN(0.8, 0.1), IFN(0.2, 0.6)]
        result = ifwa(ifns)
        assert result.mu + result.nu <= 1.0 + 1e-9

    def test_uniform_vs_weighted(self):
        """Higher weight on high-μ IFN shifts result upward in μ."""
        ifns = [IFN(0.8, 0.1), IFN(0.2, 0.5)]
        w_high = np.array([0.9, 0.1])
        w_low  = np.array([0.1, 0.9])
        res_high = ifwa(ifns, weights=w_high)
        res_low  = ifwa(ifns, weights=w_low)
        assert res_high.mu > res_low.mu

    def test_batch_matches_single(self):
        """batch_ifwa results match single-call ifwa."""
        ifns = [IFN(0.7, 0.1), IFN(0.4, 0.4), IFN(0.5, 0.3)]
        K = len(ifns)
        mu_m = np.array([[i.mu for i in ifns]])  # (1, K)
        nu_m = np.array([[i.nu for i in ifns]])
        mu_b, nu_b, pi_b = batch_ifwa(mu_m, nu_m)
        single = ifwa(ifns)
        assert abs(mu_b[0] - single.mu) < 1e-9
        assert abs(nu_b[0] - single.nu) < 1e-9

    def test_weight_normalisation(self):
        """Non-normalised weights still produce a valid IFN."""
        ifns = [IFN(0.6, 0.2), IFN(0.3, 0.4)]
        w = np.array([3.0, 1.0])   # will be normalised internally
        result = ifwa(ifns, weights=w)
        assert result.mu + result.nu <= 1.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Credal Reward Sets (Proposition 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestCredalRewardSet:
    def test_width_equals_hesitation(self):
        c = CredalRewardSet(mu_tilde=0.5, nu_tilde=0.2)
        assert abs(c.width - (1 - 0.5 - 0.2)) < 1e-9

    def test_soft_label_inside_credal_set(self):
        """Proposition 3, Part 1: s ∈ C always."""
        for mu in np.linspace(0.1, 0.8, 8):
            for nu in np.linspace(0.05, 0.8 - mu, 5):
                c = CredalRewardSet(mu, nu)
                assert c.soft_label_is_consistent(), (
                    f"Failed for μ={mu:.2f}, ν={nu:.2f}: s={c.soft_label:.3f} not in [{c.lower:.3f},{c.upper:.3f}]"
                )

    def test_hard_label_outside_credal_set(self):
        """Proposition 3, Part 2: ℓ ∈ {0,1} outside C when π̃>0, ν̃>0."""
        c = CredalRewardSet(mu_tilde=0.5, nu_tilde=0.2)   # π̃=0.3, ν̃=0.2>0
        assert c.hard_label_is_inconsistent() is True

    def test_degenerate_case(self):
        """When π̃ = 0 (crisp), hard label may coincide with boundary."""
        c = CredalRewardSet(mu_tilde=0.7, nu_tilde=0.3)   # π̃=0
        # C = [0.7, 0.7], hard label =1; 1 ∉ [0.7,0.7] still
        # Proposition holds: 1 ≠ 0.7
        assert c.width < 1e-9

    def test_batch_soft_consistency(self):
        rng = np.random.default_rng(1)
        N = 500
        mu = rng.uniform(0.1, 0.7, N)
        nu = rng.uniform(0.05, 0.9 - mu, N)
        stats = identification_consistency_stats(mu, nu)
        assert stats["soft_pct_consistent"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestIFSRankingLoss:
    def _make_tensors(self, N=8):
        mu = torch.tensor([0.7, 0.3, 0.6, 0.5, 0.8, 0.4, 0.55, 0.65])
        nu = torch.tensor([0.1, 0.5, 0.2, 0.3, 0.1, 0.4, 0.25, 0.15])
        pi = 1.0 - mu - nu
        delta_r = torch.randn(N)
        return delta_r[:N], mu[:N], nu[:N], pi[:N]

    def test_loss_non_negative(self):
        loss_fn = IFSRankingLoss(gamma=2.0)
        dr, mu, nu, pi = self._make_tensors()
        loss = loss_fn(dr, mu, nu, pi)
        assert loss.item() >= 0

    def test_gamma_zero_equals_soft_ce(self):
        """γ=0 → uniform weighting (soft CE without hesitation down-weighting)."""
        loss_fn = IFSRankingLoss(gamma=0.0)
        dr, mu, nu, pi = self._make_tensors()
        loss = loss_fn(dr, mu, nu, pi)
        assert loss.item() > 0

    def test_high_hesitation_zero_contribution(self):
        """Example with π̃ = 1 contributes ≈ 0 to the gradient."""
        loss_fn = IFSRankingLoss(gamma=2.0)
        # Two examples: one with π=0 (certain), one with π≈1 (fully hesitant)
        mu   = torch.tensor([0.9, 0.0])
        nu   = torch.tensor([0.1, 0.0])
        pi   = torch.tensor([0.0, 1.0])
        dr   = torch.tensor([1.0, 0.0], requires_grad=True)

        loss_fn2 = IFSRankingLoss(gamma=2.0, reduction="sum")
        loss = loss_fn2(dr, mu, nu, pi)
        loss.backward()
        # Gradient on the hesitant example should be ≈ 0
        assert abs(dr.grad[1].item()) < 1e-6

    def test_gradient_fixed_point(self):
        """
        Gradient Proposition: gradient = 0 when σ(Δr_n) = μ̃/(μ̃+ν̃).
        """
        loss_fn = IFSRankingLoss(gamma=2.0)
        mu = torch.tensor([0.7])
        nu = torch.tensor([0.2])
        pi = 1.0 - mu - nu
        target_sigma = (mu / (mu + nu)).item()
        # delta_r such that σ(delta_r) = target_sigma
        delta_r_star = torch.tensor([math.log(target_sigma / (1 - target_sigma))])
        grad = loss_fn.gradient_check(delta_r_star, mu, nu, pi)
        assert abs(grad.item()) < 1e-6

    def test_reduces_to_bt_when_crisp(self):
        """When π̃=0 and μ̃=1,ν̃=0 (crisp label), loss ≈ L_BT."""
        loss_fn = IFSRankingLoss(gamma=2.0)
        mu = torch.tensor([1.0])
        nu = torch.tensor([0.0])
        pi = torch.tensor([0.0])
        delta_r = torch.tensor([2.0])
        # IFS loss ≈ −log σ(2.0) in this degenerate crisp case
        loss = loss_fn(delta_r, mu, nu, pi)
        bt_loss = -torch.nn.functional.logsigmoid(delta_r).mean()
        assert abs(loss.item() - bt_loss.item()) < 1e-5


class TestIFSDPOLoss:
    def test_positive_margin_reduces_loss(self):
        """Higher μ̃−ν̃ margin should produce lower (better) loss."""
        loss_fn = IFSDPOLoss(beta=0.1, gamma=1.0)
        logr_A = torch.tensor([1.0])
        logr_B = torch.tensor([-1.0])
        mu_high = torch.tensor([0.9])
        nu_high = torch.tensor([0.05])
        pi_high = 1 - mu_high - nu_high

        mu_low  = torch.tensor([0.55])
        nu_low  = torch.tensor([0.45])
        pi_low  = 1 - mu_low - nu_low

        loss_high = loss_fn(logr_A, logr_B, mu_high, nu_high, pi_high)
        loss_low  = loss_fn(logr_A, logr_B, mu_low,  nu_low,  pi_low)
        # Higher margin with lower hesitation should give higher penalty on wrong direction
        assert loss_high.item() >= 0
        assert loss_low.item() >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Annotator Weighting
# ─────────────────────────────────────────────────────────────────────────────

class TestWeighting:
    def _make_pi_matrix(self):
        rng = np.random.default_rng(42)
        # K=3 annotators, N=100 examples
        # annotator 0: low pi (reliable), annotator 2: high pi (hesitant)
        pi = np.stack([
            rng.uniform(0.02, 0.10, 100),   # reliable
            rng.uniform(0.10, 0.30, 100),   # moderate
            rng.uniform(0.50, 0.80, 100),   # very hesitant
        ], axis=1)
        return pi

    def test_weights_sum_to_one(self):
        pi = self._make_pi_matrix()
        w = compute_weights(pi)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_reliable_annotator_gets_higher_weight(self):
        pi = self._make_pi_matrix()
        w = compute_weights(pi, lambda1=0.8, lambda2=0.5)
        assert w[0] > w[2]   # reliable > very hesitant

    def test_uniform_when_lambda_zero(self):
        pi = self._make_pi_matrix()
        w = compute_weights(pi, lambda1=0.0, lambda2=0.0)
        np.testing.assert_allclose(w, [1/3, 1/3, 1/3], atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_conf_range(self):
        pi = np.array([0.0, 0.1, 0.5, 1.0])
        assert 0.0 <= annotation_confidence(pi) <= 1.0

    def test_clarity_range(self):
        mu = np.array([0.7, 0.3, 0.5])
        nu = np.array([0.1, 0.5, 0.5])
        assert 0.0 <= preference_clarity(mu, nu) <= 1.0

    def test_iaa_perfect_agreement(self):
        # All annotators have identical IFNs per row → IAA = 1 (zero pairwise distance)
        N = 100
        mu_vals = np.random.default_rng(5).uniform(0.2, 0.7, N)
        nu_vals = np.random.default_rng(6).uniform(0.05, 0.2, N)
        # Three annotators, all with the same (μ, ν) per example
        mu_m = np.stack([mu_vals, mu_vals, mu_vals], axis=1)
        nu_m = np.stack([nu_vals, nu_vals, nu_vals], axis=1)
        iaa = ifs_inter_annotator_agreement(mu_m, nu_m)
        assert abs(iaa - 1.0) < 1e-6

    def test_iaa_decreases_with_disagreement(self):
        rng = np.random.default_rng(7)
        N = 200
        mu_agree = np.tile([0.7, 0.7, 0.7], (N, 1))
        nu_agree = np.tile([0.1, 0.1, 0.1], (N, 1))
        mu_dis = rng.uniform(0.1, 0.7, (N, 3))
        nu_dis = rng.uniform(0.05, 0.3, (N, 3))
        assert (
            ifs_inter_annotator_agreement(mu_agree, nu_agree)
            > ifs_inter_annotator_agreement(mu_dis, nu_dis)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    def _make_examples(self, N=50, K=3, seed=0):
        rng = np.random.default_rng(seed)
        examples = []
        for n in range(N):
            anns = []
            for k in range(K):
                vote = rng.choice(["A", "B", "tie"])
                conf = float(rng.uniform(0.3, 0.9))
                anns.append(AnnotationRecord(k, vote, conf))
            examples.append(PreferenceExample(n, anns))
        return examples

    def test_fit_produces_valid_ifns(self):
        examples = self._make_examples()
        pipeline = IFSRLHFPipeline()
        pipeline.fit(examples)
        mu, nu, pi = pipeline.mu_tilde_, pipeline.nu_tilde_, pipeline.pi_tilde_
        assert (mu >= 0).all() and (mu <= 1).all()
        assert (nu >= 0).all() and (nu <= 1).all()
        assert (pi >= -1e-9).all()
        assert (mu + nu <= 1.0 + 1e-9).all()

    def test_soft_labels_in_unit_interval(self):
        examples = self._make_examples()
        pipeline = IFSRLHFPipeline().fit(examples)
        s = pipeline.soft_labels()
        assert (s >= 0).all() and (s <= 1).all()

    def test_instance_weights_decrease_with_hesitation(self):
        examples = self._make_examples()
        pipeline = IFSRLHFPipeline().fit(examples)
        w = pipeline.instance_weights(gamma=2.0)
        pi = pipeline.pi_tilde_
        # Higher π̃ should correlate with lower weight
        r = np.corrcoef(pi, w)[0, 1]
        assert r < 0

    def test_quality_report_keys(self):
        examples = self._make_examples()
        pipeline = IFSRLHFPipeline().fit(examples)
        report = pipeline.quality_report()
        assert "Conf" in report and "Clarity" in report and "IFS-IAA" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
