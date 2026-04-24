"""
End-to-end demo of IFS-RLHF pipeline.

Reproduces the key empirical patterns from the paper without requiring
a real LLM or GPU:
  • IFN mapping and IFWA aggregation
  • Soft-label vs hard-label accuracy under increasing noise
  • Credal set identification consistency statistics
  • IFS dataset quality metrics
  • Annotator weight evolution across refinement rounds (simulated)

Run:
    python examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ifs_rlhf import (
    IFSRLHFPipeline, IFSRLHFConfig,
    PreferenceExample, AnnotationRecord,
    identification_consistency_stats,
    compute_weights,
)
from scripts.simulate_annotators import simulate_annotations, majority_vote

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Generate synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
N = 2000
ground_truth = rng.integers(0, 2, size=N)   # binary ground-truth labels

print("=" * 65)
print("  IFS-RLHF Demo – Reproducing core paper patterns")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Noise sweep: soft vs hard accuracy (mirrors Table 1 trend)
# ─────────────────────────────────────────────────────────────────────────────
noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
print(f"\n{'Noise p':>9} | {'MV Acc':>8} | {'SL Acc':>8} | {'IFS Acc':>9} | {'IFS-MV':>7}")
print("-" * 55)

for p in noise_levels:
    votes, confs, _ = simulate_annotations(
        ground_truth,
        noise_override={"novice1": p, "novice2": p + 0.10},
        seed=0,
    )

    # Build PreferenceExamples
    examples = []
    K = votes.shape[1]
    for n in range(N):
        anns = [
            AnnotationRecord(annotator_id=k, vote=votes[n, k], confidence=confs[n, k])
            for k in range(K)
        ]
        examples.append(PreferenceExample(example_id=n, annotations=anns))

    # Run pipeline
    pipeline = IFSRLHFPipeline(IFSRLHFConfig())
    pipeline.fit(examples)

    # Evaluate
    mv_labels  = majority_vote(votes)
    ifs_hard   = pipeline.hard_labels()
    soft       = pipeline.soft_labels()
    soft_hard  = (soft > 0.5).astype(int)

    mv_acc   = (mv_labels  == ground_truth).mean()
    sl_acc   = (soft_hard  == ground_truth).mean()
    ifs_acc  = (ifs_hard   == ground_truth).mean()

    print(f"{p:>9.2f} | {mv_acc:>8.3f} | {sl_acc:>8.3f} | {ifs_acc:>9.3f} | {ifs_acc-mv_acc:>+7.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Credal set identification consistency (Proposition 3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Identification Consistency (Proposition 3)")
print("=" * 65)

votes, confs, _ = simulate_annotations(ground_truth, noise_override={"novice1": 0.25, "novice2": 0.35}, seed=0)
examples = [
    PreferenceExample(
        example_id=n,
        annotations=[AnnotationRecord(k, votes[n, k], confs[n, k]) for k in range(votes.shape[1])]
    ) for n in range(N)
]
pipeline = IFSRLHFPipeline()
pipeline.fit(examples)

stats = pipeline.consistency_report()
print(f"  Soft labels inside credal set C:  {stats['soft_pct_consistent']:.4f}  (paper: 1.0000 by proof)")
print(f"  Hard labels outside credal set C: {stats['hard_pct_inconsistent']:.4f}  (paper: > 0 when π̃>0,ν̃>0)")
print(f"  Mean credal width (= mean π̃):     {stats['mean_width']:.4f}")
print(f"  Partially identified examples:     {stats['pct_partially_identified']:.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  IFS Quality metrics (Table 6 analogue)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  IFS Dataset Quality Metrics (§3.3)")
print("=" * 65)
report = pipeline.quality_report()
for k, v in report.items():
    print(f"  {k:40s}: {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Annotator weight evolution (Figure 5 analogue)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Annotator Weight Evolution (Figure 5 – simulated refinement)")
print("=" * 65)

K = votes.shape[1]
annotator_names = ["Expert", "Novice-1", "Novice-2", "Length-biased", "Hesitant"]

# Simulate 3 rounds by artificially providing error rates that decrease for
# Expert and increase for Length-biased (mimicking RM-based discovery).
from ifs_rlhf.weighting import compute_weights
error_rate_progression = [
    None,
    np.array([0.08, 0.26, 0.37, 0.35, 0.30]),  # Round 1: length-bias partly discovered
    np.array([0.06, 0.25, 0.36, 0.55, 0.29]),  # Round 2: length-bias clearly flagged
    np.array([0.05, 0.24, 0.35, 0.60, 0.28]),  # Round 3: converged
]

print(f"\n{'Annotator':>15}", end="")
for r in range(4):
    print(f"  {'Round '+str(r):>8}", end="")
print()
print("-" * 55)

weight_history = []
for r, err in enumerate(error_rate_progression):
    w = compute_weights(
        pipeline.pi_matrix_,
        lambda1=0.8, lambda2=0.5,
        error_rates=err, lambda3=0.3,
    )
    weight_history.append(w)

for k in range(K):
    print(f"{annotator_names[k]:>15}", end="")
    for r in range(4):
        print(f"  {weight_history[r][k]:>8.4f}", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Hesitation distribution statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Hesitation Distribution (Figure 3 pattern)")
print("=" * 65)
pi = pipeline.pi_tilde_
print(f"  Mean π̃:               {pi.mean():.4f}")
print(f"  % with π̃ < 0.15:      {(pi < 0.15).mean():.2%}  (clear-consensus mode)")
print(f"  % with π̃ > 0.40:      {(pi > 0.40).mean():.2%}  (high-hesitation mode)")
corr = np.corrcoef(pi, np.abs(pipeline.soft_labels() - 0.5))[0, 1]
print(f"  Corr(π̃, |s − 0.5|):  {corr:+.3f}  (negative ↔ high π̃ = low confidence)")

print("\nDone.")
