# IFS-RLHF
---
## Overview

Standard RLHF pipelines reduce multi-annotator preference judgments to binary labels via majority vote, discarding annotator hesitation, confidence, and disagreement. **IFS-RLHF** recasts preference labeling as an *intuitionistic fuzzy group decision-making* problem.

Each annotator's judgment is encoded as an **Intuitionistic Fuzzy Number (IFN)** with three components:

| Component | Symbol | Meaning |
|-----------|--------|---------|
| Membership | μ | Support for response A |
| Non-membership | ν | Support for response B |
| Hesitation | π = 1−μ−ν | Epistemic uncertainty |

Key properties proven in the paper and verified in tests:

- **Proposition 3 (Identification Consistency):** IFS soft labels always lie inside the Credal Reward Set `C = [μ̃, 1−ν̃]`; hard labels provably do not when π̃ > 0.
- **Gradient Proposition:** The IFS-weighted loss fixed point is the soft label `s = μ̃/(μ̃+ν̃)`, not the hard label.
- **Proposition 2 (Convergence):** Iterative annotator weight refinement converges geometrically under mild Lipschitz conditions.

---

## Project Structure

```
ifs-rlhf/
├── ifs_rlhf/
│   ├── ifn.py          # IFN dataclass + annotation→IFN mapping (§4.3, Eq. 1)
│   ├── aggregation.py  # IFWA operator, vectorised batch version (Def. 3, Eqs. 3–5)
│   ├── weighting.py    # Annotator reliability weights + iterative refinement (§4.4)
│   ├── loss.py         # IFS-weighted RM loss + IFS-DPO loss (§4.5–4.6)
│   ├── credal.py       # Credal Reward Sets, identification consistency (§3.4)
│   ├── metrics.py      # Conf, Clarity, IFS-IAA (§3.3)
│   └── pipeline.py     # Full Algorithm 1 pipeline
├── scripts/
│   └── simulate_annotators.py  # Five synthetic annotator types (§5.1)
├── examples/
│   └── demo.py         # End-to-end demo reproducing key paper patterns
└── tests/
    └── test_core.py    # 38 unit tests covering all components
```

---

## Quick Start

```bash
pip install -r requirements.txt
python examples/demo.py
```

---

## Core API

### 1. IFN Mapping (Paper §4.3, Eq. 1)

```python
from ifs_rlhf import map_annotation_to_ifn

# From a human annotator's vote + confidence
ifn = map_annotation_to_ifn(vote="A", confidence=0.85)
print(ifn)          # IFN(μ=0.850, ν=0.050, π=0.100)
print(ifn.soft_label)   # μ/(μ+ν) = 0.944

# From LLM judge token probabilities (RLAIF, §4.2)
from ifs_rlhf import logit_confidence_from_lm
conf, hesitation = logit_confidence_from_lm(prob_A=0.60, prob_B=0.30)
# hesitation = 1 − P(A) − P(B) = 0.10

# From response time (behavioral proxy, Eq. 2)
from ifs_rlhf import response_time_to_confidence
conf = response_time_to_confidence(response_time_s=15.0, baseline_s=30.0)
```

### 2. IFWA Aggregation (Paper Def. 3, Eqs. 3–5)

```python
from ifs_rlhf import IFN, ifwa
import numpy as np

annotator_ifns = [IFN(0.8, 0.1), IFN(0.6, 0.2), IFN(0.4, 0.4)]
weights = np.array([0.5, 0.3, 0.2])   # reliability-based

agg = ifwa(annotator_ifns, weights=weights)
print(agg)   # IFN(μ̃, ν̃, π̃) – aggregated over annotators
```

### 3. Annotator Reliability Weighting (Paper §4.4, Eqs. 6–7)

```python
from ifs_rlhf import compute_weights

# pi_matrix: (N, K) hesitation degrees from IFN mapping
weights = compute_weights(pi_matrix, lambda1=0.8, lambda2=0.5)
# weights[k] ∝ exp(−λ₁ π̄_k − λ₂ σ_k²)
```

### 4. IFS-Weighted Loss (Paper §4.5)

```python
import torch
from ifs_rlhf import IFSRankingLoss

loss_fn = IFSRankingLoss(gamma=2.0)

# delta_r = r_φ(x, y_A) − r_φ(x, y_B)
loss = loss_fn(delta_r, mu_tilde, nu_tilde, pi_tilde)
# L_IFS = −(1/N) Σ (1−π̃)^γ [μ̃ log σ(Δr) + ν̃ log(1−σ(Δr))]
```

### 5. IFS-DPO Loss (Paper §4.6, Eq. 8)

```python
from ifs_rlhf import IFSDPOLoss

dpo_fn = IFSDPOLoss(beta=0.1, gamma=1.0)   # γ=1 recommended for DPO
loss = dpo_fn(log_ratio_A, log_ratio_B, mu_tilde, nu_tilde, pi_tilde)
```

### 6. Full Pipeline (Algorithm 1)

```python
from ifs_rlhf import IFSRLHFPipeline, IFSRLHFConfig, PreferenceExample, AnnotationRecord

examples = [
    PreferenceExample(
        example_id=n,
        annotations=[
            AnnotationRecord(annotator_id=0, vote="A", confidence=0.9),
            AnnotationRecord(annotator_id=1, vote="A", confidence=0.6),
            AnnotationRecord(annotator_id=2, vote="B", confidence=0.7),
        ]
    )
    for n in range(N)
]

pipeline = IFSRLHFPipeline(IFSRLHFConfig())
pipeline.fit(examples)

soft = pipeline.soft_labels()        # (N,) ∈ [0,1]
hard = pipeline.hard_labels()        # (N,) ∈ {0,1}
weights = pipeline.instance_weights(gamma=2.0)  # (N,) = (1−π̃)^γ

print(pipeline.quality_report())
# {'Conf': ..., 'Clarity': ..., 'IFS-IAA': ...}
print(pipeline.consistency_report())
# {'soft_pct_consistent': 1.0, 'hard_pct_inconsistent': ...}
```

---

## Connecting to RM Training

The pipeline outputs are drop-in replacements for hard labels in any standard training loop:

```python
# Standard Bradley-Terry training (baseline)
loss_bt = F.binary_cross_entropy_with_logits(delta_r, hard_labels.float())

# IFS-RLHF training (ours)
loss_fn = IFSRankingLoss(gamma=2.0)
loss_ifs = loss_fn(delta_r, mu_tilde_t, nu_tilde_t, pi_tilde_t)
loss_total = loss_ifs + beta * kl_regularisation(reward, reward_ref)
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta` | 0.05 | Skepticism floor in IFN mapping (prevents ν=0 collapse) |
| `lambda1` | 0.8 | Mean-hesitation weight in reliability scoring |
| `lambda2` | 0.5 | Hesitation-variance weight |
| `lambda3` | 0.3 | RM error-rate weight (Phase 5 refinement) |
| `gamma` | 2.0 | Focusing exponent (RM loss); 1.0 recommended for DPO |
| `beta` | 0.01 | KL regularisation coefficient |
| `tau` | 0.15 | Hesitation threshold for clean subset D_clean |

---

## Tests

```bash
python -m pytest tests/test_core.py -v
# 38 tests covering IFN, IFWA, Credal Sets, Losses, Weighting, Pipeline
```

Key tests mirror paper propositions:
- `test_soft_label_inside_credal_set` — verifies Proposition 3, Part 1 over a grid
- `test_hard_label_outside_credal_set` — verifies Proposition 3, Part 2
- `test_gradient_fixed_point` — verifies the Gradient Proposition (loss minimised at soft label)
- `test_reliable_annotator_gets_higher_weight` — verifies reliability weighting direction
- `test_exact_idempotency_no_smoothing` — verifies IFWA Proposition 1

---

## Citation

```bibtex

```
