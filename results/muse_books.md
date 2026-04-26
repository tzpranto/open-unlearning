# MUSE Books (Llama-2-7b-hf)

Updated: 2026-04-25

Gold target: forget_knowmem ≤ 0.303, retain ≥ 0.687

## Results (seed=42)

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain). Penalizes imbalance; any weak axis tanks the score. Follows TOFU's `hmean` convention for Model Utility.

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3029 | 0.1445 | 0.6874 | 0.0107 | 0.739 | — |
| Target (pre-unlearn) | 0.4712 | 0.9970 | 0.6913 | 0.9163 | 0.009 | — |
| GradAscent | 0.0000 | 0.0000 | 0.0000 | 0.0079 | 0.000 | 22m |
| GradDiff | 0.0000 | 0.0000 | 0.0231 | 0.0079 | 0.066 | 42m |
| NPO | 0.4021 | 0.0000 | 0.6130 | **0.0000** | 0.697 | 93m |
| SimNPO | 0.2707 | 0.2697 | 0.5802 | **0.0000** | 0.672 | 80m |
| RMU | 0.2022 | 0.1151 | 0.6014 | 0.0097 | 0.741 | 39m |
| BLURNPO | 0.4493 | 0.9970 | **0.6590** | 0.9160 | 0.009 | 102m |
| PDU | 0.0000 | 0.0031 | 0.0000 | 0.0079 | 0.000 | ~55m |
| BLADE | **0.1101** | **0.0000** | 0.6580 | 0.0079 | **0.823** | ~60m |

## LLM Judge (seed=42)

Scores on 0–2 scale. FL = forget leakage (lower = better forgetting). RA = retain accuracy, RQ = response quality (higher = better).

| Method | FL↓ | FL_know↓ | FL_verb↓ | RA↑ | RQ↑ | fgt_RQ | ret_RQ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradAscent | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| GradDiff | 0.00 | 0.00 | 0.00 | 0.05 | 0.02 | 0.00 | 0.06 | 0.040 |
| NPO | 0.76 | 0.76 | **0.00** | 1.31 | 1.62 | 1.47 | **1.77** | 0.703 |
| SimNPO | 0.55 | 0.56 | 0.54 | 1.26 | 1.29 | 1.11 | 1.64 | 0.717 |
| RMU | 0.17 | 0.34 | **0.00** | 1.33 | 0.95 | 0.57 | 1.72 | 0.798 |
| BLURNPO | 1.45 | 0.90 | 2.00 | **1.49** | **1.72** | **1.72** | 1.71 | 0.488 |
| PDU | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| BLADE | **0.14** | **0.27** | **0.00** | 1.40 | 0.76 | 0.30 | 1.69 | **0.814** |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold** = best result in column (among methods, excl. Gold/Target)
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: Books
