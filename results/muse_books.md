# MUSE Books (Llama-2-7b-hf)

Updated: 2026-05-09 (BLURNPO 5-fold complete)

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
| BLURNPO | 0.1749 | 0.0004 | 0.5266 | 0.0079 | 0.730 | 96m |
| PDU | 0.1301 | 0.1232 | 0.3711 | 0.0379 | 0.602 | ~6m |
| BLADE | **0.1101** | **0.0000** | 0.6580 | 0.0079 | **0.823** | ~60m |

## LLM Judge (seed=42)

Scores on 0–2 scale. FL = forget leakage (lower = better forgetting). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1−FL/2, RA/2, ret_RQ/2).

| Method | FL↓ | FL_know↓ | FL_verb↓ | RA↑ | RQ↑ | fgt_RQ | ret_RQ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradAscent | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| GradDiff | 0.00 | 0.00 | 0.00 | 0.05 | 0.02 | 0.00 | 0.06 | 0.040 |
| NPO | 0.76 | 0.76 | **0.00** | 1.31 | 1.62 | 1.47 | **1.77** | 0.703 |
| SimNPO | 0.55 | 0.56 | 0.54 | 1.26 | 1.29 | 1.11 | 1.64 | 0.717 |
| RMU | 0.17 | 0.34 | **0.00** | 1.33 | 0.95 | 0.57 | 1.72 | 0.798 |
| BLURNPO | 0.21 | 0.41 | **0.00** | 1.09 | 0.66 | 0.26 | 1.47 | 0.696 |
| PDU | 0.28 | 0.38 | 0.18 | 0.92 | 0.74 | 0.48 | 1.28 | 0.612 |
| BLADE | **0.14** | **0.27** | **0.00** | 1.40 | 0.76 | 0.30 | 1.69 | **0.814** |

## Results (5-fold, seeds=42,123,456,789,1024)

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain).

| Method               | forget_knowmem↓ | verbmem↓        | retain↑         | extract↓        | HM↑             |
| -------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Gold (retrain)       | 0.3029          | 0.1445          | 0.6874          | 0.0107          | 0.739           |
| Target (pre-unlearn) | 0.4712          | 0.9970          | 0.6913          | 0.9163          | 0.009           |
| GradAscent           | 0.000 ± 0.000  | 0.000 ± 0.000  | 0.000 ± 0.000  | 0.008 ± 0.000  | 0.000 ± 0.000  |
| GradDiff             | 0.000 ± 0.000  | 0.000 ± 0.000  | 0.004 ± 0.002  | 0.005 ± 0.004  | 0.011 ± 0.006  |
| NPO                  | 0.303 ± 0.020  | 0.342 ± 0.027  | 0.574 ± 0.015  | 0.168 ± 0.027  | 0.638 ± 0.009  |
| SimNPO               | 0.238 ± 0.016  | 0.002 ± 0.001  | 0.600 ± 0.007  | 0.007 ± 0.003  | 0.753 ± 0.007  |
| RMU                  | 0.210 ± 0.003  | 0.113 ± 0.006  | 0.598 ± 0.006  | 0.009 ± 0.000  | 0.738 ± 0.002  |
| BLURNPO              | 0.175 ± 0.011  | 0.000 ± 0.000  | 0.547 ± 0.020  | 0.008 ± 0.000  | 0.742 ± 0.013  |
| PDU                  | 0.139 ± 0.014  | 0.129 ± 0.007  | 0.372 ± 0.007  | 0.040 ± 0.002  | 0.600 ± 0.008  |
| BLADE                | **0.089 ± 0.018** | **0.000 ± 0.000** | 0.638 ± 0.012  | 0.008 ± 0.000  | **0.818 ± 0.006** |

## LLM Judge (5-fold, seeds=42,123,456,789,1024)

Scores on 0–2 scale. FL = forget leakage (lower = better). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1−FL/2, RA/2, ret_RQ/2).

| Method     | FL↓           | RA↑           | ret_RQ↑       | HM↑           |
| ---------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent | 0.00 ± 0.00  | 0.00 ± 0.00  | 0.00 ± 0.00  | 0.000 ± 0.000 |
| GradDiff   | 0.00 ± 0.00  | 0.04 ± 0.02  | 0.01 ± 0.01  | 0.010 ± 0.009 |
| NPO        | 0.91 ± 0.03  | 1.28 ± 0.03  | 1.61 ± 0.01  | 0.647 ± 0.008 |
| SimNPO     | 0.31 ± 0.12  | 1.28 ± 0.02  | 1.62 ± 0.01  | 0.753 ± 0.020 |
| RMU        | 0.32 ± 0.02  | 1.41 ± 0.02  | 1.68 ± 0.01  | 0.791 ± 0.004 |
| BLURNPO    | 0.20 ± 0.02  | 1.15 ± 0.06  | 1.52 ± 0.07  | 0.720 ± 0.023 |
| PDU        | 0.53 ± 0.02  | 0.99 ± 0.02  | 1.15 ± 0.02  | 0.586 ± 0.007 |
| BLADE      | **0.11 ± 0.02** | **1.39 ± 0.01** | **1.66 ± 0.02** | **0.810 ± 0.005** |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold** = best result in column (among methods, excl. Gold/Target)
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: Books
