# MUSE News (Llama-2-7b-hf)

Updated: 2026-05-08

Gold target: forget_knowmem ≤ 0.324, retain ≥ 0.552

## Results (seed=42)

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain). Penalizes imbalance; any weak axis tanks the score. Follows TOFU's `hmean` convention for Model Utility.


| Method               | forget_knowmem↓ | verbmem↓ | retain↑    | extract↓ | HM↑       | train_time |
| -------------------- | --------------- | -------- | ---------- | -------- | --------- | ---------- |
| Gold (retrain)       | 0.3243          | 0.2042   | **0.5523** | 0.0247   | **0.660** | —          |
| Target (pre-unlearn) | 0.6538          | 0.5693   | 0.5436     | 0.3023   | 0.426     | —          |
| GradAscent           | 0.0027          | 0.0489   | 0.0077     | 0.0079   | 0.023     | 34m        |
| GradDiff             | 0.3555          | 0.0507   | 0.2814     | 0.0079   | 0.487     | ~1h        |
| NPO                  | 0.6593          | 0.4955   | 0.5269     | 0.2244   | 0.440     | ~1h 30m    |
| SimNPO               | 0.6510          | 0.5488   | 0.4826     | 0.2757   | 0.419     | ~1h 30m    |
| RMU                  | 0.5048          | 0.2740   | 0.4353     | 0.0504   | 0.527     | ~30m       |
| BLURNPO              | 0.3170          | 0.1809   | 0.3036     | 0.0267   | 0.448     | ~1h        |
| PDU                  | 0.5158          | 0.0019   | 0.4774     | 0.0082   | 0.581     | ~1h        |
| BLADE                | 0.5500          | 0.1728   | 0.4750     | 0.0312   | 0.542     | ~3.3h      |


## LLM Judge (seed=42)

Scores on 0–2 scale. FL = forget leakage (lower = better forgetting). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1−FL/2, RA/2, ret_RQ/2).


| Method     | FL↓  | FL_know↓ | FL_verb↓ | RA↑  | RQ↑  | fgt_RQ | ret_RQ | HM↑   |
| ---------- | ---- | -------- | -------- | ---- | ---- | ------ | ------ | ----- |
| GradAscent | 0.00 | 0.00     | 0.00     | 0.00 | 0.00 | 0.00   | 0.00   | 0.000 |
| GradDiff   | 0.59 | 1.18     | 0.01     | 0.86 | 0.94 | 0.80   | 1.24   | 0.559 |
| NPO        | 1.25 | 1.44     | 1.05     | 1.05 | 1.81 | 1.93   | 1.59   | 0.516 |
| SimNPO     | 1.49 | 1.60     | 1.38     | 1.23 | 1.84 | 1.90   | 1.71   | 0.447 |
| RMU        | 1.13 | 1.27     | 0.99     | 1.14 | 1.74 | 1.82   | 1.59   | 0.565 |
| BLURNPO    | 0.47 | 0.62     | 0.33     | 0.51 | 1.17 | 1.02   | 1.47   | 0.455 |
| PDU        | 0.84 | 1.42     | 0.27     | 1.21 | 1.27 | 1.06   | 1.68   | 0.656 |
| BLADE      | 0.85 | 1.35     | 0.36     | 0.88 | 1.11 | 0.81   | 1.72   | 0.579 |


## Results (5-fold, seeds=42,123,456,789,1024)

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain).

| Method               | forget_knowmem↓ | verbmem↓        | retain↑         | extract↓        | HM↑             |
| -------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Gold (retrain)       | 0.3243          | 0.2042          | 0.5523          | 0.0247          | 0.660           |
| Target (pre-unlearn) | 0.6538          | 0.5693          | 0.5436          | 0.3023          | 0.426           |
| GradAscent           | 0.001 ± 0.001  | 0.020 ± 0.024  | 0.003 ± 0.004  | 0.008 ± 0.000  | 0.009 ± 0.011  |
| GradDiff             | 0.329 ± 0.026  | 0.043 ± 0.019  | 0.269 ± 0.018  | 0.008 ± 0.000  | 0.479 ± 0.017  |
| NPO                  | 0.517 ± 0.023  | 0.274 ± 0.018  | 0.435 ± 0.014  | 0.073 ± 0.007  | 0.522 ± 0.014  |
| SimNPO               | 0.628 ± 0.008  | 0.542 ± 0.010  | 0.513 ± 0.006  | 0.281 ± 0.005  | 0.440 ± 0.003  |
| RMU                  | 0.495 ± 0.008  | 0.267 ± 0.007  | 0.434 ± 0.005  | 0.050 ± 0.003  | 0.531 ± 0.003  |
| PDU                  | 0.525 ± 0.020  | 0.092 ± 0.066  | 0.508 ± 0.037  | 0.024 ± 0.016  | 0.577 ± 0.006  |
| BLURNPO              | —               | —               | —               | —               | — (OOM 4×40GB)  |
| BLADE                | 0.545 ± 0.022  | **0.211 ± 0.034** | 0.490 ± 0.012  | 0.050 ± 0.014  | 0.544 ± 0.014  |

## LLM Judge (5-fold, seeds=42,123,456,789,1024)

Scores on 0–2 scale. FL = forget leakage (lower = better). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1−FL/2, RA/2, ret_RQ/2).

| Method     | FL↓           | RA↑           | ret_RQ↑       | HM↑           |
| ---------- | ------------- | ------------- | ------------- | ------------- |
| GradAscent | 0.00 ± 0.00  | 0.00 ± 0.00  | 0.00 ± 0.00  | 0.000 ± 0.000 |
| GradDiff   | 0.58 ± 0.02  | 0.79 ± 0.04  | 1.16 ± 0.14  | 0.529 ± 0.030 |
| NPO        | 1.20 ± 0.03  | 1.05 ± 0.02  | 1.58 ± 0.01  | 0.531 ± 0.011 |
| SimNPO     | 1.49 ± 0.01  | 1.24 ± 0.01  | 1.72 ± 0.01  | 0.447 ± 0.007 |
| RMU        | 1.13 ± 0.01  | 1.15 ± 0.02  | 1.60 ± 0.01  | 0.565 ± 0.004 |
| PDU        | 0.87 ± 0.12  | 1.18 ± 0.05  | 1.64 ± 0.05  | 0.638 ± 0.019 |
| BLURNPO    | —             | —             | —             | — (OOM 4×40GB) |
| BLADE      | ⏳             | ⏳             | ⏳             | ⏳ (2/5 judged) |


## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold** = best result in column (among methods, excl. Gold/Target)
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: News

