# MUSE News (Llama-2-7b-hf)

Updated: 2026-04-26

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
| GradDiff   | 0.43 | 0.86     | 0.00     | 0.56 | 0.74 | 0.53   | 1.18   | 0.459 |
| NPO        | 1.33 | 1.49     | 1.16     | 0.95 | 1.69 | 1.66   | 1.76   | 0.484 |
| SimNPO     | 1.33 | 1.46     | 1.19     | 0.91 | 1.65 | 1.64   | 1.69   | 0.473 |
| RMU        | 0.88 | 1.10     | 0.65     | 0.73 | 1.49 | 1.42   | 1.63   | 0.522 |
| BLURNPO    | 0.48 | 0.62     | 0.33     | 0.51 | 1.17 | 1.02   | 1.47   | 0.455 |
| PDU        | 0.59 | 1.18     | 0.00     | 0.76 | 0.95 | 0.65   | 1.54   | 0.561 |
| BLADE      | 0.86 | 1.35     | 0.36     | 0.88 | 1.11 | 0.81   | 1.72   | 0.579 |


## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold** = best result in column (among methods, excl. Gold/Target)
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: News

