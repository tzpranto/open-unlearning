# MUSE News (Llama-2-7b-hf)

Updated: 2026-04-26

Gold target: forget_knowmem ≤ 0.324, retain ≥ 0.552

## Results (seed=42)

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain). Penalizes imbalance; any weak axis tanks the score. Follows TOFU's `hmean` convention for Model Utility.

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3243 | 0.2042 | **0.5523** | 0.0247 | **0.660** | — |
| Target (pre-unlearn) | 0.6538 | 0.5693 | 0.5436 | 0.3023 | 0.426 | — |
| GradAscent | 0.0027 | 0.0489 | 0.0077 | 0.0079 | 0.023 | 34m |
| GradDiff | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — |
| NPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — |
| SimNPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — |
| RMU | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — |
| BLURNPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — |
| PDU | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold** = best result in column (among methods, excl. Gold/Target)
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: News
