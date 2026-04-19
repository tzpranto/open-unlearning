# MUSE Books (Llama-2-7b-hf)

Updated: 2026-04-17

Gold target: forget_knowmem ≤ 0.303, retain ≥ 0.687

## Results

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain). Penalizes imbalance; any weak axis tanks the score. Follows TOFU's `hmean` convention for Model Utility.

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3029 | 0.1445 | **0.6874** | 0.0107 | **0.739** | — |
| Target (pre-unlearn) | 0.4712 | 0.9970 | 0.6913 | 0.9163 | 0.009 | — |
| GradAscent | 0.0000 | 0.0000 | 0.0000 | 0.0079 | 0.000 | 43m |
| GradDiff | 0.0000 | 0.0000 | 0.0245 | 0.0079 | 0.070 | 52m |
| NPO | 0.4114 | 0.5697 | **0.6612** | 0.3774 | 0.542 | 1h 29m |
| SimNPO | 0.2388 | 0.0047 | **0.6036** | 0.0092 | **0.755** | 1h 23m |
| BLURNPO | 0.3109 | 0.8087 | **0.6000** | 0.7023 | 0.359 | 1h 45m |
| RMU | 0.3084 | 0.1201 | **0.6036** | 0.0111 | 0.708 | 20m |
| PDU | 0.0000 | 0.0020 | 0.0000 | 0.0079 | 0.000 | 1h 16m |
| DS-BiAL (ours) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold retain** = within 15% of gold retrain
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: Books
