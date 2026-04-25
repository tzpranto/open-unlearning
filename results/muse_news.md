# MUSE News (Llama-2-7b-hf)

Updated: 2026-04-19

Gold target: forget_knowmem ≤ 0.324, retain ≥ 0.552

## Main Results

HM = harmonic mean of (1−forget_knowmem, 1−verbmem, retain). Penalizes imbalance; any weak axis tanks the score.

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3243 | 0.2042 | **0.5523** | 0.0247 | **0.660** | — |
| Target (pre-unlearn) | 0.6538 | 0.5693 | 0.5436 | 0.3023 | 0.426 | — |
| GradAscent | 0.0027 | 0.0489 | 0.0077 | 0.0079 | 0.023 | 34m |
| GradDiff | 0.3302 | 0.0053 | 0.2466 | 0.0079 | 0.458 | 1h 02m |
| NPO | 0.5173 | 0.3567 | 0.4195 | 0.0960 | 0.499 | 1h 05m |
| SimNPO | 0.5839 | 0.2585 | 0.4698 | 0.0516 | 0.510 | 59m |
| BLURNPO | 0.5806 | 0.3559 | 0.5318 | 0.1236 | 0.516 | ~2h |
| RMU | 0.5164 | 0.2848 | 0.4567 | 0.0552 | 0.530 | 20m |
| PDU | 0.4885 | 0.0423 | 0.4140 | 0.0138 | 0.554 | 54m |
| LoRA-BiAL+clampedEntropy (ours) | 0.5281 | 0.3875 | 0.5008 | 0.1296 | 0.522 | 71m |
