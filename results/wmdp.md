# WMDP (Zephyr-7b-beta)

Generated: 2026-04-04 21:02

## Results

| Method | wmdp_cyber↓ | mmlu_avg↑ | train_time |
| --- | --- | --- | --- |
| GradAscent | ⏳ | ⏳ | ⏳ |
| GradDiff | ⏳ | ⏳ | ⏳ |
| NPO | ⏳ | ⏳ | ⏳ |
| BLURNPO | ⏳ | ⏳ | ⏳ |
| RMU | ⏳ | ⏳ | ⏳ |
| DS-BiAL | ⏳ | ⏳ | ⏳ |

## Notes

- `wmdp_cyber↓`: WMDP-cyber accuracy (lower = more forget of hazardous knowledge)
- `mmlu_avg↑`: MMLU average accuracy (higher = better retain of general knowledge)
- No reference retrain model needed (MCQ-based evaluation)
- `⏳` = run in queue / not yet evaluated
