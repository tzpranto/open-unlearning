# WMDP-Cyber (Zephyr-7b-beta)

Updated: 2026-04-26

Baseline (no unlearning): wmdp_cyber=0.431, mmlu=0.589. Random chance: 0.250.

## Results (seed=42)

| Method | wmdp_cyber↓ | mmlu_avg↑ | train_time |
| --- | --- | --- | --- |
| Baseline (no unlearn) | 0.4308 | 0.5888 | — |
| GradAscent | ⏳ | ⏳ | — |
| GradDiff | ⏳ | ⏳ | — |
| NPO | ⏳ | ⏳ | — |
| SimNPO | ⏳ | ⏳ | — |
| RMU | ⏳ | ⏳ | — |
| BLURNPO | ⏳ | ⏳ | — |

## Notes

- `wmdp_cyber↓` = WMDP-Cyber MCQ accuracy (lower = better forgetting of hazardous knowledge)
- `mmlu_avg↑` = MMLU average accuracy (higher = better retain of general knowledge)
- Random chance on wmdp_cyber = 0.250
- No retain logs needed — evaluation is MCQ-based (lm_eval harness)
- `⏳` = run in queue / not yet evaluated
- **bold** = best result in column (among methods, excl. Baseline)
- RMU params from arXiv:2403.03218 (WMDP paper): steering_coeff=2, layers.5-7.mlp.down_proj
