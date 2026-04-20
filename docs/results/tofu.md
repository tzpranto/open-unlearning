# TOFU Baselines

Updated: 2026-04-19

MU = Model Utility (hmean of 9 sub-metrics), FQ = Forget Quality (KS p-value), HM = hmean(MU, FQ)

All methods: 10 epochs, lr=1e-5, gradient checkpointing, eff_bs=32. PDU: alpha=100, eps=0.3, dual_step_size=5, warmup=5.

---

## Llama-3.2-1B-Instruct

### Forget 1% (forget01 / retain99)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.597 | 1.000 | **0.748** | 0.069 |
| Target (pre-unlearn) | 0.599 | 0.007 | 0.013 | 0.724 |
| GradAscent | 0.485 | 0.405 | 0.441 | 0.086 |
| GradDiff | 0.487 | 0.165 | 0.246 | 0.059 |
| NPO | 0.580 | 0.579 | **0.579** | 0.094 |
| SimNPO | 0.598 | 0.007 | 0.013 | 0.455 |
| RMU | 0.578 | 0.165 | 0.257 | 0.068 |
| BLURNPO | 0.598 | 0.007 | 0.013 | 0.427 |
| PDU | 0.585 | 0.766 | **0.663** | 0.038 |

### Forget 5% (forget05 / retain95)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.599 | 1.000 | **0.750** | 0.063 |
| Target (pre-unlearn) | 0.599 | 0.000 | 0.000 | 0.729 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.033 |
| GradDiff | 0.459 | 0.000 | 0.000 | 0.070 |
| NPO | 0.527 | 0.001 | 0.003 | 0.087 |
| SimNPO | 0.595 | 0.000 | 0.000 | 0.491 |
| RMU | 0.582 | 0.178 | **0.272** | 0.033 |
| BLURNPO | 0.381 | 0.000 | 0.000 | 0.094 |
| PDU | 0.591 | 0.000 | 0.000 | 0.033 |

### Forget 10% (forget10 / retain90)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.591 | 1.000 | **0.743** | 0.059 |
| Target (pre-unlearn) | 0.599 | 0.000 | 0.000 | 0.708 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.033 |
| GradDiff | 0.316 | 0.000 | 0.000 | 0.033 |
| NPO | 0.572 | 0.000 | 0.000 | 0.135 |
| SimNPO | 0.596 | 0.000 | 0.000 | 0.464 |
| RMU | 0.586 | 0.001 | 0.001 | 0.036 |
| BLURNPO | 0.016 | 0.000 | 0.000 | 0.044 |
| PDU | 0.595 | 0.000 | 0.000 | 0.035 |

---

## Llama-3.2-3B-Instruct

### Forget 1% (forget01 / retain99)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.663 | 1.000 | **0.798** | 0.067 |
| Target (pre-unlearn) | 0.665 | 0.014 | 0.028 | 0.920 |
| GradAscent | 0.639 | 0.579 | 0.607 | 0.135 |
| GradDiff | 0.594 | 0.097 | 0.167 | 0.151 |
| NPO | 0.655 | 0.579 | **0.615** | 0.126 |
| SimNPO | 0.649 | 0.029 | 0.055 | 0.535 |
| RMU | 0.657 | 0.405 | 0.501 | 0.128 |
| BLURNPO | 0.674 | 0.007 | 0.013 | 0.556 |
| PDU | 0.681 | 0.014 | 0.028 | 0.035 |

### Forget 5% (forget05 / retain95)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.659 | 1.000 | **0.795** | 0.061 |
| Target (pre-unlearn) | 0.665 | 0.000 | 0.000 | 0.887 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.033 |
| GradDiff | 0.274 | 0.000 | 0.000 | 0.039 |
| NPO | 0.617 | 0.040 | **0.074** | 0.068 |
| SimNPO | 0.651 | 0.000 | 0.000 | 0.601 |
| RMU | 0.661 | 0.713 | **0.686** | 0.070 |
| BLURNPO | 0.537 | 0.000 | 0.000 | 0.196 |
| PDU | 0.683 | 0.000 | 0.000 | 0.033 |

### Forget 10% (forget10 / retain90)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.650 | 1.000 | **0.788** | 0.065 |
| Target (pre-unlearn) | 0.665 | 0.000 | 0.000 | 0.890 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.033 |
| GradDiff | 0.587 | 0.000 | 0.000 | 0.033 |
| NPO | 0.686 | 0.000 | 0.000 | 0.066 |
| SimNPO | 0.648 | 0.000 | 0.000 | 0.536 |
| RMU | 0.666 | 0.001 | 0.002 | 0.041 |
| BLURNPO | 0.196 | 0.000 | 0.000 | 0.056 |
| PDU | 0.672 | 0.000 | 0.000 | 0.033 |

---

## Llama-2-7b-chat-hf

### Forget 1% (forget01 / retain99)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.627 | 1.000 | **0.771** | 0.072 |
| Target (pre-unlearn) | 0.628 | 0.001 | 0.003 | 1.000 |
| GradAscent | 0.584 | 0.097 | 0.167 | 0.111 |
| GradDiff | 0.597 | 0.007 | 0.013 | 0.136 |
| NPO | 0.613 | 0.014 | 0.028 | 0.132 |
| SimNPO | 0.620 | 0.003 | 0.006 | 0.710 |
| RMU | 0.626 | 0.007 | 0.013 | 0.991 |
| BLURNPO | 0.627 | 0.029 | 0.055 | 0.646 |
| PDU | 0.642 | 0.000 | 0.000 | 0.024 |

### Forget 5% (forget05 / retain95)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.627 | 1.000 | **0.770** | 0.068 |
| Target (pre-unlearn) | 0.628 | 0.000 | 0.000 | 0.981 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.027 |
| GradDiff | 0.557 | 0.000 | 0.000 | 0.099 |
| NPO | 0.575 | 0.000 | 0.000 | 0.272 |
| SimNPO | 0.614 | 0.000 | 0.000 | 0.620 |
| RMU | 0.615 | 0.000 | 0.000 | 0.798 |
| BLURNPO | 0.580 | 0.000 | 0.000 | 0.440 |
| PDU | 0.659 | 0.000 | 0.000 | 0.031 |

### Forget 10% (forget10 / retain90)

| Method | MU↑ | FQ↑ | HM↑ | ES↓ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.613 | 1.000 | **0.760** | 0.070 |
| Target (pre-unlearn) | 0.628 | 0.000 | 0.000 | 0.982 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.027 |
| GradDiff | 0.555 | 0.000 | 0.000 | 0.027 |
| NPO | 0.596 | 0.000 | 0.000 | 0.096 |
| SimNPO | 0.610 | 0.000 | 0.000 | 0.553 |
| RMU | 0.610 | 0.000 | 0.000 | 0.766 |
| BLURNPO | 0.538 | 0.000 | 0.000 | 0.290 |
| PDU | 0.660 | 0.000 | 0.000 | 0.030 |

## Notes

- MU = Model Utility: hmean of retain/RA/WF x Prob/ROUGE/TruthRatio (9 sub-metrics)
- FQ = Forget Quality: KS test p-value on truth ratios (forget vs retain logs)
- HM = hmean(MU, FQ) — overall score
- ES = Extraction Strength (lower = better forgetting)
- **bold HM** = best in split
