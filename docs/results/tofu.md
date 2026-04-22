# TOFU Baselines

Updated: 2026-04-22

All methods: 10 epochs, lr=1e-5, bs=8, accum=1 (eff_bs=8), gradient checkpointing.

---

## Llama-3.2-1B-Instruct

### Forget 1% (forget01 / retain99)

| Method | MU↑ | FQ↑ | ES↓ | fgt_Prob↓ | fgt_ROUGE↓ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.597 | 1.000 | 0.069 | — | — | — |
| Target (pre-unlearn) | 0.599 | 0.007 | 0.724 | — | — | — |
| GradAscent | 0.394 | 0.578 | 0.046 | 0.008 | 0.273 | 0.610 |
| GradDiff | 0.488 | 0.266 | 0.063 | 0.056 | 0.357 | 0.643 |
| NPO | 0.583 | 0.266 | 0.091 | 0.100 | 0.333 | 0.694 |
| SimNPO | 0.597 | 0.007 | 0.421 | 0.804 | 0.694 | 0.299 |
| RMU | 0.584 | 0.766 | 0.039 | 0.113 | 0.275 | 0.711 |
| BLURNPO | 0.589 | 0.097 | 0.119 | 0.261 | 0.391 | 0.639 |
| PDU | 0.607 | 0.054 | 0.030 | 0.003 | 0.066 | 0.806 |
| LoRA-BiAL (ours) | — | — | — | — | — | ⏳ |

---

## Llama-3.2-3B-Instruct

### Forget 1% (forget01 / retain99)

| Method | MU↑ | FQ↑ | ES↓ | fgt_Prob↓ | fgt_ROUGE↓ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.663 | 1.000 | 0.067 | — | — | — |
| Target (pre-unlearn) | 0.665 | 0.014 | 0.920 | — | — | — |
| NPO | 0.655 | 0.766 | 0.116 | 0.133 | 0.365 | 0.705 |

## Notes

- MU = Model Utility: hmean of retain/RA/WF x Prob/ROUGE/TruthRatio (9 sub-metrics)
- FQ = Forget Quality: KS test p-value on truth ratios (forget vs retain logs)
- ES = Extraction Strength (lower = better forgetting)
- fgt_Prob = forget set answer probability, fgt_ROUGE = forget set ROUGE-L
- HM = hmean(MU, 1-fgt_Prob, 1-fgt_ROUGE) — overall score
- **bold HM** = best in split
