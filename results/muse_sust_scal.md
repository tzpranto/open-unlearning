# MUSE Sustainability & Scalability (Llama-2-7b-hf, News)

Updated: 2026-05-04

## Scalability

Single-shot unlearning on progressively larger forget sets (1x, 2x, 3x, 4x the standard 889 samples).

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain).

| Scale | forget_size | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| scal/forget_1 (=seed42) | 889 | 0.542 | 0.065 | 0.522 | 0.015 | 0.581 |
| scal/forget_2 | 1778 | 0.495 | 0.050 | 0.467 | 0.016 | 0.579 |
| scal/forget_3 | 2667 | 0.538 | 0.024 | 0.488 | 0.015 | 0.573 |
| scal/forget_4 | 3554 | 0.321 | 0.021 | 0.002 | 0.014 | 0.005 |

## Sustainability

Sequential unlearning: each step applies PDU on a new 889-sample forget set, starting from the previous step's checkpoint (5-fold seed=42 model).

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain).

| Step | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- |
| step 1 (=seed42) | 0.542 | 0.065 | 0.522 | 0.015 | 0.581 |
| step 2 (forget_2) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| step 3 (forget_3) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| step 4 (forget_4) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

## LLM Judge — Scalability

FL = forget leakage (lower = better). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1-FL/2, RA/2, ret_RQ/2).

| Scale | FL↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- |
| scal/forget_1 (=seed42) | 0.84 | 1.21 | 1.68 | 0.656 |
| scal/forget_2 | 0.75 | 1.08 | 1.62 | 0.640 |
| scal/forget_3 | 0.74 | 1.22 | 1.72 | 0.683 |
| scal/forget_4 | 0.53 | 0.02 | 0.02 | 0.015 |

## LLM Judge — Sustainability

| Step | FL↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- |
| step 1 (=seed42) | 0.84 | 1.21 | 1.68 | 0.656 |
| step 2 (forget_2) | ⏳ | ⏳ | ⏳ | ⏳ |
| step 3 (forget_3) | ⏳ | ⏳ | ⏳ | ⏳ |
| step 4 (forget_4) | ⏳ | ⏳ | ⏳ | ⏳ |

## Notes

- All runs use seed=42, bsz=2, accum=16, gradient_checkpointing=true
- Model: Llama-2-7b-hf, Data: MUSE News
- Gold (retrain): forget_knowmem=0.324, retain=0.552, HM=0.660
- Scalability tests if method handles larger forget sets in one shot
- Sustainability tests if method can be applied repeatedly without destroying the model
- scal/forget_1 and sust step 1 use the canonical 5-fold seed=42 PDU result directly
- Sustainability v2: steps 2-4 chain from the 5-fold s42 checkpoint (not a fresh run)
- Scalability: retain collapses at 4x scale (3554 samples) but holds at 1-3x
